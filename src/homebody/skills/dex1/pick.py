"""The pick skill: walk in when needed, plan a grasp from visible depth, approach, close, lift,
verify the hold and defer transport tucking until navigation."""
import json
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from homebody.helpers.collision import HAND_END_LINKS, Clearance
from homebody.helpers.geometry import inverse, points_in, rotation_error, transform, wrap
from homebody.helpers.grasp import Shape, closing_angles, visible_shape
from homebody.helpers.target import TargetTrack, arm_occlusion_evidence
from homebody.helpers.trajectory import approach_path, servo_step
from homebody.motion.arm import close_or_open, move_path, reset_arm, servo
from homebody.motion.carry import (
    RECOVERY_FAILURES,
    RecoveryPlan,
    recover,
    recovery_result,
)
from homebody.motion.carry import (
    begin_pending as register_pending,
)
from homebody.primitives.observations import Frame, FreshEvidence, ObservationError
from homebody.skills import navigate
from homebody.skills.contract import Result, SkillFailure, click_argument, side_argument
from homebody.skills.dex1.clearance import servo_best_effort
from homebody.skills.dex1.transport import defer_compaction

NEEDS = ("base_velocity", "arm_target", "grip", "advance", "stop")

FAILURES = ("HAND_OCCUPIED", "CALIBRATION_MISSING", "NO_GRASP", "UNREACHABLE",
            "TARGET_MOVED", "ARM_NOT_REACHED", "LIFT_NOT_VERIFIED") + RECOVERY_FAILURES + navigate.FAILURES
PROMPT = ('pick: point_normalized_1000 [x, y], a pixel in the middle of the object\'s body in image 1; '
          'hand "right" (default) or "left". Walks to a grasp stance when needed, grasps the object and '
          'lifts it into a carry pose in front of the body; OK means it is held. A held object keeps the '
          'robot farther from furniture. With both hands loaded the robot may be unable to turn or back '
          'out, and a held object can be dragged out of the hand.')
RETRIED_APPROACH_FAILURES = ("COLLISION", "TARGET_LOST")


@dataclass(frozen=True)
class Plan:
    epoch: int
    side: str
    label: int
    approach_joints: tuple
    pre: np.ndarray
    palm: np.ndarray
    center: np.ndarray
    points: np.ndarray


@dataclass(frozen=True)
class ApproachPlan:
    """A walk-in before the grasp."""
    epoch: int
    side: str
    label: int
    drive_plan: navigate.Plan


def prepare(arguments, selected, current, planning):
    """A recheck of a carried target, a walk-in to a grasp stance, or a grasp from here."""
    side = side_argument(arguments)
    label, _ = click_argument(arguments, selected)
    if side in planning.carried:
        return RecoveryPlan(current.epoch, side, label,
                            resume_lift=planning.carried[side]["label"] == label)
    other = "left" if side == "right" else "right"
    if other in planning.carried and planning.carried[other]["label"] == label:
        raise SkillFailure("INVALID_ARGUMENT", f"The {other} hand carries the selected object; "
                                               f"repeat pick with the {other} hand to recheck it")
    return approach_stance(side, label, current, planning) or plan_target(side, label, current, planning)


def execute(plan, ctx):
    if isinstance(plan, RecoveryPlan):
        return recheck(plan, ctx)
    if isinstance(plan, ApproachPlan):
        plan = walk_in(plan, ctx)
    return grasp(plan, ctx)


def grasp(plan, ctx):
    """The pick from a planned grasp: open, approach, close, lift, verify the carry."""
    ctx.stage("Opening hand")
    close_or_open(ctx, plan.epoch, plan.side, 0.0, ctx.settings.grasp.open_seconds)
    plan, target, frame = approach_or_replan(plan, ctx)
    begin_pending(ctx, frame, plan.side, plan.label, lift_above(frame, ctx, plan.side),
                  points=target.track.last_supported_points)
    ctx.stage("Closing hand")
    frame = close_or_open(ctx, plan.epoch, plan.side, 1.0, ctx.settings.grasp.close_seconds)
    frame = squeeze(plan, ctx, frame)
    require_contact(plan, ctx, frame)
    ctx.carried[plan.side]["lift_target"] = lift_above(frame, ctx, plan.side)
    ctx.stage("Lifting")
    lift_pending(ctx, plan.epoch, plan.side)
    ctx.stage("Verifying grasp")
    status, frame = recover(ctx, plan.epoch, plan.side)
    if status == "cleared":
        ctx.stage("Returning arm")
        result = recovery_result("pick", status)
        if reset_arm(ctx, plan.epoch, plan.side):
            return result
        return Result(result.skill, result.code, f"{result.message}; the arm did not return near its rest pose")
    if status != "verified":
        raise SkillFailure("LIFT_NOT_VERIFIED", "Repeated current evidence did not verify lift; grasp remains uncertain")
    defer_compaction(ctx, plan.side)
    shape = visible_shape(frame.target_points(plan.label), ctx.settings.grasp.shape_quantiles)
    palm = frame.palm(ctx.arms[plan.side], plan.side)
    record = ctx.carried[plan.side]
    return Result("pick", "OK", "Repeated hand resistance and current visible lift verified",
                  {"side": plan.side,
                   "observed_top_rise": float(shape.upper[2] - record["baseline_upper"]),
                   "palm_rise": float(palm[2, 3] - record["baseline_palm_height"])})


def begin_pending(ctx, frame, side, label, lift_target, *, points=None):
    """Use the same robust top statistic as recovery's post-lift measurement."""
    register_pending(ctx, frame, side, label, lift_target, points=points)
    points = frame.target_points(label) if points is None else points
    ctx.carried[side]["baseline_upper"] = float(visible_shape(
        points, ctx.settings.grasp.shape_quantiles).upper[2])


def recheck(plan, ctx):
    """Re-verify a carried target, resuming its lift if enclosed."""
    ctx.stage("Checking carry")
    status, _ = recover(ctx, plan.epoch, plan.side)
    if status == "enclosed" and plan.resume_lift:
        frame = squeeze(plan, ctx, ctx.observe(plan.epoch))
        require_contact(plan, ctx, frame)
        ctx.stage("Lifting")
        lift_pending(ctx, plan.epoch, plan.side)
        ctx.stage("Verifying grasp")
        status, _ = recover(ctx, plan.epoch, plan.side)
    if status == "verified":
        if ctx.carried[plan.side]["label"] == plan.label:
            defer_compaction(ctx, plan.side)
            return Result("pick", "OK", "Current repeated hand and visual evidence verifies the original grasp")
        return Result("pick", "HAND_OCCUPIED", "The original carried object is still verified; the hand was not opened")
    return recovery_result("pick", status)


def abort(plan, ctx):
    """Reset the arm after a refused grasp. The base does not move."""
    if plan.side in ctx.carried or ctx.observations.epoch != plan.epoch:
        return
    ctx.stage("Returning arm")
    if not reset_arm(ctx, plan.epoch, plan.side):
        raise SkillFailure("ARM_NOT_REACHED", "Measured arm did not return near its rest pose")


def approach_stance(side, label, current, planning):
    """An ApproachPlan to a square stance at grasp range, or None when already there."""
    if not planning.jaw_samples.get(side):
        return None
    points = current.target_points(label)
    shape = visible_shape(points, planning.settings.grasp.shape_quantiles)
    centre = shape.center[:2]
    ahead, left = navigate.body_frame(current.base_pose, centre - current.base_pose[:2])
    bounds = planning.settings.grasp
    toward_hand = -left if side == "right" else left
    reach = grasp_distance(side, shape, current, planning)
    if ahead <= 0 or (reach - bounds.approach_short <= ahead <= reach + bounds.approach_long and
                      bounds.approach_across <= toward_hand <= bounds.approach_side):
        return None
    stance = {"goal_xy_m": current.base_pose[:2].tolist(), "facing_xy_m": centre.tolist(),
              "snap": True, "hand": side}
    walk = navigate.prepare(stance, current, current, planning, reach=reach)
    ahead = (centre - walk.goal) @ np.array([np.cos(walk.yaw), np.sin(walk.yaw)])
    if ahead > reach + bounds.approach_long:
        raise SkillFailure("UNREACHABLE", f"The nearest clear stance leaves the object {ahead:.2f} m "
                                          f"ahead; the arm's grasp range is {reach:.2f} m")
    return ApproachPlan(current.epoch, side, label, walk)


def grasp_distance(side, shape, current, planning):
    """Estimate a stance for a level or oblique wrist; exact reach is gated by IK."""
    settings, arm = planning.settings.grasp, planning.arms[side]
    jaw = np.mean([np.asarray(sample["palm_to_jaw"])[0, 3] for sample in planning.jaw_samples[side]])
    center = points_in(inverse(current.world_torso), shape.center)[2]
    upper = points_in(inverse(current.world_torso), shape.upper)[2]
    # An oblique insertion raises the wrist behind the pads. Do not reject a low
    # target on the assumption that its wrist must be at the object's mid-height.
    height = np.clip(arm.shoulder[2], center, upper + jaw * np.sin(np.pi / 3))
    rise = height - arm.shoulder[2]
    if abs(rise) >= settings.grasp_range:
        where = "above" if rise > 0 else "below"
        raise SkillFailure("UNREACHABLE", f"The object's pre-grasp point is {abs(rise):.2f} m {where} "
                                          f"the shoulder; no stance brings it within the arm's "
                                          f"grasp range of {settings.grasp_range:.2f} m", {"final": True})
    horizontal = np.sqrt(settings.grasp_range ** 2 - rise ** 2)
    return float(arm.shoulder[0] + horizontal + jaw)


def walk_in(plan, ctx):
    """Walk to the grasp stance, then plan the grasp from the new view."""
    ctx.stage("Walking to the grasp stance")
    navigate.execute(plan.drive_plan, ctx)
    ctx.advance(plan.epoch)
    current = ctx.observe(plan.epoch)
    ctx.stage("Planning grasp from the new stance")
    result = plan_target(plan.side, plan.label, current, ctx.planning())
    ctx.emit({"type": "skill_prepared", "skill": "pick", "plan": result, "walked_in": True})
    return result


def plan_target(side, label, current, planning, blocked=()):
    """The best routable grasp of the captured target from visible geometry."""
    if side in planning.carried:
        raise SkillFailure("HAND_OCCUPIED", "The selected hand has unresolved grasp ownership")
    if current.hands[side].retained(planning.settings.grasp):
        raise SkillFailure("HAND_OCCUPIED", "The selected hand already has retention evidence")
    samples = planning.jaw_samples.get(side)
    if not samples:
        raise SkillFailure("CALIBRATION_MISSING", "Robot-only jaw calibration is required")
    points = current.target_points(label)
    search = GraspSearch(side, label, current, planning, blocked)
    shoulder = points_in(current.world_torso, planning.arms[side].shoulder)
    forward = visible_shape(points, planning.settings.grasp.shape_quantiles).center - shoulder
    fitted = _width_fitted_candidates(points, samples, planning.settings.grasp, planning.cancelled, forward)
    best = search.best(list(search.reachable(fitted)))
    if planning.cancelled():
        raise SkillFailure("CANCELLED", "Operator cancelled grasp preparation")
    if best is None:
        raise SkillFailure("NO_GRASP", f"No grasp passed visible-depth geometry, reach and clearance gates: "
                           f"{search.counts}; blockers: {dict(search.blockers)}")
    grasp, route = best
    return Plan(current.epoch, side, label, route, grasp.pre, grasp.palm,
                grasp.shape.center, points)


@dataclass(frozen=True)
class GraspCandidate:
    """A proposed jaw pose that a calibrated opening fits. Lower tiers rank first."""
    tier: int
    width: float
    pre: np.ndarray
    palm: np.ndarray
    shape: Shape
    contact_overlap: float = 0.0


@dataclass(frozen=True)
class ReachableGrasp:
    """A candidate the arm reaches with a clear descent."""
    candidate: GraspCandidate
    pre_joints: np.ndarray
    clearance: float = 0.0


def _width_fitted_candidates(points, samples, settings, cancelled, forward=None):
    """Forward and oblique insertions, with either horizontal or vertical jaw closure.

    Fit the full visible span between the jaws, never a narrow slice hiding a wider
    obstruction. Use one calibrated frame per width, rather than repeating identical
    parallel-jaw poses at every closure. The pad origin sits inside the near surface;
    the wrist stays behind it along the calibrated approach axis.
    """
    shape = visible_shape(points, settings.shape_quantiles)
    forward = np.array([1., 0., 0.]) if forward is None else np.array(forward, dtype=float)
    forward[2] = 0.
    forward /= max(np.linalg.norm(forward), 1e-9)
    heading = np.arctan2(forward[1], forward[0])
    angles = np.unique(np.mod(np.r_[heading, closing_angles(points, settings.yaw_samples)], 2 * np.pi))
    apertures = np.array([sample["aperture"] for sample in samples])
    # Leave real entry clearance; aperture_error is not extra mechanical travel.
    minimum, maximum = apertures.min(), apertures.max() - settings.aperture_error
    for angle in angles:
        if cancelled():
            raise SkillFailure("CANCELLED", "Operator cancelled grasp preparation")
        horizontal = np.array([np.cos(angle), np.sin(angle), 0.])
        if horizontal @ forward < 0.5 - 1e-9:
            continue
        for pitch_index, pitch in enumerate((0., np.pi / 6, np.pi / 3)):
            approach = horizontal * np.cos(pitch) + np.array([0., 0., -np.sin(pitch)])
            lateral = np.array([-np.sin(angle), np.cos(angle), 0.])
            up = np.cross(approach, lateral)
            for roll in range(4):
                closing = (lateral, up, -lateral, -up)[roll]
                across = np.cross(approach, closing)
                rotation = np.column_stack((closing, across, approach))
                local = points @ rotation
                lower, upper = np.quantile(local, settings.shape_quantiles, axis=0)
                width = np.ptp(local[:, 0])
                if not minimum < width <= maximum:
                    continue
                sample = samples[int(np.argmin(abs(apertures - width)))]
                palm_jaw = np.asarray(sample["palm_to_jaw"])
                insertion = min(settings.depth_max, (upper[2] - lower[2]) * settings.depth_fraction)
                for tier, scale in enumerate(settings.insertion_depth_scales):
                    center = (lower + upper) / 2
                    center[2] = lower[2] + insertion * scale
                    palm = transform(rotation @ center, rotation) @ inverse(palm_jaw)
                    pre = palm.copy()
                    pre[:3, 3] -= approach * settings.approach_height
                    overlap = contact_overlap(local - center, settings)
                    yield GraspCandidate(tier * 3 + pitch_index, width, pre, palm, shape, overlap)


def contact_overlap(local, settings):
    """Visible opposing patch overlap near the pad origin, in jaw coordinates.

    Intersect both surface patches before clipping to the pad neighbourhood:
    one deep side alone is not evidence of balanced seating. This area proxy
    ranks candidates; it neither infers hidden surfaces nor verifies possession.
    """
    low, high = np.quantile(local[:, 0], settings.shape_quantiles)
    band = min(settings.pad_band, (high - low) / 4)
    patches = (local[local[:, 0] <= low + band, 1:],
               local[local[:, 0] >= high - band, 1:])
    if any(len(patch) < settings.pad_band_points for patch in patches):
        return 0.0
    bounds = [np.quantile(patch, settings.shape_quantiles, axis=0) for patch in patches]
    lower = np.maximum(np.maximum(bounds[0][0], bounds[1][0]), -settings.pad_band)
    upper = np.minimum(np.minimum(bounds[0][1], bounds[1][1]), settings.pad_band)
    return float(np.prod(np.maximum(upper - lower, 0.0)))


class GraspSearch:
    """One frame's grasp gates (reach, descent clearance, approach route), with counts per gate
    and the blockers met."""
    GATES = ("width", "pre_reachable", "grasp_reachable", "descent_clear", "approach_clear")

    def __init__(self, side, label, current, planning, blocked=()):
        self.arm, self.seed = planning.arms[side], current.arm_joints(side)
        self.settings, self.cancelled = planning.settings.grasp, planning.cancelled
        self.control = planning.settings
        self.blocked = blocked
        self.torso = inverse(current.world_torso)
        collision = planning.settings.collision
        # Recovery runs after opening, so use the measured fingers in that case.
        opened = current.hands[side].joints if blocked else np.zeros_like(current.hands[side].joints)
        self.descent = Clearance(current, planning.arms, side, collision, (label,), fingers=opened)
        self.approach = Clearance(current, planning.arms, side, collision, exempt_start=True, fingers=opened)
        self.counts = dict.fromkeys(self.GATES, 0)
        self.blockers = Counter()

    def reachable(self, candidates):
        for candidate in candidates:
            self.counts["width"] += 1
            if self.cancelled():
                return
            if any(self.equivalent(candidate, plan, phase) for plan, phase in self.blocked):
                self.blockers["equivalent blocked grasp"] += 1
                continue
            pre_joints = self.arm.solve(self.torso @ candidate.pre, self.seed)
            if pre_joints is None:
                continue
            self.counts["pre_reachable"] += 1
            grasp_joints = self.arm.solve(self.torso @ candidate.palm, pre_joints)
            if grasp_joints is None:
                continue
            self.counts["grasp_reachable"] += 1
            obstruction = self.descent.path_violation(pre_joints, grasp_joints)
            if obstruction is not None:
                self.blockers[obstruction] += 1
                continue
            # Keep the cheap joint-path gate, then check the actual servo reference model.
            margin = self.servo_descent(pre_joints, candidate.palm) if self.blocked else 0.0
            if margin is None:
                continue
            if self.blocked and any(phase == "approach" for _, phase in self.blocked):
                margin = min(margin, self.finger_margin(self.approach, pre_joints))
            self.counts["descent_clear"] += 1
            yield ReachableGrasp(candidate, pre_joints, margin)

    def equivalent(self, candidate, plan, phase):
        """Do not retry poses indistinguishable at the controller's spatial resolution."""
        names = ("palm",) if phase == "descent" else ("pre", "palm")
        return all(np.linalg.norm(getattr(candidate, name)[:3, 3] - getattr(plan, name)[:3, 3])
                   <= self.control.motion.palm_tolerance and
                   np.linalg.norm(rotation_error(getattr(candidate, name)[:3, :3],
                                                 getattr(plan, name)[:3, :3]))
                   <= self.control.servo.rotation_tolerance for name in names)

    def finger_margin(self, clearance, joints):
        """Extra visible-depth padding tolerated by every finger, capped at two cloud margins.

        This is a ranking signal, not a replacement for the full collision gate.
        """
        boxes = clearance.geometry.hand_boxes(self.arm, joints, clearance.fingers)
        unit = clearance.settings.cloud_margin
        for extra in (unit, 2 * unit):
            if any(clearance.blocked(clearance.hits(box, unit + extra)) for box in boxes):
                return extra - unit
        return 2 * unit

    def servo_descent(self, start, palm):
        """Nominal measured=commanded rollout; live checks still guard tracking/depth changes."""
        settings = self.control
        joints = np.array(start, copy=True)
        margin = np.inf if self.blocked else 0.0
        for _ in range(int(np.ceil(settings.motion.arm_timeout / settings.servo.tick))):
            if self.cancelled():
                return None
            command, distance, angle = servo_step(self.arm, joints, joints, self.torso @ palm,
                                                  settings.physics.arm_rate, settings.servo)
            if self.blocked:
                margin = min(margin, self.finger_margin(self.descent, joints))
            if distance < settings.motion.palm_tolerance and angle < settings.servo.rotation_tolerance:
                return margin
            obstruction = self.descent.path_violation(joints, command)
            if obstruction is not None:
                self.blockers[obstruction] += 1
                return None
            if np.allclose(command, joints, atol=1e-8, rtol=0):
                break
            joints = command
        self.blockers["descent servo did not converge"] += 1
        return None

    def best(self, reachable):
        """Prefer clearance and seating, trying each start lift in turn.

        For short targets prefer the insertion tier before overlap: a steep pose's
        larger projected patch can stall on contact before the pads enclose a squat
        jar. The target is exempt from descent clearance, so that gate cannot prove
        seating. Keep overlap first for taller targets and within each short-target
        tier; retain steep tiers as fallbacks when shallower insertions cannot route.
        """
        def rank(grasp):
            candidate = grasp.candidate
            short = candidate.shape.upper[2] - candidate.shape.lower[2] <= .06
            seating = ((candidate.tier, -candidate.contact_overlap) if short else
                       (-candidate.contact_overlap, candidate.tier))
            return (-grasp.clearance, *seating, candidate.width,
                    np.linalg.norm(grasp.pre_joints - self.seed))

        ranked = sorted(reachable, key=rank)
        checked = {}
        for lift in (0., *self.settings.approach_start_lifts_m):
            for grasp in ranked:
                if not self.blocked:
                    key = id(grasp)
                    if key not in checked:
                        checked[key] = self.servo_descent(grasp.pre_joints, grasp.candidate.palm)
                    if checked[key] is None:
                        continue
                approach = approach_path(self.arm, self.seed, grasp.pre_joints, self.approach, self.settings,
                                         self.cancelled, initial_lift_m=lift, blockers=self.blockers)
                if approach is not None:
                    self.counts["approach_clear"] += 1
                    return grasp.candidate, approach
        return None


def approach_or_replan(plan, ctx):
    """Bound collision/view retries and allow one separate low-lag insertion replan."""
    target = PickTarget(plan, ctx)
    attempts = ctx.settings.grasp.approach_attempts
    insertion_retried = False
    attempt = 1
    while True:
        try:
            ctx.stage("Approaching")
            frame = approach(plan, ctx, target)
            if insertion_retried:
                require_jaw_enclosure(plan, ctx, target, frame)
            break
        except SkillFailure as error:
            if error.code == "ARM_NOT_REACHED" and target.descending and not insertion_retried:
                lag = error.details.get("joint_tracking_error_rad", np.inf)
                if not np.isfinite(lag) or lag > 0.05:
                    raise  # Loaded or unknown tracking: never close on the missed pose.
                insertion_retried = True
            elif error.code in RETRIED_APPROACH_FAILURES and attempt < attempts:
                attempt += 1
            else:
                raise
            try:
                plan, target = replan(plan, ctx, target, error, attempt + int(insertion_retried))
            except SkillFailure as recovery_error:
                raise SkillFailure(recovery_error.code,
                                   f"Initial approach failure [{error.code}]: {error}; "
                                   f"recovery failure [{recovery_error.code}]: {recovery_error}") from recovery_error
    return plan, target, frame


def approach(plan, ctx, target):
    """Route to the pre-grasp, servo onto it, then descend to the grasp palm."""
    if len(plan.approach_joints):
        ctx.stage("Following grasp path")
        move_path(ctx, plan.epoch, plan.side, plan.approach_joints, tracked_target=target)
    ctx.stage("Aligning with target")
    correct_alignment(plan, ctx, target)
    servo(ctx, plan.epoch, plan.side, plan.pre, tracked_target=target)
    target.begin_descent()
    ctx.stage("Inserting jaws")
    monitored = InsertionContext(ctx)
    try:
        return servo(monitored, plan.epoch, plan.side, plan.palm,
                     tracked_target=target, exclude_labels=(plan.label,))
    except SkillFailure as error:
        if error.code == "ARM_NOT_REACHED":
            error.details.update(monitored.timeout)
        raise


class InsertionContext:
    """Forward the context, retaining the servo's measured timeout diagnostics locally."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.timeout = {}

    def __getattr__(self, name):
        return getattr(self.ctx, name)

    def emit(self, event):
        if event.get("type") == "arm_servo_timeout":
            self.timeout = dict(event)
        self.ctx.emit(event)


def require_jaw_enclosure(plan, ctx, target, frame):
    """A recovered servo must seat the measured pad plane inside supported target depth."""
    target.update(frame)
    samples = ctx.planning().jaw_samples[plan.side]
    opened = min(samples, key=lambda sample: abs(sample["closure"] - frame.hands[plan.side].closure))
    jaw = frame.palm(ctx.arms[plan.side], plan.side) @ np.asarray(opened["palm_to_jaw"])
    local = points_in(inverse(jaw), target.track.last_supported_points)
    lower, upper = np.quantile(local, ctx.settings.grasp.shape_quantiles, axis=0)
    half = opened["aperture"] / 2
    # The pad has area: a target can be seated off its centre in the
    # non-closing direction. Still require depth insertion and jaw clearance.
    band = ctx.settings.grasp.pad_band
    if not (lower[2] < 0 < upper[2] and lower[1] < band and upper[1] > -band and
            local[:, 0].min() >= -half and local[:, 0].max() <= half):
        raise SkillFailure("NO_GRASP", "Recovered insertion did not enclose the supported target")


ALIGNMENT_CORRECTION_RAD = 0.5  # only a local correction, never another approach route


def correct_alignment(plan, ctx, target):
    """Re-solve the same pre-grasp once from the measured endpoint and current torso.

    An unsolved, large or blocked correction leaves the existing full-pose servo in charge.
    Both the proposed segment and its execution retain the approach collision gates.
    """
    frame = ctx.observe(plan.epoch)
    goal = plan.pre.copy()
    goal[:3, 3] += target.update(frame)
    measured = frame.arm_joints(plan.side)
    solved = ctx.arms[plan.side].solve(inverse(frame.world_torso) @ goal, measured)
    reason = "no IK solution"
    if solved is not None:
        reason = "correction exceeds local joint bound"
        if np.max(np.abs(solved - measured)) <= ALIGNMENT_CORRECTION_RAD:
            clearance = Clearance(frame, ctx.arms, plan.side, ctx.settings.collision, exempt_start=True)
            reason = clearance.path_violation(measured, solved)
    ctx.emit({"type": "pick_alignment_correction", "frame_id": frame.frame_id,
              "side": plan.side, "reason": reason, "applied": reason is None})
    if reason is None:
        move_path(ctx, plan.epoch, plan.side, [solved], tracked_target=target)


def replan(plan, ctx, target, error, attempt):
    """Replan from a fresh view of the same target, parking the arm first when it hides it."""
    if error.code in ("COLLISION", "ARM_NOT_REACHED"):
        target.blocked.append((plan, "descent" if target.descending else "approach"))
    ctx.actions.stop(plan.epoch)
    ctx.stage("Replanning approach")
    stopped = ctx.observe(plan.epoch)
    ctx.advance(plan.epoch)
    frame = ctx.observe(plan.epoch)
    packets = FreshEvidence(plan.epoch)
    packets.update(stopped, True)
    if packets.update(frame, True) is None:
        raise SkillFailure("STALE_OBSERVATION", "Grasp replanning requires a new sensor packet") from error
    if not target.supported(frame):
        ctx.stage("Clearing target view")
        if not reset_arm(ctx, plan.epoch, plan.side):
            raise SkillFailure("ARM_NOT_REACHED", "The arm hiding the target did not return") from error
        ctx.advance(plan.epoch)
        frame = ctx.observe(plan.epoch)
        if not target.supported(frame):
            raise SkillFailure("TARGET_LOST", "A new grasp requires a supported target view") from error
    ctx.emit({"type": "grasp_replan", "attempt": attempt, "label": plan.label,
              "frame_id": frame.frame_id, "reason": str(error)})
    plan = plan_target(plan.side, plan.label, frame, ctx.planning(), target.blocked)
    ctx.emit({"type": "skill_prepared", "skill": "pick", "plan": plan, "attempt": attempt})
    return plan, target.rebased(plan)


def squeeze(plan, ctx, frame):
    """Keep the position servos loaded even when a wide object stops closure early."""
    ctx.stage("Squeezing")
    return close_or_open(ctx, plan.epoch, plan.side, 1.0 + ctx.settings.grasp.squeeze,
                         ctx.settings.grasp.close_seconds)


def opposed_jaws(hand, settings):
    """Index and middle are aliases for jaw 2, not two independent contacts."""
    return bool(np.all(hand.finger_effort_fraction[:2] > settings.hold_effort_fraction))


def require_contact(plan, ctx, frame):
    """Do not lift on a single jaw or a transient impact. Ownership stays pending."""
    samples = FreshEvidence(plan.epoch)
    settings = ctx.settings.grasp
    for index in range(settings.recovery_samples):
        if index:
            ctx.advance(plan.epoch, settings.recovery_tick)
            frame = ctx.observe(plan.epoch)
        count = samples.update(frame, opposed_jaws(frame.hands[plan.side], settings))
        if count is not None and count >= settings.recovery_confirmations:
            return
    raise SkillFailure("LIFT_NOT_VERIFIED", "Both jaws did not sustain contact before lift; grasp remains uncertain")


def lift_pending(ctx, epoch, side):
    """Translate upward with the grasp orientation and measured jaw/payload geometry."""
    record = ctx.carried[side]
    return servo_best_effort(ctx, epoch, side, record["lift_target"], exclude_labels=(record["label"],))


def lift_above(frame, ctx, side):
    """This frame's palm raised by lift_height, in world."""
    lift = frame.palm(ctx.arms[side], side)
    lift[2, 3] += ctx.settings.grasp.lift_height
    return lift


class PickTarget:
    """Track the planned target's supported surfaces during the approach and bound a finish
    on a partial view. The servo calls `update` on every read."""

    def __init__(self, plan, ctx, original_center=None, limits=None):
        self.plan, self.ctx = plan, ctx
        self.limits = ctx.settings.blind_finish if limits is None else limits
        self.capture_center = plan.center.copy() if original_center is None else original_center.copy()
        grasp = ctx.settings.grasp
        self.track = TargetTrack(plan.points, grasp.shape_quantiles, grasp.target_surface_tolerance,
                                 grasp.target_shift, self.capture_center)
        self.blocked = []
        self.mode = None
        self.descending = False
        self.last_supported_time = None
        self.window = None
        self.shift = np.zeros(3)
        self.occlusion = None

    def begin_descent(self):
        """Enter the descent, where a partial view is bounded rather than refused."""
        self.descending = True

    def rebased(self, plan):
        """A tracker for the same target freshly planned, keeping the capture anchor."""
        target = PickTarget(plan, self.ctx, self.capture_center, self.limits)
        target.blocked = list(self.blocked)
        return target

    def supported(self, frame):
        """Whether this frame shows the captured surfaces within their capture range."""
        return self._match(self._points(frame)).mode == "supported"

    def update(self, state):
        """Track one read of the approach. The returned shift (m, world) moves the planned palm."""
        palm = state.palm(self.ctx.arms[self.plan.side], self.plan.side)
        if isinstance(state, Frame):
            self._see(state, palm)
        goal = self.plan.palm.copy()
        goal[:3, 3] += self.shift
        if self.descending and (self.mode != "supported" or self.window is not None):
            self._require_corridor(palm, goal)
            if self.window is None:
                self.window = PartialWindow.opened(state, palm, self.limits.partial_travel)
        if self.window is not None and not self.window.within_bounds(
                state, palm, self.limits, self.ctx.settings.motion):
            raise SkillFailure("TARGET_LOST", "Partial-view finish exceeded its measured motion or time bound")
        return self.shift

    def _points(self, frame):
        label = self.plan.label
        return frame.target_points(label) if frame.visible(label) else np.empty((0, 3))

    def _match(self, points):
        """The track's verdict on these points. A moved target raises TARGET_MOVED."""
        match = self.track.update(points)
        if match.mode == "moved":
            raise SkillFailure("TARGET_MOVED", "Supported target geometry moved beyond the capture range")
        return match

    def _see(self, frame, palm):
        """Match this frame's view of the target and update the shift."""
        points = self._points(frame)
        match = self._match(points)
        if match.mode == "uncertain":
            match = self._wrist_match(frame) or match
        self.shift = self.track.last_supported_shape.center - self.plan.center
        if match.mode == "uncertain":
            goal = self.plan.palm.copy()
            goal[:3, 3] += self.shift
            self._admit_hidden(frame, points, palm, goal)
            mode = "occluded"
        else:
            self.last_supported_time = frame.time
            mode = match.mode
        if mode != self.mode:
            self.ctx.emit({"type": "target_tracking", "mode": mode, "frame_id": frame.frame_id,
                           "partial_travel": self.window.travel if self.window is not None else 0.0,
                           "match": match})
            self.mode = mode

    def _wrist_match(self, frame):
        """When the head view no longer supports the target, the hand's own wrist camera: its match if
        it shows the captured surfaces, else None (the head view's verdict stands)."""
        if not self.ctx.settings.grasp.wrist_camera:
            return None
        try:
            view = self.ctx.wrist(self.plan.epoch, self.plan.side)
        except (SkillFailure, ObservationError) as error:
            self.ctx.emit({"type": "wrist_view_unavailable", "reason": str(error)})
            return None
        if not view.visible(self.plan.label):
            return None
        match = self._match(view.target_points(self.plan.label))
        if match.mode == "uncertain":
            return None
        self.ctx.emit({"type": "target_tracking", "mode": f"wrist_{match.mode}", "frame_id": frame.frame_id})
        return match

    def _admit_hidden(self, frame, points, palm, goal):
        """Allow a blind finish only when the arm hides the target, the palm is close and the
        target was seen recently. Otherwise TARGET_LOST."""
        hides = self._arm_hides(frame, points)
        remaining = float(np.linalg.norm(palm[:3, 3] - goal[:3, 3]))
        age = None if self.last_supported_time is None else frame.time - self.last_supported_time
        # Reads.next schedules images at perception_period; recency is the allowance
        # for sampling jitter, not a demand for an image between scheduled reads.
        freshness_limit = self.ctx.settings.servo.perception_period + self.limits.occluded_recency
        reason = None
        if not hides:
            reason = "Current target is not explained by measured arm occlusion"
        elif self.mode != "occluded" and not (
                self.mode in ("supported", "partial") and age is not None and
                0 <= age <= freshness_limit and remaining <= self.limits.occluded_reach):
            reason = "Current visible surfaces do not support the captured target pose"
        if reason is None:
            try:
                self._require_corridor(palm, goal)
            except SkillFailure as error:
                reason = str(error)
        if reason is not None:
            details = {"preceding_mode": self.mode, "observation_age_s": age,
                       "freshness_limit_s": freshness_limit, "remaining_distance_m": remaining,
                       "target_visible": bool(len(points)), "label": self.plan.label,
                       "occlusion": self.occlusion}
            self.ctx.emit({"type": "target_tracking_failure", "reason": reason, **details})
            raise SkillFailure("TARGET_LOST", f"{reason}; tracking={json.dumps(details)}")
        if self.window is None:
            self.window = PartialWindow.opened(frame, palm, remaining + self.limits.travel_margin)

    def _arm_hides(self, frame, points):
        """Whether the measured hand explains the missing target rays and what remains visible
        agrees with the captured surfaces."""
        grasp, side = self.ctx.settings.grasp, self.plan.side
        arm, joints = self.ctx.arms[side], frame.arm_joints(side)
        boxes = (arm.collision.boxes(arm, joints)[-HAND_END_LINKS:] +
                 arm.collision.hand_boxes(arm, joints, frame.hands[side].joints))
        occlusion = arm_occlusion_evidence(self.track.last_supported_points, points, frame, boxes,
                                           grasp.target_surface_tolerance)
        self.occlusion = asdict(occlusion)
        blocked = (occlusion.measured_fraction >= grasp.occlusion_min_rays and
                   occlusion.explained_missing_fraction >= grasp.occlusion_missing_explained and
                   occlusion.exposed_missing_fraction <= grasp.occlusion_exposed_missing_max)
        if len(points):
            overlap = np.mean(cKDTree(self.track.last_supported_points).query(points)[0] <=
                              grasp.target_surface_tolerance)
            consistent = overlap >= grasp.occlusion_visible_match
        else:
            consistent = occlusion.measured_fraction >= grasp.occlusion_no_view_rays
        return blocked and consistent

    def _require_corridor(self, palm, goal):
        """TARGET_LOST unless the palm lies within the planned descent's corridor."""
        grasp, motion = self.ctx.settings.grasp, self.ctx.settings.motion
        offset = palm[:3, 3] - goal[:3, 3]
        retreat = self.plan.pre[:3, 3] - self.plan.palm[:3, 3]
        length = float(np.linalg.norm(retreat))
        axis = retreat / max(length, 1e-9)
        along = float(offset @ axis)
        across = offset - along * axis
        if (np.linalg.norm(across) > motion.palm_tolerance + grasp.target_surface_tolerance or
                not -motion.palm_tolerance <= along <= length + grasp.target_shift or
                np.linalg.norm(rotation_error(goal[:3, :3], palm[:3, :3])) > self.limits.palm_turn):
            raise SkillFailure("TARGET_LOST", "Partial target view outside the final approach corridor")


@dataclass
class PartialWindow:
    """The budget of a finish begun with the target partly visible: base and head still, palm
    travel and time bounded."""
    start: float
    base: np.ndarray
    camera: np.ndarray
    last_palm: np.ndarray
    travel_limit: float
    travel: float = 0.0

    @classmethod
    def opened(cls, frame, palm, travel_limit):
        return cls(frame.time, frame.base_pose.copy(), frame.world_camera.copy(),
                   palm[:3, 3].copy(), travel_limit)

    def within_bounds(self, frame, palm, limits, motion):
        self.travel += np.linalg.norm(palm[:3, 3] - self.last_palm)
        self.last_palm = palm[:3, 3].copy()
        return (frame.time - self.start <= motion.arm_timeout and self.travel <= self.travel_limit and
                np.linalg.norm(frame.base_pose[:2] - self.base[:2]) <= motion.palm_tolerance and
                abs(wrap(frame.base_pose[2] - self.base[2])) <= limits.base_turn and
                np.linalg.norm(frame.world_camera[:3, 3] - self.camera[:3, 3]) <= limits.head_drift and
                np.linalg.norm(rotation_error(frame.world_camera[:3, :3], self.camera[:3, :3])) <=
                limits.head_turn)
