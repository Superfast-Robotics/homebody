"""Pose-associated camera views for exploration decisions, not calibrated RGB mapping."""
from __future__ import annotations

import json
import math
import tempfile
from copy import deepcopy
from numbers import Real
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .settings import current

MAX_ASSOCIATION_AGE_S = 0.5
POSITION_STEP_M = 0.5
HEADING_STEP_RAD = math.radians(30)
PAGE_SIZE = 6
ASSOCIATION = (
    "Approximate robot viewpoint in the exploration start frame; receiver decode time, "
    "not camera exposure time. Heading is robot yaw, not calibrated camera optical direction. "
    "No camera intrinsics, depth, extrinsic or projected visibility footprint."
)
# The coverage ledger fields every action but update_region may carry.
LEDGER = {
    "coverage_updates": {"type": "array", "items": {"type": "object", "properties": {
        "area": {"type": "string"}, "requirement": {"type": "string"},
        "complementary_required": {"type": "boolean"},
        "scope": {"type": "string", "enum": ["included", "excluded", "unresolved"]},
        "status": {"type": "string", "enum": ["needs_view", "covered"]},
        "evidence_views": {"type": "array", "items": {"type": "string"}},
        "complementary_views": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "resolution": {"type": "string"},
    }, "required": ["area", "requirement", "complementary_required", "scope", "status",
                    "evidence_views", "complementary_views", "gaps", "resolution"],
        "additionalProperties": False}},
    "image_assessments": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"},
        "usability": {"type": "string", "enum": ["readable", "blurred", "uncertain"]},
        "compared_to": {"type": "string"},
        "relationship": {"type": "string", "enum": ["same_view", "different_angle", "uncertain"]},
        "reason": {"type": "string"},
    }, "required": ["id", "usability", "reason"], "additionalProperties": False}},
}


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _sharpness(rgb):
    gray = np.asarray(Image.fromarray(rgb).convert("L").resize((160, 90)), float)
    laplacian = (gray[2:, 1:-1] + gray[:-2, 1:-1] + gray[1:-1, 2:]
                 + gray[1:-1, :-2] - 4 * gray[1:-1, 1:-1])
    return float(laplacian.var())


def _frame_identity(metadata):
    return (metadata.get("sequence"), metadata.get("receiver_decode_time_s"))


def select_context_frame(samples, *, retained=False):
    """Relative clarity among nearby received samples, not calibrated focus or exposure."""
    latest = samples[-1]
    report = {"status": "latest_fallback", "selected_sequence": latest["metadata"].get(
        "sequence"), "scores": [], "omitted_candidates": [],
        "reason": "insufficient distinct phone frames"}
    if len({_frame_identity(s["metadata"]) for s in samples}) < 2:
        return latest, report
    poses = np.asarray([s["pose"] for s in samples], dtype=float)
    eligible = []
    for i, s in enumerate(samples):
        m = s["metadata"]
        values = [*s["pose"], s["sampled_at"], s["odom_age_s"],
                  m.get("receiver_decode_time_s"), m.get("receive_age_s")]
        if (any(not isinstance(v, (int, float)) for v in values)
                or not np.isfinite(values).all() or s["rgb"] is None
                or m.get("status") != "fresh" or not 0 <= s["odom_age_s"] <= .5
                or not 0 <= m["receive_age_s"] <= .5
                or abs(s["sampled_at"] - m["receiver_decode_time_s"]) > .5):
            report["reason"] = "candidate pose/time association unavailable"
            return latest, report
        if not retained and latest["sampled_at"] - s["sampled_at"] + m["receive_age_s"] > .5:
            report["omitted_candidates"].append({"sequence": m.get("sequence"),
                                                  "reason": "aged out during sample window"})
        else:
            eligible.append(i)
    if len({_frame_identity(samples[i]["metadata"]) for i in eligible}) < 2:
        report["reason"] = "insufficient distinct fresh phone frames"
        return latest, report
    delta = poses[:, None] - poses[None, :]
    angle = np.arctan2(np.sin(delta[..., 2]), np.cos(delta[..., 2]))
    if np.linalg.norm(delta[..., :2], axis=-1).max() > .05 or abs(angle).max() > math.radians(5):
        report["reason"] = "candidate poses are not stable enough for relative selection"
        return latest, report
    for i, s in enumerate(samples):
        if i not in eligible:
            report["scores"].append(None)
            continue
        report["scores"].append(_sharpness(s["rgb"]))
    selected = max(eligible, key=lambda i: (report["scores"][i], i))
    report.update(status="sharpest_stable",
                  reason="relative thumbnail sharpness; not a focus guarantee",
                  selected_sequence=samples[selected]["metadata"].get("sequence"))
    return samples[selected], report


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _transit_pose_valid(sample):
    pose = sample.get("pose")
    metadata = sample.get("metadata", {})
    route = sample.get("route_context")
    if (not isinstance(metadata, dict)
            or not (isinstance(pose, (tuple, list))
                    or isinstance(pose, np.ndarray) and pose.ndim == 1)
            or len(pose) != 3
            or not isinstance(route, dict)):
        return False
    values = [*pose, sample.get("sampled_at"), sample.get("pose_sampled_at_s"),
              sample.get("registered_pose_age_s"), sample.get("traveled_m"),
              metadata.get("receiver_decode_time_s"), metadata.get("receive_age_s")]
    if not all(_finite(v) for v in values):
        return False
    route_id = route.get("route_id")
    if (not isinstance(route_id, (str, int)) or isinstance(route_id, bool)
            or (isinstance(route_id, str) and not route_id.strip())):
        return False
    for key in ("revision", "segment", "nearest_knot"):
        value = route.get(key)
        if key == "nearest_knot" and value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return False
    sampled, pose_time = sample["sampled_at"], sample["pose_sampled_at_s"]
    return (metadata.get("status") == "fresh" and sample["traveled_m"] >= 0
            and 0 <= sample["registered_pose_age_s"] <= MAX_ASSOCIATION_AGE_S
            and 0 <= metadata["receive_age_s"] <= MAX_ASSOCIATION_AGE_S
            and abs(pose_time - sampled) <= MAX_ASSOCIATION_AGE_S
            and abs(pose_time - metadata["receiver_decode_time_s"]) <= MAX_ASSOCIATION_AGE_S)


def _distinct_pose(first, second):
    angle = math.atan2(math.sin(first[2] - second[2]), math.cos(first[2] - second[2]))
    return (math.hypot(first[0] - second[0], first[1] - second[1]) >= POSITION_STEP_M
            or abs(angle) >= HEADING_STEP_RAD)


def _positioned_transit(record):
    return (record.get("association_kind") == "approximate_receipt_pose"
            and _transit_pose_valid({
                "pose": record.get("robot_pose_xyyaw"), "metadata": record.get("phone", {}),
                "sampled_at": record.get("sampled_at_s"),
                **{k: record.get(k) for k in ("pose_sampled_at_s", "registered_pose_age_s",
                                             "route_context", "traveled_m")}}))


def select_transit_frames(retention, limit=2, *, diagnostics=None):
    """Choose distance-half views after sustained separation from departure."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= 2:
        raise ValueError("transit selection limit must be an integer in 0..2")
    context = retention.get("context")
    context = context if isinstance(context, dict) else {}
    buffer_s = current().iphone.post_action_buffer_s
    anchor = context.get("departure_pose_xyyaw")
    anchor_time = context.get("departure_pose_sampled_at_s")
    anchor_age = context.get("departure_registered_pose_age_s")
    anchor_identity = context.get("departure_camera_frame_identity")
    valid_anchor = (isinstance(anchor, (list, tuple)) and len(anchor) == 3
                    and all(_finite(v) for v in [*anchor, anchor_time, anchor_age])
                    and 0 <= anchor_age <= MAX_ASSOCIATION_AGE_S
                    and isinstance(anchor_identity, (list, tuple)) and len(anchor_identity) == 2
                    and all(_finite(v) for v in anchor_identity))
    if not valid_anchor:
        anchor = anchor_identity = None
    report = {"anchor_source": "departure_gallery" if valid_anchor else "unavailable",
              "departure_view_id": (context.get("departure_view_id")
                                        if valid_anchor else None),
              "departure_camera_frame_identity": list(anchor_identity) if valid_anchor else None,
              "departure_pose_xyyaw": list(anchor) if valid_anchor else None,
              "departure_pose_sampled_at_s": anchor_time if valid_anchor else None,
              "departure_registered_pose_age_s": anchor_age if valid_anchor else None,
              "delay_allowance_s": buffer_s,
              "candidate_count": len(retention.get("samples", [])),
              "invalid_count": 0, "departure_frame_count": 0,
              "near_departure_count": 0, "delay_wait_count": 0,
              "regression_count": 0, "eligible_count": 0, "selected_count": 0}
    unique, furthest, previous_time, separated_since = {}, -1., -math.inf, None
    # invalid observations stay in sequence and break a separation streak
    for sample in retention.get("samples", []):
        if sample.get("rgb") is None or not _transit_pose_valid(sample):
            separated_since = None
            report["invalid_count"] += 1
            continue
        now = sample["sampled_at"]
        if now < previous_time or sample["traveled_m"] < furthest:
            separated_since = None
            report["regression_count"] += 1
            continue
        previous_time, furthest = now, sample["traveled_m"]
        if anchor is None:
            anchor = sample["pose"]
            anchor_identity = _frame_identity(sample["metadata"])
            report.update(anchor_source="first_valid_retained_sample",
                          departure_camera_frame_identity=list(anchor_identity),
                          departure_pose_xyyaw=list(anchor),
                          departure_pose_sampled_at_s=sample["pose_sampled_at_s"],
                          departure_registered_pose_age_s=sample["registered_pose_age_s"])
        if _frame_identity(sample["metadata"]) == tuple(anchor_identity):
            separated_since = None
            report["departure_frame_count"] += 1
            continue
        if not _distinct_pose(sample["pose"], anchor):
            separated_since = None
            report["near_departure_count"] += 1
            continue
        if separated_since is None:
            separated_since = now
        elapsed = sample["metadata"]["receiver_decode_time_s"] - separated_since
        if elapsed < buffer_s:
            report["delay_wait_count"] += 1
            continue
        unique.setdefault(_frame_identity(sample["metadata"]), sample | {
            "departure_separated_since_s": separated_since,
            "departure_separation_elapsed_s": elapsed})
    samples = list(unique.values())
    report["eligible_count"] = len(samples)
    if diagnostics is not None:
        diagnostics.update(report)
    total = retention.get("traveled_m", furthest)
    if not samples or not limit or not _finite(total) or total <= 0 or total < furthest:
        return []
    groups = [[] for _ in range(limit)]
    for sample in samples:
        index = int(limit == 2 and sample["traveled_m"] > total / 2)
        groups[index].append(sample)
    selected = []
    for group in groups:
        ranked = sorted(group, key=lambda s: (_sharpness(s["rgb"]), s["sampled_at"]),
                        reverse=True)
        for sample in ranked:
            if any(not _distinct_pose(sample["pose"], old["pose"]) for old in selected):
                continue
            selected.append(sample | {"metadata": deepcopy(sample["metadata"]) | {
                "selection": {"status": "transit_candidate", "relative_sharpness":
                              _sharpness(sample["rgb"]),
                              "reason": "changed departure after RGB delay; distance-half clarity",
                              "departure_pose_xyyaw": report["departure_pose_xyyaw"],
                              "departure_view_id": report["departure_view_id"],
                              "departure_camera_frame_identity": report[
                                  "departure_camera_frame_identity"],
                              "anchor_source": report["anchor_source"],
                              "delay_allowance_s": buffer_s,
                              "departure_separated_since_s": sample["departure_separated_since_s"],
                              "departure_separation_elapsed_s": sample[
                                  "departure_separation_elapsed_s"]}},
                "retention_context": deepcopy(retention.get("context", {}))})
            break
    if diagnostics is not None:
        diagnostics["selected_count"] = len(selected)
    return selected


class VisualHistory:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self.epoch = max((int(p.name[6:]) for p in self.root.glob("epoch_*")
                          if p.is_dir() and p.name[6:].isdigit()), default=0)
        self.reset("new exploration context")

    def reset(self, reason: str, *, frame_metadata=None):
        """Start a new epoch; earlier epochs stay on disk, their poses unused."""
        self.epoch += 1
        self.records = []
        self.coverage_assessment = []
        self.directory = self.root / f"epoch_{self.epoch:03d}"
        self.directory.mkdir()
        self.reason = reason
        self.frame_metadata = frame_metadata
        self._save()

    def snapshot(self) -> VisualHistory:
        """A working copy of the epoch for one decision's image I/O, outside the map lock."""
        return deepcopy(self)

    def _save(self):
        temporary = self.stage_index()
        try:
            temporary.replace(self.directory / "index.json")
        finally:
            temporary.unlink(missing_ok=True)

    def stage_index(self) -> Path:
        """Prepare a unique index file; the caller owns validated publication and cleanup."""
        payload = json.dumps({
            "epoch": self.epoch, "reason": self.reason, "association": ASSOCIATION,
            "start_frame": self.frame_metadata,
            "sampling": {"position_step_m": POSITION_STEP_M,
                         "heading_step_deg": math.degrees(HEADING_STEP_RAD),
                         "when": "model decision viewpoints"},
            "views": self.records,
            "coverage_assessment": self.coverage_assessment,
        }, indent=2)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.directory,
                    prefix=".index.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            return temporary
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

    def apply_coverage(self, updates, supplied_ids, *, persist: bool = True) -> str:
        """Apply atomically; defer persistence when composing a working response snapshot."""
        if updates is None:
            return ""
        if not isinstance(updates, list) or not all(isinstance(u, dict) for u in updates):
            return "coverage_updates must be a list of area assessments"
        merged = {a["area"].strip().casefold(): deepcopy(a) for a in self.coverage_assessment}
        seen = set()
        for u in updates:
            if (any(not isinstance(u.get(k), str) for k in ("area", "requirement", "resolution"))
                    or not u["area"].strip() or not u["requirement"].strip()
                    or not isinstance(u.get("complementary_required"), bool)
                    or u.get("scope") not in ("included", "excluded", "unresolved")
                    or u.get("status") not in ("needs_view", "covered")):
                return "invalid area requirement, scope, status or resolution"
            lists = ("evidence_views", "complementary_views", "gaps")
            if any(not isinstance(u.get(k), list) or not all(
                    isinstance(v, str) and v.strip() for v in u[k]) for k in lists):
                return "area evidence and gaps must be lists of nonempty strings"
            key = u["area"].strip().casefold()
            if key in seen:
                return "update each area only once per decision"
            seen.add(key)
            old = merged.get(key)
            evidence = set(u["evidence_views"] + u["complementary_views"])
            records = {r["id"]: r for r in self.records}
            if not evidence.issubset(set(supplied_ids) & records.keys()):
                return "area citation was not supplied in this active-epoch decision"
            corrected = old and (any(old.get(k) != u[k] for k in (
                "scope", "requirement", "complementary_required"))
                or bool(set(old["gaps"]) - set(u["gaps"])))
            if (u["scope"] == "excluded" or corrected) and not u["resolution"].strip():
                return "scope or requirement correction requires an explicit resolution"
            if u["scope"] == "unresolved" and u["status"] == "covered":
                return "unresolved scope cannot be covered"
            if u["scope"] == "included" and u["status"] == "covered":
                if not u["evidence_views"] or u["gaps"] or not u["resolution"].strip():
                    return "covered area requires explicit visible-surface evidence and no gaps"
                if not evidence.issubset(self.readable_ids()):
                    return "covered area requires readable, non-advisory images"
                if u["complementary_required"]:
                    primary = set(u["evidence_views"])
                    complementary = set(u["complementary_views"])
                    if not any(self._different_angle(a, b, records)
                               for a in primary for b in complementary if a != b):
                        return ("complementary coverage requires assessed different-angle "
                                "image evidence")
            merged[key] = deepcopy(u)
        self.coverage_assessment = list(merged.values())
        if updates and persist:
            self._save()
        return ""

    def readable_ids(self) -> set[str]:
        return {r["id"] for r in self.records
                if r["usability"] == "readable" and not r["advisory_only"]}

    @staticmethod
    def _different_angle(a, b, records):
        first, second = records[a], records[b]
        # only an explicit image comparison, never a receipt pose, sets a different angle
        if any(r.get("assessment", {}).get("compared_to") == other
               and r["assessment"].get("relationship") == "same_view"
               for r, other in ((first, b), (second, a))):
            return False
        return any(r.get("assessment", {}).get("compared_to") == other
                   and r["assessment"].get("relationship") == "different_angle"
                   for r, other in ((first, b), (second, a)))

    def apply_image_assessments(self, updates, supplied_ids, *, decision_n=None,
                                persist: bool = True) -> str:
        """Assess images; deferred snapshots still invalidate downgraded coverage in memory."""
        if updates is None:
            return ""
        if not isinstance(updates, list) or not all(isinstance(u, dict) for u in updates):
            return "image_assessments must be a list"
        records = {r["id"]: r for r in self.records}
        supplied = set(supplied_ids) & records.keys()
        seen = set()
        for u in updates:
            if (not isinstance(u.get("id"), str) or u["id"] not in supplied
                    or u["id"] in seen or u.get("usability") not in (
                        "readable", "blurred", "uncertain")
                    or not isinstance(u.get("reason"), str) or not u["reason"].strip()):
                return "invalid, duplicate or unsupplied image assessment"
            seen.add(u["id"])
            relationship, compared = u.get("relationship"), u.get("compared_to")
            if relationship is not None and relationship not in (
                    "same_view", "different_angle", "uncertain"):
                return "invalid image relationship"
            if ((compared is not None and (not isinstance(compared, str)
                                          or compared not in supplied or compared == u["id"]))
                    or (relationship in ("same_view", "different_angle") and compared is None)):
                return "image comparison requires a different supplied image"
        for u in updates:
            record = records[u["id"]]
            assessment = deepcopy(u) | {"decision_n": decision_n}
            if "compared_to" not in u and "relationship" not in u:
                previous = record.get("assessment", {})
                assessment.update({k: previous[k] for k in ("compared_to", "relationship")
                                   if k in previous})
            record.update(usability=u["usability"], assessment=assessment)
            record.setdefault("assessment_history", []).append(assessment)
        readable = self.readable_ids()
        for area in self.coverage_assessment:
            evidence = set(area["evidence_views"] + area["complementary_views"])
            angles_valid = (not area["complementary_required"] or any(
                self._different_angle(a, b, records)
                for a in area["evidence_views"] for b in area["complementary_views"]
                if a != b))
            if (area.get("scope") == "included" and area["status"] == "covered"
                    and (not evidence.issubset(readable) or not angles_valid)):
                area["status"] = "needs_view"
                area["gaps"] = list(dict.fromkeys(area["gaps"] + [
                    "Cited image assessment changed; reassess the visible surface evidence."]))
        if updates and persist:
            self._save()
        return ""

    def _store(self, rgb, metadata, **fields):
        identity = _frame_identity(metadata)
        for record in self.records:
            if _frame_identity(record["phone"]) == identity:
                return record["id"], "existing immutable image"
        key = f"E{self.epoch:03d}-I{len(self.records) + 1:03d}"
        image_path = self.directory / f"{key}.png"
        Image.fromarray(rgb).save(image_path)
        self.records.append({"id": key, "image": image_path.name, "phone": deepcopy(metadata),
                             "usability": "pending", "assessment_history": [], **fields})
        self._save()
        return key, "saved new immutable image"

    def capture(self, rgb, metadata: dict, pose, *, sampled_at: float,
                odom_age_s: float, capture_role=None, route_context=None) -> tuple[str | None, str]:
        """Only associate a received frame with a fresh, contemporaneously sampled robot pose."""
        if capture_role not in (None, "pre_facing"):
            return None, "unknown stationary capture role"
        decode = metadata.get("receiver_decode_time_s")
        receive_age = metadata.get("receive_age_s")
        if rgb is None or metadata.get("status") != "fresh":
            return None, "phone frame unavailable or stale"
        selection = metadata.get("selection", {}).get("status")
        if selection is not None and selection != "sharpest_stable":
            return None, "gallery needs a stable held sample; live fallback stays separate"
        if not isinstance(decode, (int, float)) or not isinstance(receive_age, (int, float)):
            return None, "missing phone time provenance"
        values = [*pose, sampled_at, decode, receive_age, odom_age_s]
        if len(pose) != 3 or not np.isfinite(values).all():
            return None, "non-finite pose or timing"
        skew = sampled_at - decode
        if not (0 <= odom_age_s <= MAX_ASSOCIATION_AGE_S
                and 0 <= receive_age <= MAX_ASSOCIATION_AGE_S
                and abs(skew) <= MAX_ASSOCIATION_AGE_S):
            return None, "phone/registered pose too old for approximate association"
        group, anchors = None, set()
        for record in self.records:
            if (record["source_kind"] != "stationary"
                    or record.get("capture_role") == "pre_facing"):
                continue
            if record["viewpoint_group"] in anchors:
                continue
            anchors.add(record["viewpoint_group"])
            px, py, heading = record["robot_pose_xyyaw"]
            angle = math.atan2(math.sin(pose[2] - heading), math.cos(pose[2] - heading))
            if (math.hypot(pose[0] - px, pose[1] - py) < POSITION_STEP_M
                    and abs(angle) < HEADING_STEP_RAD):
                group = record["viewpoint_group"]
                break
        if group is None:
            groups = {r["viewpoint_group"] for r in self.records if r["viewpoint_group"]}
            group = f"E{self.epoch:03d}-V{len(groups) + 1:03d}"
        return self._store(rgb, metadata, source_kind="stationary", viewpoint_group=group,
                           advisory_only=False, robot_pose_xyyaw=list(map(float, pose)),
                           pose_sampled_at_s=sampled_at, registered_pose_age_s=odom_age_s,
                           decode_to_pose_sample_s=skew,
                           **({"capture_role": capture_role} if capture_role else {}),
                           **({"route_context": deepcopy(route_context)}
                              if route_context is not None else {}))

    def capture_transit(self, rgb, metadata, *, sampled_at, action_interval, pose=None,
                        pose_sampled_at_s=None, registered_pose_age_s=None,
                        route_context=None, traveled_m=None, motion_stage=None):
        """Retain optional approximate receipt position; never invent an exposure pose."""
        if (rgb is None or metadata.get("status") != "fresh" or not _finite(sampled_at)
                or not _finite(metadata.get("receiver_decode_time_s"))):
            return None, "transit frame lacks receipt-time provenance"
        position = {}
        if any(v is not None for v in (pose, pose_sampled_at_s, registered_pose_age_s,
                                       route_context, traveled_m)):
            sample = {"pose": pose, "sampled_at": sampled_at, "metadata": metadata,
                      "pose_sampled_at_s": pose_sampled_at_s,
                      "registered_pose_age_s": registered_pose_age_s,
                      "route_context": route_context, "traveled_m": traveled_m}
            if not _transit_pose_valid(sample):
                return None, "transit receipt pose or actual-route association unavailable"
            position = {"robot_pose_xyyaw": list(map(float, pose)),
                        "pose_sampled_at_s": pose_sampled_at_s,
                        "registered_pose_age_s": registered_pose_age_s,
                        "decode_to_pose_sample_s": pose_sampled_at_s
                        - metadata["receiver_decode_time_s"],
                        "route_context": deepcopy(route_context), "traveled_m": traveled_m,
                        "association_kind": "approximate_receipt_pose",
                        "exposure_pose_known": False, "capture_role": "transit"}
            if motion_stage is not None:
                position["motion_stage"] = str(motion_stage)
        return self._store(rgb, metadata, source_kind="transit", viewpoint_group=None,
                           advisory_only=False, sampled_at_s=sampled_at,
                           action_interval=deepcopy(action_interval), **position)

    def selected_ids(self, *, transit_ids=(), pre_facing_ids=()) -> list[str]:
        """Ordinary stationary history plus at most two transit and one pre-facing view."""
        selected, cited = set(), set()
        groups = {}
        for record in self.records:
            if (record["source_kind"] == "stationary"
                    and record.get("capture_role") != "pre_facing"):
                groups.setdefault(record["viewpoint_group"], []).append(record)
        readable = self.readable_ids()
        for records in groups.values():
            selected.add(records[-1]["id"])
            qualified = [r for r in records if r["id"] in readable]
            if qualified:
                selected.add(qualified[-1]["id"])
            for record in records:
                assessment = record.get("assessment", {})
                if assessment.get("relationship") == "different_angle":
                    selected.update((record["id"], assessment["compared_to"]))
        for area in self.coverage_assessment:
            cited.update(area["evidence_views"] + area["complementary_views"])
        selected.update(cited)
        ordinary = {r["id"] for records in groups.values() for r in records}
        selected.intersection_update(ordinary)
        for kind, preferred, limit in (("transit", set(transit_ids), 2),
                                        ("pre_facing", set(pre_facing_ids), 1)):
            candidates = [r for r in reversed(self.records)
                          if ((kind == "transit" and r["source_kind"] == "transit"
                               and _positioned_transit(r))
                              or (kind == "pre_facing"
                                  and r["source_kind"] == "stationary"
                                  and r.get("capture_role") == "pre_facing"))
                          and r["phone"].get("selection", {}).get("status") != "latest_fallback"]
            candidates.sort(key=lambda r: (r["id"] in preferred,
                            r["usability"] == "pending", r["id"] in cited), reverse=True)
            added = []
            for record in candidates:
                if any(not _distinct_pose(record["robot_pose_xyyaw"], old["robot_pose_xyyaw"])
                       for old in added):
                    continue
                added.append(record)
                if len(added) == limit:
                    break
            selected.update(r["id"] for r in added)
        return [r["id"] for r in self.records if r["id"] in selected
                and r["phone"].get("selection", {}).get("status") != "latest_fallback"]

    def _selected(self, ids):
        if ids is None:
            return self.records
        wanted = set(ids)
        return [r for r in self.records if r["id"] in wanted]

    def pages(self, *, ids=None) -> list[np.ndarray]:
        """Render exactly the selected active-epoch IDs, including their evidence limitations."""
        pages = []
        width, height = 400, 270
        selected = self._selected(ids)
        for start in range(0, len(selected), PAGE_SIZE):
            records = selected[start:start + PAGE_SIZE]
            sheet = Image.new("RGB", (2 * width, math.ceil(len(records) / 2) * height), "white")
            draw = ImageDraw.Draw(sheet)
            for i, record in enumerate(records):
                x, y = (i % 2) * width, (i // 2) * height
                with Image.open(self.directory / record["image"]) as original:
                    thumb = ImageOps.contain(original.convert("RGB"), (width - 8, 220))
                sheet.paste(thumb, (x + (width - thumb.width) // 2, y + 44))
                status = "advisory" if record["advisory_only"] else record["usability"]
                role = " | PRE-FACING" if record.get("capture_role") == "pre_facing" else ""
                draw.text((x + 4, y + 2), f'{record["id"]} | {status}{role}',
                          fill="black", font=_font(16))
                if record["source_kind"] == "stationary":
                    px, py, yaw = record["robot_pose_xyyaw"]
                    label = f"robot ({px:.2f}, {py:.2f}) m, yaw {math.degrees(yaw):.0f} deg"
                elif _positioned_transit(record):
                    px, py, yaw = record["robot_pose_xyyaw"]
                    label = (f"TRANSIT ~robot ({px:.2f},{py:.2f})m "
                             f"yaw {math.degrees(yaw):.0f}deg at receipt")
                else:
                    label = "TRANSIT: receipt interval only; no metric pose"
                draw.text((x + 4, y + 23), label, fill="black", font=_font(13))
            pages.append(np.asarray(sheet))
        return pages

    def manifest(self, *, ids=None) -> dict:
        selected = self._selected(ids)
        return {"epoch": self.epoch, "view_ids": [r["id"] for r in selected],
                "readable_ids": [r["id"] for r in selected if r["id"] in self.readable_ids()],
                "image_records": [{k: deepcopy(v) for k, v in r.items()
                                   if k != "assessment_history"} for r in selected],
                "index": str((self.directory / "index.json").relative_to(self.root.parent)),
                "association": ASSOCIATION,
                "coverage_assessment": deepcopy(self.coverage_assessment)}


def completion_refusal(evidence, supplied_ids, *, current_fresh: bool,
                       same_epoch: bool, observe_only: bool, coverage=None,
                       readable_ids=()) -> str:
    if observe_only:
        return "observation-only preview cannot certify an exploration run"
    if not same_epoch:
        return "SLAM/start frame changed during this decision"
    if not current_fresh:
        return "current phone/map/registered pose evidence is stale"
    if not isinstance(evidence, dict):
        return "missing visual coverage assessment"
    if not isinstance(evidence.get("coverage_summary"), str) or not evidence[
        "coverage_summary"
    ].strip():
        return "missing visual coverage summary"
    ids = evidence.get("evidence_views")
    if not isinstance(ids, list) or not all(isinstance(k, str) for k in ids):
        return "missing view citations"
    if not ids:
        return "cite at least one readable recorded image"
    if not set(ids).issubset(supplied_ids):
        return "citation was not supplied in this decision's active-epoch history"
    if not set(ids).issubset(readable_ids):
        return "completion requires readable, non-advisory cited images"
    gaps = evidence.get("remaining_gaps")
    if not isinstance(gaps, list) or not all(isinstance(gap, str) for gap in gaps):
        return "remaining_gaps must list unfinished required coverage"
    limitations = evidence.get("limitations", [])
    if not isinstance(limitations, list) or not all(isinstance(v, str) for v in limitations):
        return "limitations must be a list of inaccessible surfaces, assumptions or exclusions"
    if gaps:
        return "remaining_gaps contains unresolved required coverage: " + "; ".join(gaps)
    if coverage is not None:
        if not coverage:
            return "missing persistent inventory of required room areas"
        if not any(a.get("scope") == "included" for a in coverage):
            return "missing included task scope in the persistent coverage assessment"
        unresolved = [a for a in coverage if a.get("scope") == "unresolved" or (
            a.get("scope") == "included" and (a["status"] != "covered" or a["gaps"]))]
        if unresolved:
            return "unresolved required coverage ledger entries: " + "; ".join(
                f"{a['area']} (scope={a['scope']}, status={a['status']}, gaps={a['gaps']})"
                for a in unresolved)
    return ""
