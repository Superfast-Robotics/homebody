"""What one execute changed, in words, from the robot's own sensing: how far the base moved, each hand's
closure and finger effort before and after, and what the hand holds by the skills' own grasp records.
Coding agents judge an outcome more reliably from such a statement than from the images alone."""
import math

import numpy as np

HOLDS = {None: "holds nothing", "verified": "holds its object (grasp verified)", "uncertain": "grasp uncertain"}
MOVED_M = 0.01  # base motion below this is reported as none
TURNED_DEG = 1.0


def change_report(before, after, carried_before, carried_after) -> list[str]:
    """Lines describing how AFTER differs from BEFORE (frames) and CARRIED_BEFORE from CARRIED_AFTER (side ->
    carry record)."""
    step = np.asarray(after.base_pose[:2]) - np.asarray(before.base_pose[:2])
    moved = float(np.hypot(*step))
    turned = math.degrees(math.remainder(float(after.base_pose[2]) - float(before.base_pose[2]), math.tau))
    if moved < MOVED_M and abs(turned) < TURNED_DEG:
        lines = ["base: did not move"]
    else:
        lines = [f"base: moved {moved:.2f} m, turned {turned:.0f} deg"]
    for side in (side for side in ("left", "right") if side in before.hands and side in after.hands):
        a, b = before.hands[side], after.hands[side]
        held = [HOLDS.get((carry or {}).get("state"), "grasp uncertain")
                for carry in (carried_before.get(side), carried_after.get(side))]
        holding = held[0] if held[0] == held[1] else f"{held[0]} -> {held[1]}"
        lines.append(f"{side} hand: closure {a.closure:.2f} -> {b.closure:.2f}, finger effort "
                     f"{float(np.mean(a.finger_effort_fraction)):.2f} -> {float(np.mean(b.finger_effort_fraction)):.2f}; "
                     f"{holding}")
    return lines
