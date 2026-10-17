"""The new-skill benchmark's per-kind scores, from synthetic physics rows."""
import math

from evolve import skill_bench_tasks as tasks


def obj(xy, z=0.85, yaw=0.0, hands=(), supports=("worktable",), contacts=(), half=0.03):
    w, zq = math.cos(yaw / 2), math.sin(yaw / 2)
    corners = [[xy[0] + dx, xy[1] + dy, z + dz] for dx in (-half, half) for dy in (-half, half) for dz in (-half, half)]
    return {"position": [xy[0], xy[1], z], "quaternion_wxyz": [w, 0.0, 0.0, zq], "corners": corners,
            "hand_contacts": list(hands), "support_contacts": list(supports), "object_contacts": list(contacts)}


def rows(**objects):
    return [{"time": 0.0, "objects": objects}, {"time": 1.0, "objects": objects}]


TABLE = {"table_top": 0.82}


def test_push_counts_the_zone_and_half_way():
    task = {**TABLE, "zone_xy": [0.3, 0.0], "zone_half": [0.08, 0.08], "start_xy": [0.0, 0.0]}
    assert tasks.to_zone(rows(a=obj([0.31, 0.02])), task, "a")["stage"] == 2
    assert tasks.to_zone(rows(a=obj([0.18, 0.0])), task, "a")["stage"] == 1
    assert tasks.to_zone(rows(a=obj([0.31, 0.0], hands=("right",))), task, "a")["stage"] < 2  # still held


def test_rotate_allows_a_box_turned_by_half_a_turn_more():
    task = {**TABLE, "start_yaw": 0.0, "turn": math.radians(90), "start_xy": [0.0, 0.0]}
    assert tasks.rotate(rows(a=obj([0.0, 0.0], yaw=math.radians(-85))), task, "a")["stage"] == 2  # 90 - 180 = -90
    assert tasks.rotate(rows(a=obj([0.0, 0.0], yaw=math.radians(50))), task, "a")["stage"] == 1
    assert tasks.rotate(rows(a=obj([0.0, 0.0], yaw=0.0)), task, "a")["stage"] == 0


def test_stack_needs_rest_on_the_base_and_release():
    task = {**TABLE, "under": "b", "under_top": 0.9}
    top = obj([0.0, 0.0], z=0.94, supports=(), contacts=("b",))
    assert tasks.stack(rows(a=top, b=obj([0.0, 0.0])), task, "a")["stage"] == 2
    assert tasks.stack(rows(a=obj([0.05, 0.0]), b=obj([0.0, 0.0])), task, "a")["stage"] == 1


def test_hand_over_needs_only_the_receiving_hand():
    task = {**TABLE, "to": "left"}
    assert tasks.hand_over(rows(a=obj([0, 0], z=1.0, hands=("left",), supports=())), task, "a")["stage"] == 2
    assert tasks.hand_over(rows(a=obj([0, 0], z=1.0, hands=("left", "right"), supports=())), task, "a")["stage"] == 1
    assert tasks.hand_over(rows(a=obj([0, 0])), task, "a")["stage"] == 0


def test_lift_high_and_sweep():
    lift = {**TABLE, "height": 0.25}
    assert tasks.lift_high(rows(a=obj([0, 0], z=1.12, hands=("right",), supports=())), lift, "a")["stage"] == 2
    assert tasks.lift_high(rows(a=obj([0, 0], z=0.95, hands=("right",), supports=())), lift, "a")["stage"] == 1
    sweep = {**TABLE, "zone_xy": [0.0, 0.0], "zone_half": [0.1, 0.1], "objects": ["a", "b"]}
    assert tasks.sweep(rows(a=obj([0.2, 0.0]), b=obj([0.0, 0.25])), sweep, "a")["stage"] == 2
    assert tasks.sweep(rows(a=obj([0.2, 0.0]), b=obj([0.0, 0.0])), sweep, "a")["stage"] == 1
