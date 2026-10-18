"""Task templates: subtrees with reactive fallbacks a caller fills in.

pick_and_carry: hold the object (pick with a fresh click; once more; then the other hand), then walk each leg of a
route while still holding it; a leg that ends with the grasp uncertain rechecks it with pick and walks the leg again.
"""
from .core import Action, Condition, Fallback, Retry, Sequence

UNCERTAIN = "HOLD_UNCERTAIN"


def pick_and_carry(*, click, holding, legs, other_hand, picks=2, rechecks=1):
    """The tree. CLICK(board) -> pick arguments for board.state["side"] from a fresh view, or None when the target
    is not visible; HOLDING(board) -> whether that hand holds the object; LEGS: navigate arguments(board) per leg;
    OTHER_HAND: try the other hand when the first fails."""
    held = Condition("holding", holding)

    def picked(name):
        return Sequence(name, Action(name, "pick", click), held)

    def switch(board):
        board.state["side"] = "left" if board.state["side"] == "right" else "right"
        return True

    grasp = [Retry("pick", picked("pick"), picks)]
    if other_hand:
        grasp.append(Sequence("other hand", Condition("switch hand", switch), picked("pick with the other hand")))
    walks = [_walk(f"leg {index + 1}", leg, picked, held, rechecks) for index, leg in enumerate(legs)]
    return Sequence("pick and carry", Fallback("hold", held, *grasp), *walks)


def _walk(name, arguments, picked, held, rechecks):
    """Walk one leg holding the object; after a leg that ends with the grasp uncertain, recheck and walk again."""
    walk = Sequence(name, Action(name, "navigate", arguments), held)

    def uncertain(board):
        return bool(board.last) and board.last.get("code") == UNCERTAIN

    recover = Sequence(f"{name} recheck", Condition("grasp uncertain", uncertain), picked(f"{name} recheck pick"),
                       Sequence(f"{name} again", Action(f"{name} again", "navigate", arguments), held))
    return Fallback(name, walk, *([Retry(f"{name} rechecks", recover, rechecks)] if rechecks else []))
