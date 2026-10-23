from types import SimpleNamespace

from homebody.motion.carry import Recovery
from homebody.session.settings import Settings
from homebody.skills import navigate


def transit(monkeypatch, readings):
    """A Transit whose carry reads as held or not, tick by tick, from READINGS."""
    settings = Settings()
    ctx = SimpleNamespace(carried={"right": {}}, arms={"right": None}, settings=settings,
                          stage=lambda text: None, observe=lambda epoch: None)
    ride = object.__new__(navigate.Transit)
    ride.plan, ride.ctx, ride.settings = SimpleNamespace(epoch=1), ctx, settings.motion
    ride.frame, ride.doubts, ride.best, ride.recoveries = None, {}, 0., 0
    monkeypatch.setattr(navigate, "continuing", lambda *args, **kwargs: readings.pop(0))

    def recover_carry(side):
        ride.recoveries += 1
        return SimpleNamespace(frame=None, status=Recovery.VERIFIED)
    ride.recover_carry = recover_carry
    return ride


def test_a_carry_is_recovered_only_after_consecutive_doubts(monkeypatch):
    ticks = Settings().motion.carry_doubt_ticks
    readings = [False] * (ticks - 1) + [True] + [False] * ticks
    ride = transit(monkeypatch, list(readings))
    for _ in range(ticks):
        ride.check_carry()
    assert ride.recoveries == 0
    for _ in range(ticks):
        ride.check_carry()
    assert ride.recoveries == 1
    assert ride.doubts == {}
