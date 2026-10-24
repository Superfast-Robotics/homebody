"""The sim server's wire format: the Robot records (Frame, Measurement, WristView, HandState, NavigationMap) and
numpy arrays survive a round trip exactly, and anything unknown is refused rather than guessed."""
import numpy as np
import pytest

from homebody.primitives.observations import Frame, HandState, Measurement, NavigationMap, WristView
from homebody.simserver.codec import pack, unpack


def hand():
    return HandState(joints=np.arange(7, dtype=float), closure=0.4, finger_effort_fraction=np.array([.1, .2, .3]),
                     finger_closure_gap=np.array([.0, .5, 1.]))


def measurement():
    return Measurement(3, 17, 12.5, np.eye(4), np.eye(4) * 2, np.array([1., 2., .3]), np.linspace(0, 1, 29),
                       {"left": hand(), "right": hand()})


def test_numbers_text_lists_and_arrays_round_trip():
    value = {"a": 1, "b": [1.5, "x", None, True], "c": np.arange(12, dtype=np.int32).reshape(3, 4),
             "d": np.full((2, 2), np.nan, dtype=np.float32), "e": np.zeros((300, 400), dtype=bool)}
    back = unpack(pack(value))
    assert back["a"] == 1 and back["b"] == [1.5, "x", None, True]
    assert back["c"].dtype == np.int32 and (back["c"] == value["c"]).all()
    assert back["d"].dtype == np.float32 and np.isnan(back["d"]).all()
    assert back["e"].shape == (300, 400) and not back["e"].any()


def test_the_robot_records_round_trip():
    m = measurement()
    back = unpack(pack(m))
    assert isinstance(back, Measurement) and back.epoch == 3 and back.sequence == 17 and back.time == 12.5
    assert np.array_equal(back.joints, m.joints) and isinstance(back.hands["left"], HandState)
    assert back.hands["right"].closure == pytest.approx(0.4)
    navigation = NavigationMap(np.zeros((40, 50), dtype=bool), np.array([[.025, 0, -1], [0, -.025, 1], [0, 0, 1]]), .025)
    frame = Frame(3, 18, 12.6, np.zeros((4, 6, 3), np.uint8), np.ones((4, 6), np.float32), np.eye(3), np.eye(4),
                  np.eye(4), np.zeros(3), np.zeros(29), {"left": hand(), "right": hand()}, np.zeros((4, 6), np.int32),
                  navigation)
    back = unpack(pack(frame))
    assert isinstance(back, Frame) and isinstance(back.navigation, NavigationMap)
    assert back.navigation.occupied.shape == (40, 50) and back.rgb.dtype == np.uint8
    view = WristView(3, 12.7, "left", np.zeros((4, 6, 3), np.uint8), np.ones((4, 6)), np.zeros((4, 6), np.int32),
                     np.eye(3), np.eye(4))
    assert unpack(pack(view)).side == "left"


def test_an_unknown_record_or_array_type_is_refused():
    with pytest.raises(TypeError):
        pack(object())
    with pytest.raises(TypeError):
        pack(np.array(["text"]))  # only numeric and boolean arrays travel
