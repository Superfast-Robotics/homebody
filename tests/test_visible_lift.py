import numpy as np

from homebody.helpers.target import sight_blocked


def test_a_box_between_eye_and_point_hides_it_and_one_beside_or_behind_does_not():
    eye = np.zeros(3)
    points = np.array([[1., 0., 0.], [1., .5, 0.], [.3, 0., 0.]])
    box = (np.array([.5, 0., 0.]), np.eye(3), np.array([.05, .1, .1]))
    np.testing.assert_array_equal(sight_blocked(eye, points, [box], .01), [True, False, False])


def test_the_stop_spares_the_box_the_point_sits_in():
    eye = np.zeros(3)
    point = np.array([[1., 0., 0.]])
    box = (np.array([.99, 0., 0.]), np.eye(3), np.array([.005, .1, .1]))
    assert not sight_blocked(eye, point, [box], .03)[0]
    assert sight_blocked(eye, point, [box], .001)[0]
