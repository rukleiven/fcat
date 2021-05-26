from fcat.inner_loop_controller import nu_gap, ncrs_margin
from control import tf
import pytest


def test_nugap():
    tf_0 = tf([10], [1, 0.1])
    tf_1 = tf_0
    tf_2 = tf([1], [1, 0.59])
    tf_3 = tf([10], [1, -0.1])
    tf_4 = tf([10], [1, -0.1, 0.1])
    expect1 = 0.0
    expect2 = 0.8235
    expect3 = 0.02
    expect4 = 0.8535

    assert nu_gap(tf_0, tf_1) == pytest.approx(expect1, 5e-3)
    assert nu_gap(tf_0, tf_2) == pytest.approx(expect2, 5e-3)
    assert nu_gap(tf_2, tf_0) == pytest.approx(expect2, 5e-3)
    assert nu_gap(tf_0, tf_3) == pytest.approx(expect3, 5e-3)
    assert nu_gap(tf_0, tf_4) == pytest.approx(expect4, 5e-3)


def test_ncrs_margin():
    P1 = tf([1, 2], [1, 5, 10])
    C1 = tf([4.4], [1, 0])

    P2 = tf([4], [1, -0.001])
    C2 = tf([1], [1])
    C3 = tf([10], [1])
    C4 = tf([10], [1, 1])

    expect1 = 0.1961
    expect2 = 0.7069
    expect3 = 0.0995
    expect4 = 0.0711

    assert ncrs_margin(P1, C1) == pytest.approx(expect1, 1e-3)
    assert ncrs_margin(P2, C2) == pytest.approx(expect2, 1e-3)
    assert ncrs_margin(P2, C3) == pytest.approx(expect3, 1e-3)
    assert ncrs_margin(P2, C4) == pytest.approx(expect4, 1e-3)

