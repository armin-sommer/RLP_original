import pytest

from rp1.training.harness.schedule import cosine_interpolate


def test_cosine_interpolate_endpoints_and_disabled_final() -> None:
    assert cosine_interpolate(1.0, 0.0, 0, 10) == pytest.approx(1.0)
    assert cosine_interpolate(1.0, 0.0, 10, 10) == pytest.approx(0.0)
    assert cosine_interpolate(1.0, None, 5, 10) == 1.0
