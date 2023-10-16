"""
Simple hardware test, requires camera.
"""

import pytest

from modules.video_input import video_input


CAMERA = 0


@pytest.fixture
def camera():
    """
    Camera.
    """
    camera_device = video_input.VideoInput(
        CAMERA,
    )

    yield camera_device


def test_video_input(camera: video_input.VideoInput):
    """
    Test single image.
    """
    # Run
    result, image = camera.run()

    # Test
    assert result
    assert image is not None
