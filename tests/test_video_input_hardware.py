"""
Simple hardware test, requires camera.
"""

from modules.video_input import video_input


CAMERA = 0


if __name__ == "__main__":
    # Setup
    camera = video_input.VideoInput(
        CAMERA,
    )

    # Run
    result, image = camera.run()

    # Test
    assert result
    assert image is not None

    print("Done!")
