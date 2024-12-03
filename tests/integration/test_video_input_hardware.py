"""
Simple hardware test, requires camera.
"""

from modules.video_input import video_input


CAMERA = 0
WIDTH = 1920
HEIGHT = 1080


def main() -> int:
    """
    Main function.
    """
    # Setup
    # TODO: Common change logging option
    camera = video_input.VideoInput(CAMERA, WIDTH, HEIGHT)

    # Run
    result, image = camera.run()

    # Test
    assert result
    assert image is not None

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
