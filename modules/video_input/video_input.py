"""
Combines image and timestamp together.
"""

from .. import image_and_time
from ..common.modules.camera import base_camera
from ..common.modules.camera import camera_configurations
from ..common.modules.camera import camera_factory
from ..common.modules.logger import logger


class VideoInput:
    """
    Combines image and timestamp together.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        camera_option: camera_factory.CameraOption,
        width: int,
        height: int,
        config: camera_configurations.OpenCVCameraConfig | camera_configurations.PiCameraConfig,
        save_prefix: str,
        local_logger: logger.Logger,
    ) -> "tuple[True, VideoInput] | tuple[False, None]":
        """
        camera_option specifies which camera driver to use.
        width is the width of the images the camera takes in pixels.
        height is the height of the images the camera takes in pixels.
        camera_config specifies camera settings.
        save_prefix is name of iamge log files. Leave as empty string for not logging the images.
        """
        result, camera = camera_factory.create_camera(camera_option, width, height, config)
        if not result:
            return False, None
        return True, VideoInput(cls.__create_key, camera, save_prefix, local_logger)

    def __init__(
        self,
        class_private_create_key: object,
        camera: base_camera.BaseCameraDevice,
        save_prefix: str,
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is VideoInput.__create_key, "Use create() method."

        self.__device = camera
        self.__save_prefix = save_prefix
        self.__logger = local_logger

    def run(self) -> "tuple[True, image_and_time.ImageAndTime] | tuple[False, None]":
        """
        Returns a possible ImageAndTime with current timestamp.
        """
        result, image = self.__device.run()
        if not result:
            self.__logger.warning("Failed to take image")
            return False, None

        # If __save_prefix is not empty string, then save image
        if self.__save_prefix:
            self.__logger.save_image(image, self.__save_prefix)

        return image_and_time.ImageAndTime.create(image)
