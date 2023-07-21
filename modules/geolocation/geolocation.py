"""
Converts image space into world space.
"""

import cv2
import numpy as np


class Geolocation:
    """
    Converts image space into world space.

    Image follows xy system with top-left as origin, pixel coordinates refer to their top-left corner.
    Camera follows NED system (c forward, u right, v down).
    """
    def __init__(self,
                 resolution_x: int,
                 resolution_y: int,
                 fov_x: float,
                 fov_y: float):
        """
        Resolution is in pixels.
        Field of view in radians horizontally and vertically across the image (edge to edge).
        """
        self.__resolution_x = resolution_x
        self.__resolution_y = resolution_y

        # c is pointing at centre of image
        # u is from c to right edge
        # v is from c to bottom edge
        self.__vec_c = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.__vec_u = np.array([0.0, np.tan(fov_x / 2), 0.0], dtype=np.float64)
        self.__vec_v = np.array([0.0, 0.0, np.tan(fov_y / 2)], dtype=np.float64)

        raise NotImplementedError


    @staticmethod
    def __vector_from_image_space(pixel: int,
                                  resolution: int,
                                  vector: np.ndarray) -> "tuple[bool, np.ndarray | None]":
        """
        Get u or v vector from pixel coordinate.
        """
        if pixel < 0:
            return False, None

        if resolution < 0:
            return False, None

        if pixel > resolution:
            return False, None

        if len(vector.shape) != 1 or vector.shape[0] != 3:
            return False, None

        # Scaling factor with translation: 2 * p / r - 1 == (2 * p - r) / r
        # Codomain is [-1, 1]
        scaling = float(2 * pixel - resolution) / float(resolution)
        vec_plane = scaling * vector

        return True, vec_plane

    def camera_space_from_image_space(self,
                                      pixel_x: int,
                                      pixel_y: int) -> "tuple[bool, np.ndarray | None]":
        """
        Pixel in image space to vector in camera space.
        """
        if pixel_x < 0:
            return False, None

        if pixel_y < 0:
            return False, None

        result, vec_camera_u = self.__vector_from_image_space(
            pixel_x,
            self.__resolution_x,
            self.__vec_u,
        )
        if not result:
            return False, None

        result, vec_camera_v = self.__vector_from_image_space(
            pixel_y,
            self.__resolution_y,
            self.__vec_v,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert vec_camera_u is not None
        assert vec_camera_v is not None

        vec_pixel = self.__vec_c + vec_camera_u + vec_camera_v

        return True, vec_pixel
