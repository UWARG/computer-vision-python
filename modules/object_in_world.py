"""
Position of the object in world space
"""


# Basically a struct
# pylint: disable=too-few-public-methods
class ObjectInWorld:
    """
    Contains the estimated position of the object
    """
    __create_key = object()

    @classmethod
    def create(cls, position_x: float, position_y: float, spherical_variance: float) -> "tuple[bool, FrameAndTime | None]":
        """
        Position in local coordinates
        """
        if spherical_variance < 0.0:
            return False, None

        return True, ObjectInWorld(cls.__create_key, position_x, position_y, spherical_variance)

    def __init__(self, class_private_create_key, position_x: float, position_y: float, spherical_variance: float):
        """
        Private constructor, use create() method
        """
        assert class_private_create_key is ObjectInWorld.__create_key, "Use create() method"

        self.position_x = position_x
        self.position_y = position_y
        self.spherical_variance = spherical_variance

# pylint: enable=too-few-public-methods
