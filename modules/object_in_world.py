"""
Location of the object in world space.
"""


class ObjectInWorld:
    """
    Contains the estimated location of the object in local coordinates.
    """

    __create_key = object()

    @classmethod
    def create(
        cls, location_x: float, location_y: float, spherical_variance: float, label: int
    ) -> "tuple[bool, ObjectInWorld | None]":
        """
        location_x, location_y: Location of the object.
        spherical_variance: Uncertainty of the location.
        label: Type of object in real world.
        """
        if spherical_variance < 0.0:
            return False, None

        return True, ObjectInWorld(
            cls.__create_key, location_x, location_y, spherical_variance, label
        )

    def __init__(
        self,
        class_private_create_key: object,
        location_x: float,
        location_y: float,
        spherical_variance: float,
        label: int,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is ObjectInWorld.__create_key, "Use create() method"

        self.location_x = location_x
        self.location_y = location_y
        self.spherical_variance = spherical_variance
        self.label = label
