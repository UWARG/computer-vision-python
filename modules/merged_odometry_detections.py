"""
Drone odometry and object detections.
"""

from . import detections_and_time
from .common.kml.modules import drone_odometry_local


class MergedOdometryDetections:
    """
    Contains odometry/telemetry and detections merged by closest timestamp.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        odometry_local: drone_odometry_local.DroneOdometryLocal,
        detections: "list[detections_and_time.Detection]",
    ) -> "tuple[bool, MergedOdometryDetections | None]":
        """
        odometry_local: Drone position and orientation in local space.
        detections: List of Detections from detections_and_time.
        """
        if len(detections) == 0:
            return False, None

        return True, MergedOdometryDetections(cls.__create_key, odometry_local, detections)

    def __init__(
        self,
        class_private_create_key: object,
        odometry_local: drone_odometry_local.DroneOdometryLocal,
        detections: "list[detections_and_time.Detection]",
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert (
            class_private_create_key is MergedOdometryDetections.__create_key
        ), "Use create() method"

        self.odometry_local = odometry_local
        self.detections = detections

    def __str__(self) -> str:
        """
        To string.
        """
        return (
            f"Merged: {self.odometry_local}, detections: {len(self.detections)}\n"
            + f"{self.detections}"
        )
