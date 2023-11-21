"""
Conversion between local and global space.
"""

import pymap3d as pm

from .. import drone_odometry_local
from ..common.mavlink.modules import drone_odometry


def drone_position_global_from_local(home_location: drone_odometry.DronePosition,
                                        drone_position_local:
                                            drone_odometry_local.DronePositionLocal) \
    -> "tuple[bool, drone_odometry.DronePosition | None]":
    """
    Local coordinates to global coordinates.
    Return: Drone position in WGS 84.
    """
    latitude, longitude, altitude = pm.ned2geodetic(
        drone_position_local.north,
        drone_position_local.east,
        drone_position_local.down,
        home_location.latitude,
        home_location.longitude,
        home_location.altitude,
    )

    result, drone_position = drone_odometry.DronePosition.create(
        latitude,
        longitude,
        altitude,
    )
    if not result:
        return False, None

    # Get Pylance to stop complaining
    assert drone_position is not None

    return True, drone_position

def __drone_position_local_from_global(home_location: drone_odometry.DronePosition,
                                        drone_position: drone_odometry.DronePosition) \
    -> "tuple[bool, drone_odometry_local.DronePositionLocal | None]":
    """
    Global coordinates to local coordinates.
    Return: Drone position relative to home location (NED system).
    """
    north, east, down = pm.geodetic2ned(
        drone_position.latitude,
        drone_position.longitude,
        drone_position.altitude,
        home_location.latitude,
        home_location.longitude,
        home_location.altitude,
    )

    result, drone_position_local = drone_odometry_local.DronePositionLocal.create(
        north,
        east,
        down,
    )
    if not result:
        return False, None

    # Get Pylance to stop complaining
    assert drone_position_local is not None

    return True, drone_position_local

def drone_odometry_local_from_global(odometry: drone_odometry.DroneOdometry,
                                        home_location: drone_odometry.DronePosition) \
    -> "tuple[bool, drone_odometry_local.DroneOdometryLocal | None]":
    """
    Converts global odometry to local.
    """
    result, drone_position_local = __drone_position_local_from_global(
        home_location,
        odometry.position,
    )
    if not result:
        return False, None

    # Get Pylance to stop complaining
    assert drone_position_local is not None

    result, drone_orientation_local = drone_odometry_local.DroneOrientationLocal.create_wrap(
        odometry.orientation,
    )
    if not result:
        return False, None

    # Get Pylance to stop complaining
    assert drone_orientation_local is not None

    return drone_odometry_local.DroneOdometryLocal.create(
        drone_position_local,
        drone_orientation_local,
    )
