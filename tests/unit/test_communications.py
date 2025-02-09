"""
Tests the communications class.
"""

import pytest

from modules.communications import communications
from modules.common.modules.logger import logger
from modules import object_in_world
from modules.common.modules import position_local
from modules.common.modules.mavlink import local_global_conversion
from modules.common.modules import position_global
from modules.common.modules.data_encoding import metadata_encoding_decoding
from modules.common.modules.data_encoding import message_encoding_decoding

# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name

LATITUDE_TOLERANCE = 0.000001
LONGITUDE_TOLERANCE = 0.000001
ALTITUDE_TOLERANCE = 7


@pytest.fixture
def home_position() -> position_global.PositionGlobal:  # type: ignore
    """
    Home position.
    """
    # University of Waterloo WGS84 Coordinate
    result, position = position_global.PositionGlobal.create(43.472978, -80.540103, 336.0)
    assert result
    assert position is not None

    yield position


@pytest.fixture
def communications_maker(
    home_position: position_global.PositionGlobal,
) -> communications.Communications:  # type: ignore
    """
    Construct a Communications instance with the Home position
    """
    result, test_logger = logger.Logger.create("test_logger", False)

    assert result
    assert test_logger is not None

    result, communications_instance = communications.Communications.create(
        home_position, test_logger
    )
    assert result
    assert communications_instance is not None

    yield communications_instance  # type: ignore


def object_in_world_from_position_local(
    position_local: position_local.PositionLocal,
) -> object_in_world.ObjectInWorld:
    """
    Convert position local to object_in_world as defined in Communications.py
    """
    result, obj = object_in_world.ObjectInWorld.create(
        position_local.north, position_local.east, 0.0
    )
    assert result
    assert obj is not None

    return obj


def assert_global_positions(
    expected: position_global.PositionGlobal, actual: position_global.PositionGlobal
) -> None:
    """
    Assert each values of the global positions using the Tolerances
    """
    assert abs(expected.latitude - actual.latitude) < LATITUDE_TOLERANCE
    assert abs(expected.longitude - actual.longitude) < LONGITUDE_TOLERANCE
    assert abs(expected.altitude - actual.altitude) < ALTITUDE_TOLERANCE


class TestCommunications:
    """
    Tests for the Communications.run() method.
    """

    def test_run(
        self,
        home_position: position_global.PositionGlobal,
        communications_maker: communications.Communications,
    ) -> None:
        """
        Test if the Communications.run returns the correct instance
        """
        # Setup
        result, position = position_global.PositionGlobal.create(43.472978, -80.540103, 336.0)
        assert result
        assert position is not None

        result, actual = local_global_conversion.position_local_from_position_global(
            home_position, position
        )
        assert result
        assert actual is not None

        objects_in_world = [object_in_world_from_position_local(actual)]

        # Run
        result, metadata, generated_objects = communications_maker.run(objects_in_world)

        # Test
        assert result
        assert isinstance(metadata, bytes)
        for generated_object in generated_objects:
            assert isinstance(generated_object, bytes)

    def test_normal(
        self,
        home_position: position_global.PositionGlobal,
        communications_maker: communications.Communications,
    ) -> None:
        """
        Normal
        """
        # Setup
        result, global_position_1 = position_global.PositionGlobal.create(
            43.472978, -80.540103, 336.0
        )
        assert result
        assert global_position_1 is not None

        result, local_position_1 = local_global_conversion.position_local_from_position_global(
            home_position, global_position_1
        )
        assert result
        assert local_position_1 is not None

        result, global_position_2 = position_global.PositionGlobal.create(
            43.472800, -80.539500, 330.0
        )
        assert result
        assert global_position_2 is not None

        result, local_position_2 = local_global_conversion.position_local_from_position_global(
            home_position, global_position_2
        )
        assert result
        assert local_position_2 is not None

        global_positions = [global_position_1, global_position_2]

        objects_in_world = [
            object_in_world_from_position_local(local_position_1),
            object_in_world_from_position_local(local_position_2),
        ]
        number_of_messages = len(objects_in_world)

        # Run
        result, metadata, generated_objects = communications_maker.run(objects_in_world)
        assert result
        assert metadata is not None
        assert generated_objects is not None

        result, worker_id, actual_number_of_messages = metadata_encoding_decoding.decode_metadata(
            metadata
        )
        assert result
        assert worker_id is not None
        assert actual_number_of_messages is not None

        # Test
        assert actual_number_of_messages == number_of_messages

        # Conversion
        for i, global_position in enumerate(global_positions):
            result, worker_id, actual = message_encoding_decoding.decode_bytes_to_position_global(
                generated_objects[i]
            )
            assert result
            assert worker_id is not None
            assert actual is not None

            assert_global_positions(global_position, actual)

    def test_empty_objects(
        self,
        communications_maker: communications.Communications,
    ) -> None:
        """
        When nothing is passed in
        """
        objects_in_world = []

        result, metadata, generated_objects = communications_maker.run(objects_in_world)
        assert result
        assert metadata is not None
        assert generated_objects is not None

        result, worker_id, actual_number_of_messages = metadata_encoding_decoding.decode_metadata(
            metadata
        )

        # it will encounter an error where metadata is failed to encode
        assert result
        assert worker_id is not None
        assert actual_number_of_messages is not None

        # Test
        assert actual_number_of_messages == 0
        assert len(generated_objects) == 0

    def test_none(self, communications_maker: communications.Communications) -> None:
        """
        When None is passed in
        """
        objects_in_world = None

        result, metadata, generated_objects = communications_maker.run(objects_in_world)
        assert result
        assert metadata is not None
        assert generated_objects is not None

        result, worker_id, actual_number_of_messages = metadata_encoding_decoding.decode_metadata(
            metadata
        )

        # it will encounter an error where metadata is failed to encode
        assert result
        assert worker_id is not None
        assert actual_number_of_messages is not None

        # Test
        assert actual_number_of_messages == 0
        assert len(generated_objects) == 0

    def test_same_as_home(
        self,
        home_position: position_global.PositionGlobal,
        communications_maker: communications.Communications,
    ) -> None:
        """
        When the objects_in_world contains the home positions
        """
        # Setup
        result, local_position = local_global_conversion.position_local_from_position_global(
            home_position, home_position
        )
        assert result
        assert local_position is not None

        actual = object_in_world_from_position_local(local_position)
        objects_in_world = [actual]
        number_of_messages = len(objects_in_world)

        # Run
        result, metadata, generated_objects = communications_maker.run(objects_in_world)
        assert result
        assert metadata is not None
        assert generated_objects is not None

        # Conversion
        result, worker_id, actual_number_of_messages = metadata_encoding_decoding.decode_metadata(
            metadata
        )
        assert result
        assert worker_id is not None
        assert actual_number_of_messages is not None

        # Test
        assert actual_number_of_messages == number_of_messages

        # Conversion
        result, worker_id, actual = message_encoding_decoding.decode_bytes_to_position_global(
            generated_objects[0]
        )
        assert result
        assert worker_id is not None
        assert actual is not None

        # Test
        assert_global_positions(home_position, actual)

    def test_duplicate_coordinates(
        self,
        home_position: position_global.PositionGlobal,
        communications_maker: communications.Communications,
    ) -> None:
        """
        When the objects_in_world contains duplicate positions
        """
        # Setup
        result, global_position = position_global.PositionGlobal.create(
            43.472978, -80.540103, 336.0
        )
        assert result
        assert global_position is not None

        result, local_position = local_global_conversion.position_local_from_position_global(
            home_position, global_position
        )
        assert result
        assert local_position is not None

        position = object_in_world_from_position_local(local_position)

        objects_in_world = [position, position, position]
        number_of_messages = len(objects_in_world)

        # Run
        result, metadata, generated_objects = communications_maker.run(objects_in_world)
        assert result
        assert metadata is not None
        assert generated_objects is not None

        result, worker_id, actual_number_of_messages = metadata_encoding_decoding.decode_metadata(
            metadata
        )
        assert result
        assert worker_id is not None
        assert actual_number_of_messages is not None

        # Test
        assert actual_number_of_messages == number_of_messages

        for generated_object in generated_objects:
            # Conversion
            result, worker_id, actual = message_encoding_decoding.decode_bytes_to_position_global(
                generated_object
            )
            assert result
            assert worker_id is not None
            assert actual is not None

            # Test
            assert_global_positions(global_position, actual)
