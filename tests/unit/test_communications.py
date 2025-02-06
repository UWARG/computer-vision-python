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


@pytest.fixture
def communications_maker() -> communications.Communications:  # type: ignore
    """
    Construct a Communications instance with the Home position
    """
    result, test_logger = logger.Logger.create("test_logger", False)

    assert result
    assert test_logger is not None

    result, home_position = position_global.PositionGlobal.create(0, 0, 0)
    assert result
    assert home_position is not None

    result, communications_instance = communications.Communications.create(
        home_position, test_logger
    )
    assert result
    assert communications_instance is not None

    yield communications_instance  # type: ignore


class TestCommunications:
    """
    Tests for the Communications.run() method.
    """

    def test_normal_data(self, communications_maker: communications.Communications) -> None:
        """
        Deal with one data to test if it works
        """
        # Setup
        result, local_position = object_in_world.ObjectInWorld.create(10, 20, 1.0)
        assert result
        assert local_position is not None
        objects_in_world = [local_position]

        # Run
        result, generated_objects = communications_maker.run(objects_in_world)

        # Test
        assert result
        assert generated_objects is not None

    def test_correct_data_structure(
        self,
        communications_maker: communications.Communications,
    ) -> None:
        """
        Check if the coordinates are converted into the correct data structure
        """

        # Setup
        result, obj_1 = object_in_world.ObjectInWorld.create(30.0, 40.0, 2.0)
        assert result
        assert obj_1 is not None

        result, obj_2 = object_in_world.ObjectInWorld.create(50.0, 60.0, 3.0)
        assert result
        assert obj_2 is not None

        result, obj_3 = object_in_world.ObjectInWorld.create(70.0, 80.0, 4.0)
        assert result
        assert obj_3 is not None

        result, home_position = position_global.PositionGlobal.create(0, 0, 0)
        assert result
        assert home_position is not None

        objects_in_world = [obj_1, obj_2, obj_3]

        # Run
        result, generated_objects = communications_maker.run(objects_in_world)
        assert result
        assert generated_objects is not None

        # Test
        for i, generated_object in enumerate(generated_objects):

            # should convert the byte to positionGlobal
            result, worker_id, generated_object = message_encoding_decoding.decode_bytes_to_position_global(generated_object)
            assert result
            assert generated_object is not None
            assert worker_id is not None

            # convert the input data to position local
            result, object_position_local = position_local.PositionLocal.create(
                objects_in_world[i].location_x, east=objects_in_world[i].location_y, down=0.0
            )
            assert result

            # convert the input data to position global for comparison
            result, expected = local_global_conversion.position_global_from_position_local(
                home_position, object_position_local
            )
            assert result
            
            assert generated_object == expected
