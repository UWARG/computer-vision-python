"""
Test landing pad tracking
"""

import pytest

from modules import object_in_world
from modules.controller import landing_pad_tracking

DISTANCE_SQUARED_THRESHOLD = 2

@pytest.fixture()
def tracker():
    """
    instantiates model tracking
    """
    tracker = landing_pad_tracking.LandingPadTracking(DISTANCE_SQUARED_THRESHOLD)
    yield tracker

@pytest.fixture
def detections1():
    """
    # Sample instances of 'object_in_world.ObjectInWorld' for testing
    """
    obj1 = object_in_world.ObjectInWorld.create(0,0,8)[1]
    obj2 = object_in_world.ObjectInWorld.create(2,2,4)[1]
    obj3 = object_in_world.ObjectInWorld.create(-2,-2,2)[1]
    obj4 = object_in_world.ObjectInWorld.create(3,3,10)[1]
    obj5 = object_in_world.ObjectInWorld.create(-3,-3,6)[1]
    detections1 = [obj1, obj2, obj3, obj4, obj5]
    yield detections1

@pytest.fixture
def detections2():
    """
    Sample instances of 'object_in_world.ObjectInWorld' for testing
    """
    obj1 = object_in_world.ObjectInWorld.create(0.5,0.5,1)[1]
    obj2 = object_in_world.ObjectInWorld.create(1.5,1.5,3)[1]
    obj3 = object_in_world.ObjectInWorld.create(4,4,7)[1]
    obj4 = object_in_world.ObjectInWorld.create(-4,-4,5)[1]
    obj5 = object_in_world.ObjectInWorld.create(5,5,9)[1]
    detections2 = [obj1, obj2, obj3, obj4, obj5]
    yield detections2


class TestLandingPadTracking:
    """
    Test landing pad tracking run function
    """
    # pylint: disable=protected-access
    def test_is_similar(self):
        obj1 = object_in_world.ObjectInWorld.create(0,0,0)[1]
        obj2 = object_in_world.ObjectInWorld.create(-1,-1,0)[1]
        assert not landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(obj1, obj2, DISTANCE_SQUARED_THRESHOLD)
        obj1 = object_in_world.ObjectInWorld.create(0,0,0)[1]
        obj2 = object_in_world.ObjectInWorld.create(1,1,0)[1]
        assert not landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(obj1, obj2, DISTANCE_SQUARED_THRESHOLD)
        obj1 = object_in_world.ObjectInWorld.create(0,0,0)[1]
        obj2 = object_in_world.ObjectInWorld.create(0.5,0.5,0)[1]
        assert landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(obj1, obj2, DISTANCE_SQUARED_THRESHOLD)
        obj1 = object_in_world.ObjectInWorld.create(0,0,0)[1]
        obj2 = object_in_world.ObjectInWorld.create(-0.5,-0.5,0)[1]
        assert landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(obj1, obj2, DISTANCE_SQUARED_THRESHOLD)
        obj1 = object_in_world.ObjectInWorld.create(0,0,0)[1]
        obj2 = object_in_world.ObjectInWorld.create(2,2,0)[1]
        assert not landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(obj1, obj2, DISTANCE_SQUARED_THRESHOLD)
        obj1 = object_in_world.ObjectInWorld.create(0,0,0)[1]
        obj2 = object_in_world.ObjectInWorld.create(-2,-2,0)[1]
        assert not landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(obj1, obj2, DISTANCE_SQUARED_THRESHOLD)

    def test_run_with_empty_detections_list(self, tracker: landing_pad_tracking.LandingPadTracking):
        """
        Test run method with empty detections list
        """
        success_flag, result = tracker.run([])
        assert not success_flag
        assert result is None

    def test_run_single_input(self, tracker: landing_pad_tracking.LandingPadTracking, detections1: "list[object_in_world.ObjectInWorld]"):
        """
        Test run with only 1 input
        """
        success_flag, result = tracker.run(detections1)
        assert success_flag
        assert result == detections1[2]
        assert tracker._LandingPadTracking__unconfirmed_positives == [detections1[2], detections1[1], detections1[4], detections1[0], detections1[3]]

    def test_run_single_input_similar_detections(self, tracker: landing_pad_tracking.LandingPadTracking, detections1: "list[object_in_world.ObjectInWorld]"):
        """
        Test run with only 1 input where 2 landing pads are similar
        """        
        detections1[1] = object_in_world.ObjectInWorld.create(0.5, 0.5, 4)[1]
        success_flag, result = tracker.run(detections1)
        print(detections1[2].spherical_variance, result.spherical_variance)
        assert success_flag
        assert result == detections1[2]
        assert tracker._LandingPadTracking__unconfirmed_positives == [detections1[2], detections1[1], detections1[4], detections1[3]]

    def test_run_multiple_inputs(self, tracker: landing_pad_tracking.LandingPadTracking, detections1: "list[object_in_world.ObjectInWorld]", detections2: "list[object_in_world.ObjectInWorld]"):
        """
        Test run with 2 inputs where some landing pads are similar
        """
        success_flag, result = tracker.run(detections1)
        assert success_flag
        assert result == detections1[2]
        assert tracker._LandingPadTracking__unconfirmed_positives == [detections1[2], detections1[1], detections1[4], detections1[0], detections1[3]]

        success_flag, result = tracker.run(detections2)
        assert success_flag
        assert result == detections2[0]
        assert tracker._LandingPadTracking__unconfirmed_positives == [detections2[0], detections1[2], detections2[1], detections2[3], detections1[4], detections2[2], detections2[4], detections1[3]]

    def test_run_with_confirmed_positive(self, tracker:landing_pad_tracking.LandingPadTracking, detections1: "list[object_in_world.ObjectInWorld]"):
        """
        Test run when there is a confirmed positive
        """
        confirmed_positive = object_in_world.ObjectInWorld.create(1, 1, 1)[1]
        tracker.mark_confirmed_positive(confirmed_positive)
        assert tracker._LandingPadTracking__confirmed_positives[0] == confirmed_positive
        success_flag, result = tracker.run(detections1)
        assert success_flag
        assert result == confirmed_positive
    
    def test_mark_false_positive(self, tracker:landing_pad_tracking.LandingPadTracking, detections2: "list[object_in_world.ObjectInWorld]"):
        """
        Test if marking false positives removes similar landing pads
        """
        false_positive = object_in_world.ObjectInWorld.create(1, 1, 1)[1]
        tracker.run(detections2)
        tracker.mark_false_positive(false_positive)
        assert tracker._LandingPadTracking__false_positives[0] == false_positive
        for i in tracker._LandingPadTracking__unconfirmed_positives:
            print(i.spherical_variance)
        print(tracker._LandingPadTracking__is_similar(false_positive, detections2[1], DISTANCE_SQUARED_THRESHOLD))
        assert tracker._LandingPadTracking__unconfirmed_positives == [detections2[3], detections2[2], detections2[4]]