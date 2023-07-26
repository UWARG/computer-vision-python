"""
Test landing pad tracking
"""

import pytest

from modules import object_in_world
from modules.controller import landing_pad_tracking


DISTANCE_SQUARED_THRESHOLD = 2  # Actual distance threshold is sqrt(2)


@pytest.fixture()
def tracker():
    """
    Instantiates model tracking
    """
    tracker = landing_pad_tracking.LandingPadTracking(DISTANCE_SQUARED_THRESHOLD)
    yield tracker

@pytest.fixture
def detections1():
    """
    Sample instances of 'object_in_world.ObjectInWorld' for testing
    """
    result1, obj1 = object_in_world.ObjectInWorld.create(0,0,8)
    result2, obj2 = object_in_world.ObjectInWorld.create(2,2,4)
    result3, obj3 = object_in_world.ObjectInWorld.create(-2,-2,2)
    result4, obj4 = object_in_world.ObjectInWorld.create(3,3,10)
    result5, obj5 = object_in_world.ObjectInWorld.create(-3,-3,6)
    assert result1 and result2 and result3 and result4 and result5
    detections1 = [obj1, obj2, obj3, obj4, obj5]
    yield detections1

@pytest.fixture
def detections2():
    """
    Sample instances of 'object_in_world.ObjectInWorld' for testing
    """
    result1, obj1 = object_in_world.ObjectInWorld.create(0.5,0.5,1)
    result2, obj2 = object_in_world.ObjectInWorld.create(1.5,1.5,3)
    result3, obj3 = object_in_world.ObjectInWorld.create(4,4,7)
    result4, obj4 = object_in_world.ObjectInWorld.create(-4,-4,5)
    result5, obj5 = object_in_world.ObjectInWorld.create(5,5,9)
    assert result1 and result2 and result3 and result4 and result5
    detections2 = [obj1, obj2, obj3, obj4, obj5]
    yield detections2

@pytest.fixture
def detections3():
    """
    Sample instances of 'object_in_world.ObjectInWorld' for testing
    """
    result1, obj1 = object_in_world.ObjectInWorld.create(0,0,8)
    result2, obj2 = object_in_world.ObjectInWorld.create(0.5,0.5,4)
    result3, obj3 = object_in_world.ObjectInWorld.create(-2,-2,2)
    result4, obj4 = object_in_world.ObjectInWorld.create(3,3,10)
    result5, obj5 = object_in_world.ObjectInWorld.create(-3,-3,6)
    assert result1 and result2 and result3 and result4 and result5
    detections3 = [obj1, obj2, obj3, obj4, obj5]
    yield detections3


class TestSimilar:
    """
    Test if similar function correctly determines if 2 landing pads are close enough to be
    considered similar
    """
    # Required for testing
    # pylint: disable=protected-access
    def test_is_similar_positive_equal_to_threshold(self):
        """
        Test case where the second landing pad has positive coordinates and the distance between
        them is equal to the distance threshold
        """
        obj1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj2 = object_in_world.ObjectInWorld.create(1, 1, 0)[1]
        expected = False

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(
            obj1, obj2, DISTANCE_SQUARED_THRESHOLD
        )

        assert actual == expected

    def test_is_similar_negative_equal_to_threshold(self):
        """
        Test case where the second landing pad has negative coordinates and the distance between
        them is equal to the distance threshold
        """
        obj1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj2 = object_in_world.ObjectInWorld.create(-1, -1, 0)[1]
        expected = False

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(
            obj1, obj2, DISTANCE_SQUARED_THRESHOLD
        )

        assert actual == expected

    def test_is_similar_positive_less_than_threshold(self):
        """
        Test case where the second landing pad has positive coordinates and the distance between
        them is less than the distance threshold
        """
        obj1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj2 = object_in_world.ObjectInWorld.create(0.5, 0.5, 0)[1]
        expected = True

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(
            obj1, obj2, DISTANCE_SQUARED_THRESHOLD
        )

        assert actual == expected

    def test_is_similar_negative_less_than_threshold(self):
        """
        Test case where the second landing pad has negative coordinates and the distance between
        them is less than the distance threshold
        """
        obj1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj2 = object_in_world.ObjectInWorld.create(-0.5, -0.5, 0)[1]
        expected = True

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(
            obj1, obj2, DISTANCE_SQUARED_THRESHOLD
            )

        assert actual == expected

    def test_is_similar_positive_more_than_threshold(self):
        """
        Test case where the second landing pad has positive coordinates and the distance between
        them is more than the distance threshold
        """
        obj1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj2 = object_in_world.ObjectInWorld.create(2, 2, 0)[1]
        expected = False

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(
            obj1, obj2, DISTANCE_SQUARED_THRESHOLD
            )

        assert actual == expected

    def test_is_similar_negative_more_than_threshold(self):
        """
        Test case where the second landing pad has negative coordinates and the distance between
        them is more than the distance threshold
        """
        obj1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj2 = object_in_world.ObjectInWorld.create(-2, -2, 0)[1]
        expected = False

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(
            obj1, obj2, DISTANCE_SQUARED_THRESHOLD
        )

        assert actual == expected

    # pylint: enable=protected-access

class TestMarkFalsePositive:
    """
    Test if landing pad tracking correctly marks a detection as a false positive
    """
    # Required for testing
    # pylint: disable=protected-access
    def test_mark_false_positive_no_similar(self,
                                            tracker: landing_pad_tracking.LandingPadTracking,
                                            detections1: "list[object_in_world.ObjectInWorld]"):
        """
        Test if marking false positive adds detection to list of false positives
        """
        false_positive = object_in_world.ObjectInWorld.create(20, 20, 20)[1]
        tracker._LandingPadTracking__unconfirmed_positives = detections1
        expected = [false_positive]
        expected_unconfirmed_positives = [
            detections1[0],
            detections1[1],
            detections1[2],
            detections1[3],
            detections1[4]
        ]
        
        tracker.mark_false_positive(false_positive)

        assert tracker._LandingPadTracking__false_positives == expected
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives

    def test_mark_false_positive_with_similar(self,
                                              tracker: landing_pad_tracking.LandingPadTracking,
                                              detections2: "list[object_in_world.ObjectInWorld]"):
        """
        Test if marking false positive adds detection to list of false positives and removes
        similar landing pads
        """
        false_positive = object_in_world.ObjectInWorld.create(1, 1, 1)[1]
        tracker._LandingPadTracking__unconfirmed_positives = detections2
        expected = [false_positive]
        expected_unconfirmed_positives = [detections2[2], detections2[3], detections2[4]]
        
        tracker.mark_false_positive(false_positive)

        assert tracker._LandingPadTracking__false_positives == expected
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives
    
    def test_mark_multiple_false_positive(self,
                                          tracker: landing_pad_tracking.LandingPadTracking,
                                          detections1: "list[object_in_world.ObjectInWorld]"):
        """
        Test if marking false positive adds detection to list of false positives
        """
        false_positive1 = object_in_world.ObjectInWorld.create(0, 0, 1)[1]
        false_positive2 = object_in_world.ObjectInWorld.create(2, 2, 1)[1]
        tracker._LandingPadTracking__unconfirmed_positives = detections1
        expected = [false_positive1, false_positive2]
        expected_unconfirmed_positives = [detections1[2], detections1[3], detections1[4]]
        
        tracker.mark_false_positive(false_positive1)
        tracker.mark_false_positive(false_positive2)

        assert tracker._LandingPadTracking__false_positives == expected
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives

    # pylint: enable=protected-access


class TestMarkConfirmedPositive:
    """
    Test if landing pad tracking correctly marks a detection as a confirmed positive
    """
    # Required for testing
    # pylint: disable=protected-access
    def test_mark_confirmed_positive(self, tracker: landing_pad_tracking.LandingPadTracking):
        """
        Test if marking confirmed positive adds detection to list of confirmed positives
        """
        confirmed_positive = object_in_world.ObjectInWorld.create(1, 1, 1)[1]
        expected = [confirmed_positive]
        
        tracker.mark_confirmed_positive(confirmed_positive)

        assert tracker._LandingPadTracking__confirmed_positives == expected
    
    def test_mark_multiple_confirmed_positives(self,
                                               tracker: landing_pad_tracking.LandingPadTracking):
        """
        Test if marking confirmed positive adds detection to list of confirmed positives
        """
        confirmed_positive1 = object_in_world.ObjectInWorld.create(1, 1, 1)[1]
        confirmed_positive2 = object_in_world.ObjectInWorld.create(2, 2, 1)[1]
        expected = [confirmed_positive1, confirmed_positive2]
        
        tracker.mark_confirmed_positive(confirmed_positive1)
        tracker.mark_confirmed_positive(confirmed_positive2)

        assert tracker._LandingPadTracking__confirmed_positives == expected

    # pylint: enable=protected-access


class TestLandingPadTracking:
    """
    Test landing pad tracking run function
    """
    # Required for testing
    # pylint: disable=protected-access
    def test_run_with_empty_detections_list(self, tracker: landing_pad_tracking.LandingPadTracking):
        """
        Test run method with empty detections list
        """
        result, actual = tracker.run([])
        assert not result
        assert actual is None

    def test_run_one_input(self,
                              tracker: landing_pad_tracking.LandingPadTracking,
                              detections1: "list[object_in_world.ObjectInWorld]"):
        """
        Test run with only 1 input
        """
        expected_output = detections1[2]
        expected_unconfirmed_positives = [
            detections1[2],
            detections1[1],
            detections1[4],
            detections1[0],
            detections1[3]
        ]
        
        result, actual = tracker.run(detections1)
        
        assert result
        assert actual == expected_output
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives

    def test_run_one_input_similar_detections(self,
                                              tracker: landing_pad_tracking.LandingPadTracking,
                                              detections3: "list[object_in_world.ObjectInWorld]"):
        """
        Test run with only 1 input where 2 landing pads are similar
        """        
        expected_output = detections3[2]
        expected_unconfirmed_positives = [
            detections3[2],
            detections3[1],
            detections3[4],
            detections3[3]
        ]
        
        result, actual = tracker.run(detections3)

        assert result
        assert actual == expected_output
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives

    def test_run_multiple_inputs(self,
                                 tracker: landing_pad_tracking.LandingPadTracking,
                                 detections1: "list[object_in_world.ObjectInWorld]",
                                 detections2: "list[object_in_world.ObjectInWorld]"):
        """
        Test run with 2 inputs where some landing pads are similar
        """
        expected_output = detections2[0]
        expected_unconfirmed_positives = [
            detections2[0],
            detections1[2],
            detections2[1],
            detections2[3],
            detections1[4],
            detections2[2],
            detections2[4],
            detections1[3]
        ]

        tracker.run(detections1)
        result, actual = tracker.run(detections2)

        assert result
        assert actual == expected_output
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives

    def test_run_with_confirmed_positive(self,
                                         tracker: landing_pad_tracking.LandingPadTracking,
                                         detections1: "list[object_in_world.ObjectInWorld]"):
        """
        Test run when there is a confirmed positive
        """
        confirmed_positive = object_in_world.ObjectInWorld.create(1, 1, 1)[1]
        tracker._LandingPadTracking__confirmed_positives.append(confirmed_positive)
        expected = confirmed_positive

        result, actual = tracker.run(detections1)
        
        assert result
        assert actual == expected
    
    def test_run_with_false_positive(self,
                                     tracker: landing_pad_tracking.LandingPadTracking,
                                     detections2: "list[object_in_world.ObjectInWorld]"):
        """
        Test to see if run function doesn't add landing pads that are similar to false positives
        """
        tracker._LandingPadTracking__false_positives.append(
            object_in_world.ObjectInWorld.create(1, 1, 1)[1]
        )
        expected_unconfirmed_positives = [detections2[3], detections2[2], detections2[4]]

        tracker.run(detections2)
        
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives

    # pylint: enable=protected-access
