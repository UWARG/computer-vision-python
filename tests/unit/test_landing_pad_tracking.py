"""
Test landing pad tracking.
"""

import pytest

from modules import object_in_world
from modules.decision import landing_pad_tracking


DISTANCE_SQUARED_THRESHOLD = 2  # Actual distance threshold is sqrt(2)


# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


@pytest.fixture()
def tracker() -> landing_pad_tracking.LandingPadTracking:  # type: ignore
    """
    Instantiates model tracking.
    """
    tracking = landing_pad_tracking.LandingPadTracking(DISTANCE_SQUARED_THRESHOLD)
    yield tracking  # type: ignore


@pytest.fixture
def detections_1() -> "list[object_in_world.ObjectInWorld]":  # type: ignore
    """
    Sample instances of ObjectInWorld for testing.
    """
    result, obj_1 = object_in_world.ObjectInWorld.create(0, 0, 8)
    assert result
    assert obj_1 is not None

    result, obj_2 = object_in_world.ObjectInWorld.create(2, 2, 4)
    assert result
    assert obj_2 is not None

    result, obj_3 = object_in_world.ObjectInWorld.create(-2, -2, 2)
    assert result
    assert obj_3 is not None

    result, obj_4 = object_in_world.ObjectInWorld.create(3, 3, 10)
    assert result
    assert obj_4 is not None

    result, obj_5 = object_in_world.ObjectInWorld.create(-3, -3, 6)
    assert result
    assert obj_5 is not None

    detections = [obj_1, obj_2, obj_3, obj_4, obj_5]
    yield detections  # type: ignore


@pytest.fixture
def detections_2() -> "list[object_in_world.ObjectInWorld]":  # type: ignore
    """
    Sample instances of ObjectInWorld for testing.
    """
    result, obj_1 = object_in_world.ObjectInWorld.create(0.5, 0.5, 1)
    assert result
    assert obj_1 is not None

    result, obj_2 = object_in_world.ObjectInWorld.create(1.5, 1.5, 3)
    assert result
    assert obj_2 is not None

    result, obj_3 = object_in_world.ObjectInWorld.create(4, 4, 7)
    assert result
    assert obj_3 is not None

    result, obj_4 = object_in_world.ObjectInWorld.create(-4, -4, 5)
    assert result
    assert obj_4 is not None

    result, obj_5 = object_in_world.ObjectInWorld.create(5, 5, 9)
    assert result
    assert obj_5 is not None

    detections = [obj_1, obj_2, obj_3, obj_4, obj_5]
    yield detections  # type: ignore


@pytest.fixture
def detections_3() -> "list[object_in_world.ObjectInWorld]":  # type: ignore
    """
    Sample instances of ObjectInWorld for testing.
    """
    result, obj_1 = object_in_world.ObjectInWorld.create(0, 0, 8)
    assert result
    assert obj_1 is not None

    result, obj_2 = object_in_world.ObjectInWorld.create(0.5, 0.5, 4)
    assert result
    assert obj_2 is not None

    result, obj_3 = object_in_world.ObjectInWorld.create(-2, -2, 2)
    assert result
    assert obj_3 is not None

    result, obj_4 = object_in_world.ObjectInWorld.create(3, 3, 10)
    assert result
    assert obj_4 is not None

    result, obj_5 = object_in_world.ObjectInWorld.create(-3, -3, 6)
    assert result
    assert obj_5 is not None

    detections = [obj_1, obj_2, obj_3, obj_4, obj_5]
    yield detections  # type: ignore


class TestSimilar:
    """
    Test if similar function correctly determines if 2 landing pads are close enough to be
    considered similar.
    """

    def test_is_similar_positive_equal_to_threshold(self) -> None:
        """
        Test case where the second landing pad has positive coordinates and the distance between
        them is equal to the distance threshold.
        """
        obj_1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj_2 = object_in_world.ObjectInWorld.create(1, 1, 0)[1]
        expected = False

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(  # type: ignore
            obj_1,
            obj_2,
            DISTANCE_SQUARED_THRESHOLD,
        )

        assert actual == expected

    def test_is_similar_negative_equal_to_threshold(self) -> None:
        """
        Test case where the second landing pad has negative coordinates and the distance between
        them is equal to the distance threshold.
        """
        obj_1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj_2 = object_in_world.ObjectInWorld.create(-1, -1, 0)[1]
        expected = False

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(  # type: ignore
            obj_1,
            obj_2,
            DISTANCE_SQUARED_THRESHOLD,
        )

        assert actual == expected

    def test_is_similar_positive_less_than_threshold(self) -> None:
        """
        Test case where the second landing pad has positive coordinates and the distance between
        them is less than the distance threshold.
        """
        obj_1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj_2 = object_in_world.ObjectInWorld.create(0.5, 0.5, 0)[1]
        expected = True

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(  # type: ignore
            obj_1,
            obj_2,
            DISTANCE_SQUARED_THRESHOLD,
        )

        assert actual == expected

    def test_is_similar_negative_less_than_threshold(self) -> None:
        """
        Test case where the second landing pad has negative coordinates and the distance between
        them is less than the distance threshold.
        """
        obj_1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj_2 = object_in_world.ObjectInWorld.create(-0.5, -0.5, 0)[1]
        expected = True

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(  # type: ignore
            obj_1,
            obj_2,
            DISTANCE_SQUARED_THRESHOLD,
        )

        assert actual == expected

    def test_is_similar_positive_more_than_threshold(self) -> None:
        """
        Test case where the second landing pad has positive coordinates and the distance between
        them is more than the distance threshold.
        """
        obj_1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj_2 = object_in_world.ObjectInWorld.create(2, 2, 0)[1]
        expected = False

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(  # type: ignore
            obj_1,
            obj_2,
            DISTANCE_SQUARED_THRESHOLD,
        )

        assert actual == expected

    def test_is_similar_negative_more_than_threshold(self) -> None:
        """
        Test case where the second landing pad has negative coordinates and the distance between
        them is more than the distance threshold.
        """
        obj_1 = object_in_world.ObjectInWorld.create(0, 0, 0)[1]
        obj_2 = object_in_world.ObjectInWorld.create(-2, -2, 0)[1]
        expected = False

        actual = landing_pad_tracking.LandingPadTracking._LandingPadTracking__is_similar(  # type: ignore
            obj_1,
            obj_2,
            DISTANCE_SQUARED_THRESHOLD,
        )

        assert actual == expected


class TestMarkFalsePositive:
    """
    Test if landing pad tracking correctly marks a detection as a false positive.
    """

    def test_mark_false_positive_no_similar(
        self,
        tracker: landing_pad_tracking.LandingPadTracking,
        detections_1: "list[object_in_world.ObjectInWorld]",
    ) -> None:
        """
        Test if marking false positive adds detection to list of false positives.
        """
        _, false_positive = object_in_world.ObjectInWorld.create(20, 20, 20)
        assert false_positive is not None

        tracker._LandingPadTracking__unconfirmed_positives = detections_1  # type: ignore
        expected = [false_positive]
        expected_unconfirmed_positives = [
            detections_1[0],
            detections_1[1],
            detections_1[2],
            detections_1[3],
            detections_1[4],
        ]

        tracker.mark_false_positive(false_positive)

        assert tracker._LandingPadTracking__false_positives == expected  # type: ignore
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives  # type: ignore

    def test_mark_false_positive_with_similar(
        self,
        tracker: landing_pad_tracking.LandingPadTracking,
        detections_2: "list[object_in_world.ObjectInWorld]",
    ) -> None:
        """
        Test if marking false positive adds detection to list of false positives and removes.
        similar landing pads
        """
        _, false_positive = object_in_world.ObjectInWorld.create(1, 1, 1)
        assert false_positive is not None

        tracker._LandingPadTracking__unconfirmed_positives = detections_2  # type: ignore
        expected = [false_positive]
        expected_unconfirmed_positives = [detections_2[2], detections_2[3], detections_2[4]]

        tracker.mark_false_positive(false_positive)

        assert tracker._LandingPadTracking__false_positives == expected  # type: ignore
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives  # type: ignore

    def test_mark_multiple_false_positive(
        self,
        tracker: landing_pad_tracking.LandingPadTracking,
        detections_1: "list[object_in_world.ObjectInWorld]",
    ) -> None:
        """
        Test if marking false positive adds detection to list of false positives.
        """
        _, false_positive_1 = object_in_world.ObjectInWorld.create(0, 0, 1)
        assert false_positive_1 is not None

        _, false_positive_2 = object_in_world.ObjectInWorld.create(2, 2, 1)
        assert false_positive_2 is not None

        tracker._LandingPadTracking__unconfirmed_positives = detections_1  # type: ignore
        expected = [false_positive_1, false_positive_2]
        expected_unconfirmed_positives = [detections_1[2], detections_1[3], detections_1[4]]

        tracker.mark_false_positive(false_positive_1)
        tracker.mark_false_positive(false_positive_2)

        assert tracker._LandingPadTracking__false_positives == expected  # type: ignore
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives  # type: ignore


class TestMarkConfirmedPositive:
    """
    Test if landing pad tracking correctly marks a detection as a confirmed positive.
    """

    def test_mark_confirmed_positive(
        self, tracker: landing_pad_tracking.LandingPadTracking
    ) -> None:
        """
        Test if marking confirmed positive adds detection to list of confirmed positives.
        """
        _, confirmed_positive = object_in_world.ObjectInWorld.create(1, 1, 1)
        assert confirmed_positive is not None

        expected = [confirmed_positive]

        tracker.mark_confirmed_positive(confirmed_positive)

        assert tracker._LandingPadTracking__confirmed_positives == expected  # type: ignore

    def test_mark_multiple_confirmed_positives(
        self, tracker: landing_pad_tracking.LandingPadTracking
    ) -> None:
        """
        Test if marking confirmed positive adds detection to list of confirmed positives.
        """
        _, confirmed_positive_1 = object_in_world.ObjectInWorld.create(1, 1, 1)
        assert confirmed_positive_1 is not None

        _, confirmed_positive_2 = object_in_world.ObjectInWorld.create(2, 2, 1)
        assert confirmed_positive_2 is not None

        expected = [confirmed_positive_1, confirmed_positive_2]

        tracker.mark_confirmed_positive(confirmed_positive_1)
        tracker.mark_confirmed_positive(confirmed_positive_2)

        assert tracker._LandingPadTracking__confirmed_positives == expected  # type: ignore


class TestLandingPadTracking:
    """
    Test landing pad tracking run function.
    """

    def test_run_with_empty_detections_list(
        self, tracker: landing_pad_tracking.LandingPadTracking
    ) -> None:
        """
        Test run method with empty detections list.
        """
        result, actual = tracker.run([])
        assert not result
        assert actual is None

    def test_run_one_input(
        self,
        tracker: landing_pad_tracking.LandingPadTracking,
        detections_1: "list[object_in_world.ObjectInWorld]",
    ) -> None:
        """
        Test run with only 1 input.
        """
        expected_output = detections_1[2]
        expected_unconfirmed_positives = [
            detections_1[2],
            detections_1[1],
            detections_1[4],
            detections_1[0],
            detections_1[3],
        ]

        result, actual = tracker.run(detections_1)

        assert result
        assert actual == expected_output
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives  # type: ignore

    def test_run_one_input_similar_detections(
        self,
        tracker: landing_pad_tracking.LandingPadTracking,
        detections_3: "list[object_in_world.ObjectInWorld]",
    ) -> None:
        """
        Test run with only 1 input where 2 landing pads are similar.
        """
        expected_output = detections_3[2]
        expected_unconfirmed_positives = [
            detections_3[2],
            detections_3[1],
            detections_3[4],
            detections_3[3],
        ]

        result, actual = tracker.run(detections_3)

        assert result
        assert actual == expected_output
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives  # type: ignore

    def test_run_multiple_inputs(
        self,
        tracker: landing_pad_tracking.LandingPadTracking,
        detections_1: "list[object_in_world.ObjectInWorld]",
        detections_2: "list[object_in_world.ObjectInWorld]",
    ) -> None:
        """
        Test run with 2 inputs where some landing pads are similar.
        """
        expected_output = detections_2[0]
        expected_unconfirmed_positives = [
            detections_2[0],
            detections_1[2],
            detections_2[1],
            detections_2[3],
            detections_1[4],
            detections_2[2],
            detections_2[4],
            detections_1[3],
        ]

        tracker.run(detections_1)
        result, actual = tracker.run(detections_2)

        assert result
        assert actual == expected_output
        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives  # type: ignore

    def test_run_with_confirmed_positive(
        self,
        tracker: landing_pad_tracking.LandingPadTracking,
        detections_1: "list[object_in_world.ObjectInWorld]",
    ) -> None:
        """
        Test run when there is a confirmed positive.
        """
        _, confirmed_positive = object_in_world.ObjectInWorld.create(1, 1, 1)
        assert confirmed_positive is not None

        tracker._LandingPadTracking__confirmed_positives.append(confirmed_positive)  # type: ignore
        expected = confirmed_positive

        result, actual = tracker.run(detections_1)

        assert result
        assert actual == expected

    def test_run_with_false_positive(
        self,
        tracker: landing_pad_tracking.LandingPadTracking,
        detections_2: "list[object_in_world.ObjectInWorld]",
    ) -> None:
        """
        Test to see if run function doesn't add landing pads that are similar to false positives.
        """
        _, false_positive = object_in_world.ObjectInWorld.create(1, 1, 1)
        assert false_positive is not None

        tracker._LandingPadTracking__false_positives.append(false_positive)  # type: ignore
        expected_unconfirmed_positives = [detections_2[3], detections_2[2], detections_2[4]]

        tracker.run(detections_2)

        assert tracker._LandingPadTracking__unconfirmed_positives == expected_unconfirmed_positives  # type: ignore
