"""
Unit tests for searchExplosive module
"""

import pytest
from modules.searchExplosive.searchExplosive import SearchExplosive
import cv2


def test_empty_frame_edge_detection():
    # test empty frame in edge detection method
    with pytest.raises(cv2.error):
        detector = SearchExplosive(None)
        detector.edge_detection()


def test_empty_edges_contour_detection():
    # test empty frame in edge detection method
    detector = SearchExplosive(None)
    result = detector.contour_detection()
    assert not result


def test_edge_detection():
    # test happy path in edge detection method
    image = cv2.imread("sampleImages/sample_4.jpg")
    detector = SearchExplosive(image)
    detector.edge_detection()
    assert detector.edges.size != 0  # should be edges detected


def test_contour_detection():
    # test happy path in contour detection method
    image = cv2.imread("sampleImages/sample_4.jpg")
    detector = SearchExplosive(image)
    detector.edge_detection()
    detector.contour_detection()
    assert detector.count == 3  # should be 3 detected objects in the image


def test_contour_detection_2():
    # test happy path in contour detection method
    image = cv2.imread("sampleImages/sample_2.jpg")
    detector = SearchExplosive(image)
    detector.edge_detection()
    detector.contour_detection()
    assert detector.count == 4  # should be 4 detected objects in the image


def test_edge_detection_with_plain_image():
    # test plain frame in edge detection method, no edges should be detected
    image = cv2.imread("sampleImages/plain_white_img.jpg")
    detector = SearchExplosive(image)
    detector.edge_detection()
    assert not detector.edges.any()


def test_contour_detection_with_plain_image():
    # test contour detection when no contours should be detected
    image = cv2.imread("sampleImages/plain_white_img.jpg")
    detector = SearchExplosive(image)
    detector.edge_detection()
    detector.contour_detection()
    assert detector.count == 0
