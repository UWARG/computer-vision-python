import cv2
import numpy as np

# Define colours
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def contour_detection(processed, original):
    """
    Detect contours and mark them with a bounding box

    Parameters
    ----------
    processed : ndarray
      ndarray representation of an image with edges detected
    original : ndarray
      ndarray of the original image we are processing

    Returns
    -------
        None
    """
    # Find the contours
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw bounding box around each contour
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Get rid of any small specs that were picked up as contours
        if area < 1:
            continue
        x, y, w, h = cv2.boundingRect(cnt)

        # Draw bounding box around the detected contours
        cv2.rectangle(original, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(image, "Object", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE)

    cv2.imshow('Window', original)  # show for testing
    cv2.waitKey(0)  # show for testing


def edge_detection(image):
    """
    Performs noise reduction and applies canny edge detection on a given image

    Parameters
    ----------
    image : ndarray
      ndarray of the original image we are processing

    Returns
    -------
    imgEdges : ndarray
      ndarray representation of an image with edges detected
    """
    # Convert to grayscale
    imgGrayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise while trying to maintaining the objects of interest
    kernel = np.ones((2, 2))
    imgErode = cv2.erode(imgGrayscale, kernel, iterations=1)
    imgDil = cv2.dilate(imgErode, kernel, iterations=1)

    # Apply filter for further noise reduction
    img_blur = cv2.bilateralFilter(imgDil, 8, 75, 75)

    # Calculate the thresholds using empirical rule
    # Find the mean and standard deviation of pixel intensities
    mean = np.nanmean(img_blur)
    std = np.nanstd(img_blur)

    # Finding lower threshold
    if mean - 2 * std < 0:
        lower = 0
    else:
        lower = mean - 2 * std  # worked better with subtracting 2 standard deviations

    # Finding upper threshold
    if mean + std > 255:
        upper = 255
    else:
        upper = mean + std

    # Perform edge detection with the computed bounds
    imgEdges = cv2.Canny(img_blur, lower, upper)

    return imgEdges


def find_explosive(image):
    """
    Detects objects on the given image assuming a relatively uniform background (i.e field of grass)

    Parameters
    ----------
    image : ndarray
      ndarray of the original image we are processing

    Returns
    -------
        None
    """
    edges = edge_detection(image)
    contour_detection(edges, image)


# For testing purposes
if __name__ == "__main__":
    image = cv2.imread("sampleImages/sample_4.jpg")  # change image to the sample image you wish to test
    find_explosive(image)
