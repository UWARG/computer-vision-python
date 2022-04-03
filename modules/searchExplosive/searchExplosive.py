import cv2
import numpy as np

# Define colours
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class SearchExplosive:
    """
    Detects objects on the given image assuming a relatively uniform background (i.e field of grass)
    """

    def __init__(self, image):
        """
        Initializes attributes

        Parameters
        ----------
        image : ndarray
          ndarray of the image we are processing
        """
        self.image = image
        self.edges = None
        self.count = 0  # used to count number of bounding boxes drawn
        self.detectedContours = image  # will hold the image with bounding boxes

    def edge_detection(self):
        """
        Performs noise reduction and applies canny edge detection on a given image

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        # Convert to grayscale
        imgGrayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Remove noise while trying to maintaining the objects of interest
        kernel = np.ones((2, 2))
        imgErode = cv2.erode(imgGrayscale, kernel, iterations=1)
        imgDil = cv2.dilate(imgErode, kernel, iterations=1)

        # Apply filter for further noise reduction
        imgBlur = cv2.bilateralFilter(imgDil, 8, 75, 75)

        # Calculate the thresholds using empirical rule
        # Find the mean and standard deviation of pixel intensities
        mean = np.nanmean(imgBlur)
        std = np.nanstd(imgBlur)

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
        imgEdges = cv2.Canny(imgBlur, lower, upper)

        self.edges = imgEdges

    def contour_detection(self):
        """
        Detect contours and mark them with a bounding box

        Parameters
        ----------
            None

        Returns
        -------
            None
        """
        if self.edges is None:
             print("ERROR: Detect edges before contours")
             return

        # Find the contours
        self.contours, hierarchy = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Draw bounding box around each contour
        for cnt in self.contours:
            x, y, w, h = cv2.boundingRect(cnt)

            imgHeight, imgWidth, channels = self.detectedContours.shape
            imgArea = imgHeight * imgWidth
            rectArea = w * h

            # Get rid of any unwanted contour detections if their bounding box is less than 0.1% of the image area
            if (rectArea / imgArea) * 100 < 0.1:
                continue

            # Draw bounding box around the detected contours
            cv2.rectangle(self.detectedContours, (x, y), (x + w, y + h), BLUE, 2)
            cv2.putText(self.detectedContours, "Object", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE)
            self.count += 1
