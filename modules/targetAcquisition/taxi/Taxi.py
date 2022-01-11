import cv2
import numpy as np
from modules.targetAcquisition.taxi.boxDetection.detect import Detection
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Constant declaration

FOCAL_LENGTH = 24
REAL_HEIGHT = 101.6
IMAGE_HEIGHT = 1080
SENSOR_HEIGHT = 4.63


class Taxi:
    """
    Performs cardboard box detection on a given video frame

    Attributes
    ----------
    state : string
        state of object recognition ("BOX", "TRACK", ...)
    bbox : list<tuple<tuple, tuple>>
        a list of ((x1, y1), (x2, y2)) coordinates for (top left, bottom right) of bounding boxes; one per box
    frame : np.ndarray
        the current video frame
    yolo : Detection object
        the YOLOv5 detector
    tracker : TrackerKCF object
        the cv2 KCF bounding box tracker
    nextUncheckedID : int
        the ID of the next box to scan
    expectedCount : int
        the number of boxes it needs to detect to go into tracking state
    numStableFrames : int
        the number of consecutive frames with all <expectedCount> number of boxes in view before switching to TRACK
    distanceFromBox : int
        the plane's current distance from the target box
    minDistanceFromBox : int
        the minimum distance the plane can be from the box until it's considered too close
    moveWaitTarget : int
        the number of frames to wait before issuing another move or turn command
    frameCount : int
        the number of consecutive frames where all <expectedCount> boxes are in view
    totalWait : int
        maximum number of frames to wait for before time-out while searching for <expectedCount> number of boxes
    moveWaitCount : int
        wait for a certain number of detections before issuing the next movement command
    distanceCount : int
        if the plane has moved by <distanceCount> number of steps forward, it needs to recalibrate the bounding box by calculating a new target distance
        
    Constants
    ----------
    FOCAL_LENGTH : int
        focal length of the camera (in mm)
    REAL_HEIGHT : int
        real height of the box (in mm)
    IMAGE_HEIGHT : int
        height of the image (in pixels)
    SENSOR_HEIGHT : int
        height of the sensor (in mm)
   
    Methods
    -------
    __init__()
        Initialize class variables
    find_overlapped_bbox()
        Find which new YOLO bbox overlaps the previously tracked bbox
    calculate_distance()
        Calculate approximate distance between box and drone
    set_state(state: string)
        Prepare variables for box detection, etc.
    main()
        Main operations: getting camera input and passing the image to appropriate methods
    """

    def __init__(self, state="BOX", bbox=[((0, 0), (0, 0))], frame=[], nextUncheckedID=0,
                 expectedCount=1, numStableFrames=20, distanceFromBox=0,
                 minDistanceFromBox=50, moveWaitTarget=75, recalibrate=False, lastBbox=[],
                 frameCount=0, totalWait=0, moveWaitCount=0, distanceCount=0):
        """
        Initializes variables
        """
        self.state = state
        self.bbox = bbox
        self.frame = frame
        self.yolo = Detection()
        self.tracker = None
        self.nextUncheckedID = nextUncheckedID
        self.expectedCount = expectedCount
        self.numStableFrames = numStableFrames
        self.distanceFromBox = distanceFromBox
        self.minDistanceFromBox = minDistanceFromBox
        self.moveWaitTarget = moveWaitTarget
        self.recalibrate = recalibrate
        self.lastBbox = lastBbox

        # Instead of keeping the following as local variables in the while loop,
        # they become object attributes and get updated each time Taxi.main() is called
        self.frameCount = frameCount
        self.totalWait = totalWait
        self.moveWaitCount = moveWaitCount
        self.distanceCount = distanceCount

    def calculate_distance(self, pts):
        """
        Calculate approximate distance between box and drone
        """

        # Calculating object height in pixels by extracting y coordinates from each tuple 'pts'
        objectHeight = pts[0][1] - \
                       pts[1][1] if (pts[0][1] > pts[1][1]) else pts[1][1] - pts[0][1]

        # Calculate distance
        distance = (FOCAL_LENGTH * REAL_HEIGHT * IMAGE_HEIGHT) / \
                   (objectHeight * SENSOR_HEIGHT)

        # Convert to m
        distance_m = distance * 1000
        return distance_m

    def find_overlapped_bbox(self):
        """
        Given the last known bbox and a list of new ones, return the new bbox which overlaps the old box the most.
        Helps recalibrate the tracked bbox to fit the cardboard box better, so distance calculation is more accurate.
        """
        # If the new scan found zero bounding boxes, return no bbox
        if len(self.bbox) < 1 or len(self.lastBbox) < 1:
            print("ERROR: box list too short. Aborting")
            return None, 0

        # Area of the tracked bbox
        last = self.lastBbox[0]
        originalArea = (last[1][1] - last[0][1]) * (last[1][0] - last[0][0])

        bbox = self.bbox[0]
        maxOverlap = 0
        for box in self.bbox:
            x0 = max(last[0][0], box[0][0])
            x1 = min(last[1][0], box[1][0])
            y0 = max(last[0][1], box[0][1])
            y1 = min(last[1][1], box[1][1])
            newArea = (y1 - y0) * (x1 - x0)
            overlap = newArea / originalArea

            if overlap > maxOverlap:
                maxOverlap = overlap
                bbox = box

        # If even the best match is not good enough, return no bboxes
        if maxOverlap < 0.75:
            return None, maxOverlap

        # Return the bbox with max overlap
        print(f"Matched bbox: {bbox}, max overlap: {maxOverlap}")
        return bbox, maxOverlap

    def set_state(self, state):
        """
        Prepare variables for box detection, tracking etc.

        Parameters
        ----------
        state: string
            state of object recognition ("BOX", "TRACK", ...)
        """
        if state == "BOX" or state == 0:
            self.state = "BOX"

        elif state == "TRACK" or state == 1:
            self.state = "TRACK"
            # If switching to track from box for the first time, use the box ID
            if not self.recalibrate:
                bbox = self.bbox[self.nextUncheckedID]
                print(f"Switching to track and NOT recalibrating. bbox: {bbox}")
            # Else update distance with recalibrated bbox
            else:
                print("Switching to track and recalibrating")
                bbox, _ = self.find_overlapped_bbox()
                print(f"New bbox after recalibration: {bbox}")

            # If recalibration failed to find the new box
            if bbox == None:
                print("Failed to find a bbox")

            # If recalibration found the new box, recalculate distance and carry on moving
            else:
                print("At least one bbox is found")
                bboxReformat = (bbox[0][0], bbox[0][1],
                                bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])
                # Note: must create the tracker here instead of __init__, else can't switch freely between track & box
                self.tracker = cv2.TrackerKCF_create()
                # Initialize tracker with first frame and bounding box
                self.tracker.init(self.frame, bboxReformat)
                print("Tracker created and initialized")
                # Calculate distance from tracked bbox
                self.distanceFromBox = self.calculate_distance(bbox)
                print(f"Calculated distance: {self.distanceFromBox}")
                self.state = "TRACK"
                print("Set variable self.state to TRACK")
        else:
            print("Error: invalid state selected")

    def main(self, cap):
        """
        Main operations: getting camera input and passing the image to appropriate methods

        Parameters
        ----------
        cap : np.ndarray
            the current captured rgb image from the camera
        """

        self.frame = cap
        if self.state == "BOX":
            self.bbox = self.yolo.detect_boxes(self.frame)
            for (topLeft, botRight) in self.bbox:
                cv2.rectangle(self.frame, topLeft,
                              botRight, (0, 0, 255), 2)

            # If initial run, expect 1 box
            # If recalibrating, expect at least 1 box (less accurate but there's no other way)
            if not self.recalibrate:
                expCount = self.expectedCount
            else:
                print("Recalibrating in BOX mode. Expecting 1 box")
                expCount = 1

            # YOLO can't move on until all 1 box stay in frame consistently/continuously for more than frameCount frames
            if len(self.bbox) >= expCount:
                self.frameCount += 1
            else:
                frameCount = 0

            if frameCount == self.numStableFrames:
                print("Got the required number of stable frames")
                self.set_state("TRACK")
                frameCount = 0

            self.totalWait += 1
            if self.totalWait > 150:
                print(
                    f"Waited {self.totalWait} frames without finding the expected number of bbox. Switching to human control.")
                return {'latestDistance': 0.0}

        # Switch to track when all 1 box are in view
        if self.state == "TRACK":
            # print("In tracking mode")
            found, bbox = self.tracker.update(self.frame)

            # Tracking succeeds and plane isn't already close enough
            if found and self.distanceFromBox > self.minDistanceFromBox:
                # print("Tracking succeeds and plane isn't already close enough")
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                self.bbox = [(p1, p2)]
                cv2.rectangle(self.frame, p1, p2, (255, 0, 0), 2, 1)

                # While not at half the distance from the box, keep moving forward
                if self.distanceCount < 500:
                    # Issue the next movement command every couple of frames
                    if self.moveWaitCount < self.moveWaitTarget:
                        print("Waiting to move")
                        self.moveWaitCount += 1
                    else:
                        # Note TURNING HAS BEEN TURNED OFF. This will simply move forward.
                        # Don't need to turn if the bbox midpoint is within the center margin
                        # If center to left past margin, turn right by 1 degree
                        # If center to right past margin, turn left by 1 degree
                        margin = 50
                        turn = 0
                        forward = 0
                        # Tempoararily disable turning due to complexity
                        # if (p1[0] + p2[0]) / 2 < self.frame.shape[0] / 2 - margin:
                        #     turn = 1
                        # elif (p1[0] + p2[0]) / 2 > self.frame.shape[0] / 2 + margin:
                        #     turn = -1
                        # else:
                        #     forward = 5
                        forward = self.calculate_distance(self.bbox) / 4
                        self.distanceCount += forward
                        print(
                            f"Turn {turn} degree(s) and move forward by {forward}")

                        # TODO: fix self.frame.shape not matching cv.imshow video shape
                        cv2.rectangle(self.frame, (240, 0),
                                      (240, 640), (255, 0, 255), 2, 1)

                        self.moveWaitCount = 0
                        return {'latestDistance': forward}
                # At the halfway point, do YOLO again to reset the size of the bounding box
                # KCF and most other tracking algos have constant bbox size even when getting closer, which is a problem for distance calculation
                else:
                    print("Distance limit reached, time to recalibrate")
                    self.distanceCount = 0
                    self.recalibrate = True
                    self.lastBbox = self.bbox
                    self.set_state("BOX")
                    return {'latestDistance': 0.0}

            # Tracking fails or plane is already close enough
            else:
                # Assuming the plane is facing the right box, tracking fails either due to algo error or the plane got too close to object
                # Switch to human control to drive the plane to the right spot
                print(f"Found: {found}\nDistance: {self.distanceFromBox}")
                return {'latestDistance': 0.0}

        cv2.imshow('Image', self.frame)

        '''
        # Temporary manual box selection
        key = cv2.waitKey(10)
        if key == ord('b') and self.state != "BOX":
            print("manual switch to box state")
            self.set_state("BOX")

        # Temporary manual tracking selection
        if key == ord('t') and self.state != "TRACK":
            print("manual switch to tracking state")
            self.set_state("TRACK")

        if key == ord('q'):
            print("manual exit")
            break
        '''


# Instantiate the Taxi object and run operations
if __name__ == '__main__':
    testTaxi = Taxi()
    testTaxi.main()
