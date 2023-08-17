"""
Keeps track of detected and visited landing pads
"""

import numpy as np

from .. import object_in_world


class LandingPadTracking:
    """
    Tracks the real world location of detected landing pads, labelling them as either confirmed
    positive, unconfirmed positive, or false positive
    """
    def __init__(self, distance_squared_threshold: float):
        self.__unconfirmed_positives = []
        self.__false_positives = []
        self.__confirmed_positives = []

        # Landing pads within the square root of this distance are considered the same landing pad
        self.__distance_squared_threshold = distance_squared_threshold

    @staticmethod
    def __is_similar(detection1: object_in_world.ObjectInWorld,
                     detection2: object_in_world.ObjectInWorld,
                     distance_squared_threshold: float) -> bool:
        """
        Returns whether detection1 and detection2 are close enough to be considered the same
        landing pad
        """
        distance_squared = (detection2.position_x - detection1.position_x) ** 2 \
                           + (detection2.position_y - detection1.position_y) ** 2
        return distance_squared < distance_squared_threshold

    def mark_false_positive(self, detection: object_in_world.ObjectInWorld):
        """
        Marks a detection as false positive and removes similar landing pads from the list of
        unconfirmed positives
        """
        self.__false_positives.append(detection)
        self.__unconfirmed_positives = [
            landing_pad for landing_pad in self.__unconfirmed_positives
            if not self.__is_similar(landing_pad, detection, self.__distance_squared_threshold)
        ]

    def mark_confirmed_positive(self, detection: object_in_world.ObjectInWorld):
        """
        Marks a detection as a confimred positive for future use
        """
        self.__confirmed_positives.append(detection)
        
    def run(self, detections: "list[object_in_world.ObjectInWorld]"):
        """
        Updates the list of unconfirmed positives and returns the a first confirmed positive if
        one exists, else the unconfirmed positive with the lowest variance
        """
        for detection in detections:
            match_found = False

            # If detection matches a false positive, don't add it
            for false_positive in self.__false_positives:
                if self.__is_similar(detection, false_positive, self.__distance_squared_threshold):
                    match_found = True
                    break
            if match_found:
                continue

            # If detection matches an unconfirmed positive, replace old detection
            for i, landing_pad in enumerate(self.__unconfirmed_positives):
                if self.__is_similar(detection, landing_pad, self.__distance_squared_threshold):
                    match_found = True
                    self.__unconfirmed_positives[i] = detection
                    break
            if match_found:
                continue

            # If new landing pad, add to list of unconfirmed positives
            self.__unconfirmed_positives.append(detection)

        # If there are confirmed positives, return the first one
        if len(self.__confirmed_positives) > 0:
            return True, self.__confirmed_positives[0]

        # If the list is empty, all landing pads have been visited, none are viable
        if len(self.__unconfirmed_positives) == 0:
            return False, None
                
        # Sort list by variance in ascending order
        self.__unconfirmed_positives.sort(key=lambda x: x.spherical_variance)

        # Return detection with lowest variance
        return True, self.__unconfirmed_positives[0]
