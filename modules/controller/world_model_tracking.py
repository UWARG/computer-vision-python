import numpy as np

from .. import object_in_world


class WorldModelTracking:
    """
    keeps track of the real world location of landing pads that
    are to be visited or have already been vsisited
    """

    def __init__(self):
        self.__unconfirmed_positives = []
        self.__false_positives = []
        self.confirmed_positive = None

    def __similar(self,
                  detection1: object_in_world.ObjectInWorld,
                  detection2: object_in_world.ObjectInWorld) \
            -> bool:
        """
        returns whether detection1 and detection2 are close enough
        to be considered the same landing
        """

        DISTANCE_THRESHOLD = 0.2
        distance = (detection2.position_x - detection1.position_x) ** 2 \
            + (detection2.position_y - detection1.position_y) ** 2
        return distance < DISTANCE_THRESHOLD

    def mark_false_positive(self, detection: object_in_world.ObjectInWorld) \
                            -> object_in_world.ObjectInWorld:
        """
        marks a detection as false positive
        """

        self.__false_positives += [detection]
        for landing_pad in self.__unconfirmed_positives:
            if self.__similar(landing_pad, detection):
                self.__unconfirmed_positives.remove(detection)
        return detection

    def mark_confirmed_positive(self, detection: object_in_world.ObjectInWorld) \
                                -> object_in_world.ObjectInWorld:
        """
        marks a detection as a confimred positive for future use
        """

        self.confirmed_positive = detection
        return self.confirmed_positive

    def run(self, detections: np.ndarray):
        """
        updates the list of unconfirmed positives and returns the
        detection with the lowest variance
        """
        
        for detection in detections:
            match_found = False
            for false_positive in self.__false_positives:
                if self.__similar(detection, false_positive):
                    match_found = True
                    break
            if match_found:
                continue
            for i, landing_pad in enumerate(self.__unconfirmed_positives):
                if self.__similar(detection, landing_pad):
                    match_found = True
                    self.__unconfirmed_positives[i] = detection
                    break
            if match_found:
                continue
            self.__unconfirmed_positives.append(detection)

        self.__unconfirmed_positives.sort(key=lambda x: x.spherical_variance)

        return self.__unconfirmed_positives[0]
