

class WorldModelTracking:

    def __init__(self):
        self.__unconfirmed_positives = []
        self.__false_positives = []
        self.__unconfirmed_positives = None         #object in world
    
    def similar(self, detection1, detection2):
        pass

    def run(self, detections):
        for detection in detections:
            match_found = False
            for false_positive in self.__false_positives:
                if self.similar(detection, false_positive):
                    match_found = True
                    break
            if match_found:
                continue
            for i, landing_pad in enumerate(self.__unconfirmed_positives):
                if self.similar(detection, landing_pad):
                    match_found = True
                    self.__unconfirmed_positives[i] = detection
                    break
            if match_found:
                continue
            self.__unconfirmed_positives.append(detection)
        
        self.__unconfirmed_positives.sort(key=lambda x: x.spherical_variance)

        return True, self.__unconfirmed_positives[-1]