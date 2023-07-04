"""
TODO: Write description
"""
from position_object import PositionObject
# Placeholder:
from detection_in_world import DetectionInWorld

class ClusterEstimation:
    """
    TODO: Write description
    """
    def __init__(self):
        # TODO: Settings etc.
        raise NotImplementedError
        
    def run(self, detections: "list[DetectionInWorld]", run_override:  bool) -> "tuple[bool, list[PositionObject | None]]":
        """
        TODO: Write description
        """
        if not run_override and not self.decide_to_run(detections):
            return False, None

        # TODO: Implementation
        raise NotImplementedError

    def decide_to_run(self, detections: "list[DetectionInWorld]") -> bool:
        # Minimum detections to run
        min_to_run = 50
        if(len(detections) > min_to_run):
            return True
        else:
            return False
