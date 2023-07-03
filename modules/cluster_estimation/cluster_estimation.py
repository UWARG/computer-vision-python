"""
TODO: Write description
"""
# TODO: imports


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
        """
        TODO: Write description
        """
        # TODO: Implementation
        raise NotImplementedError
