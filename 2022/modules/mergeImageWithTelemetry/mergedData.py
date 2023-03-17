import numpy.typing as npt

"""
    contains the merged data
"""

class MergedData:
    """
        contains the merged data

        atributes
        --------
        image
            the image data
        telemetry
            the telemetry data

        methods
        ------
        init
            initializes the data
    """
    def __init__(self, image : npt.ArrayLike, telemetry: dict):
        self.image = image
        self.telemetry = telemetry

