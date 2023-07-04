class PositionObject:
    def __init__(self, x, y, covariance):
        self.location_x = x
        self.location_y = y
        self.spherical_variance = covariance