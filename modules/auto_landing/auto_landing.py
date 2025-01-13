import math

class AutoLanding:
    """"
    Auto-landing script. 
    """

    def __init__ (
            self,
            FOV_X: float,
            FOV_Y: float,
            im_h: float,
            im_w: float,
            x_center: float,
            y_center: float,
            height: float,
            ) -> None:
        """"
        FOV_X: The horizontal camera field of view in degrees.
        FOV_Y: The vertical camera field of view in degrees.
        im_w: Width of image.
        im_h: Height of image.
        x_center: x-coordinate of center of bounding box.
        y_center: y-coordinate of center of bounding box.
        height: Height above ground level in meters.

        """
        self.FOV_X = FOV_X
        self.FOV_Y = FOV_Y
        self.im_h = im_h
        self.im_w = im_w
        self.x_center = x_center
        self.y_center = y_center
        self.height = height

    def run (self, FOV_X, FOV_Y, im_w, im_h, x_center, y_center, height) -> "tuple[float, float, float]":
        """
        Calculates the angles in radians of the bounding box based on its center.

        Return: Tuple of the x and y angles in radians respectively and the target distance in meters.
        """
        angle_x = (self.x_center - self.im_w / 2) * (self.FOV_X * (math.pi / 180)) / self.im_w
        angle_y = (self.y_center - self.im_h / 2) * (self.FOV_Y * (math.pi / 180)) / self.im_h

        print("X angle (rad): ", angle_x)
        print("Y angle (rad): ", angle_y)

        x_dist = math.tan(angle_x) * self.height
        y_dist = math.tan(angle_y) * self.height
        ground_hyp = math.sqrt(math.pow(x_dist, 2) + math.pow(y_dist, 2))
        print("Required horizontal correction (m): ", ground_hyp)
        target_to_vehicle_dist = math.sqrt(math.pow(ground_hyp, 2) + math.pow(self.height, 2))
        print("Distance from vehicle to target (m): ", target_to_vehicle_dist)

        return angle_x, angle_y, target_to_vehicle_dist


    