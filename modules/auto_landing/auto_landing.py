import math

class AutoLanding:
    """"
    Auto-landing script. 

    TODO: make create method
    """
    def __init__ (
            self,
            FOV_X: float,
            FOV_Y: float,
            im_h: float,
            im_w: float,
            ) -> None:
        """"
        FOV_X: The horizontal camera field of view in degrees.
        FOV_Y: The vertical camera field of view in degrees.
        im_w: Width of image.
        im_h: Height of image.

        """
        self.FOV_X = FOV_X
        self.FOV_Y = FOV_Y
        self.im_h = im_h
        self.im_w = im_w

    def run (self, 
             x_center: float, 
             y_center: float, 
             height: float
             ) -> "tuple[float, float, float]":
        """
        Calculates the angles in radians of the bounding box based on its center.

        x_center: x-coordinate of center of bounding box.
        y_center: y-coordinate of center of bounding box.
        height: Height above ground level in meters.

        Return: Tuple of the x and y angles in radians respectively and the target distance in meters.
        """
        angle_x = (x_center - self.im_w / 2) * (self.FOV_X * (math.pi / 180)) / self.im_w
        angle_y = (y_center - self.im_h / 2) * (self.FOV_Y * (math.pi / 180)) / self.im_h

        print("X angle (rad): ", angle_x)
        print("Y angle (rad): ", angle_y)

        x_dist = math.tan(angle_x) * height
        y_dist = math.tan(angle_y) * height
        ground_hyp = (x_dist**2 + y_dist**2)**0.5
        print("Required horizontal correction (m): ", ground_hyp)
        target_to_vehicle_dist = (ground_hyp**2 + height**2)**0.5
        print("Distance from vehicle to target (m): ", target_to_vehicle_dist)

        return angle_x, angle_y, target_to_vehicle_dist


    