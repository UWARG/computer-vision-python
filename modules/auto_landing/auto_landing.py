import math
import numpy as np

class AutoLanding:
    """"
    Auto-landing script. 
    """
    # do i make a create() as well?
    
    def __init__ (
            self,
            image: np.ndarray, # or do i use MatLike?
            bounding_box: tuple,
            FOV_X: float,
            FOV_Y: float,
            im_h: float,
            im_w: float,
            ) -> None:
        """"
        image: The input image.
        bounding_box: The bounding box defined as (x, y, w, h).
        FOV_X: The horizontal camera field of view (in degrees).
        FOV_Y: The vertical camera field of view (in degrees).
        im_w: Width of image.
        im_h: Height of image.

        """
        self.image = image
        self.bounding_box = bounding_box
        self.FOV_X = FOV_X
        self.FOV_Y = FOV_Y
        self.im_h = im_h
        self.im_w = im_w

    def run (self, image, bounding_box, FOV_X, FOV_Y, im_w, im_h) -> dict:
        """
        Calculates the angles (in radians) of the vertices of the bounding box.

        Return: Dictionary with vertex coordinates as keys and (angle_x, angle_y) as values.
        """
        x, y, w, h = self.bounding_box

        vertices = {
            "top_left": (x, y),
            "top_right": (x + w, y),
            "bottom_left": (x, y + h),
            "bottom_right": (x + w, y + h)
        }
        
        box_angles = {}

        for vertex, (vertex_x, vertex_y) in vertices.items():
            angle_x = (vertex_x - self.im_w / 2) * (self.FOV_X * (math.pi / 180)) / self.im_w
            angle_y = (vertex_y - self.im_h / 2) * (self.FOV_Y * (math.pi / 180)) / self.im_h
            box_angles[vertex] = (angle_x, angle_y)

        return box_angles


    