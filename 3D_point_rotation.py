import numpy as np
import cv2


class AngleNotAllowed(Exception):
    """Angle is not in (0,90,270,360)"""
    pass


def rotate_around_z(point, angle, precision=7):
    theta = np.deg2rad(angle)
    # rotacia
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    rotated_points = point @ rotation_matrix
    # vymena osi:
    if angle == 0:
        rotated_points = np.array([rotated_points[1], rotated_points[2]])
    elif angle == 90:
        rotated_points = np.array([rotated_points[1], rotated_points[2]])
    elif angle == 180:
        rotated_points = np.array([rotated_points[1], rotated_points[2]])
    elif angle == 270:
        rotated_points = np.array([rotated_points[1], rotated_points[2]])

    else:
        raise AngleNotAllowed

    return ";".join([str(round(i, precision)) for i in rotated_points])

# points_3d = np.array([1,2,0])
# views = [0, 90, 180, 270]
#
# for angle in views:
#    print(f"View at {angle}°:\n{rotate_around_z(points_3d, angle)}")


