import numpy as np
import math
import time


Dofs = ['left-arm1', 'left-arm2', 'left-arm3', 'left-elbow1', 'left-wrist1', 'left-wrist2', 'left-thumb1', 'left-thumb2', 'left-forefinger', 'left-littlefingers', 'right-arm1', 'right-arm2', 'right-arm3', 'right-elbow1', 'right-wrist1', 'right-wrist2', 'right-thumb1', 'right-thumb2', 'right-forefinger', 'right-littlefingers', 'neck1', 'neck2']




def calculate_nico_joint(angle, angle_range, nico_joint_range ):
    x1, x2 = angle_range
    y1, y2 = nico_joint_range
    if angle < x1:
        return y1
    if angle > x2:
        return y2
    k = (y2 - y1) / (x2 - x1)
    q = y1 - (k * x1)
    y = k * angle + q
    return round(y)







# calculates angle of three points,while the calculated angle is at the middle point
def calculate_angle(top, middle, bottom):
    degrees = math.degrees(math.atan2(bottom[1] - top[1], bottom[0] - middle[0]) - math.atan2(top[1] - middle[1], top[0] - middle[0]))
    angle = int(np.abs(degrees))
    if angle > 180:
        angle = 360 - angle
    return angle




# functions for converting angles to NICO's ranges

def convert_r_shoulder_fwd_bwd(value):
    return calculate_nico_joint(value, [100, 200], [12, 4088])


def convert_r_shoulder_left_right(value):
    return calculate_nico_joint(value, [100, 200], [12, 4088])


def convert_r_elbow(value):
    return calculate_nico_joint(value, [100, 200], [55, 100])


def convert_r_wrist_rotate(value):
    return calculate_nico_joint(value, [100, 200], [12, 4088])


def convert_r_wrist_left_right(value):
    return calculate_nico_joint(value, [115, 160], [15, 4090])


def convert_r_index_finger(value):
    return calculate_nico_joint(value, [52, 167], [180, 0])


def convert_r_other_fingers(value):
    return calculate_nico_joint(value, [32, 168], [180, 0])


def convert_r_thumb_lift(value):
    return calculate_nico_joint(value, [41, 76], [180, 0])


def convert_r_thumb_close(value):
    return calculate_nico_joint(value, [97, 155], [180, 0])


