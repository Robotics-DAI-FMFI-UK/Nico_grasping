import cv2
import mediapipe as mp
import math
from nico_structure import *
import time
from nicomotors import NicoMotors



def setRobotDofs(angles, delay = 0):
    #print(angles)
    for dof,angle in zip(Dofs,angles):
	    #motors.setPositionDg(dof,angle)
	    motors.setPositionDg(dof,angle)
	    pass
##    while ((np.linalg.norm(np.array(getRobotDofs())-np.array(angles))/len(angles)) > 1.3):
##        time.sleep(0.001)
    time.sleep(delay)

def getRobotDofs():
    angles = []
    for dof in Dofs:
        angle = motors.getPositionDg(dof)
        angles.append(angle)
    return angles









motors = NicoMotors()
dofs = motors.dofs()
motors.open()


def enableTorque():
    for dof in Dofs:
        motors.enableTorque(dof)
        motors.setMovingSpeed(dof,20)
def disableTorque():
    for dof in Dofs:
        motors.disableTorque(dof)

enableTorque()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

wCam, hCam = 1280, 960
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)



r_elbow_converted = 0
r_other_fingers_converted = 0
r_index_finger_converted = 0
r_thumb_lift_converted = 0
r_thumb_close_converted = 0


with mp_holistic.Holistic(
        model_complexity=0,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
) as holistic:
    while cam.isOpened():
        success, image = cam.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Left hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        rightHandList = []
        leftHandList = []

        if results.right_hand_landmarks:
            rightHand = results.right_hand_landmarks
            for id, value in enumerate(rightHand.landmark):
                h, w, c = image.shape

                cx, cy = int(value.x * w), int(value.y * h)
                rightHandList.append([id, cx, cy])

        if results.left_hand_landmarks:
            leftHand = results.left_hand_landmarks
            for id, value in enumerate(leftHand.landmark):
                h, w, c = image.shape
            

                cx, cy = int(value.x * w), int(value.y * h)
                leftHandList.append([id, cx, cy])

        poseList = []
        if results.pose_landmarks:
            pose = results.pose_landmarks
            for id, value in enumerate(pose.landmark):
                h, w, c = image.shape
                cx, cy = int(value.x * w), int(value.y * h)
                poseList.append([id, cx, cy])


        if rightHandList:

            RIGHT_WRIST = rightHandList[0][1], rightHandList[0][2]
            RIGHT_THUMB_CMC = rightHandList[1][1], rightHandList[1][2]
            RIGHT_THUMB_MCP = rightHandList[2][1], rightHandList[2][2]
            RIGHT_THUMB_IP = rightHandList[3][1], rightHandList[3][2]
            RIGHT_THUMB_TIP = rightHandList[4][1], rightHandList[4][2]
            RIGHT_INDEX_FINGER_MCP = rightHandList[5][1], rightHandList[5][2]
            RIGHT_INDEX_FINGER_TIP = rightHandList[8][1], rightHandList[8][2]
            RIGHT_RING_FINGER_MCP = rightHandList[13][1], rightHandList[13][2]
            RIGHT_RING_FINGER_TIP = rightHandList[16][1], rightHandList[16][2]
            RIGHT_PINKY_MCP = rightHandList[16][1], rightHandList[16][2]

            right_forefinger = calculate_angle(RIGHT_INDEX_FINGER_TIP, RIGHT_INDEX_FINGER_MCP, RIGHT_WRIST)
            right_littlefingers = calculate_angle(RIGHT_RING_FINGER_TIP, RIGHT_RING_FINGER_MCP, RIGHT_WRIST)
            right_thumb2 = calculate_angle(RIGHT_THUMB_MCP, RIGHT_WRIST, RIGHT_PINKY_MCP)
            right_thumb1 = calculate_angle(RIGHT_THUMB_TIP, RIGHT_THUMB_IP, RIGHT_THUMB_MCP)



            r_other_fingers_converted = convert_r_other_fingers(right_littlefingers)
            r_index_finger_converted = convert_r_index_finger(right_forefinger)
            r_thumb_lift_converted = convert_r_thumb_lift(right_thumb2)
            r_thumb_close_converted = convert_r_thumb_close(right_thumb1)

##            app.update_finger_angle(0, right_littlefingers)
##            app.update_finger_angle(1, right_forefinger)
##            app.update_finger_angle(2, right_thumb2)
##            app.update_finger_angle(3, right_thumb1)
##            app.update()
            #print(right_littlefingers,right_forefinger, right_thumb2, right_thumb1)
            
            
##            print(
##                f'{int(time.time() * 1000 - start_time)} 0 {r_other_fingers_converted} 0 {r_index_finger_converted} 0 '
##        4        f'{r_thumb_lift_converted} 0 {r_thumb_close_converted} 2004 2007 2211 1521 2298 '
##                f'2030 2184 1961 2650 1943 1383 2935 1990 2143')

        if leftHandList:
            ...

        if poseList:
            LEFT_SHOULDER = poseList[11][1], poseList[11][2]
            RIGHT_SHOULDER = poseList[12][1], poseList[12][2]
            RIGHT_ELBOW = poseList[14][1], poseList[14][2]
            RIGHT_WRIST = poseList[16][1], poseList[16][2]


            r_shoulder_fwd_bwd = None
            r_shoulder_left_right = None
            r_shoulder_lift = None
            r_elbow = calculate_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
            r_elbow_converted = convert_r_elbow(r_elbow)
            r_wrist_rotate = None
            r_wrist_left_right = None

        #Dofs = ['left-arm1', 'left-arm2', 'left-arm3', 'left-elbow1', 'left-wrist1', 'left-wrist2', 'left-thumb1', 'left-thumb2', 'left-forefinger', 'left-littlefingers', 'right-arm1', 'right-arm2', 'right-arm3', 'right-elbow1', 'right-wrist1', 'right-wrist2', 'right-thumb1', 'right-thumb2', 'right-forefinger', 'right-littlefingers', 'neck1', 'neck2']

        setRobotDofs([4, 14, 21, 92, 67, 20, 11, 8, 0, 0, -6, 14, 20, r_elbow_converted, -27, 47, r_thumb_lift_converted, r_thumb_close_converted, r_index_finger_converted, r_other_fingers_converted ,  -3, 0])
        cv2.imshow('Position detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cam.release()
cv2.destroyAllWindows()
exit()
