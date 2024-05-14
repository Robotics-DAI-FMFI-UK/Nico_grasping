from nicomotors import NicoMotors
import numpy as np
import time, random
import threading


motors = NicoMotors()
dofs = motors.dofs()
motors.open()

leftArmDofs = ['left-arm1', 'left-arm2', 'left-arm3', 'left-elbow1', 'left-wrist1', 'left-wrist2', 'left-thumb1', 'left-thumb2', 'left-forefinger', 'left-littlefingers', 'right-arm1', 'right-arm2', 'right-arm3', 'right-elbow1', 'right-wrist1', 'right-wrist2', 'right-thumb1', 'right-thumb2', 'right-forefinger', 'right-littlefingers', 'neck1', 'neck2']
cube = []

def enableTorque():
    for dof in leftArmDofs:
        motors.enableTorque(dof)
        motors.setMovingSpeed(dof,20)
def disableTorque():
    for dof in leftArmDofs:
        motors.disableTorque(dof)

enableTorque()

def getLeftArm():
    angles = []
    for dof in leftArmDofs:
        angle = motors.getPositionDg(dof)
        angles.append(angle)
    return angles

def setLeftArm(angles, delay = 1):
    for dof,angle in zip(leftArmDofs,angles):
	    motors.setPositionDg(dof,angle)
    global cube
    while ((np.linalg.norm(np.array(getLeftArm())-np.array(angles))/len(angles)) > 1.3):
        time.sleep(0.001)
        #print(motors.getLeftIR())
        if (motors.getLeftIR()) < 65:
            cube = getLeftArm()
    
    time.sleep(delay)


poses = [[-5.0, -2.0, 16.0, 70.0, 67.0, 15.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 113.0, -27.0, 48.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[-3.0, 3.0, 21.0, 80.0, 67.0, 19.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 113.0, -27.0, 48.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[-1.0, 5.0, 21.0, 82.0, 67.0, 19.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 113.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[1.0, 9.0, 21.0, 88.0, 67.0, 19.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 113.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[2.0, 12.0, 21.0, 89.0, 67.0, 20.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 113.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[4.0, 14.0, 21.0, 92.0, 67.0, 20.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 112.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[5.0, 17.0, 21.0, 96.0, 67.0, 11.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 112.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[7.0, 21.0, 20.0, 102.0, 67.0, 13.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 112.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[10.0, 27.0, 20.0, 106.0, 67.0, 13.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 112.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[13.0, 32.0, 19.0, 114.0, 67.0, 12.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 112.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[17.0, 32.0, 15.0, 117.0, 67.0, 11.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 112.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[19.0, 34.0, 13.0, 122.0, 67.0, 3.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 112.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[23.0, 34.0, 8.0, 124.0, 67.0, 3.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 112.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0],
[28.0, 34.0, 3.0, 129.0, 67.0, -5.0, 11.0, 8.0, 0.0, 0.0, -6.0, 14.0, 20.0, 112.0, -27.0, 47.0, 65.0, 75.0, 1.0, 0.0, -3.0, 0.0]]



setLeftArm(poses[0])
for pose in poses:
    setLeftArm(pose)
    setLeftArm([60.0] + pose[1:])

    setLeftArm(pose)



setLeftArm(a)

motors.close()
