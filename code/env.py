import pybullet as p
import time
import pybullet_data as pd
import random
import numpy as np

num_link = 3
force = 100
maxVelocity = 1.5


def set_constrain_blue04():
    feet_pos = [0.038, 0.0, 0.246]
    feet_ori = p.getQuaternionFromEuler([3.14, 0, -1.57])
    feet_id = p.loadURDF(urdf_path_2, feet_pos, feet_ori, useFixedBase=0)
    enableCollision = 0
    p.setCollisionFilterPair(robot_id, feet_id, 2, 0, enableCollision)
    p.setCollisionFilterPair(robot_id, feet_id, 3, 0, enableCollision)
    p.setCollisionFilterPair(robot_id, feet_id, 6, 1, enableCollision)
    p.setCollisionFilterPair(robot_id, feet_id, 7, 1, enableCollision)
    p.setCollisionFilterPair(robot_id, feet_id, 10, 2, enableCollision)
    p.setCollisionFilterPair(robot_id, feet_id, 11, 2, enableCollision)

    j_type = p.JOINT_POINT2POINT
    dis_axis = 0.005

    # 1,2
    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=2,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=0,
                       jointType=j_type,
                       jointAxis=[0, 0, 1],
                       parentFramePosition=[0.042, 0.0, 0.03 - dis_axis],
                       childFramePosition=[0.00808, 0.0 - dis_axis, 0.0088])

    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=2,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=0,
                       jointType=j_type,
                       jointAxis=[0, 0, 1],
                       parentFramePosition=[0.042, 0.0, 0.03 + dis_axis],
                       childFramePosition=[0.00808, 0.0 + dis_axis, 0.0088])

    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=3,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=0,
                       jointType=j_type,
                       jointAxis=[0, 0, 1],
                       parentFramePosition=[0.042, 0, 0.03 - dis_axis],
                       childFramePosition=[0.00808, 0 - dis_axis, -0.0088])

    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=3,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=0,
                       jointType=j_type,
                       jointAxis=[0, 0, 1],
                       parentFramePosition=[0.042, 0, 0.03 + dis_axis],
                       childFramePosition=[0.00808, 0 + dis_axis, -0.0088])

    # 3,4
    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=6,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=1,
                       jointType=j_type,
                       jointAxis=[0, 0, 1],
                       parentFramePosition=[0.042, 0, 0.03 + dis_axis],
                       childFramePosition=[0.00808, 0 - dis_axis, -0.0088])

    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=6,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=1,
                       jointType=j_type,
                       jointAxis=[0, 0, 1],
                       parentFramePosition=[0.042, 0, 0.03 - dis_axis],
                       childFramePosition=[0.00808, 0 + dis_axis, -0.0088])

    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=7,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=1,
                       jointType=j_type,
                       jointAxis=[0, 0, 0],
                       parentFramePosition=[0.042, 0, 0.03 + dis_axis],
                       childFramePosition=[0.00808, 0 - dis_axis, 0.0088])

    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=7,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=1,
                       jointType=j_type,
                       jointAxis=[0, 0, 0],
                       parentFramePosition=[0.042, 0, 0.03 - dis_axis],
                       childFramePosition=[0.00808, 0 + dis_axis, 0.0088])

    # 5,6
    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=10,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=2,
                       jointType=j_type,
                       jointAxis=[0, 0, 0],
                       parentFramePosition=[0.042, 0, 0.03 - dis_axis],
                       childFramePosition=[0.00808, 0 + dis_axis, -0.0088])

    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=10,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=2,
                       jointType=j_type,
                       jointAxis=[0, 0, 0],
                       parentFramePosition=[0.042, 0, 0.03 + dis_axis],
                       childFramePosition=[0.00808, 0 - dis_axis, -0.0088])

    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=11,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=2,
                       jointType=j_type,
                       jointAxis=[0, 0, 1],
                       parentFramePosition=[0.042, 0, 0.03 - dis_axis],
                       childFramePosition=[0.00808, 0 + dis_axis, 0.0088])

    p.createConstraint(parentBodyUniqueId=robot_id,
                       parentLinkIndex=11,
                       childBodyUniqueId=feet_id,
                       childLinkIndex=2,
                       jointType=j_type,
                       jointAxis=[0, 0, 1],
                       parentFramePosition=[0.042, 0, 0.03 + dis_axis],
                       childFramePosition=[0.00808, 0 - dis_axis, 0.0088])


def set_constrain_blue_body():
    # enableCollision = 0
    # p.setCollisionFilterPair(robot_id, robot_id, 3, 4, enableCollision)
    j_type = p.JOINT_POINT2POINT
    bb_cid = p.createConstraint(parentBodyUniqueId=robot_id,
                                parentLinkIndex=4,
                                childBodyUniqueId=robot_id,
                                childLinkIndex=3,
                                jointType=j_type,
                                jointAxis=[0, 0, 0],
                                parentFramePosition=[0.042, 0.0, 0.03],
                                childFramePosition=[0.0088, -0.00808, 0.001])

    # p.changeConstraint(bb_cid, maxForce=100,)

def sim_angle(robotId, pos_value):
    joint_ids = [0, 4, 8]
    for i in range(500):
        for joint in range(num_link):
            p.setJointMotorControl2(robotId, joint_ids[joint], controlMode=p.POSITION_CONTROL,
                                    targetPosition=pos_value[joint],
                                    force=force,
                                    maxVelocity=maxVelocity)
        p.stepSimulation()
        time.sleep(1. / 240.)


if __name__ == "__main__":
    GUI = True
    if GUI:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pd.getDataPath())  # optionally
    # p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])

    urdf_path_1 = 'blue-body/urdf/blue-body.urdf'
    # urdf_path_2 = 'blue-feet/urdf/blue-feet.urdf'
    # urdf_path_4 = 'blue04/urdf/blue04.urdf'

    robot_id = p.loadURDF(urdf_path_1, startPos, startOrientation, useFixedBase=1, useMaximalCoordinates=False)
    basePos, baseOrn = p.getBasePositionAndOrientation(robot_id)  # Get model position
    basePos_list = [basePos[0], basePos[1], 0.3]
    p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=75, cameraPitch=-20,
                                 cameraTargetPosition=basePos_list)  # fix camera onto model

    constrain = True
    if constrain:
        set_constrain_blue_body()
        # set_constrain_blue04()

    control = False
    if control:
        for i in range(10):
            angle01 = random.random() - 0.5
            angle02 = random.random() - 0.5
            angle03 = random.random() - 0.5
            angle_list = np.array([angle01, angle02, angle03]) * np.pi / 3
            sim_angle(robotId=robot_id, pos_value=angle_list)
    else:
        for i in range(100000):
            # if i < 10:
            #     time.sleep(10)
            p.stepSimulation()
            time.sleep(1. / 240.)

    p.disconnect()
