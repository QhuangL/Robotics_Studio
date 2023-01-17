import pybullet as p
import time
import pybullet_data as pd

if __name__ == "__main__":
    GUI = True
    if GUI:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pd.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])

    robot_id = p.loadMJCF("p.xml")
    # basePos, baseOrn = p.getBasePositionAndOrientation(robot_id)  # Get model position
    # basePos_list = [basePos[0], basePos[1], 0.3]
    # p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=75, cameraPitch=-20,
    #                              cameraTargetPosition=basePos_list)  # fix camera onto model

    for i in range(100000):
        # if i < 10:
        #     time.sleep(10)
        p.stepSimulation()
        time.sleep(1. / 240.)

    p.disconnect()
