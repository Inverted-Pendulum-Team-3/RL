import pybullet as p
import pybullet_data
import numpy as np
import csv

MAXFORCE = 1.56

def create_env():
    #Create physicsClient
    physicsClient = p.connect(p.GUI)
    #physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    #Load ground plane
    planeId = p.loadURDF("plane.urdf")
    
    #Load robot from URDF, arbitrary start position, and standing starting orientation
    startPos = [0, 0, .15]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0.7])
    boxId = p.loadURDF("C:\\Users\\dylan\\Documents\\Semester_9\\CAPSTONE\\Robot_RL\\Balro3.urdf", startPos, startOrientation)
    
    #Set friction and gravity
    p.changeDynamics(planeId, -1, lateralFriction=10.0)
    p.changeDynamics(boxId, 1, lateralFriction=3.0)
    p.changeDynamics(boxId, 0, lateralFriction=3.0)
    p.setGravity(0,0,-9.8) 
    
    return boxId
    
def step_sim(velocity1, velocity2, boxId, target_velocity, target_rot_rate):
    #Set motor speeds and step simulation
    p.setJointMotorControl2(bodyUniqueId = boxId, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity = velocity1 * 12.775, force= MAXFORCE)
    p.setJointMotorControl2(bodyUniqueId = boxId, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity = velocity2 * 12.775, force= MAXFORCE)
    p.stepSimulation()
    
    #get velocity of robot (velocity of forward movement)
    linear_velocity = get_linear_velocity(boxId)
    
    #Collect orientation data from base link
    roll, pitch = get_roll_pitch(boxId)
    
    #Collect roll rate and rotation rate data
    roll_rate, rot_rate = get_pitch_rate(boxId)
    
    #Collect wheel velocity data from joints
    wheel_velocity_1 = p.getJointState(boxId, 0)[1]
    wheel_velocity_2 = p.getJointState(boxId, 1)[1]

    
    #Display pendulum angle and sleep
    #p.addUserDebugText("Pendulum angle: {:.2f}".format(roll * 57), [0, 0, 1.2], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.041)
    #p.addUserDebugText("Target Velocity: {:.2f}".format(target_velocity), [0, 0, 1], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Linear Velocity: {:.2f}".format(linear_velocity), [0, 0, 1.4], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Velocity Compliance: {:.2f}".format(-1 * abs(target_velocity - (linear_velocity * 0.2))), [0, 0, 1.6], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Linear Velocity: {:.2f}".format(linear_velocity), [0, 0, 1], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Target Velocity: {:.2f}".format(target_velocity), [0, 0, 1.2], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Roll: {:.2f}".format(roll), [0, 0, 1.2], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Pitch: {:.2f}".format(pitch * 10), [0, 0, 1.4], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Roll Rate: {:.2f}".format(roll_rate * 0.3), [0, 0, 1.6], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Rot. Rate: {:.2f}".format(rot_rate), [0, 0, 1.8], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Wheel Velocity 1: {:.2f}".format(velocity1), [0, 0, 2], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Wheel Velocity 2: {:.2f}".format(velocity2), [0, 0, 2.2], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Target Velocity: {:.2f}".format(target_velocity), [0, 0, 2.4], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("Target Velocity: {:.2f}".format(target_rot_rate), [0, 0, 2.6], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    
    #log_data("state_vals/linear_velocity.csv", linear_velocity*0.8)
    #log_data("state_vals/roll.csv", roll * 3.33)
    #log_data("state_vals/pitch.csv", pitch * 5)
    #log_data("state_vals/roll_rate.csv", roll_rate * 0.5)
    #log_data("state_vals/rot_rate.csv", rot_rate*0.1333)
    #log_data("state_vals/wheel_velocity_1.csv", wheel_velocity_1)
    #log_data("state_vals/wheel_velocity_2.csv", wheel_velocity_2)
  
    
    state = linear_velocity, roll, pitch, roll_rate, rot_rate, wheel_velocity_1, wheel_velocity_2, target_velocity, target_rot_rate
    return state

def get_linear_velocity(boxId):
    linear_vel_world, angular_vel_world = p.getBaseVelocity(boxId)
    pos, orn = p.getBasePositionAndOrientation(boxId)
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    
    x, y, z = rot_matrix.T @ np.array(linear_vel_world)
    #log_data("state_vals/z.csv", z)
    #p.addUserDebugText("Linear Velocity: {:.2f}".format(z), [0, 0, 1.4], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    linear_velocity = (z ** 2 + y ** 2) ** 0.5
    if y < 0:
        linear_velocity = -linear_velocity
    #p.addUserDebugText("Linear Velocity: {:.2f}".format(linear_velocity), [0, 0, 1.4], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    return linear_velocity


def get_roll_pitch(boxId):
    _, orn = p.getBasePositionAndOrientation(boxId)
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    g_world = np.array([0, 0, -1])
    g_body = rot_matrix.T @ g_world

    # Use -g_body[2] so upright = 0
    roll = np.arctan2(g_body[1], -g_body[2])
    pitch = np.arctan2(-g_body[0], np.sqrt(g_body[1]**2 + g_body[2]**2))

    return roll, pitch


def get_pitch_rate(robot_id):
    
    #Get angular velocity in world frame
    _, ang_vel_world = p.getBaseVelocity(robot_id)
    ang_vel_world = np.array(ang_vel_world)

    #Get orientation quaternion and rotation matrix
    _, orn = p.getBasePositionAndOrientation(robot_id)
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    #Transform angular velocity to body frame
    #Body-frame angular velocity = R^T * world-frame angular velocity
    ang_vel_body = rot_matrix.T @ ang_vel_world

    #Extract pitch rate and roll rate
    roll_rate = float(ang_vel_body[0])
    pitch_rate = float(ang_vel_body[1])
    rot_rate  = float(ang_vel_body[2])
    #p.addUserDebugText("roll rate: {:.2f}".format(roll_rate), [0, 0, 1.4], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("pitch rate: {:.2f}".format(pitch_rate), [0, 0, 1.6], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)
    #p.addUserDebugText("rot rate: {:.2f}".format(rot_rate), [0, 0, 1.8], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.1)

    return roll_rate, rot_rate

    
def log_data(file, data):
    with open (file, mode = 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([data])
   
            
def env_disconnect():
    p.disconnect()
    
