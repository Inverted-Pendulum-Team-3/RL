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
    startPos = [0, 0, .17]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0.7])
    boxId = p.loadURDF("C:\\Users\\dylan\\Desktop\\WorkingRL\\balro4.urdf", startPos, startOrientation)
    #Set friction and gravity
    p.changeDynamics(planeId, -1, lateralFriction=1.0)
    p.changeDynamics(boxId, 1, lateralFriction=1.0, jointDamping=0.3)
    p.changeDynamics(boxId, 0, lateralFriction=1.0, jointDamping=0.3)
    p.setGravity(0,0,-9.8) 
    
    return boxId
    
def step_sim(velocity1, velocity2, boxId, target_velocity, target_rot_rate, total_timesteps):
    #Set motor speeds and step simulation
    p.setJointMotorControl2(bodyUniqueId = boxId, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity = velocity1 * 12.775, force= MAXFORCE)
    p.setJointMotorControl2(bodyUniqueId = boxId, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity = velocity2 * 12.775, force= MAXFORCE)
    
    for i in range(4):
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

    velocity_error = target_velocity - linear_velocity
    rot_rate_error = target_rot_rate - rot_rate
    
    experience_scale = min(1.0, total_timesteps / 200000)
    
    
    state = linear_velocity + np.random.normal(0,0.05 * experience_scale), roll + np.random.normal(0,0.005 * experience_scale), roll_rate+ np.random.normal(0,0.01 * experience_scale), rot_rate + np.random.normal(0,0.01 * experience_scale), wheel_velocity_1 + np.random.normal(0,0.05 * experience_scale), wheel_velocity_2 + np.random.normal(0,0.05 * experience_scale), velocity_error, rot_rate_error
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
    rot_rate  = float(ang_vel_body[2])
    
    return roll_rate, rot_rate

    
def log_data(file, data):
    with open (file, mode = 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([data])
   
            
def env_disconnect():
    p.disconnect()
