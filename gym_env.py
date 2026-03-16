import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import csv
from pyb_env import create_env, step_sim, env_disconnect
#import agent

class GymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Action space: 2 motor velocities (normalized between -1 and 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.training = True
        # Observation space: roll, pitch, roll_rate, rot_rate, wheel_vel_1, wheel_vel_2, targ_vel, targ_rotation
        self.observation_space = spaces.Box(
    low=np.array([
        -10.0,   # linear_velocity (m/s)
        -1.0,   # roll (≈ -40°)
        -1.0,  # roll_rate (rad/s)
        -3.0,  # rot_rate (rad/s)
        -5.0,  # wheel_velocity_1 (rad/s)
        -5.0,  # wheel_velocity_2 (rad/s)
        -10.0,   # velocity_error
        -10.0    # rot_rate_error
    ], dtype=np.float32),
    high=np.array([
        10.0, 1.0, 1.0, 3.0, 5.0, 5.0, 10.0, 10.0
    ], dtype=np.float32),
    dtype=np.float32
)
        
        self.boxId = None
        self.step_count = 0
        self.max_steps = 1000
        self.episode_reward = 0

    def reset(self, *, seed=None, options=None):
        
        super().reset(seed=seed)
        
        if self.boxId is None: 
            self.boxId = create_env()
        
        
        experience_scale = min(1.0, self.model.num_timesteps / 200000)
        angle = np.random.normal(loc=0, scale = 0.05 *(1- experience_scale) + 0.19 * experience_scale)#scale=0.05 + (experience_scale * 0.2), size=1)
        
        if abs(angle) > 0.18:
            angle = np.random.uniform(low = -0.18, high = 0.18)
       
    
        
        p.resetBasePositionAndOrientation(self.boxId, [0, 0, .17], p.getQuaternionFromEuler([angle, 0, 0.7]) )
        p.resetBaseVelocity(self.boxId, [0,0,0], [0,0,0])
       
       
        if self.step_count != 0 and self.training == True:
            log_data("Tests/stability_training_36.csv", self.step_count)
            log_data("Tests/stability_rewards_36.csv", self.episode_reward)
            print(self.step_count)
        
        self.step_count = 0
        self.episode_reward = 0
        self.max_steps = 1000
        
        self.target_velocity = 0
        self.target_rot = 0
        if self.training == False:
            self.target_velocity = 0
            self.target_rot = 0
            self.max_steps = 10000
        
        state = np.array(step_sim(0, 0, self.boxId, self.target_velocity, self.target_rot, self.model.num_timesteps), dtype=np.float32)
        
        
        info = {}
        
        return state, info

    def step(self, action):
        # Denormalize actions if needed
        experience_scale = min(1.0, self.model.num_timesteps / 200000)
        
        velocity1 = float(action[0] + np.random.normal(0, 0.02 * experience_scale))
        velocity2 = float(action[1] + np.random.normal(0, 0.02 * experience_scale))
        state = np.array(step_sim(velocity1, velocity2, self.boxId, self.target_velocity, self.target_rot, self.model.num_timesteps), dtype=np.float32)


        reward = compute_reward_5(state)
        #p.addUserDebugText("Reward: {:.2f}".format(reward), [0, 0, 0.8], textColorRGB=[0,1,0], textSize=1.5, lifeTime=0.1)
        self.episode_reward += reward
        terminated = abs(state[1]) > 0.25
        
            
        truncated = False
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        info = {}
        
        
        return state, reward, terminated, truncated, info
    


    def close(self):
        env_disconnect()
        

def compute_reward_5(state):
    roll = state[1]
    velocity_error = state[6]
    rotation_error = state[7]
    tracking_gate = np.exp(-(roll / 0.125)**2)
    # Balance reward — gaussian centered at 0, falls off quickly
    balance_reward = (2 * tracking_gate) - 1

    # Gate tracking penalties by how well balanced the robot is
    

    velocity_penalty = tracking_gate * min(abs(velocity_error), 2.0)
    rotation_penalty = tracking_gate * min(abs(rotation_error), 2.0)

    reward = (
        balance_reward
        - velocity_penalty
        - rotation_penalty
    )

    return reward

def log_data(file, data):
    with open (file, mode = 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([data])
