import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import csv
from pyb_env import create_env, step_sim, env_disconnect
import time

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
        -1.0,   # pitch (≈ -40°)
        -1.0,  # roll_rate (rad/s)
        -3.0,  # rot_rate (rad/s)
        -3.0,  # wheel_velocity_1 (rad/s)
        -3.0,  # wheel_velocity_2 (rad/s)
        -7.0,   # target_velocity
        -10.0    # target_rot_rate
    ], dtype=np.float32),
    high=np.array([
        10.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 7.0, 10.0
    ], dtype=np.float32),
    dtype=np.float32
)
        
        self.boxId = None
        self.step_count = 0
        rand = np.random.uniform(low = -100, high = 100)
        self.max_steps = 10000 + rand
        if self.training == False:
            self.max_steps = 10000
        self.episode_reward = 0
        #self.target_velocity = np.random.uniform(low = -.1, high = .1)
        #self.target_velocity = 0

    def reset(self, *, seed=None, options=None):
        
        super().reset(seed=seed)
        
        if self.boxId is None: 
            self.boxId = create_env()
        
        angle = np.random.normal(loc=0, scale=0.05 , size=1)
        
        if abs(angle) > 0.25:
            angle = np.random.uniform(low = -0.22, high = 0.22)
       
        
        is_negative = np.random.choice([True, False])
        if is_negative:
           angle *= -1
        
        
        p.resetBasePositionAndOrientation(self.boxId, [0, 0, .15], p.getQuaternionFromEuler([angle, 0, 0.7]) )
        p.resetBaseVelocity(self.boxId, [0,0,0], [0,0,0])
       
       
        if self.step_count != 0 and self.training == True:
            log_data("Tests/stability_training_1.csv", self.step_count)
            log_data("Tests/stability_rewards_1.csv", self.episode_reward)
            print(self.step_count)
        
        self.step_count = 0
        self.episode_reward = 0
        rand = np.random.normal(loc = 0, scale = 20, size = 1)
        self.max_steps = 10000+ rand
        
        self.target_velocity = np.random.normal(loc=0, scale=1e-6)
        self.target_rot = np.random.normal(loc=0, scale=1e-6)
        if self.training == False:
            self.target_velocity = 0
            self.target_rot = 0
            self.max_steps = 10000
        
        state = np.array(step_sim(0, 0, self.boxId, self.target_velocity, self.target_rot), dtype=np.float32)
        
        
        info = {}
        
        return state, info

    def step(self, action):
        # Denormalize actions if needed
        velocity1 = float(action[0])
        velocity2 = float(action[1])
        state = np.array(step_sim(velocity1, velocity2, self.boxId, self.target_velocity, self.target_rot), dtype=np.float32)


        reward = compute_reward_4(state)
        p.addUserDebugText("Reward: {:.2f}".format(reward), [0, 0, 2.8], textColorRGB=[0,1,0], textSize=1.5, lifeTime=0.1)
        self.episode_reward += reward
        terminated = abs(state[1]) > 0.27 or abs(state[2]) > 0.25
        
            
        truncated = False
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        info = {}
        #linear_velocity, roll, pitch, roll_rate * 0.3, rot_rate, wheel_velocity_1 * 0.078, wheel_velocity_2 * 0.078, target_velocity, target_rot_rate
        #log_data("state_vals/linear_velocity.csv", state[0])
        #log_data("state_vals/roll.csv", state[1])
        #log_data("state_vals/pitch.csv",state[2])
        #log_data("state_vals/roll_rate.csv",state[3])
        #log_data("state_vals/rot_rate.csv",state[4])
        #log_data("state_vals/wheel_velocity_1.csv",state[5])
        #log_data("state_vals/wheel_velocity_2.csv",state[6])
        
        
        
        return state, reward, terminated, truncated, info
    


    def close(self):
        env_disconnect()
        

def compute_reward_3(state):
    roll = state[1]
    pitch = state[2]
    linear_velocity = state[0]
    rot_rate = state[4]
    wheel_v1 = state[5]
    wheel_v2 = state[6]
    target_velocity = state[7]
    target_rot_rate = state[8]
    
    gate = np.exp(- (roll / 0.14)**2)
    velocity_penalty = gate * abs(target_velocity - linear_velocity) / 2
    rotation_penalty = gate * abs(target_rot_rate - rot_rate)
    
    reward = gate - velocity_penalty - rotation_penalty
    log_data("state_vals/reward.csv", reward)
    #p.addUserDebugText("Reward: {:.2f}".format(reward), [0, 0, 1.4], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0.041)
    return reward

def compute_reward_4(state):
    linear_velocity = state[0]
    roll = state[1]
    pitch = state[2]
    roll_rate = state[3]
    rot_rate = state[4]
    wheel_v1 = state[5]
    wheel_v2 = state[6]
    target_velocity = state[7]
    target_rot_rate = state[8]
    
    # === Balance reward (most important) ===
    balance_reward = np.exp(- (roll / 0.25)**2 - (pitch / 0.3)**2)
    
    # === Stability: penalize ROLL rate only (not rotation rate!) ===
    # Roll rate = tipping sideways (bad)
    # Rotation rate = turning (controlled by target)
    stability_penalty = 0.2 * (roll_rate**2)
    
    # === Velocity tracking (gated by balance) ===
    tracking_gate = np.exp(- (roll / 0.2)**2)
    velocity_error = abs(target_velocity - linear_velocity)
    rotation_error = abs(target_rot_rate - rot_rate)
    
    velocity_penalty = tracking_gate * 0.5 * velocity_error
    rotation_penalty = tracking_gate * 0.5 * rotation_error  # Increased weight
    
    # === Wheel balance: only penalize when NOT commanded to turn ===
    # If we want to go straight (target_rot_rate ≈ 0), wheels should be similar
    # If we want to turn, differential is expected
    if abs(target_rot_rate) < 0.05:  # Threshold for "going straight"
        wheel_diff_penalty = 0.05 * abs(wheel_v1 - wheel_v2)
    else:
        wheel_diff_penalty = 0.0  # Allow differential when turning
    
    # === Compose final reward ===
    reward = (
        2.0 * balance_reward          # Prioritize balance
        - velocity_penalty            # Track velocity when balanced
        - rotation_penalty            # Track rotation when balanced  
        - stability_penalty           # Penalize roll oscillations
        - wheel_diff_penalty          # Symmetric only when going straight
    )
    
    return reward

def log_data(file, data):
    with open (file, mode = 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([data])