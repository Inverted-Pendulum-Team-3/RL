from stable_baselines3 import SAC
from gym_env import GymEnv
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer
import os
import torch as th

env = DummyVecEnv([lambda: GymEnv()])

balro_model = "balro_sac_model_36"
norm_stats = "balro_norm_stats_36.pkl"

# --- Check if we're resuming or starting fresh ---
model_exists = os.path.exists(f"{balro_model}.zip")
stats_exist = os.path.exists(norm_stats)

if model_exists and stats_exist:
    print("Loading existing model and normalization stats...")
    
    # First: Load normalization stats into a wrapper
    env = VecNormalize.load(norm_stats, env)
    env.training = True  # Ensure training mode
    env.norm_obs = True
    env.norm_reward = True
    
    # Then: Load model with the properly configured env
    model = SAC.load(balro_model, env=env)
    env.envs[0].model = model
    print(f"Resumed from {model.num_timesteps} timesteps")
    print(f"Replay buffer size: {model.replay_buffer.size()}")
    
else:
    print("Creating new model...")
    
    # Create fresh VecNormalize
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8
    )
    
    # Create fresh model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gradient_steps=1,
        gamma=0.99,
        ent_coef= 0.05,
        device='auto',
        train_freq=1,
        optimize_memory_usage=False,
    )
    env.envs[0].model = model  # Pass model reference to env for timestep access
# Set training mode
env.envs[0].training = True
  # Pass model reference to env for timestep access
# --- Training loop ---
try:
    for i in range(750):
        print(f"\n--- Cycle {i+1} (Total timesteps: {model.num_timesteps}) ---")
        experience_scale = min(1.0, model.num_timesteps / 200000)
        new_ent_coef =  0.05 * (1 - experience_scale) + 0.003 * experience_scale
        model.ent_coef = new_ent_coef
        model.ent_coef_tensor = th.tensor(new_ent_coef, device=model.device, requires_grad=False)
        model.learn(
            total_timesteps=10000,
            log_interval=3,
            reset_num_timesteps=False
        )
        
        # Save both model (with replay buffer!) and normalization stats
        if (i + 1) % 10 == 0:
            model.save(balro_model, include=["replay_buffer"])  # FIXED!
            env.save(norm_stats)
            print(f"Cycle {i+1}: Saved at {model.num_timesteps} timesteps")
            print(f"  Replay buffer size: {model.replay_buffer.size()}")

except KeyboardInterrupt:
    print("\nKeyboard interrupt received. Saving...")
    model.save(balro_model, include=["replay_buffer"])  # FIXED!
    env.save(norm_stats)
    print("Model and normalization stats saved.")

finally:
    env.close()
    print("Training ended safely.")
