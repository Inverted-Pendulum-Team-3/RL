from stable_baselines3 import SAC
from gym_env import GymEnv
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer
import os
"""
env = DummyVecEnv([lambda: GymEnv()])
#env.training = True
model = SAC.load("balro_sac_model_12", env=env)
state = env.reset()
while True:
    action, _ = model.predict(state, deterministic=True)  # deterministic = best action
    state, reward, terminated, truncated = env.step(action)



"""
"""
# --- Create environment ---
env = DummyVecEnv([lambda: GymEnv()])
env.envs[0].training = True

balro_model = "balro_sac_model_19"

# --- Load or create model ---
try:
    model = SAC.load(balro_model, env=env)
except FileNotFoundError:
    model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,        # try 3e-4
    buffer_size=100000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gradient_steps = -1,
    gamma=0.99,
    ent_coef='auto_0.1',
    device='auto',
    train_freq=(1, "episode"),
    optimize_memory_usage=False,

)

# --- Training loop ---
try:
    for i in range(750):
        model.learn(
            total_timesteps=10000,
            log_interval=3,
            reset_num_timesteps=False
        )
        # Save model with replay buffer
        if (i + 1) % 10 == 0:
        
            model.save(balro_model, include=["replay_buffer"])
            print(f"Cycle {i+1}: Model saved successfully.")

        # Optional: reset replay buffer every 10 cycles (if desired)
        if i == 35:
            print("Resetting replay buffer...")
            model.replay_buffer = ReplayBuffer(
                buffer_size=model.replay_buffer.buffer_size,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=model.device,
                optimize_memory_usage=False,
            )
            
            print("Replay buffer reset successfully.")
            """
            #Refill a SAC replay buffer with fresh transitions
            #WITHOUT updating model weights.
"""
            steps = 50000
            # Ensure no training happens during refill
            old_learning_starts = model.learning_starts
            old_train_freq = model.train_freq
            old_gradient_steps = model.gradient_steps

            model.learning_starts = 10**18   # guarantee no updates
            model.gradient_steps = 0         # no gradients

            obs = env.reset()
            added = 0

            print(f"\nRefilling replay buffer with {steps} steps...")
            
            env.envs[0].training = False

            
            while added < steps:
                # Policy chooses an action (no backprop)
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, done, info = env.step(action)
                #done = terminated or truncated
                # Insert sample manually
                model.replay_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    infos= info   # SB3 requires list
                )

                obs = next_obs
                if done:
                    obs = env.reset()

                added += 1

                # Restore original training config
            model.learning_starts = old_learning_starts
            model.train_freq = old_train_freq
            model.gradient_steps = old_gradient_steps
            env.envs[0].training = True

            print(f"Replay buffer refill complete. {steps} samples added.\n")
            

except KeyboardInterrupt:
    print("\nKeyboard interrupt received. Saving model and exiting...")
    model.save(balro_model, include=["replay_buffer"])
    print("Model saved successfully.")

finally:
    env.close()
    print("Environment closed. Training session ended safely.")
    


# --- Create environment with normalization ---

env = DummyVecEnv([lambda: GymEnv()])
env = VecNormalize(
    env,
    norm_obs=True,          # Normalize observations
    norm_reward=True,       # Also normalize rewards (important!)
    clip_obs=10.0,          # Clip normalized obs to ±10
    clip_reward=10.0,       # Clip normalized rewards
    gamma=0.99,             # Discount factor for reward normalization
    epsilon=1e-8            # Numerical stability
)

# Evaluation environment (uses same normalization stats)
eval_env = DummyVecEnv([lambda: GymEnv()])
eval_env = VecNormalize(
    eval_env,
    training=False,         # Don't update stats during eval
    norm_obs=True,
    norm_reward=False,      # Don't normalize rewards during eval
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99
)

env.envs[0].training = True

balro_model = "balro_sac_model_21"
norm_stats = "balro_norm_stats_21.pkl"

# --- Load or create model ---
try:
    model = SAC.load(balro_model, env=env)
    # CRITICAL: Load normalization statistics too!
    env = VecNormalize.load(norm_stats, env)
    print("Loaded model and normalization stats")
except FileNotFoundError:
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=500000,     # Larger buffer as discussed
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gradient_steps=-1,      # Train once per episode
        gamma=0.99,
        ent_coef='auto',
        device='auto',
        train_freq=(1, "episode"),
        optimize_memory_usage=False,
    )
    print("Created new model")

# --- Training loop ---
try:
    for i in range(750):
        model.learn(
            total_timesteps=10000,
            log_interval=3,
            reset_num_timesteps=False
        )
        
        # Save both model AND normalization stats
        if (i + 1) % 10 == 0:
            model.save(balro_model)
            env.save(norm_stats)  # CRITICAL!
            print(f"Cycle {i+1}: Model and norm stats saved.")

except KeyboardInterrupt:
    print("\nKeyboard interrupt received. Saving...")
    model.save(balro_model, include=["replay_buffer"])
    env.save(norm_stats)
    print("Model and normalization stats saved.")

finally:
    env.close()
    print("Training ended safely.")
"""

env = DummyVecEnv([lambda: GymEnv()])

balro_model = "balro_sac_model_22"
norm_stats = "balro_norm_stats_22.pkl"

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
        buffer_size=500000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gradient_steps=-1,
        gamma=0.99,
        ent_coef='auto',
        device='auto',
        train_freq=(1, "episode"),
        optimize_memory_usage=False,
    )

# Set training mode
env.envs[0].training = True

# --- Training loop ---
try:
    for i in range(750):
        print(f"\n--- Cycle {i+1} (Total timesteps: {model.num_timesteps}) ---")
        
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