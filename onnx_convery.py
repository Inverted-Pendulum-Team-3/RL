#!/usr/bin/env python3
"""
Run this on your training machine to export the model for deployment.
"""
"""
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_env import GymEnv
import sys
print(sys.executable)
POLICY_PATH    = "balro_policy.onnx"
NORM_MEAN_PATH = "balro_norm_mean.npy"
NORM_VAR_PATH  = "balro_norm_var.npy"
NORM_CLIP_PATH = "balro_norm_clip.npy"
MODEL_PATH     = "balro_sac_model_36"
NORM_PKL       = "balro_norm_stats_36.pkl"

# ── Load model + normalization ──────────────────────────────────────────────
env = DummyVecEnv([lambda: GymEnv()])
env = VecNormalize.load(NORM_PKL, env)
env.training = False
env.norm_reward = False
model = SAC.load(MODEL_PATH, env=env)

# ── Save normalization stats as plain numpy ─────────────────────────────────
import pickle
with open("balro_norm_stats.pkl", "wb") as f:
    pickle.dump({
        "mean":     env.obs_rms.mean,
        "var":      env.obs_rms.var,
        "clip_obs": env.clip_obs,
    }, f)
print("✅ Saved normalization stats to balro_norm_stats.pkl")
print(f"   obs mean shape: {env.obs_rms.mean.shape}")
print(f"   obs mean[:3]:   {env.obs_rms.mean[:3]}")
# ── Export policy to ONNX ───────────────────────────────────────────────────
obs_dim   = env.observation_space.shape[0]
dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

policy = model.policy
policy.eval()

class ActorWrapper(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
    def forward(self, obs):
        mean_action = self.actor.mu(self.actor.features_extractor(obs))
        return torch.tanh(mean_action)

wrapped = ActorWrapper(policy.actor)
wrapped.eval()

torch.onnx.export(
    wrapped,
    dummy_obs,
    POLICY_PATH,
    input_names  = ["observation"],
    output_names = ["action"],
    dynamic_axes = {
        "observation": {0: "batch"},
        "action":      {0: "batch"},
    },
    opset_version = 17,
)
print("✅ Exported policy to balro_policy.onnx")
"""

import numpy as np
import torch
import pickle
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_env import GymEnv
import sys
print(sys.executable)

POLICY_PATH = "balro_policy.onnx"
MODEL_PATH  = "balro_sac_model_36"
NORM_PKL    = "balro_norm_stats_36.pkl"

# ── Load model + normalization ──────────────────────────────────────────────
env = DummyVecEnv([lambda: GymEnv()])
env = VecNormalize(env, training=False, norm_reward=False)

with open(NORM_PKL, "rb") as f:
    stats = pickle.load(f)

env.obs_rms.mean = stats["mean"]
env.obs_rms.var  = stats["var"]
env.clip_obs     = stats["clip_obs"]
env.training     = False
env.norm_reward  = False

model = SAC.load(MODEL_PATH, env=env)

# ── Save normalization stats as plain numpy ─────────────────────────────────
with open("balro_norm_stats.pkl", "wb") as f:
    pickle.dump({
        "mean":     env.obs_rms.mean,
        "var":      env.obs_rms.var,
        "clip_obs": env.clip_obs,
    }, f)
print("✅ Saved normalization stats to balro_norm_stats.pkl")
print(f"   obs mean shape: {env.obs_rms.mean.shape}")
print(f"   obs mean[:3]:   {env.obs_rms.mean[:3]}")

# ── Export policy to ONNX ───────────────────────────────────────────────────
obs_dim   = env.observation_space.shape[0]
dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

policy = model.policy
policy.eval()

class ActorWrapper(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
    def forward(self, obs):
        features    = self.actor.features_extractor(obs)
        latent_pi   = self.actor.latent_pi(features)   # MLP hidden layers
        mean_action = self.actor.mu(latent_pi)         # output layer
        return torch.tanh(mean_action)

wrapped = ActorWrapper(policy.actor)
wrapped.eval()

torch.onnx.export(
    wrapped,
    dummy_obs,
    POLICY_PATH,
    input_names  = ["observation"],
    output_names = ["action"],
    dynamic_axes = {
        "observation": {0: "batch"},
        "action":      {0: "batch"},
    },
    opset_version = 17,
)
print("✅ Exported policy to balro_policy.onnx")
