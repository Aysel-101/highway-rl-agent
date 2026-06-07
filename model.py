"""
Model creation and checkpoint management for Highway-RL.

Handles PPO model instantiation with configured hyperparameters
and provides utilities for saving/loading model checkpoints.
"""

import os
from typing import Optional

import gymnasium as gym
from stable_baselines3 import PPO

from src.config import (
    BATCH_SIZE,
    CLIP_RANGE,
    ENT_COEF,
    GAE_LAMBDA,
    GAMMA,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MODEL_DIR,
    N_EPOCHS,
    N_STEPS,
    NET_ARCH,
    SEED,
    TENSORBOARD_LOG,
    VF_COEF,
)


def create_model(
    env: gym.Env,
    tensorboard_log: Optional[str] = TENSORBOARD_LOG,
) -> PPO:
    """
    Create a new PPO model with configured hyperparameters.

    Architecture:
        Input (5 vehicles × 5 features = 25 flattened)
        → Dense(256, Tanh)
        → Dense(256, Tanh)
        → Policy Head (5 discrete actions, softmax)
        → Value Head (scalar critic)

    Args:
        env: The Gymnasium environment for the model.
        tensorboard_log: Directory for TensorBoard logging.

    Returns:
        Configured PPO model ready for training.
    """
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs={"net_arch": NET_ARCH},
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=SEED,
    )

    return model


def save_model(model: PPO, name: str, directory: str = MODEL_DIR) -> str:
    """
    Save a model checkpoint to disk.

    Args:
        model: The PPO model to save.
        name: Name for the checkpoint file (without extension).
        directory: Directory to save the checkpoint in.

    Returns:
        The full path to the saved checkpoint file.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    model.save(path)
    print(f"[INFO] Model saved to: {path}.zip")
    return path


def load_model(name: str, env: gym.Env, directory: str = MODEL_DIR) -> PPO:
    """
    Load a model checkpoint from disk.

    Args:
        name: Name of the checkpoint file (without extension).
        env: The Gymnasium environment to attach to the model.
        directory: Directory containing the checkpoint.

    Returns:
        Loaded PPO model with the specified environment.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    path = os.path.join(directory, name)
    full_path = path if path.endswith(".zip") else f"{path}.zip"

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint not found: {full_path}")

    model = PPO.load(path, env=env)
    print(f"[INFO] Model loaded from: {full_path}")
    return model


def get_checkpoint_path(name: str, directory: str = MODEL_DIR) -> str:
    """
    Get the full path to a checkpoint file.

    Args:
        name: Name of the checkpoint (without extension).
        directory: Directory containing checkpoints.

    Returns:
        Full path to the checkpoint .zip file.
    """
    return os.path.join(directory, f"{name}.zip")
