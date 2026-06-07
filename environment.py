"""
Environment setup for Highway-RL.

Provides factory functions to create configured highway-env environments.
"""

from typing import Any, Dict, Optional

import gymnasium as gym

import highway_env  # noqa: F401

from src.config import ENV_ID, SEED, get_env_config


def make_env(
    render_mode: Optional[str] = None,
    seed: int = SEED,
) -> gym.Env:
    """
    Create and configure a highway-env environment.

    Args:
        render_mode: Gymnasium render mode ('rgb_array' for recording, None for training).
        seed: Random seed for reproducibility.

    Returns:
        Configured Gymnasium environment.
    """
    env = gym.make(ENV_ID, render_mode=render_mode)

    # Apply custom configuration
    env_config: Dict[str, Any] = get_env_config()
    env.unwrapped.config.update(env_config)
    env.reset(seed=seed)

    return env


def make_training_env(seed: int = SEED) -> gym.Env:
    """
    Create an environment optimized for training (no rendering).

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Training-optimized environment with no rendering overhead.
    """
    return make_env(render_mode=None, seed=seed)


def make_recording_env(seed: int = SEED) -> gym.Env:
    """
    Create an environment configured for video recording.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Environment with rgb_array rendering enabled.
    """
    return make_env(render_mode="rgb_array", seed=seed)
