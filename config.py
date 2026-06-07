"""
Centralized configuration for the Highway-RL training pipeline.

All hyperparameters, environment settings, and training parameters
are defined here to maintain clean separation from core logic.
"""

from typing import Dict, List, Any


# =============================================================================
# Environment Configuration
# =============================================================================
ENV_ID: str = "highway-fast-v0"
LANES_COUNT: int = 3
VEHICLES_COUNT: int = 15
DURATION: int = 60
SIMULATION_FREQUENCY: int = 15
POLICY_FREQUENCY: int = 5

# Observation Configuration (Kinematics)
OBSERVATION_CONFIG: Dict[str, Any] = {
    "type": "Kinematics",
    "vehicles_count": 5,
    "features": ["presence", "x", "y", "vx", "vy"],
    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20],
    },
    "absolute": False,
    "normalize": True,
    "flatten": True,
}

# Action Configuration (Discrete Meta-Actions)
ACTION_CONFIG: Dict[str, str] = {
    "type": "DiscreteMetaAction",
}

# Action mapping for reference
ACTION_MAP: Dict[int, str] = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

# =============================================================================
# Reward Configuration
# =============================================================================
COLLISION_REWARD: float = -1.0
HIGH_SPEED_REWARD: float = 0.4
RIGHT_LANE_REWARD: float = 0.1
REWARD_SPEED_RANGE: List[float] = [20, 30]
NORMALIZE_REWARD: bool = True

# =============================================================================
# PPO Hyperparameters
# =============================================================================
LEARNING_RATE: float = 5e-4
GAMMA: float = 0.8
GAE_LAMBDA: float = 0.8
N_STEPS: int = 512
BATCH_SIZE: int = 64
N_EPOCHS: int = 10
CLIP_RANGE: float = 0.2
ENT_COEF: float = 0.01
VF_COEF: float = 0.5
MAX_GRAD_NORM: float = 0.5

# Neural Network Architecture
NET_ARCH: List[int] = [256, 256]

# =============================================================================
# Training Configuration
# =============================================================================
TOTAL_TIMESTEPS: int = 150_000
N_ENVS: int = 8
SEED: int = 42
LOG_DIR: str = "logs"
MODEL_DIR: str = "models"
TENSORBOARD_LOG: str = "tensorboard_logs"

# Checkpoint steps
UNTRAINED_STEP: int = 0
HALF_TRAINED_STEP: int = 75_000
FULLY_TRAINED_STEP: int = TOTAL_TIMESTEPS

# =============================================================================
# Evaluation & Recording
# =============================================================================
EVAL_EPISODES: int = 5
EVAL_FREQ: int = 5_000
VIDEO_STEPS: int = 300
VIDEO_DIR: str = "videos"
ASSETS_DIR: str = "assets"


def get_env_config() -> Dict[str, Any]:
    """
    Build the complete environment configuration dictionary.

    Returns:
        Dictionary with all environment parameters for highway-env.
    """
    return {
        "lanes_count": LANES_COUNT,
        "vehicles_count": VEHICLES_COUNT,
        "duration": DURATION,
        "simulation_frequency": SIMULATION_FREQUENCY,
        "policy_frequency": POLICY_FREQUENCY,
        "collision_reward": COLLISION_REWARD,
        "high_speed_reward": HIGH_SPEED_REWARD,
        "right_lane_reward": RIGHT_LANE_REWARD,
        "reward_speed_range": REWARD_SPEED_RANGE,
        "normalize_reward": NORMALIZE_REWARD,
        "observation": OBSERVATION_CONFIG,
        "action": ACTION_CONFIG,
    }
