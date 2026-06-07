"""
Training pipeline for the Highway-RL autonomous driving agent.

Trains a PPO agent on the highway-fast-v0 environment with
custom reward shaping, saving checkpoints at untrained,
half-trained, and fully-trained stages for evolution video.
"""

import os
from typing import Any, Callable, Dict, List

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import highway_env  # noqa: F401

from src.config import (
    ENV_ID,
    EVAL_EPISODES,
    EVAL_FREQ,
    HALF_TRAINED_STEP,
    LOG_DIR,
    MODEL_DIR,
    N_ENVS,
    SEED,
    TOTAL_TIMESTEPS,
    get_env_config,
)
from src.model import create_model, save_model
from src.utils import plot_training_rewards, save_rewards_to_csv


class RewardLoggerCallback(BaseCallback):
    """
    Custom callback to log episode rewards during training.

    Collects per-episode rewards for post-training analysis
    and visualization.

    Attributes:
        episode_rewards: List of total rewards per completed episode.
        episode_lengths: List of episode lengths.
    """

    def __init__(self, verbose: int = 0) -> None:
        """Initialize the reward logger callback."""
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def _on_step(self) -> bool:
        """
        Called at each training step. Logs completed episode rewards.

        Returns:
            True to continue training.
        """
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
                if self.verbose > 0 and len(self.episode_rewards) % 50 == 0:
                    avg = np.mean(self.episode_rewards[-50:])
                    print(
                        f"[TRAIN] Episode {len(self.episode_rewards)} | "
                        f"Avg Reward (last 50): {avg:.3f}"
                    )
        return True


class HalfTrainedCallback(BaseCallback):
    """
    Callback to save a checkpoint at the half-trained mark.

    Attributes:
        half_step: The timestep at which to save the half-trained checkpoint.
        saved: Whether the checkpoint has already been saved.
    """

    def __init__(
        self,
        half_step: int = HALF_TRAINED_STEP,
        save_dir: str = MODEL_DIR,
        verbose: int = 0,
    ) -> None:
        """
        Initialize half-trained checkpoint callback.

        Args:
            half_step: Timestep to trigger the checkpoint save.
            save_dir: Directory to save the checkpoint.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.half_step: int = half_step
        self.save_dir: str = save_dir
        self.saved: bool = False

    def _on_step(self) -> bool:
        """
        Save checkpoint when half-trained timestep is reached.

        Returns:
            True to continue training.
        """
        if not self.saved and self.num_timesteps >= self.half_step:
            save_model(self.model, "half_trained", self.save_dir)
            self.saved = True
            print(
                f"[INFO] Half-trained checkpoint saved at step {self.num_timesteps}"
            )
        return True


def _make_env_fn(
    seed: int,
    env_config: Dict[str, Any],
) -> Callable[[], gym.Env]:
    """
    Create a factory function for vectorized environment creation.

    Each env is wrapped with LaneChangePenaltyWrapper and Monitor
    for proper episode stats tracking.

    Args:
        seed: Random seed.
        env_config: Highway-env configuration dictionary.

    Returns:
        Callable that creates a configured environment.
    """

    def _init() -> gym.Env:
        env = gym.make(ENV_ID, render_mode=None)
        env.unwrapped.config.update(env_config)
        env.reset(seed=seed)
        env = Monitor(env)
        return env

    return _init


def train(
    total_timesteps: int = TOTAL_TIMESTEPS,
    n_envs: int = N_ENVS,
    seed: int = SEED,
) -> None:
    """
    Execute the full training pipeline.

    Steps:
        1. Create vectorized training environments (DummyVecEnv)
        2. Save untrained checkpoint (initial random policy)
        3. Train PPO agent with callbacks for logging and checkpointing
        4. Save fully-trained checkpoint
        5. Generate reward plot and save metrics

    Args:
        total_timesteps: Total number of training timesteps.
        n_envs: Number of parallel environments.
        seed: Random seed for reproducibility.
    """
    print("=" * 60)
    print("  Highway-RL Agent Training Pipeline")
    print("=" * 60)
    print(f"  Environment:     {ENV_ID}")
    print(f"  Algorithm:       PPO (MlpPolicy)")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Parallel Envs:   {n_envs}")
    print(f"  Seed:            {seed}")
    print("=" * 60)

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Build vectorized environment
    env_config = get_env_config()
    env_fns = [_make_env_fn(seed + i, env_config) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    print(f"\n[INFO] Created {n_envs} parallel environments")
    print(f"[INFO] Observation space: {vec_env.observation_space}")
    print(f"[INFO] Action space: {vec_env.action_space}")

    # Create model
    model = create_model(vec_env)

    # Save untrained checkpoint
    save_model(model, "untrained", MODEL_DIR)
    print("[INFO] Untrained checkpoint saved")

    # Setup callbacks
    reward_logger = RewardLoggerCallback(verbose=1)
    half_trained_cb = HalfTrainedCallback(
        half_step=HALF_TRAINED_STEP,
        save_dir=MODEL_DIR,
    )

    # Create eval environment for periodic evaluation
    eval_env_fn = _make_env_fn(seed + 100, env_config)
    eval_env = DummyVecEnv([eval_env_fn])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "best"),
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // n_envs, 1),
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = [reward_logger, half_trained_cb, eval_callback]

    # Train
    print(f"\n[INFO] Starting training for {total_timesteps:,} timesteps...")
    print("[INFO] This should take approximately 5-15 minutes on CPU\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save fully trained model
    save_model(model, "fully_trained", MODEL_DIR)
    print("[INFO] Fully trained checkpoint saved")

    # Save training metrics
    if reward_logger.episode_rewards:
        save_rewards_to_csv(reward_logger.episode_rewards)
        plot_training_rewards(
            reward_logger.episode_rewards,
            title="Highway-RL PPO Training: Reward vs Episode",
        )
        print(f"\n[INFO] Training completed!")
        print(f"[INFO] Total episodes: {len(reward_logger.episode_rewards)}")
        print(
            f"[INFO] Final avg reward (last 50): "
            f"{np.mean(reward_logger.episode_rewards[-50:]):.3f}"
        )
    else:
        print("[WARN] No episode rewards were logged")

    # Cleanup
    vec_env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("  Checkpoints saved in: models/")
    print("  Reward plot saved in: assets/reward_plot.png")
    print("  Next step: Run evaluate.py to generate videos")
    print("=" * 60)


if __name__ == "__main__":
    train()
