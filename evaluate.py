"""
Evaluation and video recording for the Highway-RL agent.

Loads model checkpoints at untrained, half-trained, and fully-trained
stages, records gameplay videos, and creates the evolution GIF.
"""

import glob
import os
from typing import List, Optional

import gymnasium as gym
import imageio
import numpy as np

import highway_env  # noqa: F401

from src.config import (
    ASSETS_DIR,
    ENV_ID,
    EVAL_EPISODES,
    MODEL_DIR,
    SEED,
    VIDEO_DIR,
    VIDEO_STEPS,
    get_env_config,
)
from src.model import load_model
from src.utils import create_simple_evolution_gif


# Training stages to evaluate
STAGES = [
    ("untrained", "Untrained Agent"),
    ("half_trained", "Half-Trained Agent"),
    ("fully_trained", "Fully Trained Agent"),
]


def record_episode(
    model_name: str,
    stage_label: str,
    output_dir: str = VIDEO_DIR,
    max_steps: int = VIDEO_STEPS,
    seed: int = SEED,
) -> Optional[str]:
    """
    Record a single episode video for a given model checkpoint.

    Creates a fresh environment with rgb_array rendering, loads the
    specified checkpoint, and records the agent's behavior.

    Args:
        model_name: Name of the model checkpoint to load.
        stage_label: Human-readable label for this training stage.
        output_dir: Directory to save the recorded video.
        max_steps: Maximum steps to record per episode.
        seed: Random seed for reproducibility.

    Returns:
        Path to the saved MP4 video, or None if recording failed.
    """
    print(f"\n[EVAL] Recording: {stage_label}")
    print(f"[EVAL] Loading checkpoint: {model_name}")

    os.makedirs(output_dir, exist_ok=True)

    # Create environment with rendering
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env_config = get_env_config()
    env.unwrapped.config.update(env_config)

    # Wrap with Monitor for stats if needed, or leave as is

    # Load the model checkpoint
    try:
        model = load_model(model_name, env)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        env.close()
        return None

    # Record frames
    frames: List[np.ndarray] = []
    obs, info = env.reset(seed=seed)
    frames.append(env.render())

    total_reward = 0.0
    step = 0

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            print(
                f"[EVAL] Episode ended at step {step + 1} | "
                f"Total Reward: {total_reward:.3f}"
            )
            break

    env.close()

    if not frames:
        print(f"[ERROR] No frames captured for {stage_label}")
        return None

    # Save video as MP4
    video_path = os.path.join(output_dir, f"{model_name}.mp4")
    imageio.mimsave(video_path, frames, fps=15)
    print(f"[EVAL] Video saved: {video_path} ({len(frames)} frames)")

    return video_path


def evaluate_model(
    model_name: str,
    n_episodes: int = EVAL_EPISODES,
    seed: int = SEED,
) -> dict:
    """
    Evaluate a model checkpoint over multiple episodes.

    Args:
        model_name: Name of the model checkpoint to evaluate.
        n_episodes: Number of evaluation episodes.
        seed: Random seed.

    Returns:
        Dictionary with evaluation metrics (mean_reward, std_reward, etc.).
    """
    env = gym.make(ENV_ID, render_mode=None)
    env_config = get_env_config()
    env.unwrapped.config.update(env_config)
    # No lane change penalty wrapper needed

    try:
        model = load_model(model_name, env)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        env.close()
        return {}

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    crash_count: int = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1

            if terminated or truncated:
                if info.get("crashed", False):
                    crash_count += 1
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    metrics = {
        "model": model_name,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "crash_rate": crash_count / n_episodes,
        "n_episodes": n_episodes,
    }

    print(f"\n[EVAL] Results for {model_name}:")
    print(f"  Mean Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
    print(f"  Mean Length: {metrics['mean_length']:.1f} steps")
    print(f"  Crash Rate:  {metrics['crash_rate']:.1%}")

    return metrics


def run_full_evaluation() -> None:
    """
    Run the complete evaluation pipeline.

    Records videos for all three training stages, evaluates each
    model, and creates the evolution GIF.
    """
    print("=" * 60)
    print("  Highway-RL Agent Evaluation Pipeline")
    print("=" * 60)

    video_paths: List[str] = []
    all_metrics: List[dict] = []

    # Record videos and evaluate each stage
    for model_name, stage_label in STAGES:
        # Record video
        video_path = record_episode(model_name, stage_label)
        if video_path:
            video_paths.append(video_path)

        # Evaluate performance
        metrics = evaluate_model(model_name)
        if metrics:
            all_metrics.append(metrics)

    # Print comparison table
    if all_metrics:
        print("\n" + "=" * 60)
        print("  Performance Comparison")
        print("=" * 60)
        print(f"{'Stage':<20} {'Mean Reward':<15} {'Crash Rate':<15} {'Avg Length':<15}")
        print("-" * 65)
        for m in all_metrics:
            print(
                f"{m['model']:<20} "
                f"{m['mean_reward']:<15.3f} "
                f"{m['crash_rate']:<15.1%} "
                f"{m['mean_length']:<15.1f}"
            )

    # Create evolution GIF
    if video_paths:
        print(f"\n[INFO] Creating evolution GIF from {len(video_paths)} videos...")
        os.makedirs(ASSETS_DIR, exist_ok=True)
        create_simple_evolution_gif(
            video_paths=video_paths,
            output_path=os.path.join(ASSETS_DIR, "evolution.gif"),
            fps=15,
            max_frames_per_stage=150,
        )

    print("\n" + "=" * 60)
    print("  Evaluation Complete!")
    print(f"  Videos saved in: {VIDEO_DIR}/")
    print(f"  Evolution GIF: {ASSETS_DIR}/evolution.gif")
    print("=" * 60)


if __name__ == "__main__":
    run_full_evaluation()
