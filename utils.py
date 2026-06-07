"""
Utility functions for plotting, video processing, and metrics.

Provides tools for visualizing training progress, creating
evolution GIFs, and smoothing reward curves.
"""

import csv
import os
from typing import List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np

from src.config import ASSETS_DIR


def smooth_curve(
    values: List[float],
    window_size: int = 20,
) -> np.ndarray:
    """
    Apply a rolling average to smooth noisy reward curves.

    Args:
        values: Raw reward values from training episodes.
        window_size: Number of episodes for the rolling window.

    Returns:
        Smoothed numpy array of the same length as input.
    """
    if len(values) < window_size:
        return np.array(values)

    smoothed = np.convolve(
        values,
        np.ones(window_size) / window_size,
        mode="valid",
    )

    # Pad the beginning to maintain original length
    pad_size = len(values) - len(smoothed)
    padding = np.full(pad_size, smoothed[0])
    return np.concatenate([padding, smoothed])


def plot_training_rewards(
    rewards: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Reward Curve",
    window_size: int = 20,
) -> None:
    """
    Generate a professional training reward plot.

    Creates a dual-layer plot with raw episode rewards (transparent)
    and a smoothed rolling-average curve, annotated with training
    stage markers.

    Args:
        rewards: List of episode rewards from training.
        save_path: Path to save the plot image. Defaults to assets/reward_plot.png.
        title: Title for the plot.
        window_size: Window size for the smoothing function.
    """
    if save_path is None:
        os.makedirs(ASSETS_DIR, exist_ok=True)
        save_path = os.path.join(ASSETS_DIR, "reward_plot.png")

    episodes = list(range(1, len(rewards) + 1))
    smoothed = smooth_curve(rewards, window_size)

    # Professional styling
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw rewards (transparent background)
    ax.plot(
        episodes,
        rewards,
        alpha=0.2,
        color="#4A90D9",
        linewidth=0.8,
        label="Raw Episode Reward",
    )

    # Smoothed curve (prominent)
    ax.plot(
        episodes,
        smoothed,
        color="#E74C3C",
        linewidth=2.5,
        label=f"Smoothed (window={window_size})",
    )

    # Training stage annotations
    total_episodes = len(rewards)
    mid_episode = total_episodes // 2

    if total_episodes > 10:
        ax.axvline(
            x=mid_episode,
            color="#F39C12",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
        )
        ax.text(
            mid_episode + total_episodes * 0.01,
            max(rewards) * 0.95,
            "Half-Trained",
            fontsize=10,
            color="#F39C12",
            fontweight="bold",
        )

    # Labels and title
    ax.set_xlabel("Episode", fontsize=13, fontweight="bold")
    ax.set_ylabel("Episode Reward", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="lower right")
    ax.tick_params(labelsize=11)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Reward plot saved to: {save_path}")


def save_rewards_to_csv(
    rewards: List[float],
    save_path: str = "logs/training_rewards.csv",
) -> None:
    """
    Save episode rewards to a CSV file for reproducibility.

    Args:
        rewards: List of episode rewards.
        save_path: Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, reward in enumerate(rewards, start=1):
            writer.writerow([i, reward])

    print(f"[INFO] Rewards saved to: {save_path}")


def load_rewards_from_csv(
    csv_path: str = "logs/training_rewards.csv",
) -> List[float]:
    """
    Load episode rewards from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of episode rewards.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Rewards CSV not found: {csv_path}")

    rewards: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["reward"]))

    return rewards


def create_evolution_gif(
    video_dirs: List[str],
    output_path: Optional[str] = None,
    fps: int = 15,
    max_frames_per_stage: int = 150,
) -> None:
    """
    Create an evolution GIF from video frame directories.

    Combines frames from untrained, half-trained, and fully-trained
    agent recordings into a single GIF with stage labels.

    Args:
        video_dirs: List of directories containing video frames or MP4 files.
        output_path: Path to save the output GIF.
        fps: Frames per second for the GIF.
        max_frames_per_stage: Maximum frames to include per training stage.
    """
    if output_path is None:
        os.makedirs(ASSETS_DIR, exist_ok=True)
        output_path = os.path.join(ASSETS_DIR, "evolution.gif")

    stage_labels = ["Untrained Agent", "Half-Trained Agent", "Fully Trained Agent"]
    all_frames: List[np.ndarray] = []

    for video_path, label in zip(video_dirs, stage_labels):
        if not os.path.exists(video_path):
            print(f"[WARN] Video not found: {video_path}, skipping stage '{label}'")
            continue

        # Read frames from MP4
        reader = imageio.get_reader(video_path)
        frames = []
        for i, frame in enumerate(reader):
            if i >= max_frames_per_stage:
                break
            frames.append(frame)
        reader.close()

        if not frames:
            print(f"[WARN] No frames in {video_path}")
            continue

        # Add label overlay to frames
        labeled_frames = _add_label_to_frames(frames, label)
        all_frames.extend(labeled_frames)

        # Add brief pause between stages (repeat last frame)
        for _ in range(fps):
            all_frames.append(labeled_frames[-1])

    if all_frames:
        imageio.mimsave(output_path, all_frames, fps=fps, loop=0)
        print(f"[INFO] Evolution GIF saved to: {output_path}")
    else:
        print("[ERROR] No frames collected, GIF not created")


def _add_label_to_frames(
    frames: List[np.ndarray],
    label: str,
) -> List[np.ndarray]:
    """
    Add a text label banner to the top of each frame.

    Uses matplotlib to render text onto frames since PIL/Pillow
    may not be available.

    Args:
        frames: List of video frames as numpy arrays.
        label: Text label to add to each frame.

    Returns:
        List of frames with label overlay.
    """
    labeled: List[np.ndarray] = []
    banner_height = 40

    for frame in frames:
        h, w = frame.shape[:2]
        new_frame = np.zeros((h + banner_height, w, 3), dtype=np.uint8)

        # Dark banner background
        new_frame[:banner_height, :] = [30, 30, 30]

        # Add the original frame below the banner
        new_frame[banner_height:, :] = frame[:, :, :3]

        # Render label using matplotlib
        fig, ax = plt.subplots(figsize=(w / 100, banner_height / 100), dpi=100)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(
            0.5, 0.5, label,
            fontsize=14,
            color="white",
            fontweight="bold",
            ha="center",
            va="center",
        )
        ax.set_facecolor("#1E1E1E")
        ax.axis("off")
        fig.patch.set_facecolor("#1E1E1E")
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        banner = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        banner = banner.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Resize banner to match frame width
        from PIL import Image
        banner_img = Image.fromarray(banner)
        banner_img = banner_img.resize((w, banner_height), Image.LANCZOS)
        new_frame[:banner_height, :] = np.array(banner_img)

        labeled.append(new_frame)

    return labeled


def create_simple_evolution_gif(
    video_paths: List[str],
    output_path: Optional[str] = None,
    fps: int = 15,
    max_frames_per_stage: int = 150,
) -> None:
    """
    Create evolution GIF by simply concatenating video frames.

    A simpler alternative to create_evolution_gif that works
    without PIL/matplotlib text rendering.

    Args:
        video_paths: Paths to MP4 video files for each stage.
        output_path: Output GIF path. Defaults to assets/evolution.gif.
        fps: Frames per second.
        max_frames_per_stage: Max frames per training stage.
    """
    if output_path is None:
        os.makedirs(ASSETS_DIR, exist_ok=True)
        output_path = os.path.join(ASSETS_DIR, "evolution.gif")

    all_frames: List[np.ndarray] = []

    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"[WARN] Video not found: {video_path}")
            continue

        reader = imageio.get_reader(video_path)
        for i, frame in enumerate(reader):
            if i >= max_frames_per_stage:
                break
            all_frames.append(frame)
        reader.close()

    if all_frames:
        imageio.mimsave(output_path, all_frames, fps=fps, loop=0)
        print(f"[INFO] Evolution GIF saved to: {output_path}")
    else:
        print("[ERROR] No frames collected")


def get_video_files(video_dir: str) -> List[str]:
    """
    Find all MP4 video files in a directory.

    Args:
        video_dir: Directory to search for video files.

    Returns:
        Sorted list of absolute paths to MP4 files.
    """
    if not os.path.exists(video_dir):
        return []

    videos = [
        os.path.join(video_dir, f)
        for f in sorted(os.listdir(video_dir))
        if f.endswith(".mp4")
    ]
    return videos
