# 🚗 Highway-RL: Autonomous Driving with Deep Reinforcement Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3-green?logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

**CMP4501 – Applied Reinforcement Learning | Semester Project**

*Student: Aysel Suleymanova 2105382
*Track: Option A – Autonomous Driving with Highway-Env*

</div>

---

## 🎬 Training Evolution

The following animation demonstrates how the agent improves across three stages of training:

<div align="center">

![Training Evolution](assets/evolution.gif)

</div>

| Stage | Description |
|-------|-------------|
| **Untrained** | Random actions — the vehicle crashes almost immediately into surrounding traffic |
| **Half-Trained** | The agent begins to avoid collisions and maintain speed, but still makes errors |
| **Fully Trained** | Smooth, confident driving at high speed through dense traffic with zero crashes |

---

## 📐 Methodology

### A. The Reward Function

The agent's behavior is shaped by a carefully designed **multi-objective reward function** that balances speed, safety, and lane discipline:

$$R(s, a) = \underbrace{\alpha \cdot \hat{v}}_{\text{speed}} + \underbrace{\beta \cdot \mathbb{1}_{\text{crash}}}_{\text{collision}} + \underbrace{\gamma \cdot \frac{l}{L-1}}_{\text{right lane}}$$

**Where:**

| Symbol | Formula | Description |
|--------|---------|-------------|
| $\hat{v}$ | $\text{clip}\left(\frac{v - v_{\min}}{v_{\max} - v_{\min}},\ 0,\ 1\right)$ | Normalized vehicle speed within range $[20, 30]$ m/s |
| $\mathbb{1}_{\text{crash}}$ | $\in \{0, 1\}$ | Binary indicator: 1 if the ego vehicle has collided |
| $l / (L - 1)$ | — | Lane index normalized to $[0, 1]$; higher values = rightmost lane |

**Reward Coefficients:**

| Coefficient | Value | Role |
|-------------|-------|------|
| $\alpha$ | `0.4` | Encourages the agent to maintain high speed |
| $\beta$ | `-1.0` | Heavy penalty makes collision avoidance the top priority |
| $\gamma$ | `0.1` | Mild preference for the rightmost lane (realistic driving) |

**Justification:** This design creates a clear optimization hierarchy — **safety first** (crash penalty dominates at $-1.0$), **speed second** (the largest positive component), and **driving discipline third** (lane and stability bonuses). The normalization to approximately $[0, 1]$ ensures stable gradient updates during PPO training.

---

### B. The Model

#### Algorithm: Proximal Policy Optimization (PPO)

**Why PPO?**
- ✅ Stable training with clipped objective — avoids destructive policy updates
- ✅ Works well with discrete action spaces (5 meta-actions)
- ✅ Sample-efficient with vectorized environments
- ✅ Proven effectiveness on highway-env benchmarks (RL Baselines3 Zoo)
- ✅ CPU-friendly — no GPU required

**Hyperparameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning Rate | `5e-4` | Standard for PPO; high enough for fast convergence on simple tasks |
| Discount Factor ($\gamma$) | `0.8` | Low value for short-horizon highway episodes; prioritizes immediate rewards |
| GAE Lambda ($\lambda$) | `0.95` | High value for low-bias advantage estimation |
| Batch Size | `64` | Balances gradient stability and CPU memory |
| Mini-Epochs | `10` | Multiple passes over each batch for sample efficiency |
| Clip Range ($\epsilon$) | `0.2` | Standard PPO clipping threshold |
| Entropy Coeff. | `0.01` | Small bonus to maintain exploration |
| N Steps | `256` | Rollout length per environment before each update |
| Parallel Envs | `4` | DummyVecEnv for 4× throughput without subprocess overhead |
| Total Timesteps | `150,000` | Sufficient for convergence (~15 min on CPU) |

#### Neural Network Architecture

```
Input Layer:     25 neurons (5 vehicles × 5 features, flattened)
                        ↓
Hidden Layer 1:  256 neurons (Tanh activation)
                        ↓
Hidden Layer 2:  256 neurons (Tanh activation)
                   ↙          ↘
Policy Head:   5 outputs     Value Head: 1 output
(action probs via softmax)   (state value estimate)
```

- **Input**: Flattened kinematics matrix (25 values)
- **Hidden Layers**: Two fully-connected layers of 256 units with Tanh activation
- **Policy Head**: Outputs a probability distribution over 5 discrete actions
- **Value Head**: Outputs a scalar estimate of the state's value (used by the critic)

---

### C. States and Actions

#### Observation Space (What the Agent Sees)

The agent receives a **5×5 kinematics matrix** representing the ego vehicle and its 4 nearest neighbors:

| Feature | Description | Range |
|---------|-------------|-------|
| `presence` | Whether a vehicle exists in this slot | $\{0, 1\}$ |
| `x` | Longitudinal position relative to ego | $[-100, 100]$ m |
| `y` | Lateral position relative to ego | $[-100, 100]$ m |
| `vx` | Longitudinal velocity relative to ego | $[-20, 20]$ m/s |
| `vy` | Lateral velocity relative to ego | $[-20, 20]$ m/s |

- Observations are **relative** to the ego vehicle (not absolute world coordinates)
- All features are **normalized** to $[-1, 1]$ for training stability
- Row 0 is always the ego vehicle; rows 1-4 are the nearest neighbors

#### Action Space (What the Agent Can Do)

5 discrete **meta-actions** that provide high-level tactical control:

| ID | Action | Effect |
|----|--------|--------|
| 0 | `LANE_LEFT` | Change one lane to the left |
| 1 | `IDLE` | Maintain current lane and speed |
| 2 | `LANE_RIGHT` | Change one lane to the right |
| 3 | `FASTER` | Increase speed |
| 4 | `SLOWER` | Decrease speed |

The low-level vehicle dynamics (steering, acceleration) are handled automatically by the simulation's motion planner.

---

## 📊 Training Analysis

### A. Reward Graph

<div align="center">

![Training Reward Curve](assets/reward_plot.png)

</div>

*The blue shaded region shows raw per-episode rewards. The red line is a 20-episode rolling average.*

### B. Commentary

**Phase 1 — Random Exploration (Episodes 0–100):**
During the initial phase, the agent acts essentially randomly. Episode rewards are low and highly variable, with frequent crashes resulting in short episodes. The mean reward hovers near the minimum as the agent has not yet learned any useful policy.

**Phase 2 — Rapid Learning (Episodes 100–400):**
The smoothed reward curve shows a steep upward trend as the agent begins to learn collision avoidance — the most heavily penalized behavior. The agent first learns to *not crash*, then gradually learns to *accelerate*. Reward variance decreases as behavior becomes more consistent.

**Phase 3 — Refinement and Convergence (Episodes 400+):**
The reward curve plateaus near its maximum as the agent has learned an effective policy: drive fast, stay in the rightmost lanes, and avoid lane changes unless necessary. Small fluctuations remain due to stochastic traffic configurations, but the policy is stable.

**Key Observations:**
- The discount factor of $\gamma = 0.8$ proved critical — with $\gamma = 0.99$, the agent was too far-sighted and failed to learn immediate collision avoidance
- Using 8 parallel environments provided ~7× speedup over a single environment with minimal overhead

---

## 🔥 Challenges and Failures

### Challenge 1: The "Slow and Safe" Trap

**Problem:** In early training runs, the agent discovered that the safest strategy was to **drive extremely slowly**. By always choosing the `SLOWER` action, it could avoid all collisions and accumulate zero crash penalties. However, this resulted in poor driving performance — the vehicle would crawl along the highway while all other traffic passed it.

**Root Cause:** The high-speed reward ($\alpha = 0.4$) was not large enough relative to the crash penalty ($\beta = -1.0$) to incentivize risk-taking.

**Solution:** Lowered the discount factor from $\gamma = 0.99$ to $\gamma = 0.8$. This made the agent more "short-sighted," causing it to value the immediate speed reward more highly relative to the distant possibility of a crash. Combined with the normalized reward range $[0, 1]$, this rebalanced the trade-off between speed and safety.


---

## 🚀 How to Run

### Prerequisites

- Python 3.9 or higher
- Standard laptop CPU (no GPU required)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/highway-rl-agent.git
cd highway-rl-agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training (Optional)

*Note: You can skip this step and go straight to Evaluation, as the pre-trained models are already included in the `models/` directory.*

```bash
# Train the agent (~15 minutes on CPU)
python -m src.train
```

This will:
- Train a PPO agent for 150,000 timesteps
- Save checkpoints at: `models/untrained.zip`, `models/half_trained.zip`, `models/fully_trained.zip`
- Generate `assets/reward_plot.png`
- Log metrics to `logs/training_rewards.csv`

### Evaluation & Video Generation

```bash
# Evaluate and record videos
python -m src.evaluate
```

This will:
- Record gameplay videos for each training stage → `videos/`
- Create the evolution GIF → `assets/evolution.gif`
- Print a performance comparison table

---

## 📁 Repository Structure

```
highway-rl-agent/
│
├── README.md                  ← This report
├── requirements.txt           ← Python dependencies
├── .gitignore                 ← Git exclusions
│
├── src/
│   ├── __init__.py            ← Package marker
│   ├── config.py              ← All hyperparameters (centralized)
│   ├── environment.py         ← Env setup & reward wrapper
│   ├── model.py               ← PPO model creation & checkpointing
│   ├── train.py               ← Training pipeline
│   ├── evaluate.py            ← Evaluation & video recording
│   └── utils.py               ← Plotting & video utilities
│
├── assets/
│   ├── evolution.gif           ← Training evolution animation
│   └── reward_plot.png         ← Reward vs. episodes plot
│
└── videos/
    ├── untrained.mp4           ← Random agent recording
    ├── half_trained.mp4        ← Mid-training recording
    └── fully_trained.mp4       ← Final agent recording
```

---

## 🛠️ Built With

| Library | Version | Purpose |
|---------|---------|---------|
| [Gymnasium](https://gymnasium.farama.org/) | 0.29.1 | RL environment interface |
| [highway-env](https://highway-env.farama.org/) | 1.9.1 | Highway driving simulation |
| [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) | 2.3.2 | PPO implementation |
| [PyTorch](https://pytorch.org/) | ≥2.0 | Neural network backend |
| [Matplotlib](https://matplotlib.org/) | ≥3.7 | Training visualization |
| [imageio](https://imageio.readthedocs.io/) | ≥2.31 | Video/GIF creation |

---

<div align="center">

*CMP4501 – Applied Reinforcement Learning | 2025*

</div>
