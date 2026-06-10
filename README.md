# Safe In-Hand Rotation with Constrained Policy Optimization

Comparing **CPO (Constrained Policy Optimization)** against a **PPO** baseline for in-hand cube rotation using a simulated LEAP robotic hand in [MjLab](https://github.com/mujocolab/mjlab).

PPO maximizes rotation speed using soft reward penalties. CPO enforces hard safety constraints via a Lagrangian relaxation, automatically adjusting a penalty multiplier to keep torque and drop-risk costs within a budget.

## Key Results (1000 iterations)

| Metric | PPO | CPO |
|--------|-----|-----|
| Yaw-rate reward | 0.097 | 0.015 |
| Torque penalty | 0.077 | 0.060 |
| Drop penalty | 0.052 | 0.017 |
| Lagrange multiplier | — | 0.0096 |

CPO rotates slower but keeps safety costs lower, which is the expected safe-RL tradeoff.

## Project Structure

```
safe-hand-rotation/
├── scripts/
│   ├── train.py                    # Train PPO or CPO (--algo ppo|cpo)
│   ├── play.py                     # Evaluate a trained policy
│   └── collect_grasp_cache.py      # Build grasp cache for episode resets
├── src/safe_hand_rotation/
│   ├── robots/
│   │   └── leap_hand/              # LEAP hand config, XML assets, constants
│   ├── tasks/
│   │   └── cube_rotate/
│   │       ├── env_cfg.py          # Environment configuration
│   │       ├── rl_cfg.py           # PPO/CPO training hyperparameters
│   │       └── grasp_cache.npz     # Pre-computed stable grasps (255)
│   ├── mdp/
│   │   ├── observations.py         # Joint/cube observation functions
│   │   ├── rewards.py              # Yaw-rate, fallen penalty, torque penalty
│   │   ├── terminations.py         # Cube dropped, timeout
│   │   ├── costs.py                # Safety costs (torque limit, drop proximity)
│   │   └── events.py               # Grasp cache reset logic
│   └── algos/
│       ├── cpo.py                  # CPO algorithm (Lagrangian relaxation)
│       └── cost_wrapper.py         # Wrapper that computes costs each step
└── pyproject.toml
```

## Setup

Requires Python 3.12+ and a CUDA-capable GPU.

```bash
git clone https://github.com/<your-username>/safe-hand-rotation.git
cd safe-hand-rotation
uv sync
uv add scipy  # unlisted mjlab dependency
```

## Usage

### Train PPO
```bash
uv run python scripts/train.py --algo ppo --num-envs 4096 --max-iterations 1000
```

### Train CPO
```bash
uv run python scripts/train.py --algo cpo --num-envs 4096 --max-iterations 1000
```

### Collect grasp cache (only needed once)
```bash
uv run python scripts/collect_grasp_cache.py
```

### Environment variables for headless GPU training (e.g. Colab)
```bash
export WANDB_MODE=offline
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
```

## How CPO Works

CPO extends PPO by modifying the reward each timestep:

```
penalized_reward = reward - λ × cost
```

Where λ (the Lagrange multiplier) automatically adjusts:
- Cost above budget → λ increases → policy becomes safer
- Cost below budget → λ decreases → policy pursues reward

This is a Lagrangian relaxation of the constrained optimization problem, implemented as a subclass of rsl_rl's PPO.

## Acknowledgments

Built on [MjLab](https://github.com/mujocolab/mjlab) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl). Task design adapted from [HORA](https://github.com/haozhiqi/hora) (Qi et al., 2022). Robot model from [LEAP Hand](https://github.com/leap-hand/LEAP_Hand_Sim) (Shaw et al., 2023). Reference implementation by [Hao Jiang](https://github.com/Msornerrrr/in-hand-rotation-mjlab).

## References

- Schulman et al., "Proximal Policy Optimization Algorithms," 2017
- Achiam et al., "Constrained Policy Optimization," ICML 2017
- Stooke et al., "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods," 2020
- Qi et al., "In-Hand Object Rotation via Rapid Motor Adaptation," CoRL 2022
- Shaw et al., "LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning," RSS 2023
