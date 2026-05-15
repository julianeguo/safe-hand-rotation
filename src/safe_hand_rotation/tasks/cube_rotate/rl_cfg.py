"""
RL training configuration for PPO (and later CPO).

Defines the neural network architecture and PPO hyperparameters.
"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO training config for cube rotation."""
    return RslRlOnPolicyRunnerCfg(

        # ── Neural network architecture ─────────────────────────────
        actor=RslRlModelCfg(
            hidden_dims=(512, 512, 256),
            activation="elu",
            obs_normalization=True,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 512, 256),
            activation="elu",
            obs_normalization=True,
        ),

        # ── PPO algorithm settings ─────────────────────────────────
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=5,         # passes over collected data
            num_mini_batches=4,            # split data into 4 chunks
            learning_rate=1e-3,            # step size for updates
            schedule="adaptive",           # adjust LR based on KL divergence
            gamma=0.99,                    # discount factor
            lam=0.95,                      # GAE lambda
            entropy_coef=0.003,            # encourage exploration
            desired_kl=0.01,               # target KL divergence
            max_grad_norm=1.0,             # clip gradients
            value_loss_coef=1.0,           # critic loss weight
            use_clipped_value_loss=True,   # PPO value clipping
            clip_param=0.2,               # PPO clip range
        ),

        # ── Training loop settings ──────────────────────────────────
        experiment_name="leap_cube_rotate_ppo",
        num_steps_per_env=32,              # steps collected before each update
        max_iterations=5000,               # total training iterations
        save_interval=100,                 # save checkpoint every 100 iters
        clip_actions=1.0,                  # clamp actions to [-1, 1]
    )