"""Train a policy for cube rotation."""

import argparse

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.rl import MjlabOnPolicyRunner

# This import triggers task registration
import safe_hand_rotation.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Train cube rotation policy")
    parser.add_argument(
        "--task",
        type=str,
        default="SafeHandRotation-LeapLeft-v0",
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

    # Load configs from registry
    env_cfg = load_env_cfg(args.task)
    rl_cfg = load_rl_cfg(args.task)

    # Override num_envs
    env_cfg.scene.num_envs = args.num_envs

    # Override max_iterations if provided
    if args.max_iterations is not None:
        rl_cfg.max_iterations = args.max_iterations

    # Create runner and train
    runner = MjlabOnPolicyRunner(env_cfg, rl_cfg)
    runner.learn()


if __name__ == "__main__":
    main()