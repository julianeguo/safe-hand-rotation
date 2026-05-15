"""Evaluate a trained policy."""

import argparse

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import safe_hand_rotation.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Evaluate cube rotation policy")
    parser.add_argument(
        "--task",
        type=str,
        default="SafeHandRotation-LeapLeft-v0",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    env_cfg = load_env_cfg(args.task)
    rl_cfg = load_rl_cfg(args.task)

    env_cfg.scene.num_envs = args.num_envs
    rl_cfg.load_checkpoint = args.checkpoint

    Runner = load_runner_cls(args.task)
    runner = Runner(env_cfg, rl_cfg)
    runner.play()


if __name__ == "__main__":
    main()