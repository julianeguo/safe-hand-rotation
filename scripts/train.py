"""Train a policy for cube rotation (PPO or CPO)."""

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import safe_hand_rotation.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Train cube rotation policy")
    parser.add_argument("--task", type=str, default="SafeHandRotation-LeapLeft-v0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--algo", type=str, choices=["ppo", "cpo"], default="ppo")
    args = parser.parse_args()

    # If CPO requested, use the CPO task
    if args.algo == "cpo":
        args.task = "SafeHandRotation-LeapLeft-CPO-v0"

    # GPU setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load configs from registry
    env_cfg = load_env_cfg(args.task)
    rl_cfg = load_rl_cfg(args.task)

    # Override settings
    env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        rl_cfg.max_iterations = args.max_iterations

    # Create environment
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    # Wrap with cost computation if using CPO
    if args.algo == "cpo":
        from safe_hand_rotation.algos.cost_wrapper import CostEnvWrapper
        env = CostEnvWrapper(env, max_torque=0.25, safe_height=0.05)

    # Set up logging
    log_dir = Path("logs") / rl_cfg.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Convert config to dict
    rl_dict = asdict(rl_cfg)

    # For CPO, inject CPO-specific settings into the config dict
    if args.algo == "cpo":
        rl_dict["algorithm"]["class_name"] = "safe_hand_rotation.algos.cpo.CPO"
        rl_dict["algorithm"]["cost_limit"] = 0.2
        rl_dict["algorithm"]["lagrange_lr"] = 0.01
        rl_dict["algorithm"]["initial_lagrange"] = 0.0
        rl_dict["algorithm"]["rnd_cfg"] = None

    # Create runner and train
    runner = MjlabOnPolicyRunner(env, rl_dict, str(log_dir), device)
    runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()