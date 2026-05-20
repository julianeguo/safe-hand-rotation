"""Train a policy for cube rotation."""

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

# This import triggers task registration
import safe_hand_rotation.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Train cube rotation policy")
    parser.add_argument("--task", type=str, default="SafeHandRotation-LeapLeft-v0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

    # GPU setup
    os.environ["MUJOCO_GL"] = "egl"
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"

    # Load configs from registry
    env_cfg = load_env_cfg(args.task)
    rl_cfg = load_rl_cfg(args.task)

    # Override settings
    env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        rl_cfg.max_iterations = args.max_iterations

    # 1. Create the actual environment (the simulation)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # 2. Wrap it so rsl_rl can talk to it
    env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    # 3. Set up logging directory
    log_dir = Path("logs") / rl_cfg.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 4. Convert rl config to dict (runner expects a dict)
    rl_dict = asdict(rl_cfg)

    # 5. Create runner and train
    runner = MjlabOnPolicyRunner(env, rl_dict, str(log_dir), device)
    runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()