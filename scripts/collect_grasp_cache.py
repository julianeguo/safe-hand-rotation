"""Collect hand-cube grasp cache by passive settling.

Places the cube near the palm, runs zero actions to let the hand settle,
and saves stable grasps for use during training reset.
"""

import numpy as np
import torch
import os

from mjlab.envs import ManagerBasedRlEnv
from safe_hand_rotation.tasks.cube_rotate.env_cfg import cube_rotate_env_cfg

os.environ["WANDB_MODE"] = "offline"


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_envs = 256
    settle_steps = 240

    # Stability thresholds
    lin_vel_threshold = 0.08
    ang_vel_threshold = 1.2
    max_palm_distance = 0.14
    min_cube_height = 0.05

    # Load env config and modify for collection
    env_cfg = cube_rotate_env_cfg()
    env_cfg.scene.num_envs = num_envs

    # Make episodes long enough for settling
    step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, (settle_steps + 2) * step_dt)

    # Create environment
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    robot = env.scene["robot"]
    cube = env.scene["cube"]
    action_dim = env.action_manager.total_action_dim

    # Find palm body for distance checks
    palm_body_id, _ = robot.find_bodies("palm", preserve_order=True)
    palm_body_id = palm_body_id[0]

    zero_action = torch.zeros((num_envs, action_dim), device=device)
    all_env_ids = torch.arange(num_envs, device=device)

    # Reset and let the hand settle with zero actions
    env.reset()
    for step in range(settle_steps):
        env.step(zero_action)
        if (step + 1) % 60 == 0:
            print(f"  Settling step {step + 1}/{settle_steps}")

    # Check stability
    cube_pos_w = cube.data.root_link_pos_w
    cube_quat_w = cube.data.root_link_quat_w
    cube_lin_vel_w = cube.data.root_link_lin_vel_w
    cube_ang_vel_w = cube.data.root_link_ang_vel_w

    palm_pos_w = robot.data.body_link_pos_w[:, palm_body_id]
    cube_dist = torch.norm(cube_pos_w - palm_pos_w, dim=-1)

    stable = torch.ones(num_envs, dtype=torch.bool, device=device)
    stable &= torch.norm(cube_lin_vel_w, dim=-1) <= lin_vel_threshold
    stable &= torch.norm(cube_ang_vel_w, dim=-1) <= ang_vel_threshold
    stable &= cube_dist <= max_palm_distance
    stable &= cube_pos_w[:, 2] >= min_cube_height

    stable_ids = stable.nonzero(as_tuple=False).squeeze(-1)
    kept = int(stable_ids.numel())

    print(f"\nStable grasps: {kept}/{num_envs}")

    if kept == 0:
        print("No stable grasps found! The cube is probably not starting inside the hand.")
        print(f"  Cube mean pos: {cube_pos_w.mean(dim=0).cpu().numpy()}")
        print(f"  Palm mean pos: {palm_pos_w.mean(dim=0).cpu().numpy()}")
        print(f"  Mean distance: {cube_dist.mean().item():.4f}")
        print(f"  Mean cube height: {cube_pos_w[:, 2].mean().item():.4f}")
        print(f"  Mean lin vel: {torch.norm(cube_lin_vel_w, dim=-1).mean().item():.4f}")
        env.close()
        return

    # Save stable grasps
    cube_pose_rel = torch.cat(
        [
            cube_pos_w[stable_ids] - env.scene.env_origins[stable_ids],
            cube_quat_w[stable_ids],
        ],
        dim=-1,
    )

    out_path = "src/safe_hand_rotation/tasks/cube_rotate/grasp_cache.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez_compressed(
        out_path,
        joint_pos=robot.data.joint_pos[stable_ids].cpu().numpy().astype(np.float32),
        cube_pose_rel=cube_pose_rel.cpu().numpy().astype(np.float32),
        joint_names=np.asarray(robot.joint_names),
    )
    print(f"Saved {kept} grasps to {out_path}")

    env.close()


if __name__ == "__main__":
    main()