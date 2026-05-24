"""
Custom event functions for cube rotation task.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_from_grasp_cache(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    cache_file: str,
) -> None:
    """Reset cube pose and robot joints from pre-computed grasp cache."""

    # Load cache (only once, then store on env)
    cache = getattr(env, "_grasp_cache", None)
    if cache is None:
        cache_path = Path(cache_file)
        if not cache_path.exists():
            print(f"[WARNING] Grasp cache not found: {cache_path}")
            return
        data = np.load(cache_path)
        cache = {
            "joint_pos": torch.tensor(data["joint_pos"], device=env.device, dtype=torch.float32),
            "cube_pose_rel": torch.tensor(data["cube_pose_rel"], device=env.device, dtype=torch.float32),
        }
        env._grasp_cache = cache
        print(f"[grasp_cache] Loaded {cache['joint_pos'].shape[0]} grasps from {cache_path}")

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    num_resets = len(env_ids)
    num_grasps = cache["joint_pos"].shape[0]

    # Pick random grasps
    indices = torch.randint(0, num_grasps, (num_resets,), device=env.device)

    # Set robot joint positions
    robot: Entity = env.scene["robot"]
    selected_joints = cache["joint_pos"][indices]
    robot.data.joint_pos[env_ids] = selected_joints
    robot.data.joint_vel[env_ids] = 0.0
    robot.data.joint_pos_target[env_ids] = selected_joints

    # Set cube pose
    cube: Entity = env.scene["cube"]
    pose = cache["cube_pose_rel"][indices].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]

    cube.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    cube.write_root_link_velocity_to_sim(
        torch.zeros((num_resets, 6), device=env.device),
        env_ids=env_ids,
    )