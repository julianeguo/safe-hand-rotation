"""
Observation functions for the cube rotation task.

Each function takes the environment (and optional config) and returns
a tensor of shape [num_envs, obs_dim] — one row per parallel environment.

We start with the essentials. cube_size, cube_mass, friction etc. for domain randomization
and similar elements in the reference repo will be added later if time permits.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ── Robot observations ─────────────────────────────────────────────────────

def joint_pos_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> torch.Tensor:
    """Current joint positions (relative to default)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> torch.Tensor:
    """Current joint velocities."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


# ── Cube observations ─────────────────────────────────────────────────────

def cube_position(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Cube position in world frame [x, y, z]."""
    cube = env.scene[object_cfg.name]
    return cube.data.root_link_pos_w


def cube_orientation(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Cube orientation in world frame [qw, qx, qy, qz]."""
    cube = env.scene[object_cfg.name]
    return cube.data.root_link_quat_w


def cube_linear_velocity(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Cube linear velocity in world frame [vx, vy, vz]."""
    cube = env.scene[object_cfg.name]
    return cube.data.root_link_lin_vel_w


def cube_angular_velocity(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Cube angular velocity in world frame [wx, wy, wz]."""
    cube = env.scene[object_cfg.name]
    return cube.data.root_link_ang_vel_w


def last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The most recent action (joint commands) sent to the robot."""
    return env.action_manager.action