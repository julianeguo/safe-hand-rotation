"""
Cost functions for CPO safety constraints.

Each function returns a tensor of shape [num_envs]:
0.0 = safe, positive = constraint violation (bigger = worse).

These are used by CPO only — PPO ignores them.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def torque_limit_cost(
    env: ManagerBasedRlEnv,
    max_torque: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> torch.Tensor:
    """Cost for exceeding joint torque limits.
    
    Returns 0 when all torques are within limits.
    Returns a positive value when any joint exceeds max_torque.
    """
    robot = env.scene[asset_cfg.name]
    torques = torch.abs(robot.data.actuator_force[:, asset_cfg.joint_ids])
    exceeded = torch.sum(torch.clamp(torques - max_torque, min=0), dim=-1) # sum the last dimension (1 number per env)
    return exceeded

def cube_drop_proximity_cost(
    env: ManagerBasedRlEnv,
    object_name: str,
    safe_height: float,
) -> torch.Tensor:
    """Cost that increases as the cube approaches dropping height.
    
    Returns 0 when cube is above safe_height.
    Returns positive value when cube dips below safe_height.
    """
    cube = env.scene[object_name]
    height = cube.data.root_link_pos_w[:, 2] - safe_height
    return torch.clamp(safe_height - height, min=0)