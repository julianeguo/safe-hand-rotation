"""
Reward functions for the cube rotation task.

Each function takes the environment and returns a tensor of shape
[num_envs] — one reward value per parallel environment.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi
from safe_hand_rotation.mdp.observations import cube_orientation, cube_position
from mjlab.managers.scene_entity_config import SceneEntityCfg


if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class yaw_rate_reward:
    def __init__(self, cfg, env):
        self._env = env
        _, _, yaw = euler_xyz_from_quat(cube_orientation(env))
        self.prev_yaw = yaw.clone()
    
    def reset(self, env_ids):
        _, _, yaw = euler_xyz_from_quat(cube_orientation(self._env))
        self.prev_yaw[env_ids] = yaw[env_ids]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        object_name: str,
        clip_min: float = -0.25,
        clip_max: float = 0.25,
    ) -> torch.Tensor:
        roll, pitch, cur_yaw = euler_xyz_from_quat(cube_orientation(env))
        delta_yaw = wrap_to_pi(cur_yaw - self.prev_yaw)
        self.prev_yaw = cur_yaw
        return torch.clamp(delta_yaw, min=clip_min, max=clip_max)

def object_fallen(
        env: ManagerBasedRlEnv,
        object_name: str,
        minimum_height: float,
    ) -> torch.Tensor:
        cube = env.scene[object_name]
        return (cube.data.root_link_pos_w[:, 2] < minimum_height).float()

def joint_torque_penalty(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
    ) -> torch.Tensor:
        robot = env.scene[asset_cfg.name]
        torques = torch.square(robot.data.actuator_force[:, asset_cfg.joint_ids]) # squares all values -> L2 regularization
        return torch.sum(torques, dim= -1) # one sum per env