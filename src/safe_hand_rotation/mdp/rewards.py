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
        self.history_steps = 4
        _, _, yaw = euler_xyz_from_quat(cube_orientation(env))
        self.prev_yaw = yaw.clone()
        # Ring buffer for history averaging
        self._delta_hist = torch.zeros(
            (env.num_envs, self.history_steps), device=env.device
        )
        self._hist_ptr = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._hist_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        # Track initial pose for drift gating
        self._init_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self._init_roll = torch.zeros(env.num_envs, device=env.device)
        self._init_pitch = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids):
        _, _, yaw = euler_xyz_from_quat(cube_orientation(self._env))
        self.prev_yaw[env_ids] = yaw[env_ids]
        self._delta_hist[env_ids] = 0.0
        self._hist_ptr[env_ids] = 0
        self._hist_count[env_ids] = 0
        # Save initial pose for drift detection
        cube = self._env.scene["cube"]
        roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
        self._init_pos[env_ids] = cube.data.root_link_pos_w[env_ids]
        self._init_roll[env_ids] = roll[env_ids]
        self._init_pitch[env_ids] = pitch[env_ids]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        object_name: str,
        clip_min: float = -0.25,
        clip_max: float = 0.25,
        drift_position_threshold: float = 0.02,
        drift_tilt_threshold: float = 0.35,
    ) -> torch.Tensor:
        cube = env.scene[object_name]
        roll, pitch, cur_yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)

        # Handle resets
        reset_mask = env.episode_length_buf <= 1
        if torch.any(reset_mask):
            self.reset(reset_mask.nonzero(as_tuple=False).squeeze(-1))

        # 1. Compute yaw delta
        delta_yaw = wrap_to_pi(cur_yaw - self.prev_yaw)
        self.prev_yaw = cur_yaw.clone()

        # 2. History averaging — smooth over last 4 steps
        env_ids = torch.arange(env.num_envs, device=env.device)
        self._delta_hist[env_ids, self._hist_ptr] = delta_yaw
        self._hist_ptr = (self._hist_ptr + 1) % self.history_steps
        self._hist_count = torch.clamp(self._hist_count + 1, max=self.history_steps)
        hist_len = torch.clamp(self._hist_count, min=1).float()
        avg_delta = self._delta_hist.sum(dim=-1) / hist_len

        # 3. Convert to yaw rate
        yaw_rate = avg_delta / max(env.step_dt, 1e-6)
        yaw_reward = torch.clamp(-yaw_rate, min=clip_min, max=clip_max)

        # 4. Drift gating — reduce reward if cube is sliding or tilting
        pos_error = torch.norm(
            cube.data.root_link_pos_w - self._init_pos, dim=-1
        )
        roll_error = wrap_to_pi(roll - self._init_roll).abs()
        pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
        tilt_error = torch.norm(
            torch.stack([roll_error, pitch_error], dim=-1), dim=-1
        )

        in_bounds = (pos_error <= drift_position_threshold) & (
            tilt_error <= drift_tilt_threshold
        )
        drift_factor = torch.where(in_bounds, 1.0, 0.1)

        return yaw_reward * drift_factor

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