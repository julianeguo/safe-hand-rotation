"""
Environment configuration for the cube rotation task.

This file wires everything together: the scene (robot + cube + floor),
the observation/reward/termination functions from mdp/,
and the action space.

Follows the sample repo's pattern: cube is created programmatically,
terrain provides the floor, robot is loaded from XML via entity config.
"""

from __future__ import annotations

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import JointPositionActionCfg
from mjlab.entity import EntityCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from safe_hand_rotation.robots import get_leap_left_hand_cfg
from safe_hand_rotation.mdp.observations import (
    joint_pos_rel,
    joint_vel_rel,
    cube_position,
    cube_orientation,
    cube_linear_velocity,
    cube_angular_velocity,
    last_action,
)
from safe_hand_rotation.mdp.rewards import (
    yaw_rate_reward,
    object_fallen,
    joint_torque_penalty,
)
from safe_hand_rotation.mdp.terminations import cube_dropped


# --------------------------------------------------------------------------- #
# Scene entity configs (shortcuts used throughout)
# --------------------------------------------------------------------------- #
ROBOT_CFG = SceneEntityCfg("robot", joint_names=(".*",))
CUBE_CFG = SceneEntityCfg("cube")


# --------------------------------------------------------------------------- #
# Cube spec (created programmatically, not from XML)
# --------------------------------------------------------------------------- #
def get_cube_spec(cube_size: float = 0.0375, mass: float = 0.1) -> mujoco.MjSpec:
    """Build a cube MJCF programmatically.

    Args:
        cube_size: half-size of the cube in meters (0.0375 = 7.5cm wide cube)
        mass: cube mass in kg
    """
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(cube_size,) * 3,
        mass=mass,
        rgba=(0.85, 0.3, 0.15, 1.0),
    )
    return spec


# --------------------------------------------------------------------------- #
# Environment configuration
# --------------------------------------------------------------------------- #
def cube_rotate_env_cfg() -> ManagerBasedRlEnvCfg:
    """Build and return the full environment config."""

    return ManagerBasedRlEnvCfg(

        # ── Simulation settings ─────────────────────────────────────────
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,    # Physics at 200 Hz
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
            nconmax=55,
            njmax=600,
        ),
        decimation=10,             # Agent acts every 10 steps → 20 Hz

        # ── Episode length ──────────────────────────────────────────────
        episode_length_s=20.0,     # 20 seconds per episode (matches sample repo)
        scale_rewards_by_dt=True,  # Scale rewards by timestep for consistency

        # ── Scene: what exists in the world ─────────────────────────────
        scene=SceneCfg(
            # Floor
            terrain=TerrainEntityCfg(terrain_type="plane"),
            # Robot and cube
            entities={
                "robot": get_leap_left_hand_cfg(),
                "cube": EntityCfg(
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.1),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                    spec_fn=get_cube_spec,
                ),
            },
            num_envs=1,
            env_spacing=0.6,
        ),

        # ── Viewer ──────────────────────────────────────────────────────
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="palm",
            distance=0.45,
            elevation=-25,
            azimuth=110,
        ),

        # ── Actions: what the robot can do ──────────────────────────────
        # Robot sends target joint positions to its 16 motors.
        actions={
            "joint_pos": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(".*",),
            ),
        },

        # ── Observations: what the robot sees ───────────────────────────
        observations={
            "policy": ObservationGroupCfg(
                observation_terms={
                    "joint_pos": ObservationTermCfg(
                        func=joint_pos_rel,
                        params={"asset_cfg": ROBOT_CFG},
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=joint_vel_rel,
                        params={"asset_cfg": ROBOT_CFG},
                    ),
                    "cube_pos": ObservationTermCfg(
                        func=cube_position,
                    ),
                    "cube_orient": ObservationTermCfg(
                        func=cube_orientation,
                    ),
                    "cube_lin_vel": ObservationTermCfg(
                        func=cube_linear_velocity,
                    ),
                    "cube_ang_vel": ObservationTermCfg(
                        func=cube_angular_velocity,
                    ),
                    "last_action": ObservationTermCfg(
                        func=last_action,
                    ),
                },
            ),
        },

        # ── Rewards: the scoring system ─────────────────────────────────
        rewards={
            "yaw_rate": RewardTermCfg(
                func=yaw_rate_reward,
                weight=1.25,
                params={"object_name": "cube"},
            ),
            "fallen_penalty": RewardTermCfg(
                func=object_fallen,
                weight=-10.0,
                params={"object_name": "cube", "minimum_height": 0.04},
            ),
            "torque_penalty": RewardTermCfg(
                func=joint_torque_penalty,
                weight=-0.1,
                params={"asset_cfg": ROBOT_CFG},
            ),
        },

        # ── Terminations: when does the episode end ─────────────────────
        terminations={
            "time_out": TerminationTermCfg(
                func=envs_mdp.time_out,
                time_out=True,
            ),
            "cube_dropped": TerminationTermCfg(
                func=cube_dropped,
                params={"object_name": "cube", "minimum_height": 0.02},
            ),
            "nan": TerminationTermCfg(
                func=envs_mdp.nan_detection,
            ),
        },

        # ── Events: what happens on reset ───────────────────────────────
        events={
            "reset_base": EventTermCfg(
                func=envs_mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "reset_robot_joints": EventTermCfg(
                func=envs_mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (0.0, 0.0),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                },
            ),
            "reset_cube_pose": EventTermCfg(
                func=envs_mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("cube"),
                    "pose_range": {
                        "x": (-0.006, 0.006),
                        "y": (-0.006, 0.006),
                        "z": (-0.005, 0.005),
                        "yaw": (-3.14, 3.14),
                    },
                    "velocity_range": {},
                },
            ),
        },

        # ── Empty for now (can add later) ───────────────────────────────
        commands={},
        curriculum={},
        metrics={},
        recorders={},
    )