"""
Wrapper that computes cost signals and adds them to environment extras.

PPO ignores extras["costs"]. CPO reads them for constraint enforcement.
This means the same wrapper can be used for both algorithms safely.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from safe_hand_rotation.mdp.costs import torque_limit_cost, cube_drop_proximity_cost


class CostEnvWrapper:
    """Wraps an RslRlVecEnvWrapper to add cost signals."""

    def __init__(
        self,
        env,
        max_torque: float = 0.25,
        safe_height: float = 0.05,
    ):
        self.env = env
        self.max_torque = max_torque
        self.safe_height = safe_height

    def step(self, actions):
        obs, rew, dones, extras = self.env.step(actions)

        # Access the underlying mjlab environment
        mjlab_env = self.env.env

        # Compute costs from our cost functions
        torque_cost = torque_limit_cost(
            mjlab_env,
            max_torque=self.max_torque,
        )
        drop_cost = cube_drop_proximity_cost(
            mjlab_env,
            object_name="cube",
            safe_height=self.safe_height,
        )

        # Total cost per environment
        extras["costs"] = torque_cost + drop_cost

        return obs, rew, dones, extras

    # Delegate everything else to the wrapped environment
    def __getattr__(self, name):
        return getattr(self.env, name)