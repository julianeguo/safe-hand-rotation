"""
Termination functions for the cube rotation task.

Each function returns a boolean tensor of shape [num_envs]:
True = episode ends for that environment, False = keep going.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def cube_dropped(
    env: ManagerBasedRlEnv,
    object_name: str,
    minimum_height: float,
) -> torch.Tensor:
    cube = env.scene[object_name]
    return (cube.data.root_link_pos_w[:, 2] < minimum_height).float()