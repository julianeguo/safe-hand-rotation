from mjlab.tasks.registry import register_mjlab_task

from safe_hand_rotation.tasks.cube_rotate.env_cfg import cube_rotate_env_cfg
from safe_hand_rotation.tasks.cube_rotate.rl_cfg import ppo_cfg

register_mjlab_task(
    task_id="SafeHandRotation-LeapLeft-v0",
    env_cfg=cube_rotate_env_cfg(),
    rl_cfg=ppo_cfg(),
)