"""
LEAP Hand (Left) configuration for mjlab.

Adapted from Msornerrrr/in-hand-rotation-mjlab's leap_right_constants.py
and leap_left_constants.py. Uses all the original SysID-calibrated values
but targets the public mjlab API (IdealPdActuatorCfg instead of
DelayedActuatorCfg, no update_assets).

Goals:
- Spawn LEAP left hand stably
- Use mjlab IdealPd actuators with current-mapped torque limits
- Per-joint SysID scaling for stiffness, damping, effort, armature, friction
- Detailed collision bitmasks (no intra-finger collision, cross-finger enabled)
"""

from __future__ import annotations
from pathlib import Path

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_ROBOT_DIR = Path(__file__).parent / "leap_hand"
LEAP_LEFT_XML = _ROBOT_DIR / "xmls" / "left_hand.xml"


# --------------------------------------------------------------------------- #
# Actuator SysID constants (Dynamixel XC330-M288)
# --------------------------------------------------------------------------- #
# PD control in current domain:
#   i_des = Kp_i * (q_des - q) - Kd_i * qdot
#   tau   = K_t  * i_des
#
# Datasheet: https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/

TORQUE_PER_AMP = 0.53                 # K_t [Nm/A]
CURRENT_LIMIT_A = 0.55                # 550 mA
EFFORT_LIMIT = TORQUE_PER_AMP * CURRENT_LIMIT_A  # ~0.2915 Nm

# Current-domain PD gains: 800 mA/rad position, 200 mA/(rad/s) velocity
KP_CURRENT = 0.8    # A/rad (800 mA / 1000)
KD_CURRENT = 0.2    # A/(rad/s) (200 mA / 1000)

# Convert to torque domain for MuJoCo
STIFFNESS = TORQUE_PER_AMP * KP_CURRENT   # ~0.424 Nm/rad
DAMPING = TORQUE_PER_AMP * KD_CURRENT     # ~0.106 Nm/(rad/s)

# Reduced gains for rotational MCP joints (if_rot, mf_rot, rf_rot)
MCP_SIDE_SCALE = 0.75
MCP_SIDE_STIFFNESS = STIFFNESS * MCP_SIDE_SCALE
MCP_SIDE_DAMPING = DAMPING * MCP_SIDE_SCALE

# Reflected armature: Ia = gear_ratio^2 * rotor_inertia
GEAR_RATIO = 288.35
ROTOR_INERTIA = 1.7e-8
REFLECTED_ARMATURE = (GEAR_RATIO ** 2) * ROTOR_INERTIA

# Dry friction: ~10% of max torque
FRICTIONLOSS = 0.1 * EFFORT_LIMIT


# --------------------------------------------------------------------------- #
# Joint names (16 DOF)
# --------------------------------------------------------------------------- #
JOINT_ORDER = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)

MCP_SIDE_ROT_JOINTS = ("if_rot", "mf_rot", "rf_rot")


# --------------------------------------------------------------------------- #
# Per-joint SysID scaling factors
# --------------------------------------------------------------------------- #
# These come from real-robot system identification (fitting sim to hardware).
# Each joint's base value gets multiplied by its scale factor.

STIFFNESS_SCALE = {
    "if_mcp": 1.3821, "if_rot": 1.3903, "if_pip": 1.3601, "if_dip": 1.4000,
    "mf_mcp": 1.3860, "mf_rot": 1.4000, "mf_pip": 1.3462, "mf_dip": 1.4000,
    "rf_mcp": 1.3983, "rf_rot": 1.4000, "rf_pip": 1.4000, "rf_dip": 1.3490,
    "th_cmc": 1.3544, "th_axl": 1.2862, "th_mcp": 1.3870, "th_ipl": 1.3084,
}
DAMPING_SCALE = {
    "if_mcp": 0.7264, "if_rot": 0.7000, "if_pip": 0.7000, "if_dip": 0.7699,
    "mf_mcp": 0.8130, "mf_rot": 0.7281, "mf_pip": 0.7918, "mf_dip": 0.7000,
    "rf_mcp": 0.7887, "rf_rot": 0.7251, "rf_pip": 0.7000, "rf_dip": 0.8001,
    "th_cmc": 0.7000, "th_axl": 0.7064, "th_mcp": 0.9578, "th_ipl": 0.7178,
}
EFFORT_SCALE = {
    "if_mcp": 0.8500, "if_rot": 1.0684, "if_pip": 0.8500, "if_dip": 0.8500,
    "mf_mcp": 0.8621, "mf_rot": 1.0606, "mf_pip": 0.8500, "mf_dip": 0.8500,
    "rf_mcp": 0.8892, "rf_rot": 0.8500, "rf_pip": 0.8500, "rf_dip": 1.1062,
    "th_cmc": 0.8500, "th_axl": 0.8633, "th_mcp": 0.8612, "th_ipl": 1.0408,
}
ARMATURE_SCALE = {
    "if_mcp": 1.2629, "if_rot": 1.4800, "if_pip": 1.1925, "if_dip": 0.6543,
    "mf_mcp": 0.8186, "mf_rot": 1.0668, "mf_pip": 1.3619, "mf_dip": 1.1253,
    "rf_mcp": 0.6000, "rf_rot": 1.2074, "rf_pip": 1.1534, "rf_dip": 1.3067,
    "th_cmc": 0.9726, "th_axl": 0.8835, "th_mcp": 1.0873, "th_ipl": 1.1179,
}
FRICTION_SCALE = {
    "if_mcp": 1.3455, "if_rot": 0.7058, "if_pip": 1.1626, "if_dip": 0.3881,
    "mf_mcp": 0.9242, "mf_rot": 0.3264, "mf_pip": 0.9817, "mf_dip": 1.5468,
    "rf_mcp": 1.8238, "rf_rot": 1.0831, "rf_pip": 0.9794, "rf_dip": 0.8705,
    "th_cmc": 0.3610, "th_axl": 2.2000, "th_mcp": 0.5970, "th_ipl": 1.8449,
}


# --------------------------------------------------------------------------- #
# Actuator builder
# --------------------------------------------------------------------------- #
def _base_stiffness(joint: str) -> float:
    return MCP_SIDE_STIFFNESS if joint in MCP_SIDE_ROT_JOINTS else STIFFNESS

def _base_damping(joint: str) -> float:
    return MCP_SIDE_DAMPING if joint in MCP_SIDE_ROT_JOINTS else DAMPING

def _scaled(value: float, scale_map: dict[str, float], joint: str) -> float:
    return value * scale_map.get(joint, 1.0)

def _make_actuator(joint: str) -> IdealPdActuatorCfg:
    """Create a SysID-calibrated PD actuator for one joint."""
    return IdealPdActuatorCfg(
        target_names_expr=(joint,),
        stiffness=_scaled(_base_stiffness(joint), STIFFNESS_SCALE, joint),
        damping=_scaled(_base_damping(joint), DAMPING_SCALE, joint),
        effort_limit=_scaled(EFFORT_LIMIT, EFFORT_SCALE, joint),
        armature=_scaled(REFLECTED_ARMATURE, ARMATURE_SCALE, joint),
        frictionloss=_scaled(FRICTIONLOSS, FRICTION_SCALE, joint),
    )

LEAP_ACTUATORS = tuple(_make_actuator(j) for j in JOINT_ORDER)


# --------------------------------------------------------------------------- #
# Articulation
# --------------------------------------------------------------------------- #
ARTICULATION = EntityArticulationInfoCfg(
    actuators=LEAP_ACTUATORS,
    soft_joint_pos_limit_factor=0.95,
)


# --------------------------------------------------------------------------- #
# Spec helpers (remove XML-defined actuators, embed mesh assets)
# --------------------------------------------------------------------------- #
def _configure_spec(spec: mujoco.MjSpec) -> None:
    """Remove XML actuators so mjlab IdealPd actuators take over."""
    for actuator in tuple(spec.actuators):
        spec.delete(actuator)

def _get_assets(meshdir: str) -> dict[str, bytes]:
    """Read mesh files into memory (needed for mujoco-warp / viser)."""
    assets: dict[str, bytes] = {}
    asset_dir = LEAP_LEFT_XML.parent / "assets"
    for p in asset_dir.rglob("*"):
        if p.is_file():
            assets[p.name] = p.read_bytes()
    return assets

def get_spec() -> mujoco.MjSpec:
    """Load MJCF, configure actuators, embed assets."""
    spec = mujoco.MjSpec.from_file(str(LEAP_LEFT_XML))
    _configure_spec(spec)
    spec.assets = _get_assets(spec.meshdir)
    return spec


# --------------------------------------------------------------------------- #
# Default pose (left hand, fingers curled to hold cube)
# --------------------------------------------------------------------------- #
HOME_POSE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.1),
    joint_pos={
        # Index
        "if_mcp": 0.131, "if_rot": 0.0, "if_pip": 0.65, "if_dip": 1.0,
        # Middle
        "mf_mcp": 0.131, "mf_rot": 0.0, "mf_pip": 0.65, "mf_dip": 1.0,
        # Ring
        "rf_mcp": 0.131, "rf_rot": 0.0, "rf_pip": 0.65, "rf_dip": 1.0,
        # Thumb
        "th_cmc": 0.8, "th_axl": -0.78, "th_mcp": 0.5, "th_ipl": 0.367,
    },
    joint_vel={".*": 0.0},
)


# --------------------------------------------------------------------------- #
# Collision configuration
# --------------------------------------------------------------------------- #
# Bitmask channels:
#   bit-0 (1)  = external objects (cube, table)
#   bit-1 (2)  = index finger
#   bit-2 (4)  = middle finger
#   bit-3 (8)  = ring finger
#   bit-4 (16) = thumb
#
# Same-finger pairs share a contype bit absent from their own conaffinity,
# so no intra-finger collision. Cross-finger pairs collide normally.

COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision.*", ".*_tip"),
    contype={
        "palm_collision.*": 1,
        "if_.*": 2,
        "mf_.*": 4,
        "rf_.*": 8,
        "th_.*": 16,
    },
    conaffinity={
        "palm_collision.*": 1,
        "if_.*": 29,    # 1 | 4 | 8 | 16  (ext + mid + ring + thumb)
        "mf_.*": 27,    # 1 | 2 | 8 | 16  (ext + idx + ring + thumb)
        "rf_.*": 23,    # 1 | 2 | 4 | 16  (ext + idx + mid + thumb)
        "th_.*": 15,    # 1 | 2 | 4 | 8   (ext + idx + mid + ring)
    },
    condim={
        ".*_tip": 6,
        ".*": 3,
    },
    friction={
        ".*_tip": (0.8, 5e-3, 1e-4),
        ".*": (0.2,),
    },
    solref={
        ".*_tip": (0.01, 1),
        ".*": (0.05, 1),
    },
    priority={
        ".*_tip": 2,
        ".*": 0,
    },
    disable_other_geoms=False,
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_leap_left_hand_cfg() -> EntityCfg:
    """Return a complete LEAP left hand entity config for mjlab."""
    return EntityCfg(
        init_state=HOME_POSE,
        collisions=(COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


# --------------------------------------------------------------------------- #
# Debug: run this file directly to view the hand in MuJoCo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    leap = Entity(get_leap_left_hand_cfg())
    mujoco.viewer.launch(leap.spec.compile())
