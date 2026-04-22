"""
Global settings for pseudo-dVRK (pDVRK) project.

Centralizes common configuration used across multiple scripts.
Script-specific settings should remain in their respective files.
"""

from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
URDFS_ROOT = PROJECT_ROOT / "urdfs"

# Robot URDF directories
DEFAULT_PHANTOM_ROOT = URDFS_ROOT / "phantom_touch"
DEFAULT_PSM_ROOT = URDFS_ROOT / "psm"
DEFAULT_ECM_ROOT = URDFS_ROOT / "ecm"
DEFAULT_MTM_ROOT = URDFS_ROOT / "mtm"

# ============================================================================
# Haptic Device Configuration
# ============================================================================

# Device names
LEFT_DEVICE_NAME = "Left Device"
RIGHT_DEVICE_NAME = "Right Device"

# Joint2 calibration coefficients
# Device-specific values - update after running joint2_calibration
# Formula: joints[2] = joints[2] - COEFF * joints[1]
LEFT_J2_COEFF = 0.993545966786979
RIGHT_J2_COEFF = 0.995028312512455

# ============================================================================
# Teleoperation Control Parameters
# ============================================================================

# Teleoperation scaling factor for position/pose control
TELEOPERATION_GAIN = 0.3

# PSM jaw/gripper control
JAW_IDX = 7  # Joint index for jaw/gripper
JAW_CLOSE_SPEED = 0.1
JAW_OPEN_SPEED = 0.1
JAW_MIN = 0.0
JAW_MAX = 1.0

# Phantom/PSM base placement distances used by teleoperation examples
PHANTOM_DUAL_Y_DISTANCE = 1.0
PHANTOM_BOTH_Y_DISTANCE = 0.5

PSM_BASE_X = -0.3
PSM_BASE_Y_DISTANCE = 0.4
PSM_BASE_Z = 0.1
PSM_BASE_X_ROTATION_SPLIT_DEG = 90.0

# ============================================================================
# ECM Camera Simulation Parameters
# ============================================================================

# Monocular and stereo camera model
ECM_CAMERA_HFOV_DEG = 70.0
ECM_CAMERA_NEAR = 0.01
ECM_CAMERA_FAR = 3.0
ECM_STEREO_BASELINE_M = 0.008
ECM_CAMERA_FOCUS_DISTANCE_M = 0.3

# ECM camera simulation defaults
ECM_CAMERA_DEFAULT_COUNT = 2  # allowed values: 1 (mono), 2 (stereo)
ECM_CAMERA_DEFAULT_SCOPE_DEG = 0.0  # allowed values: 0.0, 30.0
ECM_CAMERA_YAW_STEP_DEG = 1.0
ECM_CAMERA_PITCH_STEP_DEG = 1.0
ECM_CAMERA_INSERTION_STEP_M = 0.002

CAMERA_ROLL_DEG = 90.0  # Compensates for the additional world-Z.
CAMERA_Z_M = 0.3  # Initial camera Z in world frame for teleoperation examples

# ============================================================================
# World / Collision Parameters
# ============================================================================

WORLD_FLOOR_SIZE_X_M = 0.6
WORLD_FLOOR_SIZE_Y_M = 0.6
WORLD_FLOOR_Z_M = 0.0
WORLD_FLOOR_THICKNESS_M = 0.01
WORLD_FLOOR_FRICTION = 0.0
WORLD_FLOOR_DENSITY_KG_M3 = 1.0
WORLD_FLOOR_RESTITUTION = 0.05


WORLD_WALL_THICKNESS_M = 0.01
WORLD_WALL_HEIGHT_M = 0.08
WORLD_WALL_FRICTION = 0.6
WORLD_WALL_DENSITY_KG_M3 = 2400.0
WORLD_WALL_RESTITUTION = 0.1


# Target A (assigned to left PSM)
TELEOP_TARGET_A_NAME = "target_a"
TELEOP_TARGET_A_SHAPE = "sphere"  # "sphere" or "cube"
TELEOP_TARGET_A_POSITION_M = (0.2, 0.15, 0.03)
TELEOP_TARGET_A_RADIUS_M = 0.018  # used when SHAPE == "sphere"
TELEOP_TARGET_A_SIZE_M = (0.03, 0.03, 0.03)  # used when SHAPE == "cube"
TELEOP_TARGET_A_FRICTION = 0.1
TELEOP_TARGET_A_DENSITY_KG_M3 = 1.0
TELEOP_TARGET_A_RESTITUTION = 0.4
TELEOP_TARGET_A_COLOR = "deepskyblue"

# Target B (assigned to right PSM)
TELEOP_TARGET_B_NAME = "target_b"
TELEOP_TARGET_B_SHAPE = "cube"  # "sphere" or "cube"
TELEOP_TARGET_B_POSITION_M = (0.2, -0.15, 0.03)
TELEOP_TARGET_B_RADIUS_M = 0.018  # used when SHAPE == "sphere"
TELEOP_TARGET_B_SIZE_M = (0.03, 0.03, 0.03)  # used when SHAPE == "cube"
TELEOP_TARGET_B_FRICTION = 0.1
TELEOP_TARGET_B_DENSITY_KG_M3 = 1.0
TELEOP_TARGET_B_RESTITUTION = 0.18
TELEOP_TARGET_B_COLOR = "firebrick"

# ============================================================================
# Haptic Collision Feedback Parameters
# ============================================================================

HAPTIC_MAX_FORCE_N = 3.0
HAPTIC_STIFFNESS_N_PER_M = 180.0
HAPTIC_DAMPING_N_S_PER_M = 12.0
HAPTIC_CONTACT_DISTANCE_M = 0.12
HAPTIC_MESH_CONTACT_MODE = 1
HAPTIC_MESH_CONTACT_TOLERANCE_M = 0.001
