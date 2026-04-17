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
ECM_CAMERA_DEFAULT_COUNT = 1  # allowed values: 1 (mono), 2 (stereo)
ECM_CAMERA_DEFAULT_SCOPE_DEG = 0.0  # allowed values: 0.0, 30.0
ECM_CAMERA_YAW_STEP_DEG = 1.0
ECM_CAMERA_PITCH_STEP_DEG = 1.0
ECM_CAMERA_INSERTION_STEP_M = 0.002

CAMERA_ROLL_DEG = 90.0  # Compensates for the additional world-Z.
CAMERA_Z_M = 0.3  # Initial camera Z in world frame for teleoperation examples
