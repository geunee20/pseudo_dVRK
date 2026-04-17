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
