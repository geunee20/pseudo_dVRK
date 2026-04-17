## pseudo_dVRK

Also referred to as pseudo-dVRK (pDVRK).

This project is a Windows-based pseudo-dVRK (pDVRK) simulation platform inspired by the da Vinci Research Kit (dVRK), designed to operate without ROS.

The system uses a 3D Touch (Phantom) haptic device as an input interface to control a Patient Side Manipulator (PSM) in real time.

### Motivation
The official dVRK stack is tightly coupled with ROS and primarily supports Linux environments. This project aims to provide a lightweight, ROS-free alternative that runs natively on Windows.

### Features
- Real-time control of PSM using a 3D Touch (Phantom) device
- ROS-independent architecture
- Python-based simulation and visualization pipeline
- URDF-based robot modeling

### Goal
To build a fully functional, bimanual pseudo-dVRK (pDVRK) simulation platform on Windows with haptic feedback and intuitive teleoperation.

## Environment Setup (Windows)

### 1. Create Python environment

```bash
conda create -n pDVRK python=3.11 -y
conda activate pDVRK
```

If you prefer venv:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. OpenHaptics SDK prerequisite

To run scripts that use the Phantom/3D Touch device, OpenHaptics SDK runtime and drivers must be installed on Windows.

- Install OpenHaptics SDK and device driver first
- Confirm the device is detected before running scripts
- `device_info.py` is a quick connectivity check

## Run

Run from repository root.

### Device check

```bash
python -m scripts.device_info
```

### Real-time visualization examples

```bash
python -m scripts.example_real_time_viz
python -m scripts.example_real_time_viz_phantom
```

### Teleoperation examples

```bash
python -m scripts.example_teleoperation_position
python -m scripts.example_teleoperation_pose
```

### Calibration scripts

#### Joint2 Calibration (Automatic)
Calibrates the joint2 (wrist 2) coupling coefficient caused by gravity and joint coupling.

**Usage:**
```bash
python -m scripts.calibrations.joint2_calibration
```

**Procedure:**
1. Script starts, vacuum is enabled automatically
2. Position phantom arm in straight (horizontal) position
3. Press the clutch button to start
4. 3-second countdown displayed
5. Move upper arm (joint1) slowly up and down for 10 seconds
6. Script automatically collects data and computes calibration coefficient
7. Results printed to console

**Output:**
- Calibration coefficient (joint2_coeff) with R² goodness-of-fit
- Example output:
  ```
  Slope (joint2_coeff):  0.993545966786979
  Intercept:             0.005271849362127
  R-squared:             0.987234
  ```

**Note:** Each Phantom device requires its own calibration. Update `LEFT_J2_COEFF` or `RIGHT_J2_COEFF` in `src/utils/script_common.py` with the computed value.


#### Phantom Workspace Profiler
Maps the workspace limits during device operation.

```bash
python -m scripts.calibrations.phantom_workspace_profiler
```

## Configuration

Global settings are centralized in `settings.py` at the project root. Common configuration parameters include:

- **Robot paths**: `DEFAULT_PHANTOM_ROOT`, `DEFAULT_PSM_ROOT`, `DEFAULT_ECM_ROOT`, `DEFAULT_MTM_ROOT`
- **Device configuration**: `LEFT_DEVICE_NAME`, `RIGHT_DEVICE_NAME`
- **Calibration coefficients**: `LEFT_J2_COEFF`, `RIGHT_J2_COEFF` (update after running joint2_calibration)
- **Teleoperation parameters**: `TELEOPERATION_GAIN`, `JAW_*` constants

Script-specific settings (window size, colors, camera positions, etc.) remain in their respective files.

**Usage in scripts:**
```python
from settings import DEFAULT_PHANTOM_ROOT, LEFT_J2_COEFF, TELEOPERATION_GAIN
```

## Notes

- If `pyvista` import fails, ensure your active environment is the one where `requirements.txt` was installed.
- `pinocchio` package is required for `PinocchioIK`-based scripts.
- This repository currently uses `scripts/example_*.py` entrypoints.

## Acknowledgements / Resources

This project reuses and adapts URDF and external libraries from the following repositories:

### dVRK URDF
- Source: https://github.com/WPI-AIM/dvrk_env  
- Usage: Only URDF and related assets are extracted and used

**Citation:**
Gondokaryono RA, Agrawal A, Munawar A, Nycz CJ, Fischer GS,  
*An Approach to Modeling Closed-Loop Kinematic Chain Mechanisms, applied to Simulations of the da Vinci Surgical System*,  
Special Issue on Platforms for Robotics Research - Acta Polytechnica Hungarica,  
Vol 16, No 8, pp 29–48, Nov 2019

---

### Phantom (3D Touch) URDF
- Source: https://github.com/eaa3/phantom_touch_ros2  
- Usage: URDF and mesh files adapted for this project

---

### pyOpenHaptics
- Source: https://github.com/mikelitu/pyOpenHaptics/tree/main  
- Usage: Modified version used for haptic device interface (device initialization, state callback, and Windows compatibility adjustments)

## Disclaimer
This project is not affiliated with Intuitive Surgical or the official dVRK project.