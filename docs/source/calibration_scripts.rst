Calibration Scripts
===================

This section summarizes calibration/diagnostic scripts under ``calibrations/``.

Run scripts from repository root using ``python -m ...``.

- ``calibrations.device_info``:
  Device connectivity check for Phantom/OpenHaptics runtime and basic state readout.
- ``calibrations.joint2_calibration``:
  Estimates the joint-2 coupling coefficient used to compensate wrist coupling.
- ``calibrations.phantom_workspace_profiler``:
  Profiles reachable workspace bounds during live device motion.

Quick commands:

.. code-block:: bash

   python -m calibrations.device_info
   python -m calibrations.joint2_calibration
   python -m calibrations.phantom_workspace_profiler
