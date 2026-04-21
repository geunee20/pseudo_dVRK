Example Scripts
===============

This section summarizes runnable demos under ``examples/``.

Run scripts from repository root using ``python -m ...``.

Realtime visualization:

- ``examples.real_time_viz``: Realtime robot visualization.
- ``examples.real_time_viz_phantom``: Realtime visualization with Phantom input.
- ``examples.visualize``: Basic visualization entrypoint/helper demo.

Single-arm teleoperation:

- ``examples.teleoperation_position``: Position-only teleoperation.
- ``examples.teleoperation_pose``: Full-pose teleoperation.

Dual-arm teleoperation:

- ``examples.teleoperation_position_dual``: Dual position teleoperation.
- ``examples.teleoperation_pose_dual``: Dual pose teleoperation.
- ``examples.teleoperation_position_dual_ecm``: Dual position teleoperation with ECM context.
- ``examples.teleoperation_pose_dual_ecm``: Dual pose teleoperation with ECM context.

Phantom side setup demos:

- ``examples.phantom_left``: Left Phantom-only setup.
- ``examples.phantom_right``: Right Phantom-only setup.
- ``examples.phantom_both``: Dual Phantom setup.

Camera simulation:

- ``examples.ecm_camera_sim``: ECM camera simulation and camera control.

Quick commands:

.. code-block:: bash

   python -m examples.real_time_viz
   python -m examples.real_time_viz_phantom
   python -m examples.teleoperation_position
   python -m examples.teleoperation_pose
   python -m examples.teleoperation_position_dual
   python -m examples.teleoperation_pose_dual
   python -m examples.teleoperation_position_dual_ecm
   python -m examples.teleoperation_pose_dual_ecm
   python -m examples.ecm_camera_sim
