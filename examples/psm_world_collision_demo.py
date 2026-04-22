import argparse

import numpy as np
import pyvista as pv

from src.robots.psm import PSM
from src.utils.visualization.viewers import add_robot_meshes_to_plotter
from src.utils.world_engine import check_psm_world_collisions, create_demo_world


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a simple world and report approximate PSM/world collisions."
    )
    parser.add_argument(
        "--robot-root",
        default="urdfs/psm",
        help="Directory containing psm.urdf and meshes/.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Robot mesh opacity in [0, 1].",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    psm = PSM(robot_root=args.robot_root)
    theta = np.zeros(psm.dof, dtype=float)
    world = create_demo_world()

    results = check_psm_world_collisions(psm, world, theta=theta)
    print(f"Registered collision geometries: {len(psm.get_link_collisions())}")
    print(f"World bodies: {len(world.bodies)}")
    for body in world.bodies:
        print(
            f"  body={body.name} density={body.density or 0.0:.1f}kg/m^3 "
            f"mass={body.mass_kg:.3f}kg weight={body.weight_n:.3f}N "
            f"restitution={body.restitution:.2f} friction={body.friction:.2f}"
        )
    print(f"Detected contacts: {len(results)}")
    for item in results:
        print(
            f"  {item.body_name:>6s} <-> {item.link_name:<20s} "
            f"depth={item.penetration_depth:.4f} friction={item.friction:.2f} "
            f"restitution={item.restitution:.2f}"
        )

    plotter = pv.Plotter(title="PSM World Collision Demo", window_size=[1200, 900])
    world.add_to_plotter(plotter, opacity=0.85)
    add_robot_meshes_to_plotter(
        plotter,
        name="psm",
        robot=psm,
        theta=theta,
        alpha=float(args.alpha),
    )
    plotter.add_text(
        f"Contacts: {len(results)}",
        position="upper_left",
        font_size=12,
    )
    plotter.add_axes(line_width=2)  # type: ignore[misc]
    plotter.show()


if __name__ == "__main__":
    main()
