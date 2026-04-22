from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pyvista as pv

from src.kinematics.fk import CollisionPose, collision_transforms
from src.robots.protocols import CollisionRobotLike
from src.utils.transforms import make_transform
from settings import (
    HAPTIC_MESH_CONTACT_MODE,
    HAPTIC_MESH_CONTACT_TOLERANCE_M,
    WORLD_FLOOR_DENSITY_KG_M3,
    WORLD_FLOOR_FRICTION,
    WORLD_FLOOR_RESTITUTION,
    WORLD_FLOOR_SIZE_X_M,
    WORLD_FLOOR_SIZE_Y_M,
    WORLD_FLOOR_THICKNESS_M,
    WORLD_FLOOR_Z_M,
    WORLD_WALL_DENSITY_KG_M3,
    WORLD_WALL_FRICTION,
    WORLD_WALL_HEIGHT_M,
    WORLD_WALL_RESTITUTION,
    WORLD_WALL_THICKNESS_M,
)


_MESH_POINTS_CACHE: dict[str, np.ndarray] = {}
_POLYDATA_CACHE: dict[str, pv.PolyData] = {}


@dataclass
class RigidBody:
    """Minimal rigid body description used by the lightweight world engine.

    This is intentionally kinematic only: it stores geometry, pose, and a
    friction coefficient for contact reasoning, but it does not integrate full
    rigid-body dynamics.
    """

    name: str
    shape: str
    pose: np.ndarray
    friction: float = 0.5
    density: float | None = None
    mass: float | None = None
    restitution: float = 0.2
    size: np.ndarray | None = None
    radius: float | None = None
    height: float | None = None
    color: str = "lightgray"
    is_static: bool = True
    linear_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    @property
    def volume_m3(self) -> float:
        if self.shape in {"floor", "cube"} and self.size is not None:
            return float(np.prod(np.asarray(self.size, dtype=float)))
        if self.shape == "sphere" and self.radius is not None:
            radius = float(self.radius)
            return float((4.0 / 3.0) * np.pi * radius**3)
        if (
            self.shape == "cylinder"
            and self.radius is not None
            and self.height is not None
        ):
            return float(np.pi * float(self.radius) ** 2 * float(self.height))
        return 0.0

    @property
    def mass_kg(self) -> float:
        if self.mass is not None:
            return float(self.mass)
        if self.density is not None:
            return float(self.density) * self.volume_m3
        return 0.0

    @property
    def weight_n(self) -> float:
        return 9.81 * self.mass_kg


@dataclass
class CollisionResult:
    """Robot-vs-world contact result using mesh collision narrowphase."""

    body_name: str
    link_name: str
    geometry_type: str
    friction: float
    restitution: float
    body_density_kg_m3: float
    body_mass_kg: float
    penetration_depth: float
    contact_point: np.ndarray
    contact_normal: np.ndarray
    body_bounds_min: np.ndarray
    body_bounds_max: np.ndarray
    collision_bounds_min: np.ndarray
    collision_bounds_max: np.ndarray


@dataclass
class BodyRenderItem:
    poly: pv.PolyData
    local_vertices: np.ndarray


def _copy_polydata(poly: pv.PolyData) -> pv.PolyData:
    return pv.PolyData(poly.copy(deep=True))


def _ensure_polydata(dataset: pv.DataObject) -> pv.PolyData:
    """Normalize any PyVista dataset into a ``PolyData`` surface."""
    if isinstance(dataset, pv.PolyData):
        return _copy_polydata(dataset)
    if isinstance(dataset, pv.DataSet):
        surface = dataset.extract_surface(algorithm="dataset_surface").triangulate()
        if isinstance(surface, pv.PolyData):
            return _copy_polydata(surface)
    return pv.PolyData()


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    hom = np.c_[pts, np.ones(len(pts), dtype=float)]
    return (T @ hom.T).T[:, :3]


def _make_local_box_points(size: np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(size, dtype=float).reshape(3)
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    return signs * half


def _make_local_cylinder_points(radius: float, height: float) -> np.ndarray:
    z = 0.5 * float(height)
    r = float(radius)
    return np.array(
        [
            [-r, -r, -z],
            [-r, -r, z],
            [-r, r, -z],
            [-r, r, z],
            [r, -r, -z],
            [r, -r, z],
            [r, r, -z],
            [r, r, z],
        ],
        dtype=float,
    )


def _mesh_points(mesh_path: str) -> np.ndarray:
    cache_key = str(Path(mesh_path).resolve())
    cached = _MESH_POINTS_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    mesh = pv.read(mesh_path)
    if isinstance(mesh, pv.MultiBlock):
        blocks = []
        for block in mesh:
            if block is None:
                continue
            surf = block.extract_surface(algorithm="dataset_surface").triangulate()
            if isinstance(surf, pv.PolyData) and surf.n_points > 0:
                blocks.append(np.asarray(surf.points, dtype=float))
        if not blocks:
            return np.zeros((0, 3), dtype=float)
        points = np.vstack(blocks)
        _MESH_POINTS_CACHE[cache_key] = points.copy()
        return points

    surf = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    if not isinstance(surf, pv.PolyData) or surf.n_points == 0:
        return np.zeros((0, 3), dtype=float)
    points = np.asarray(surf.points, dtype=float)
    _MESH_POINTS_CACHE[cache_key] = points.copy()
    return points


def _mesh_polydata(mesh_path: str) -> pv.PolyData:
    cache_key = str(Path(mesh_path).resolve())
    cached = _POLYDATA_CACHE.get(cache_key)
    if cached is not None:
        return _copy_polydata(cached)

    mesh = pv.read(mesh_path)
    if isinstance(mesh, pv.MultiBlock):
        merged: pv.PolyData | None = None
        for block in mesh:
            if block is None:
                continue
            surf = block.extract_surface(algorithm="dataset_surface").triangulate()
            if not isinstance(surf, pv.PolyData) or surf.n_points == 0:
                continue
            merged = surf if merged is None else merged.merge(surf, merge_points=False)
        if merged is None:
            merged = pv.PolyData()
        _POLYDATA_CACHE[cache_key] = _copy_polydata(merged)
        return _copy_polydata(merged)

    surf = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    if not isinstance(surf, pv.PolyData):
        surf = pv.PolyData()
    _POLYDATA_CACHE[cache_key] = _copy_polydata(surf)
    return _copy_polydata(surf)


def _transform_polydata(poly: pv.PolyData, T: np.ndarray) -> pv.PolyData:
    transformed = _copy_polydata(poly)
    if transformed.n_points == 0:
        return transformed
    transformed.points = _transform_points(np.asarray(transformed.points), T)
    return transformed


def _make_box_polydata(size: np.ndarray) -> pv.PolyData:
    size = np.asarray(size, dtype=float).reshape(3)
    cube = pv.Cube(
        center=(0.0, 0.0, 0.0),
        x_length=float(size[0]),
        y_length=float(size[1]),
        z_length=float(size[2]),
    ).triangulate()
    return _ensure_polydata(cast(pv.DataObject, cube))


def _make_sphere_polydata(radius: float) -> pv.PolyData:
    sphere = pv.Sphere(radius=float(radius), center=(0.0, 0.0, 0.0)).triangulate()
    return _ensure_polydata(cast(pv.DataObject, sphere))


def _make_cylinder_polydata(radius: float, height: float) -> pv.PolyData:
    cylinder = pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=float(radius),
        height=float(height),
    ).triangulate()
    return _ensure_polydata(cast(pv.DataObject, cylinder))


def _collision_local_points(collision: Any) -> np.ndarray:
    if collision.geometry_type == "mesh":
        if collision.mesh_path is None or not Path(collision.mesh_path).exists():
            return np.zeros((0, 3), dtype=float)
        return _mesh_points(collision.mesh_path)
    if collision.geometry_type == "box" and collision.size is not None:
        return _make_local_box_points(collision.size)
    if collision.geometry_type == "sphere" and collision.radius is not None:
        radius = float(collision.radius)
        return _make_local_box_points(np.array([2 * radius, 2 * radius, 2 * radius]))
    if (
        collision.geometry_type == "cylinder"
        and collision.radius is not None
        and collision.length is not None
    ):
        return _make_local_cylinder_points(collision.radius, collision.length)
    return np.zeros((0, 3), dtype=float)


def _body_local_points(body: RigidBody) -> np.ndarray:
    if body.shape in {"floor", "cube", "wall"}:
        if body.size is None:
            return np.zeros((0, 3), dtype=float)
        return _make_local_box_points(body.size)
    if body.shape == "sphere":
        radius = float(body.radius or 0.0)
        return _make_local_box_points(np.array([2 * radius, 2 * radius, 2 * radius]))
    if body.shape == "cylinder":
        return _make_local_cylinder_points(
            float(body.radius or 0.0), float(body.height or 0.0)
        )
    return np.zeros((0, 3), dtype=float)


def _body_local_mesh(body: RigidBody) -> pv.PolyData:
    cache_key = (
        f"body:{body.shape}:{tuple(np.asarray(body.size).tolist()) if body.size is not None else None}:"
        f"{body.radius}:{body.height}"
    )
    cached = _POLYDATA_CACHE.get(cache_key)
    if cached is not None:
        return _copy_polydata(cached)

    if body.shape in {"floor", "cube", "wall"} and body.size is not None:
        poly = _make_box_polydata(body.size)
    elif body.shape == "sphere" and body.radius is not None:
        poly = _make_sphere_polydata(body.radius)
    elif (
        body.shape == "cylinder" and body.radius is not None and body.height is not None
    ):
        poly = _make_cylinder_polydata(body.radius, body.height)
    else:
        poly = pv.PolyData()

    _POLYDATA_CACHE[cache_key] = _copy_polydata(poly)
    return _copy_polydata(poly)


def _collision_local_mesh(collision: Any) -> pv.PolyData:
    if collision.geometry_type == "mesh":
        if collision.mesh_path is None or not Path(collision.mesh_path).exists():
            return pv.PolyData()
        return _mesh_polydata(collision.mesh_path)
    if collision.geometry_type == "box" and collision.size is not None:
        return _make_box_polydata(collision.size)
    if collision.geometry_type == "sphere" and collision.radius is not None:
        return _make_sphere_polydata(collision.radius)
    if (
        collision.geometry_type == "cylinder"
        and collision.radius is not None
        and collision.length is not None
    ):
        return _make_cylinder_polydata(collision.radius, collision.length)
    return pv.PolyData()


def _bounds_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        zeros = np.zeros(3, dtype=float)
        return zeros, zeros
    return pts.min(axis=0), pts.max(axis=0)


def _overlap_depth(
    min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray
) -> float:
    overlap = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
    if np.any(overlap <= 0.0):
        return 0.0
    return float(overlap.min())


def _contact_normal_and_point(
    body: RigidBody,
    body_min: np.ndarray,
    body_max: np.ndarray,
    collision_min: np.ndarray,
    collision_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    overlap_min = np.maximum(body_min, collision_min)
    overlap_max = np.minimum(body_max, collision_max)
    overlap = overlap_max - overlap_min
    if np.any(overlap <= 0.0):
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float), 0.0

    axis = int(np.argmin(overlap))
    body_center = 0.5 * (body_min + body_max)
    collision_center = 0.5 * (collision_min + collision_max)

    normal = np.zeros(3, dtype=float)
    sign = 1.0 if collision_center[axis] >= body_center[axis] else -1.0
    if body.shape == "floor" and axis == 2:
        sign = 1.0 if collision_center[2] >= body_center[2] else -1.0
    normal[axis] = sign

    contact_point = 0.5 * (overlap_min + overlap_max)
    depth = float(overlap[axis])
    return normal, contact_point, depth


def _world_to_body_local(body: RigidBody, point_world: np.ndarray) -> np.ndarray:
    T_inv = np.linalg.inv(body.pose)
    point_h = np.r_[np.asarray(point_world, dtype=float).reshape(3), 1.0]
    return (T_inv @ point_h)[:3]


def _body_contact_normal(
    body: RigidBody, contact_point_world: np.ndarray
) -> np.ndarray:
    p_local = _world_to_body_local(body, contact_point_world)

    if body.shape == "sphere" and body.radius is not None:
        n_local = p_local.copy()
        norm = float(np.linalg.norm(n_local))
        if norm > 1e-12:
            n_local /= norm
        else:
            n_local = np.array([0.0, 0.0, 1.0], dtype=float)
    elif body.shape in {"floor", "cube", "wall"} and body.size is not None:
        half = 0.5 * np.asarray(body.size, dtype=float).reshape(3)
        margins = half - np.abs(p_local)
        axis = int(np.argmin(margins))
        n_local = np.zeros(3, dtype=float)
        n_local[axis] = 1.0 if p_local[axis] >= 0.0 else -1.0
    elif (
        body.shape == "cylinder" and body.radius is not None and body.height is not None
    ):
        radial = np.array([p_local[0], p_local[1], 0.0], dtype=float)
        radial_norm = float(np.linalg.norm(radial))
        side_margin = float(body.radius) - radial_norm
        cap_margin = 0.5 * float(body.height) - abs(float(p_local[2]))
        if cap_margin < side_margin:
            n_local = np.array(
                [0.0, 0.0, 1.0 if p_local[2] >= 0.0 else -1.0], dtype=float
            )
        elif radial_norm > 1e-12:
            n_local = radial / radial_norm
        else:
            n_local = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        n_local = np.array([0.0, 0.0, 1.0], dtype=float)

    n_world = body.pose[:3, :3] @ n_local
    norm = float(np.linalg.norm(n_world))
    if norm > 1e-12:
        n_world /= norm
    return n_world


def _primitive_penetration_depth(body: RigidBody, points_world: np.ndarray) -> float:
    if len(points_world) == 0:
        return 0.0
    local = np.array([_world_to_body_local(body, p) for p in points_world], dtype=float)

    if body.shape == "sphere" and body.radius is not None:
        radial = np.linalg.norm(local, axis=1)
        return float(np.max(np.maximum(float(body.radius) - radial, 0.0)))

    if body.shape in {"floor", "cube", "wall"} and body.size is not None:
        half = 0.5 * np.asarray(body.size, dtype=float).reshape(3)
        margins = half[None, :] - np.abs(local)
        inside = np.all(margins >= 0.0, axis=1)
        if not np.any(inside):
            return 0.0
        return float(np.max(np.min(margins[inside], axis=1)))

    if body.shape == "cylinder" and body.radius is not None and body.height is not None:
        radial = np.linalg.norm(local[:, :2], axis=1)
        radial_margin = float(body.radius) - radial
        axial_margin = 0.5 * float(body.height) - np.abs(local[:, 2])
        inside = (radial_margin >= 0.0) & (axial_margin >= 0.0)
        if not np.any(inside):
            return 0.0
        return float(np.max(np.minimum(radial_margin[inside], axial_margin[inside])))

    return 0.0


def _mesh_contact(
    body: RigidBody,
    body_mesh_world: pv.PolyData,
    collision_mesh_world: pv.PolyData,
    collision_points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    if body_mesh_world.n_points == 0 or collision_mesh_world.n_points == 0:
        return None

    try:
        contacts, n_contacts = body_mesh_world.collision(
            collision_mesh_world,
            contact_mode=int(HAPTIC_MESH_CONTACT_MODE),
            box_tolerance=float(HAPTIC_MESH_CONTACT_TOLERANCE_M),
            cell_tolerance=0.0,
            n_cells_per_node=2,
            generate_scalars=False,
        )
    except Exception:
        return None

    if int(n_contacts) <= 0 or contacts.n_points == 0:
        return None

    contact_point = np.asarray(contacts.points, dtype=float).mean(axis=0)
    normal = _body_contact_normal(body, contact_point)
    depth = _primitive_penetration_depth(body, collision_points_world)
    if depth <= 0.0:
        return None
    return normal, contact_point, depth


class WorldEngine:
    """Minimal container for world rigid bodies and mesh-based collision checks.

    The engine models floor/cube/sphere bodies with friction coefficients and
    performs kinematic robot-vs-world collision checks using transformed URDF
    collision geometry, broadphase bounds, and PyVista mesh collision as the
    narrowphase test.
    """

    def __init__(self) -> None:
        self.bodies: list[RigidBody] = []

    def add_floor(
        self,
        *,
        name: str = "floor",
        size: tuple[float, float] = (0.6, 0.6),
        z: float = 0.0,
        thickness: float = 0.01,
        friction: float = 0.8,
        density: float = 2400.0,
        mass: float | None = None,
        restitution: float = 0.05,
        color: str = "lightgray",
    ) -> RigidBody:
        pose = make_transform(np.eye(3), np.array([0.0, 0.0, z - 0.5 * thickness]))
        body = RigidBody(
            name=name,
            shape="floor",
            pose=pose,
            size=np.array([float(size[0]), float(size[1]), float(thickness)]),
            friction=float(friction),
            density=float(density),
            mass=mass,
            restitution=float(restitution),
            color=color,
            is_static=True,
        )
        self.bodies.append(body)
        return body

    def add_cube(
        self,
        *,
        name: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        friction: float = 0.6,
        density: float = 700.0,
        mass: float | None = None,
        restitution: float = 0.2,
        color: str = "tomato",
        is_static: bool = True,
    ) -> RigidBody:
        body = RigidBody(
            name=name,
            shape="cube",
            pose=make_transform(np.eye(3), np.asarray(center, dtype=float)),
            size=np.asarray(size, dtype=float),
            friction=float(friction),
            density=float(density),
            mass=mass,
            restitution=float(restitution),
            color=color,
            is_static=bool(is_static),
        )
        self.bodies.append(body)
        return body

    def add_sphere(
        self,
        *,
        name: str,
        center: tuple[float, float, float],
        radius: float,
        friction: float = 0.6,
        density: float = 1000.0,
        mass: float | None = None,
        restitution: float = 0.35,
        color: str = "royalblue",
        is_static: bool = True,
    ) -> RigidBody:
        body = RigidBody(
            name=name,
            shape="sphere",
            pose=make_transform(np.eye(3), np.asarray(center, dtype=float)),
            radius=float(radius),
            friction=float(friction),
            density=float(density),
            mass=mass,
            restitution=float(restitution),
            color=color,
            is_static=bool(is_static),
        )
        self.bodies.append(body)
        return body

    def add_wall(
        self,
        *,
        name: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        friction: float = 0.6,
        density: float = 2400.0,
        restitution: float = 0.1,
        color: str = "lightgray",
    ) -> RigidBody:
        """Add a static wall body (thin box) to the world."""
        body = RigidBody(
            name=name,
            shape="wall",
            pose=make_transform(np.eye(3), np.asarray(center, dtype=float)),
            size=np.asarray(size, dtype=float),
            friction=float(friction),
            density=float(density),
            restitution=float(restitution),
            color=color,
            is_static=True,
        )
        self.bodies.append(body)
        return body

    def add_walls_for_floor(
        self,
        floor: RigidBody,
        wall_height: float = WORLD_WALL_HEIGHT_M,
        wall_thickness: float = WORLD_WALL_THICKNESS_M,
        friction: float = WORLD_WALL_FRICTION,
        density: float = WORLD_WALL_DENSITY_KG_M3,
        restitution: float = WORLD_WALL_RESTITUTION,
        color: str = "lightgray",
    ) -> list[RigidBody]:
        """Add four walls enclosing the XY perimeter of *floor* at the floor surface."""
        if floor.size is None:
            return []
        sx = float(floor.size[0])
        sy = float(floor.size[1])
        floor_top_z = float(floor.pose[2, 3]) + 0.5 * float(floor.size[2])
        cx = float(floor.pose[0, 3])
        cy = float(floor.pose[1, 3])
        wall_cz = floor_top_z + 0.5 * float(wall_height)
        t = float(wall_thickness)
        h = float(wall_height)
        walls = [
            self.add_wall(
                name=f"{floor.name}_wall_neg_x",
                center=(cx - 0.5 * sx - 0.5 * t, cy, wall_cz),
                size=(t, sy + 2 * t, h),
                friction=friction,
                density=density,
                restitution=restitution,
                color=color,
            ),
            self.add_wall(
                name=f"{floor.name}_wall_pos_x",
                center=(cx + 0.5 * sx + 0.5 * t, cy, wall_cz),
                size=(t, sy + 2 * t, h),
                friction=friction,
                density=density,
                restitution=restitution,
                color=color,
            ),
            self.add_wall(
                name=f"{floor.name}_wall_neg_y",
                center=(cx, cy - 0.5 * sy - 0.5 * t, wall_cz),
                size=(sx, t, h),
                friction=friction,
                density=density,
                restitution=restitution,
                color=color,
            ),
            self.add_wall(
                name=f"{floor.name}_wall_pos_y",
                center=(cx, cy + 0.5 * sy + 0.5 * t, wall_cz),
                size=(sx, t, h),
                friction=friction,
                density=density,
                restitution=restitution,
                color=color,
            ),
        ]
        return walls

    def body_bounds(self, body: RigidBody) -> tuple[np.ndarray, np.ndarray]:
        return _bounds_from_points(
            _transform_points(_body_local_points(body), body.pose)
        )

    def body_mesh(self, body: RigidBody) -> pv.PolyData:
        return _transform_polydata(_body_local_mesh(body), body.pose)

    def body_by_name(self, name: str) -> RigidBody | None:
        for body in self.bodies:
            if body.name == name:
                return body
        return None

    def add_to_plotter(
        self, plotter: pv.Plotter, opacity: float = 0.9
    ) -> dict[str, BodyRenderItem]:
        """Render world bodies and return mutable mesh state for live updates."""
        items: dict[str, BodyRenderItem] = {}
        for body in self.bodies:
            local = _body_local_mesh(body)
            if local.n_points == 0:
                continue

            poly = _copy_polydata(local)
            local_vertices = np.asarray(poly.points, dtype=float).copy()
            poly.points = _transform_points(local_vertices, body.pose)
            plotter.add_mesh(
                poly,
                color=body.color,
                opacity=float(opacity),
                smooth_shading=True,
            )
            items[body.name] = BodyRenderItem(poly=poly, local_vertices=local_vertices)
        return items

    def update_plotter_meshes(self, render_items: dict[str, BodyRenderItem]) -> None:
        """Update already-added world meshes after body poses change."""
        for body in self.bodies:
            item = render_items.get(body.name)
            if item is None:
                continue
            item.poly.points = _transform_points(item.local_vertices, body.pose)

    def _wall_interior_xy_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return XY interior (playable) bounds derived from wall bodies around the floor."""
        floor_body: RigidBody | None = None
        for body in self.bodies:
            if body.shape == "floor":
                floor_body = body
                break
        if floor_body is None or floor_body.size is None:
            return None
        half_x = 0.5 * float(floor_body.size[0])
        half_y = 0.5 * float(floor_body.size[1])
        cx = float(floor_body.pose[0, 3])
        cy = float(floor_body.pose[1, 3])
        return (
            np.array([cx - half_x, cy - half_y], dtype=float),
            np.array([cx + half_x, cy + half_y], dtype=float),
        )

    def _floor_top_z(self) -> float:
        z_top = 0.0
        for body in self.bodies:
            if body.shape != "floor" or body.size is None:
                continue
            z = float(body.pose[2, 3]) + 0.5 * float(body.size[2])
            z_top = max(z_top, z)
        return z_top

    def _support_offset(self, body: RigidBody) -> float:
        if body.shape == "sphere" and body.radius is not None:
            return float(body.radius)
        if body.shape == "cylinder" and body.height is not None:
            return 0.5 * float(body.height)
        if body.shape in {"cube", "floor"} and body.size is not None:
            return 0.5 * float(body.size[2])
        return 0.0

    def step_dynamics(
        self,
        dt: float,
        contact_forces: dict[str, np.ndarray],
        linear_damping: float = 2.0,
    ) -> None:
        """Integrate simple translational dynamics for non-static bodies.

        Uses Coulomb-like planar friction threshold against the floor support
        force and a basic restitution bounce on vertical floor contact.
        """
        dt = float(max(dt, 1e-5))
        floor_top_z = self._floor_top_z()
        xy_bounds = self._wall_interior_xy_bounds()

        for body in self.bodies:
            if body.is_static:
                continue
            mass = float(max(body.mass_kg, 1e-9))

            force = np.asarray(
                contact_forces.get(body.name, np.zeros(3, dtype=float)),
                dtype=float,
            ).reshape(3)
            force[2] -= mass * 9.81

            lateral = force[:2].copy()
            lateral_norm = float(np.linalg.norm(lateral))
            normal_support = mass * 9.81 + max(float(force[2]), 0.0)
            static_limit = float(body.friction) * normal_support
            if lateral_norm <= static_limit:
                force[:2] = 0.0
            elif lateral_norm > 1e-12:
                force[:2] *= (lateral_norm - static_limit) / lateral_norm

            acc = force / mass - float(linear_damping) * body.linear_velocity
            body.linear_velocity = body.linear_velocity + acc * dt
            body.pose[:3, 3] = body.pose[:3, 3] + body.linear_velocity * dt

            min_z = floor_top_z + self._support_offset(body)
            if float(body.pose[2, 3]) < min_z:
                body.pose[2, 3] = min_z
                if float(body.linear_velocity[2]) < 0.0:
                    body.linear_velocity[2] = -float(body.restitution) * float(
                        body.linear_velocity[2]
                    )
                body.linear_velocity[:2] *= max(0.0, 1.0 - float(body.friction) * dt)

            # Clamp XY within wall-enclosed floor bounds
            if xy_bounds is not None:
                xy_min, xy_max = xy_bounds
                offset = self._support_offset(body)
                for axis in range(2):
                    lo = float(xy_min[axis]) + offset
                    hi = float(xy_max[axis]) - offset
                    pos = float(body.pose[axis, 3])
                    if pos < lo:
                        body.pose[axis, 3] = lo
                        if float(body.linear_velocity[axis]) < 0.0:
                            body.linear_velocity[axis] = -float(
                                body.restitution
                            ) * float(body.linear_velocity[axis])
                    elif pos > hi:
                        body.pose[axis, 3] = hi
                        if float(body.linear_velocity[axis]) > 0.0:
                            body.linear_velocity[axis] = -float(
                                body.restitution
                            ) * float(body.linear_velocity[axis])

    def check_robot_collisions(
        self,
        robot: CollisionRobotLike,
        theta: Optional[np.ndarray] = None,
        base_transform: Optional[np.ndarray] = None,
    ) -> list[CollisionResult]:
        results: list[CollisionResult] = []
        poses = collision_transforms(robot, theta=theta, base_transform=base_transform)
        for pose in poses:
            collision_mesh_world = _transform_polydata(
                _collision_local_mesh(pose.collision),
                pose.T_world,
            )
            collision_points_world = (
                np.asarray(collision_mesh_world.points, dtype=float)
                if collision_mesh_world.n_points > 0
                else np.zeros((0, 3), dtype=float)
            )
            collision_min, collision_max = collision_bounds(pose)
            if np.allclose(collision_min, collision_max) and np.allclose(
                collision_min, 0.0
            ):
                continue
            for body in self.bodies:
                body_min, body_max = self.body_bounds(body)
                if (
                    _overlap_depth(body_min, body_max, collision_min, collision_max)
                    <= 0.0
                ):
                    continue

                mesh_contact = _mesh_contact(
                    body,
                    self.body_mesh(body),
                    collision_mesh_world,
                    collision_points_world,
                )
                if mesh_contact is None:
                    continue

                normal, contact_point, depth = mesh_contact
                if depth <= 0.0:
                    continue
                results.append(
                    CollisionResult(
                        body_name=body.name,
                        link_name=pose.link_name,
                        geometry_type=pose.collision.geometry_type,
                        friction=float(body.friction),
                        restitution=float(body.restitution),
                        body_density_kg_m3=float(body.density or 0.0),
                        body_mass_kg=float(body.mass_kg),
                        penetration_depth=depth,
                        contact_point=contact_point,
                        contact_normal=normal,
                        body_bounds_min=body_min,
                        body_bounds_max=body_max,
                        collision_bounds_min=collision_min,
                        collision_bounds_max=collision_max,
                    )
                )
        return results


def collision_bounds(pose: CollisionPose) -> tuple[np.ndarray, np.ndarray]:
    """Return world-frame AABB bounds for one robot collision geometry."""
    local_points = _collision_local_points(pose.collision)
    world_points = _transform_points(local_points, pose.T_world)
    return _bounds_from_points(world_points)


def create_demo_world() -> WorldEngine:
    """Create a small default scene with an XY floor, four walls, one cube, and one sphere."""
    world = WorldEngine()
    floor = world.add_floor(
        size=(WORLD_FLOOR_SIZE_X_M, WORLD_FLOOR_SIZE_Y_M),
        z=WORLD_FLOOR_Z_M,
        thickness=WORLD_FLOOR_THICKNESS_M,
        friction=WORLD_FLOOR_FRICTION,
        density=WORLD_FLOOR_DENSITY_KG_M3,
        restitution=WORLD_FLOOR_RESTITUTION,
    )
    world.add_walls_for_floor(floor)

    return world


def check_psm_world_collisions(
    robot: CollisionRobotLike,
    world: WorldEngine,
    theta: Optional[np.ndarray] = None,
    base_transform: Optional[np.ndarray] = None,
) -> list[CollisionResult]:
    """Convenience wrapper for querying PSM-vs-world collisions."""
    return world.check_robot_collisions(
        robot, theta=theta, base_transform=base_transform
    )


def compute_haptic_feedback_force(
    collisions: list[CollisionResult],
    *,
    tool_position_world: np.ndarray,
    tool_velocity_world: np.ndarray | None = None,
    stiffness_n_per_m: float = 120.0,
    damping_n_s_per_m: float = 8.0,
    max_distance_m: float = 0.12,
    max_force_n: float = 3.0,
) -> np.ndarray:
    """Aggregate nearby contact responses into a single Cartesian haptic force.

    Contacts are weighted by distance from the tool tip so proximal/base-link
    contacts do not dominate distal tool interaction.  Restitution sharpens the
    damping term to make impacts feel crisper.
    """
    tool_position_world = np.asarray(tool_position_world, dtype=float).reshape(3)
    if tool_velocity_world is None:
        tool_velocity_world = np.zeros(3, dtype=float)
    else:
        tool_velocity_world = np.asarray(tool_velocity_world, dtype=float).reshape(3)

    total_force = np.zeros(3, dtype=float)
    for result in collisions:
        delta = tool_position_world - result.contact_point
        distance = float(np.linalg.norm(delta))
        if distance > float(max_distance_m):
            continue

        distance_weight = (1.0 - distance / max(float(max_distance_m), 1e-9)) ** 2
        mass_scale = float(np.clip(np.sqrt(max(result.body_mass_kg, 1e-6)), 0.5, 2.0))
        restitution_scale = 1.0 + float(np.clip(result.restitution, 0.0, 1.0))
        v_normal = float(np.dot(tool_velocity_world, result.contact_normal))
        inward_speed = max(-v_normal, 0.0)
        magnitude = (
            distance_weight
            * mass_scale
            * (
                float(stiffness_n_per_m) * float(result.penetration_depth)
                + float(damping_n_s_per_m) * restitution_scale * inward_speed
            )
        )
        total_force += magnitude * result.contact_normal

    norm = float(np.linalg.norm(total_force))
    if norm > float(max_force_n) and norm > 0.0:
        total_force *= float(max_force_n) / norm
    return total_force


def compute_body_contact_forces(
    collisions: list[CollisionResult],
    *,
    tool_position_world: np.ndarray,
    tool_velocity_world: np.ndarray | None = None,
    stiffness_n_per_m: float = 220.0,
    damping_n_s_per_m: float = 18.0,
    max_distance_m: float = 0.15,
) -> dict[str, np.ndarray]:
    """Compute force applied to world bodies from robot contacts.

    Force direction is opposite to the body contact normal so pushed bodies can
    move away when contact pressure exceeds friction/support limits.
    """
    tool_position_world = np.asarray(tool_position_world, dtype=float).reshape(3)
    if tool_velocity_world is None:
        tool_velocity_world = np.zeros(3, dtype=float)
    else:
        tool_velocity_world = np.asarray(tool_velocity_world, dtype=float).reshape(3)

    out: dict[str, np.ndarray] = {}
    for result in collisions:
        delta = tool_position_world - result.contact_point
        distance = float(np.linalg.norm(delta))
        if distance > float(max_distance_m):
            continue

        w = (1.0 - distance / max(float(max_distance_m), 1e-9)) ** 2
        v_n = float(np.dot(tool_velocity_world, result.contact_normal))
        inward_speed = max(-v_n, 0.0)
        mag = w * (
            float(stiffness_n_per_m) * float(result.penetration_depth)
            + float(damping_n_s_per_m) * inward_speed
        )
        f_world = -mag * np.asarray(result.contact_normal, dtype=float).reshape(3)
        if result.body_name not in out:
            out[result.body_name] = np.zeros(3, dtype=float)
        out[result.body_name] += f_world
    return out
