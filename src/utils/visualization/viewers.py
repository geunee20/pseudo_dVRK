from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import pyvista as pv

from src.kinematics.fk import joint_frames, visual_transforms
from src.robots.protocols import VisualRobotLike


# ============================================================
# Mesh loading helpers
# ============================================================


def _dataset_to_polydata(dataset: pv.DataObject) -> pv.PolyData:
    """Convert a generic PyVista dataset to :class:`pyvista.PolyData`.

    For :class:`pyvista.PolyData` inputs the data is deep-copied.  For other
    :class:`pyvista.DataSet` types the surface is extracted and triangulated.
    Empty or degenerate results are returned as an empty PolyData.

    Args:
        dataset: Any PyVista data object.

    Returns:
        :class:`pyvista.PolyData` with at least one point and cell, or an
        empty :class:`pyvista.PolyData` on failure.
    """
    if isinstance(dataset, pv.PolyData):
        poly = dataset.copy(deep=True)
    elif isinstance(dataset, pv.DataSet):
        surface = dataset.extract_surface().triangulate()
        if not isinstance(surface, pv.PolyData):
            return pv.PolyData()
        poly = surface
    else:
        return pv.PolyData()

    if poly.n_points == 0 or poly.n_cells == 0:
        return pv.PolyData()
    return cast(pv.PolyData, poly)


def _load_polydata(mesh_path: Path) -> Optional[pv.PolyData]:
    """Load a mesh file and return it as a single merged :class:`pyvista.PolyData`.

    Handles multi-block files (e.g. DAE) by merging each non-empty block.
    Prints a warning and returns ``None`` on read failure.

    Args:
        mesh_path: Absolute path to the mesh file.

    Returns:
        Merged :class:`pyvista.PolyData`, or ``None`` if loading failed.
    """
    try:
        mesh = pv.read(str(mesh_path))
    except Exception as exc:
        print(f"[warn] failed to load mesh {mesh_path}: {exc}")
        return None

    if isinstance(mesh, pv.MultiBlock):
        merged: Optional[pv.PolyData] = None
        for block in mesh:
            if block is None:
                continue

            poly = _dataset_to_polydata(block)
            if poly.n_points == 0:
                continue

            merged = poly if merged is None else merged.merge(poly, merge_points=False)

        return merged

    return _dataset_to_polydata(mesh)


def _make_axis_actor(plotter: pv.Plotter, T: np.ndarray, scale: float = 0.03):
    """Add red/green/blue axis arrows representing the columns of a transform to the plotter.

    Args:
        plotter: PyVista plotter to add arrows to.
        T: ``(4, 4)`` homogeneous transform; origin is ``T[:3, 3]``,
            axes are columns of ``T[:3, :3]``.
        scale: Arrow length in metres.

    Returns:
        Dict ``{'x': actor, 'y': actor, 'z': actor}`` of added mesh actors.
    """
    origin = T[:3, 3]
    R = T[:3, :3]

    actors = {}
    colors = {"x": "red", "y": "green", "z": "blue"}

    for i, key in enumerate(["x", "y", "z"]):
        direction = R[:, i]
        arrow = pv.Arrow(start=origin, direction=direction, scale=scale)
        actors[key] = plotter.add_mesh(arrow, color=colors[key], name=None)

    return actors


def _make_joint_marker(
    plotter: pv.Plotter, T: np.ndarray, active: bool, radius: float = 0.004
):
    """Add a sphere marker at the origin of a joint transform.

    The marker is red when *active* (non-zero joint value) and white otherwise.

    Args:
        plotter: PyVista plotter.
        T: ``(4, 4)`` transform; sphere is placed at ``T[:3, 3]``.
        active: ``True`` → red sphere; ``False`` → white sphere.
        radius: Sphere radius in metres.

    Returns:
        PyVista mesh actor for the added sphere.
    """
    center = T[:3, 3]
    sphere = pv.Sphere(radius=radius, center=center)
    color = "red" if active else "white"
    return plotter.add_mesh(sphere, color=color, smooth_shading=True, name=None)


# ============================================================
# Per-robot state in the viewer
# ============================================================


@dataclass
class _VisualItem:
    poly: pv.PolyData
    actor: object
    base_vertices: np.ndarray


@dataclass
class _FrameItem:
    x_actor: Any
    y_actor: Any
    z_actor: Any
    marker_actor: Any


@dataclass
class _RobotVisualState:
    robot: VisualRobotLike
    mesh_items: Dict[str, _VisualItem]
    frame_items: Dict[str, _FrameItem]


@dataclass
class _WaypointSet:
    points_poly: pv.PolyData
    points_actor: Any
    label_actors: list[Any]
    points: np.ndarray
    active_mask: np.ndarray
    color: str
    point_size: int
    opacity: float


@dataclass
class PlotterRobotMeshState:
    robot: VisualRobotLike
    poly_by_key: Dict[str, pv.PolyData]
    base_vertices_by_key: Dict[str, np.ndarray]


def add_robot_meshes_to_plotter(
    plotter: pv.Plotter,
    name: str,
    robot: VisualRobotLike,
    theta: Optional[np.ndarray] = None,
    base_transform: Optional[np.ndarray] = None,
    color: Optional[str] = None,
    alpha: float = 1.0,
) -> PlotterRobotMeshState:
    """Add only robot meshes to a plain PyVista plotter and return update state."""
    if theta is None:
        theta = robot.get_theta()
    else:
        theta = np.asarray(theta, dtype=float).reshape(-1)

    if base_transform is None:
        base_transform = np.eye(4)
    else:
        base_transform = np.asarray(base_transform, dtype=float).reshape(4, 4)

    poly_by_key: Dict[str, pv.PolyData] = {}
    base_vertices_by_key: Dict[str, np.ndarray] = {}

    for vp in visual_transforms(robot, theta, base_transform):
        mesh_path = Path(vp.visual.mesh_path)
        if not mesh_path.exists():
            continue

        poly = _load_polydata(mesh_path)
        if poly is None or poly.n_points == 0:
            continue

        key = f"{name}:{vp.link_name}:{mesh_path.name}"
        base_vertices = np.asarray(poly.points).copy()
        poly.points = DvrkRealtimeViz._transform_points(base_vertices, vp.T_world)

        plotter.add_mesh(
            poly,
            color=color,
            opacity=float(alpha),
            smooth_shading=True,
            name=key,
        )
        poly_by_key[key] = poly
        base_vertices_by_key[key] = base_vertices

    return PlotterRobotMeshState(
        robot=robot,
        poly_by_key=poly_by_key,
        base_vertices_by_key=base_vertices_by_key,
    )


def update_robot_meshes_on_plotter(
    state: PlotterRobotMeshState,
    theta: np.ndarray,
    base_transform: Optional[np.ndarray] = None,
) -> None:
    """Update mesh vertices previously added by add_robot_meshes_to_plotter."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != state.robot.dof:
        raise ValueError(f"Expected theta length {state.robot.dof}, got {theta.size}")

    if base_transform is None:
        base_transform = np.eye(4)
    else:
        base_transform = np.asarray(base_transform, dtype=float).reshape(4, 4)

    for vp in visual_transforms(state.robot, theta, base_transform):
        mesh_path = Path(vp.visual.mesh_path)
        key = f"{mesh_path.name}"

        for full_key, poly in state.poly_by_key.items():
            if not full_key.endswith(f":{vp.link_name}:{key}"):
                continue

            base_vertices = state.base_vertices_by_key[full_key]
            poly.points = DvrkRealtimeViz._transform_points(base_vertices, vp.T_world)
            poly.Modified()


class DvrkRealtimeViz:
    """Interactive PyVista-based real-time visualiser for dVRK robots.

    Manages a single :class:`pyvista.Plotter` window and supports adding
    multiple robots by name, updating their joint configurations in real time,
    and overlaying joint-frame axes, waypoint sets, and FK markers.

    Example usage::

        viz = DvrkRealtimeViz(show_frames=True)
        viz.add_robot("psm", psm_robot, theta=theta0)
        # in a callback:
        viz.update_robot("psm", theta_new)
        viz.show()
    """

    def __init__(
        self,
        title: str = "dVRK Real-Time Visualization",
        window_size: tuple[int, int] = (1400, 900),
        background: str = "white",
        show_frames: bool = True,
        alpha: float = 0.6,
        frame_scale: float = 0.03,
        marker_radius: float = 0.004,
    ) -> None:
        """Initialise the visualiser and open a PyVista plotter window.

        Args:
            title: Window title string.
            window_size: ``(width, height)`` in pixels.
            background: Background colour passed to :meth:`pyvista.Plotter.set_background`.
            show_frames: Whether to draw joint-frame axis arrows and markers.
            alpha: Default mesh opacity ``[0, 1]``.
            frame_scale: Length of joint-frame axis arrows in metres.
            marker_radius: Radius of joint-origin sphere markers in metres.
        """
        self.plotter: Any = pv.Plotter(title=title, window_size=list(window_size))
        self.plotter.set_background(background)

        self.show_frames = show_frames
        self.alpha = float(alpha)
        self.frame_scale = float(frame_scale)
        self.marker_radius = float(marker_radius)
        self._waypoint_sets: Dict[str, _WaypointSet] = {}
        self._robots: Dict[str, _RobotVisualState] = {}

        T_world = np.eye(4)
        _make_axis_actor(self.plotter, T_world, scale=0.1)

    def _build_frame_items(
        self,
        robot: VisualRobotLike,
        theta: np.ndarray,
        base_transform: np.ndarray,
    ) -> Dict[str, _FrameItem]:
        frame_items: Dict[str, _FrameItem] = {}
        q_full = robot.expand_theta(theta)

        for jf in joint_frames(robot, theta, base_transform):
            axis_actors = _make_axis_actor(
                self.plotter, jf.T_world, scale=self.frame_scale
            )
            marker_actor = _make_joint_marker(
                self.plotter,
                jf.T_world,
                active=abs(q_full.get(jf.joint_name, 0.0)) > 1e-6,
                radius=self.marker_radius,
            )
            frame_items[jf.joint_name] = _FrameItem(
                x_actor=axis_actors["x"],
                y_actor=axis_actors["y"],
                z_actor=axis_actors["z"],
                marker_actor=marker_actor,
            )

        return frame_items

    def _clear_frame_items(self, frame_items: Dict[str, _FrameItem]) -> None:
        for frame in frame_items.values():
            self.plotter.remove_actor(frame.x_actor, render=False)
            self.plotter.remove_actor(frame.y_actor, render=False)
            self.plotter.remove_actor(frame.z_actor, render=False)
            self.plotter.remove_actor(frame.marker_actor, render=False)

    def add_robot(
        self,
        name: str,
        robot: VisualRobotLike,
        theta: Optional[np.ndarray] = None,
        base_transform: Optional[np.ndarray] = None,
        color: Optional[str] = None,
    ) -> None:
        """Load and add a robot's meshes (and optionally joint frames) to the plotter.

        Args:
            name: Unique robot identifier used for subsequent :meth:`update_robot` calls.
            robot: Robot model conforming to :class:`~src.robots.protocols.VisualRobotLike`.
            theta: Initial joint configuration, shape ``(dof,)``.  Defaults to
                the robot's current configuration.
            base_transform: Optional ``(4, 4)`` world-from-base transform.  Defaults to
                identity.
            color: Solid colour for all meshes, or ``None`` to use mesh file colours.

        Raises:
            ValueError: If *name* is already registered in this viewer.
        """
        if name in self._robots:
            raise ValueError(f"Robot name already exists in viewer: {name}")

        if theta is None:
            theta = robot.get_theta()
        else:
            theta = np.asarray(theta, dtype=float).reshape(-1)

        if base_transform is None:
            base_transform = np.eye(4)
        else:
            base_transform = np.asarray(base_transform, dtype=float).reshape(4, 4)

        mesh_items: Dict[str, _VisualItem] = {}
        frame_items: Dict[str, _FrameItem] = {}

        for vp in visual_transforms(robot, theta, base_transform):
            mesh_path = Path(vp.visual.mesh_path)
            if not mesh_path.exists():
                print(f"[warn] missing mesh: {mesh_path}")
                continue

            poly = _load_polydata(mesh_path)
            if poly is None or poly.n_points == 0:
                continue

            base_vertices = np.asarray(poly.points).copy()
            poly.points = self._transform_points(base_vertices, vp.T_world)

            actor_name = f"{name}:{vp.link_name}:{mesh_path.name}"
            actor = self.plotter.add_mesh(
                poly,
                color=color,
                opacity=self.alpha,
                smooth_shading=True,
                name=actor_name,
            )
            mesh_items[actor_name] = _VisualItem(
                poly=poly,
                actor=actor,
                base_vertices=base_vertices,
            )

        if self.show_frames:
            frame_items = self._build_frame_items(robot, theta, base_transform)

        self._robots[name] = _RobotVisualState(
            robot=robot,
            mesh_items=mesh_items,
            frame_items=frame_items,
        )

    def update_robot(
        self,
        name: str,
        theta: np.ndarray,
        base_transform: Optional[np.ndarray] = None,
    ) -> None:
        """Update mesh positions (and joint frames) for a previously added robot.

        Transforms mesh vertices for the new configuration in-place and calls
        ``poly.Modified()`` so PyVista picks up the change without re-adding actors.

        Args:
            name: Robot identifier previously passed to :meth:`add_robot`.
            theta: New joint configuration, shape ``(dof,)``.
            base_transform: Optional ``(4, 4)`` world-from-base transform.  Defaults
                to the identity matrix.

        Raises:
            KeyError: If *name* is not a registered robot.
            ValueError: If ``theta.size != robot.dof``.
        """
        if name not in self._robots:
            raise KeyError(f"Unknown robot in viewer: {name}")

        state = self._robots[name]
        robot = state.robot

        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != robot.dof:
            raise ValueError(f"Expected theta length {robot.dof}, got {theta.size}")

        if base_transform is None:
            base_transform = np.eye(4)
        else:
            base_transform = np.asarray(base_transform, dtype=float).reshape(4, 4)

        for vp in visual_transforms(robot, theta, base_transform):
            mesh_path = Path(vp.visual.mesh_path)
            actor_name = f"{name}:{vp.link_name}:{mesh_path.name}"
            if actor_name not in state.mesh_items:
                continue

            item = state.mesh_items[actor_name]
            item.poly.points = self._transform_points(item.base_vertices, vp.T_world)
            item.poly.Modified()

        if self.show_frames:
            self._clear_frame_items(state.frame_items)
            state.frame_items = self._build_frame_items(robot, theta, base_transform)

        self.plotter.render()

    def add_fk_marker(self, size: int = 10, color: str = "blue") -> pv.PolyData:
        """Add a single movable point marker to the plotter and return its PolyData.

        The returned :class:`pyvista.PolyData` can be updated in-place to move
        the marker without re-adding it to the scene.

        Args:
            size: Rendered point size in pixels.
            color: Marker colour.

        Returns:
            Single-point :class:`pyvista.PolyData` (initially at the origin).
        """
        marker = pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=float))
        self.plotter.add_points(
            marker,
            color=color,
            point_size=size,
            render_points_as_spheres=True,
        )
        return marker

    def make_line(self) -> pv.PolyData:
        """Create a two-point degenerate line segment PolyData suitable for in-place updates.

        Both endpoints start at the origin.  Update ``poly.points`` to reposition
        the line without re-adding it to the scene.

        Returns:
            :class:`pyvista.PolyData` with two coincident points and one line cell.
        """
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
        poly = pv.PolyData(pts)
        poly.lines = np.array([2, 0, 1])
        return poly

    def add_polyline_mesh(self, points: np.ndarray) -> pv.PolyData:
        """Create a connected polyline PolyData from an ordered array of points.

        Args:
            points: Point array, shape ``(N, 3)``.

        Returns:
            :class:`pyvista.PolyData` with ``N`` points and ``N-1`` line cells
            (or just points if ``N < 2``).
        """
        poly = pv.PolyData(points)

        if len(points) >= 2:
            cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
            cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
            cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
            poly.lines = cells.ravel()

        return poly

    def add_canvas(
        self,
        center: Optional[np.ndarray] = None,
        width: float = 0.72,
        height: float = 0.35,
        color: str = "white",
        opacity: float = 0.9,
        show_border: bool = True,
        border_color: str = "black",
        border_width: int = 3,
        border_offset: float = 1e-3,
    ):
        """Add a flat rectangular canvas mesh to the plotter (for 2-D projections).

        The canvas lies in the YZ plane (X = ``center[0]``).  An optional border
        polyline is drawn slightly in front (offset along X).

        Args:
            center: Centre of the canvas ``(x, y, z)`` in metres.
                Defaults to a position suitable for the standard dVRK setup.
            width: Canvas width (Y direction) in metres.
            height: Canvas height (Z direction) in metres.
            color: Fill colour of the canvas mesh.
            opacity: Opacity ``[0, 1]`` of the canvas.
            show_border: Whether to draw a border polyline.
            border_color: Colour of the border polyline.
            border_width: Line width of the border in pixels.
            border_offset: Offset of the border along X to avoid Z-fighting.

        Returns:
            ``(canvas, border)`` — the canvas :class:`pyvista.PolyData` and
            the border :class:`pyvista.PolyData` (or ``None`` if *show_border* is False).
        """
        if center is None:
            center = np.array([0.0, 2.0, 0.175 - 0.03], dtype=float)
        else:
            center = np.asarray(center, dtype=float).reshape(-1)
            if center.size != 3:
                raise ValueError("center must be shape (3,)")

        x, y, z = center
        p0 = np.array([x, y - width / 2.0, z - height / 2.0], dtype=float)
        p1 = np.array([x, y + width / 2.0, z - height / 2.0], dtype=float)
        p2 = np.array([x, y + width / 2.0, z + height / 2.0], dtype=float)
        p3 = np.array([x, y - width / 2.0, z + height / 2.0], dtype=float)

        canvas = pv.PolyData()
        canvas.points = np.vstack([p0, p1, p2, p3])
        canvas.faces = np.array([4, 0, 1, 2, 3])
        self.plotter.add_mesh(canvas, color=color, opacity=opacity)

        border = None
        if show_border:
            border_points = np.vstack([p0, p1, p2, p3]).copy()
            border_points[:, 0] += border_offset

            border = pv.PolyData()
            border.points = border_points
            border.lines = np.hstack([[2, 0, 1], [2, 1, 2], [2, 2, 3], [2, 3, 0]])
            self.plotter.add_mesh(border, color=border_color, line_width=border_width)

        return canvas, border

    def add_canvas_waypoints(
        self,
        name: str,
        waypoints_xyz: np.ndarray,
        color: str = "blue",
        point_size: int = 15,
        alpha: float = 0.8,
        label_color: str = "white",
        label_size: int = 15,
        label_z_offset: float = 0.0,
    ):
        """Add a named set of labelled waypoint markers to the plotter.

        Waypoints are rendered as spheres with sequential numeric labels
        ``1, 2, …, N``.  Individual points can be hidden with
        :meth:`hide_canvas_waypoint` and restored with :meth:`reset_canvas_waypoints`.

        Args:
            name: Unique identifier for this waypoint set.
            waypoints_xyz: Waypoint positions, shape ``(N, 3)``.
            color: Sphere colour.
            point_size: Rendered sphere size in pixels.
            alpha: Opacity ``[0, 1]``.
            label_color: Text colour for the numeric labels.
            label_size: Font size of the labels.
            label_z_offset: Additional Z offset applied to label positions.

        Returns:
            The newly created :class:`_WaypointSet`.

        Raises:
            ValueError: If *name* already exists or *waypoints_xyz* is not ``(N, 3)``.
        """
        if waypoints_xyz.ndim != 2 or waypoints_xyz.shape[1] != 3:
            raise ValueError("waypoints_xyz must be shape (N, 3)")
        if name in self._waypoint_sets:
            raise ValueError(f"Waypoint set already exists: {name}")

        poly = pv.PolyData(waypoints_xyz.copy())
        points_actor = self.plotter.add_points(
            poly,
            color=color,
            point_size=point_size,
            render_points_as_spheres=True,
            opacity=alpha,
        )

        label_actors: list[Any] = []
        for i, p in enumerate(waypoints_xyz):
            p_lab = np.asarray(p, dtype=float).copy()
            p_lab[2] += label_z_offset
            actor = self.plotter.add_point_labels(
                np.array([p_lab], dtype=float),
                [str(i + 1)],
                text_color=label_color,
                point_size=0,
                font_size=label_size,
                shape=None,
                justification_horizontal="center",
                justification_vertical="center",
                always_visible=True,
            )
            label_actors.append(actor)

        self._waypoint_sets[name] = _WaypointSet(
            points_poly=poly,
            points_actor=points_actor,
            label_actors=label_actors,
            points=waypoints_xyz.copy(),
            active_mask=np.ones(len(waypoints_xyz), dtype=bool),
            color=color,
            point_size=point_size,
            opacity=float(alpha),
        )
        return self._waypoint_sets[name]

    def hide_canvas_waypoint(self, name: str, index: int) -> None:
        """Hide a single waypoint from a named set.

        Removes the point from the rendered poly and hides its label actor.
        Has no effect if the waypoint is already hidden.

        Args:
            name: Waypoint set identifier.
            index: 0-based index of the waypoint to hide.

        Raises:
            KeyError: If *name* is not registered.
            IndexError: If *index* is out of range.
        """
        if name not in self._waypoint_sets:
            raise KeyError(f"Unknown waypoint set: {name}")

        wp = self._waypoint_sets[name]
        if not (0 <= index < len(wp.points)):
            raise IndexError(f"Waypoint index out of range: {index}")
        if not wp.active_mask[index]:
            return

        wp.active_mask[index] = False
        wp.points_poly.points = wp.points[wp.active_mask]
        wp.points_poly.Modified()

        try:
            wp.label_actors[index].SetVisibility(False)
        except Exception:
            pass

        self.plotter.render()

    def reset_canvas_waypoints(self, name: str) -> None:
        """Restore all waypoints in a named set to their original positions/visibility.

        Args:
            name: Waypoint set identifier.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._waypoint_sets:
            raise KeyError(f"Unknown waypoint set: {name}")

        wp = self._waypoint_sets[name]
        wp.active_mask[:] = True
        wp.points_poly.points = wp.points.copy()
        wp.points_poly.Modified()

        for actor in wp.label_actors:
            try:
                actor.SetVisibility(True)
            except Exception:
                pass

        self.plotter.render()

    def add_waypoint_highlight_marker(
        self, size: int = 28, color: str = "yellow"
    ) -> pv.PolyData:
        """Add a large point marker used to highlight an active waypoint.

        A convenience wrapper around :meth:`add_fk_marker` with defaults
        suitable for highlighting a selected waypoint.

        Args:
            size: Rendered point size in pixels.
            color: Marker colour.

        Returns:
            Single-point :class:`pyvista.PolyData` (initially at the origin).
        """
        return self.add_fk_marker(size=size, color=color)

    def update_waypoint_highlight(
        self, marker: pv.PolyData, point: np.ndarray | None
    ) -> None:
        """Move or hide the waypoint highlight marker.

        Args:
            marker: Marker previously returned by :meth:`add_waypoint_highlight_marker`.
            point: New 3-D position, or ``None`` to hide the marker.
        """
        if point is None:
            marker.points = np.empty((0, 3), dtype=float)
        else:
            marker.points = np.array([point], dtype=float)
        marker.Modified()

    def show(self) -> None:
        """Open the PyVista window and block until it is closed."""
        self.plotter.show()

    def reset_camera(self) -> None:
        """Reset the camera to fit all visible scene contents and re-render."""
        self.plotter.reset_camera()
        self.plotter.render()

    def set_camera(
        self,
        position=(1.0, 1.0, 1.0),
        focal_point=(0.0, 0.0, 0.0),
        viewup=(0.0, 0.0, 1.0),
    ) -> None:
        """Set the camera position, focal point, and up-vector.

        Args:
            position: Camera position in world coordinates ``(x, y, z)``.
            focal_point: Point the camera looks at ``(x, y, z)``.
            viewup: Camera up-vector ``(x, y, z)``.
        """
        self.plotter.camera_position = [position, focal_point, viewup]
        self.plotter.render()

    @staticmethod
    def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Apply a ``(4, 4)`` homogeneous transform to an ``(N, 3)`` point array.

        Args:
            points: Point array, shape ``(N, 3)``.
            T: Homogeneous transform ``(4, 4)``.

        Returns:
            Transformed points, shape ``(N, 3)``.
        """
        ones = np.ones((points.shape[0], 1), dtype=float)
        points_h = np.hstack([points, ones])
        transformed = (T @ points_h.T).T
        return transformed[:, :3]


def visualize(
    robot,
    theta: Optional[np.ndarray] = None,
    show_frames: bool = False,
    alpha: float = 1.0,
) -> DvrkRealtimeViz:
    """Create a :class:`DvrkRealtimeViz` pre-loaded with a single robot.

    Convenience factory for quick visualisation scripts.

    Args:
        robot: Robot model conforming to :class:`~src.robots.protocols.VisualRobotLike`.
        theta: Joint configuration, shape ``(dof,)``.  Defaults to the robot's
            current configuration.
        show_frames: Whether to draw joint-frame axis arrows.
        alpha: Mesh opacity ``[0, 1]``.

    Returns:
        :class:`DvrkRealtimeViz` with the robot added but the window not yet shown.
        Call :meth:`DvrkRealtimeViz.show` to open the window.
    """
    if theta is None:
        theta = robot.get_theta()
    viz = DvrkRealtimeViz(show_frames=show_frames, alpha=float(alpha))
    viz.add_robot("robot", robot, theta=np.asarray(theta, dtype=float).reshape(-1))
    return viz


def set_camera_view(scene, eye, target):
    """Point a viewer or plotter camera at a target from a given eye position.

    Accepts either a :class:`DvrkRealtimeViz` (calls :meth:`~DvrkRealtimeViz.set_camera`)
    or a raw :class:`pyvista.Plotter`.  Up-vector is always ``+Z``.

    Args:
        scene: :class:`DvrkRealtimeViz` or :class:`pyvista.Plotter`.
        eye: Camera position ``(x, y, z)`` in world coordinates.
        target: Focal point ``(x, y, z)`` in world coordinates.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)

    if hasattr(scene, "set_camera"):
        scene.set_camera(
            position=tuple(eye.tolist()),
            focal_point=tuple(target.tolist()),
            viewup=(0.0, 0.0, 1.0),
        )
        return

    if isinstance(scene, pv.Plotter):
        scene.camera_position = [
            tuple(eye.tolist()),
            tuple(target.tolist()),
            (0.0, 0.0, 1.0),
        ]
        scene.render()
        return

    if hasattr(scene, "camera_position"):
        scene.camera_position = [
            tuple(eye.tolist()),
            tuple(target.tolist()),
            (0.0, 0.0, 1.0),
        ]
        if hasattr(scene, "render"):
            scene.render()


def demo_two_robots(ecm, psm, theta_ecm=None, theta_psm=None) -> None:
    viz = DvrkRealtimeViz(show_frames=True, alpha=0.6)

    if theta_ecm is None:
        theta_ecm = np.zeros(ecm.dof)
    if theta_psm is None:
        theta_psm = np.zeros(psm.dof)

    T_ecm = np.eye(4)
    T_psm = np.eye(4)
    T_psm[:3, 3] = np.array([0.25, 0.0, 0.0])

    viz.add_robot(
        "ecm", ecm, theta=theta_ecm, base_transform=T_ecm, color="lightsteelblue"
    )
    viz.add_robot("psm", psm, theta=theta_psm, base_transform=T_psm, color="orange")
    viz.set_camera(
        position=(1.0, 1.0, 1.0), focal_point=(0.0, 0.0, 0.0), viewup=(0.0, 0.0, 1.0)
    )
    viz.show()
