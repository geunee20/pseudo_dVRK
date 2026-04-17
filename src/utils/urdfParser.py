import re
import math
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path


from src.robots.robot import JointInfo, JointLimit, Mimic, LinkVisual


def _clean_name(text: str | None) -> str:
    if text is None:
        return ""
    return text.replace("${prefix}", "").replace("${parent_link}", "world").strip()


def _strip_package_prefix(mesh_filename: str) -> str:
    s = mesh_filename.strip().replace("\\", "/")

    # package://name/meshes/foo.stl -> meshes/foo.stl
    m = re.match(r"^package://[^/]+/(.+)$", s)
    if m:
        return m.group(1)

    m = re.match(r"^package:/[^/]+/(.+)$", s)
    if m:
        return m.group(1)

    return s


def _eval_expr(expr: str) -> float:
    """
    Evaluate a very small subset of xacro scalar expressions safely.
    Supports numbers, +, -, *, /, parentheses, and PI.
    Example:
        ${-75/180*PI} -> -1.3089969389957472
    """
    s = expr.strip()

    if s.startswith("${") and s.endswith("}"):
        s = s[2:-1].strip()

    s = s.replace("PI", str(math.pi))

    # allow only safe characters
    if not re.fullmatch(r"[0-9eE\.\+\-\*/\(\) ]+", s):
        raise ValueError(f"Unsupported xacro scalar expression: {expr}")

    return float(eval(s, {"__builtins__": {}}, {}))


def _parse_scalar(text: str | None, default: float = 0.0) -> float:
    if text is None:
        return float(default)

    s = text.strip()

    if "${" in s:
        return _eval_expr(s)

    return float(s)


def _parse_vec3(text: str | None, default=None) -> np.ndarray:
    if default is None:
        default = np.zeros(3)

    if text is None:
        return np.asarray(default, dtype=float)

    text = text.strip()

    if text.startswith("${") and text.endswith("}") and " " not in text:
        return np.asarray(default, dtype=float)

    parts = text.split()
    if len(parts) != 3:
        raise ValueError(f"Failed to parse vec3 from: {text}")

    out = []
    for i, part in enumerate(parts):
        try:
            out.append(_parse_scalar(part, default=float(default[i])))
        except ValueError:
            out.append(float(default[i]))

    return np.asarray(out, dtype=float)


def _prefer_stl_if_available(mesh_path: Path) -> Path:
    """
    If the given mesh is .dae and a same-stem .STL exists, use the STL instead.
    """
    if mesh_path.suffix.lower() == ".dae":
        stl_path = mesh_path.with_suffix(".STL")
        if stl_path.exists():
            return stl_path

        stl_path_lower = mesh_path.with_suffix(".stl")
        if stl_path_lower.exists():
            return stl_path_lower

    return mesh_path


def parse_urdf(robot, urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for elem in root:
        tag = elem.tag.split("}")[-1]
        if tag == "link":
            link_name = _clean_name(elem.attrib.get("name"))
            robot.add_link(link_name)

            for child in elem:
                if child.tag.split("}")[-1] != "visual":
                    continue

                xyz = np.zeros(3)
                rpy = np.zeros(3)
                mesh = None

                for v in child:
                    vtag = v.tag.split("}")[-1]

                    if vtag == "origin":
                        xyz = _parse_vec3(v.attrib.get("xyz"))
                        rpy = _parse_vec3(v.attrib.get("rpy"))

                    elif vtag == "geometry":
                        for g in v:
                            if g.tag.split("}")[-1] == "mesh":
                                mesh = g.attrib.get("filename")

                if mesh is not None:
                    mesh_rel = _strip_package_prefix(mesh)
                    mesh_path = robot.resolve_mesh_path(mesh_rel)
                    mesh_path = _prefer_stl_if_available(mesh_path)

                    robot.add_visual(
                        LinkVisual(
                            link_name=link_name,
                            mesh_path=mesh_path,
                            origin_xyz=xyz,
                            origin_rpy=rpy,
                        )
                    )

        elif tag == "joint":
            joint_name = _clean_name(elem.attrib.get("name"))
            joint_type = elem.attrib["type"]

            parent = ""
            child = ""
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            axis = np.array([0.0, 0.0, 1.0])
            limit = None
            mimic = None

            for sub in elem:
                st = sub.tag.split("}")[-1]

                if st == "parent":
                    parent = _clean_name(sub.attrib.get("link"))

                elif st == "child":
                    child = _clean_name(sub.attrib.get("link"))

                elif st == "origin":
                    xyz = _parse_vec3(sub.attrib.get("xyz"))
                    rpy = _parse_vec3(sub.attrib.get("rpy"))

                elif st == "axis":
                    axis = _parse_vec3(
                        sub.attrib.get("xyz"), default=np.array([0.0, 0.0, 1.0])
                    )

                elif st == "limit":
                    lower = _parse_scalar(sub.attrib.get("lower"), 0.0)
                    upper = _parse_scalar(sub.attrib.get("upper"), 0.0)
                    limit = JointLimit(lower=lower, upper=upper)

                elif st == "mimic":
                    mimic = Mimic(
                        joint=_clean_name(sub.attrib.get("joint")),
                        multiplier=float(sub.attrib.get("multiplier", 1.0)),
                        offset=float(sub.attrib.get("offset", 0.0)),
                    )

            robot.add_joint(
                JointInfo(
                    name=joint_name,
                    joint_type=joint_type,
                    parent=parent,
                    child=child,
                    origin_xyz=xyz,
                    origin_rpy=rpy,
                    axis=axis,
                    limit=limit,
                    mimic=mimic,
                )
            )
