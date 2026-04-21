from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "calibrations"))
sys.path.insert(0, str(ROOT / "examples"))

project = "pseudo_dVRK"
author = "pseudo_dVRK contributors"
release = "main"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
# Publish URDF assets with the generated site so links can point to raw files.
html_extra_path = [str(ROOT / "urdfs")]

# Keep docs buildable in CI without native graphics/haptics stacks.
autodoc_mock_imports = [
    "pyvista",
    "pinocchio",
    "matplotlib",
    "src.pyOpenHaptics",
    "src.pyOpenHaptics.hd",
    "src.pyOpenHaptics.hd_device",
    "src.pyOpenHaptics.hd_callback",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
