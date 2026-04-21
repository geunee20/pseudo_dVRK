from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    docs_source = root / "docs" / "source"
    docs_build = root / "docs" / "build" / "html"

    # Remove stale generated trees from previous docs scopes.
    for stale in (docs_source / "api_calibrations", docs_source / "api_examples"):
        if stale.exists():
            shutil.rmtree(stale)

    apidoc_targets = [
        (docs_source / "api", root / "src"),
    ]

    for out_dir, module_dir in apidoc_targets:
        out_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                "-m",
                "sphinx.ext.apidoc",
                "-f",
                "-o",
                str(out_dir),
                str(module_dir),
            ],
            cwd=root,
        )

    run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            str(docs_source),
            str(docs_build),
        ],
        cwd=root,
    )

    print(f"Built docs at: {docs_build}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
