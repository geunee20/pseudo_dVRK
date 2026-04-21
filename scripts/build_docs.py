from __future__ import annotations

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
    api_out = docs_source / "api"

    api_out.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "-m",
            "sphinx.ext.apidoc",
            "-f",
            "-o",
            str(api_out),
            str(root / "src"),
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
