"""Install exactly one built distribution selected by kind."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("wheel", "sdist"))
    parser.add_argument("--extra")
    parser.add_argument("--no-deps", action="store_true")
    arguments = parser.parse_args()

    pattern = "*.whl" if arguments.kind == "wheel" else "*.tar.gz"
    distributions = sorted(Path("dist").glob(pattern))
    if len(distributions) != 1:
        raise SystemExit(
            f"expected one {arguments.kind} in dist, found {len(distributions)}"
        )
    requirement = str(distributions[0].resolve())
    if arguments.extra:
        requirement = f"{requirement}[{arguments.extra}]"
    command = [sys.executable, "-m", "pip", "install"]
    if arguments.no_deps:
        command.append("--no-deps")
    subprocess.run([*command, requirement], check=True)


if __name__ == "__main__":
    main()
