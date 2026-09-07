"""Audit distribution members with generic allowlists and path-leak checks."""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

LOCAL_PATH_PATTERNS = (
    re.compile(rb"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]"),
    re.compile(rb"/(?:home|Users)/[^/\s]+/"),
    re.compile(rb"file://", re.IGNORECASE),
)
FORBIDDEN_COMPONENTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "test-install",
}
SDIST_ROOT_FILES = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
    "setup.cfg",
}
PACKAGE_FILES = {
    "src/adaptcompile/__init__.py",
    "src/adaptcompile/_validation.py",
    "src/adaptcompile/compiler/__init__.py",
    "src/adaptcompile/compiler/data.py",
    "src/adaptcompile/dataset.py",
    "src/adaptcompile/episode.py",
    "src/adaptcompile/errors.py",
    "src/adaptcompile/family.py",
    "src/adaptcompile/geometry.py",
    "src/adaptcompile/model.py",
    "src/adaptcompile/program.py",
    "src/adaptcompile/py.typed",
    "src/adaptcompile/result.py",
    "src/adaptcompile/serialization.py",
    "src/adaptcompile/study.py",
}
SDIST_PUBLIC_FILES = {
    "docs/architecture.md",
    "docs/compiler-data.md",
    "examples/basic_comparison.py",
    "examples/compiler_dataset.py",
    "examples/from_experiment_results.py",
    "tests/conftest.py",
    "tests/integration/test_compiler_dataset.py",
    "tests/integration/test_experiment_slice.py",
    "tests/integration/test_multi_model_slice.py",
    "tests/test_compiler.py",
    "tests/test_dataset.py",
    "tests/test_episode.py",
    "tests/test_family.py",
    "tests/test_geometry.py",
    "tests/test_model.py",
    "tests/test_optional_dataframe.py",
    "tests/test_program.py",
    "tests/test_result.py",
    "tests/test_serialization.py",
    "tests/test_study.py",
}
WHEEL_PACKAGE_FILES = {name.removeprefix("src/") for name in PACKAGE_FILES}


def _normalized_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member path: {name}")
    if FORBIDDEN_COMPONENTS.intersection(path.parts):
        raise ValueError(f"development artifact in distribution: {name}")
    if path.suffix in {".pyc", ".pyo"}:
        raise ValueError(f"bytecode in distribution: {name}")
    return path


def _check_text(name: str, payload: bytes) -> None:
    for pattern in LOCAL_PATH_PATTERNS:
        if pattern.search(payload):
            raise ValueError(f"local filesystem path found in {name}")


def _wheel_member_allowed(path: PurePosixPath) -> bool:
    if path.parts[0] == "adaptcompile":
        return path.as_posix() in WHEEL_PACKAGE_FILES
    if path.parts[0].endswith(".dist-info"):
        return path.name in {
            "INSTALLER",
            "METADATA",
            "RECORD",
            "REQUESTED",
            "WHEEL",
            "top_level.txt",
            "LICENSE",
        }
    return False


def check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            member = _normalized_member(name)
            if not _wheel_member_allowed(member):
                raise ValueError(f"unexpected wheel member: {name}")
            _check_text(name, archive.read(name))
    required = {
        "adaptcompile/__init__.py",
        "adaptcompile/compiler/__init__.py",
        "adaptcompile/compiler/data.py",
        "adaptcompile/py.typed",
    }
    missing = required.difference(names)
    if missing:
        raise ValueError(f"wheel is missing required members: {sorted(missing)}")


def _sdist_member_allowed(path: PurePosixPath) -> bool:
    if len(path.parts) < 2:
        return True
    relative = PurePosixPath(*path.parts[1:])
    if len(relative.parts) == 1:
        return relative.name in SDIST_ROOT_FILES
    if relative.as_posix() in PACKAGE_FILES | SDIST_PUBLIC_FILES:
        return True
    if len(relative.parts) >= 2 and relative.parts[0] == "src":
        return relative.parts[1].endswith(".egg-info")
    return False


def check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        for item in archive.getmembers():
            member = _normalized_member(item.name)
            if item.isdir():
                continue
            if not _sdist_member_allowed(member):
                raise ValueError(f"unexpected sdist member: {item.name}")
            if not item.isfile():
                continue
            extracted = archive.extractfile(item)
            if extracted is None:
                raise ValueError(f"could not read sdist member: {item.name}")
            _check_text(item.name, extracted.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    arguments = parser.parse_args()
    for artifact in arguments.artifacts:
        if artifact.suffix == ".whl":
            check_wheel(artifact)
        elif artifact.name.endswith(".tar.gz"):
            check_sdist(artifact)
        else:
            raise ValueError(f"unsupported distribution type: {artifact}")


if __name__ == "__main__":
    main()
