"""The repository layout rules: configs/ and tests/ mirror src/rp1, and imports point one way."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "rp1"

# what each layer may import from rp1, besides itself
ALLOWED = {
    "utils": set(),
    "data": {"utils"},
    "environment": {"utils", "data"},
    "core.world_model": {"utils", "data"},
    "core.agent": {"utils", "data", "core.world_model"},
    "training": {"utils", "data", "environment", "core"},
    "methods": {"utils", "data", "environment", "core", "training"},
    "inference": {"utils", "data", "environment", "core", "training", "methods"},
}


def _mirrors(relative: Path) -> bool:
    return (PACKAGE / relative).is_dir() or (PACKAGE / relative).with_suffix(".py").is_file()


def _directories(root: Path) -> list[Path]:
    return sorted(
        path.relative_to(root) for path in root.rglob("*") if path.is_dir() and "__pycache__" not in path.parts
    )


@pytest.mark.parametrize("relative", _directories(ROOT / "configs"), ids=str)
def test_config_directories_mirror_the_package(relative: Path) -> None:
    assert _mirrors(relative), f"configs/{relative} has no counterpart in src/rp1"


@pytest.mark.parametrize("relative", _directories(ROOT / "tests"), ids=str)
def test_test_directories_mirror_the_package(relative: Path) -> None:
    assert (PACKAGE / relative).is_dir(), f"tests/{relative} has no counterpart in src/rp1"


@pytest.mark.parametrize("path", sorted((ROOT / "tests").rglob("test_*.py")), ids=lambda path: path.name)
def test_test_files_mirror_a_module(path: Path) -> None:
    relative = path.relative_to(ROOT / "tests")
    if relative.parent == Path():
        return
    module = relative.parent / relative.name.removeprefix("test_").removesuffix(".py")
    assert _mirrors(module), f"tests/{relative} does not correspond to a module in src/rp1"


def _layer(module: str) -> str:
    parts = module.split(".")
    return ".".join(parts[:2]) if parts[0] == "core" else parts[0]


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return {name.removeprefix("rp1.") for name in names if name == "rp1" or name.startswith("rp1.")}


@pytest.mark.parametrize("path", sorted(PACKAGE.rglob("*.py")), ids=lambda path: str(path.relative_to(PACKAGE)))
def test_imports_point_one_way(path: Path) -> None:
    parts = [part for part in path.relative_to(PACKAGE).with_suffix("").parts if part != "__init__"]
    if not parts or parts == ["core"]:
        return
    relative = Path(*parts)
    layer = _layer(".".join(parts))
    allowed = ALLOWED[layer] | {layer}
    for imported in _imports(path):
        target = _layer(imported)
        assert any(target == name or target.startswith(name + ".") for name in allowed), (
            f"{relative} ({layer}) imports rp1.{imported}"
        )
