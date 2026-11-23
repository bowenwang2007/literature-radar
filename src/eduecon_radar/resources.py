from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path


@dataclass(frozen=True)
class RuntimeResource:
    logical_name: str
    relative_path: str
    category: str = "runtime_required"


RESOURCE_ROOTS = ("migrations", "prompts", "config", "manifests")


def project_root() -> Path:
    """Locate the checkout for editable installs without relying on cwd."""
    return Path(__file__).resolve().parents[2]


def resource(relative_path: str) -> Traversable:
    clean = relative_path.replace("\\", "/").lstrip("/")
    packaged = files("eduecon_radar").joinpath("resources", *clean.split("/"))
    if packaged.is_file() or packaged.is_dir():
        return packaged
    checkout = project_root().joinpath(*clean.split("/"))
    if checkout.exists():
        return checkout
    raise FileNotFoundError(f"Runtime resource is missing: {clean}")


def read_text(relative_path: str) -> str:
    return resource(relative_path).read_text(encoding="utf-8")


def runtime_path(project_root: Path, relative_path: str) -> Path:
    """Resolve a runtime file, preferring a project-local override.

    A checkout may override a shipped file by placing it in the project — that
    is how the domain/profile configuration is customised — but nothing may
    depend on the checkout *existing*: when the project copy is absent the
    packaged resource is used, which is what makes a wheel install work.

    ``scripts/check_packaging.py`` asserts every runtime file is declared in the
    wheel's force-include list, so "it happens to be in my checkout" cannot pass
    unnoticed.
    """
    local = project_root.joinpath(*relative_path.split("/"))
    if local.is_file() or local.is_dir():
        return local
    return Path(str(resource(relative_path)))


def resource_sha256(relative_path: str) -> str:
    return hashlib.sha256(resource(relative_path).read_bytes()).hexdigest()


def runtime_inventory() -> tuple[RuntimeResource, ...]:
    items: list[RuntimeResource] = []
    for root in RESOURCE_ROOTS:
        candidate = resource(root)
        for entry in _walk(candidate, root):
            items.append(RuntimeResource(entry, entry))
    items.append(RuntimeResource("journals.yaml", "journals.yaml"))
    return tuple(sorted(items, key=lambda item: item.relative_path))


def _walk(node: Traversable, prefix: str) -> list[str]:
    result: list[str] = []
    for child in sorted(node.iterdir(), key=lambda item: item.name):
        relative = f"{prefix}/{child.name}"
        if child.is_dir():
            if child.name == "fixtures":
                continue
            result.extend(_walk(child, relative))
        elif child.is_file():
            if child.name.casefold() == "readme.md":
                continue
            result.append(relative)
    return result


def verify_runtime_resources() -> dict[str, object]:
    records = []
    for item in runtime_inventory():
        target = resource(item.relative_path)
        records.append(
            {
                "logical_name": item.logical_name,
                "resolved_package_path": str(target),
                "expected_hash": resource_sha256(item.relative_path),
                "actual_hash": hashlib.sha256(target.read_bytes()).hexdigest(),
                "status": "ok",
            }
        )
    return {
        "status": "ok" if records else "error",
        "resource_count": len(records),
        "resources": records,
    }
