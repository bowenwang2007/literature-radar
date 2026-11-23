"""Assert every runtime resource ships with the package.

The product must not depend on the source checkout happening to contain a file.
``pyproject.toml`` declares what the wheel carries; this script walks the real
runtime roots and fails when a file is not covered by that declaration.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\check_packaging.py [--json]
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Directories and files the shipped product resolves at runtime.
RUNTIME_ROOTS = ("migrations", "prompts", "manifests", "config")
RUNTIME_FILES = ("journals.yaml",)

#: Checkout-only by design: user data and files a user is expected to create.
CHECKOUT_ONLY = (
    "config/personal_journals.json",
)


def _force_include() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {})
    return dict(wheel.get("wheel", {}).get("force-include", {}))


def _covered(relative: str, entries: dict[str, str]) -> str | None:
    """Return the force-include entry that ships ``relative``, if any."""
    for source, target in entries.items():
        if relative == source or relative.startswith(f"{source.rstrip('/')}/"):
            return f"{source} -> {target}"
    return None


def _runtime_files() -> list[str]:
    found: list[str] = []
    for name in RUNTIME_FILES:
        if (ROOT / name).is_file():
            found.append(name)
    for root in RUNTIME_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if "__pycache__" in relative or relative.endswith(".pyc"):
                continue
            # Design fixtures and readmes are documentation and test data; the
            # product never resolves them (the same rule runtime_inventory uses).
            if "/fixtures/" in relative or path.name.casefold() == "readme.md":
                continue
            if relative in CHECKOUT_ONLY:
                continue
            found.append(relative)
    return found


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    entries = _force_include()
    files = _runtime_files()
    missing = [item for item in files if _covered(item, entries) is None]
    stale = [
        source
        for source in entries
        if not (ROOT / source).exists()
    ]
    report = {
        "runtime_files": len(files),
        "force_include_entries": len(entries),
        "unpackaged_runtime_files": missing,
        "force_include_missing_on_disk": stale,
    }
    if "--json" in args:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"runtime files          : {len(files)}")
        print(f"force-include entries  : {len(entries)}")
        print(f"unpackaged runtime files: {len(missing)}")
        for item in missing:
            print(f"  ! {item}")
        print(f"declared but absent on disk: {len(stale)}")
        for item in stale:
            print(f"  ! {item}")
    return 1 if missing or stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
