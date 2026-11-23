"""The product core must stay free of research/governance/historical modules.

These tests encode the boundary that the clean-core refactor established.  They
are deliberately import-graph based (AST) rather than file-name based, and one
of them runs a real subprocess import so "the app does not load the domain
stack" is measured instead of asserted.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = SRC / "eduecon_radar"
UI_ROOT = "eduecon_radar.ui.app"

FORBIDDEN_PREFIXES = (
    "eduecon_radar.domain",
    "eduecon_radar.services.pilot_runtime",
    "eduecon_radar.services.runtime_bundle",
    "eduecon_radar.services.prototype_workflow",
    "eduecon_radar.services.workflow",
    "eduecon_radar.services.task_lock",
    "eduecon_radar.services.task_review",
    "eduecon_radar.services.semantic_screening",
    "eduecon_radar.services.gold_governance",
    "eduecon_radar.services.gold_pilot_governance",
    "eduecon_radar.services.validation_batch",
    "eduecon_radar.services.ai_screening",
    "eduecon_radar.services.personal_mode",
    "eduecon_radar.services.ingestion",
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _targets(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names if alias.name.startswith("eduecon_radar")]
    if node.level or not node.module or not node.module.startswith("eduecon_radar"):
        return []
    return [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]


def _module_level_edges() -> dict[str, set[str]]:
    modules = {_module_name(path): path for path in PKG.rglob("*.py")}

    def resolve(target: str) -> str | None:
        candidate = target
        while candidate and candidate not in modules:
            if "." not in candidate:
                return None
            candidate = candidate.rsplit(".", 1)[0]
        return candidate or None

    edges: dict[str, set[str]] = {}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        found: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for target in _targets(node):
                    resolved = resolve(target)
                    if resolved and resolved != name:
                        found.add(resolved)
        edges[name] = found
    return edges


def _forbidden(modules: set[str]) -> list[str]:
    return sorted(
        name
        for name in modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_PREFIXES
        )
    )


def test_ui_startup_closure_excludes_research_and_legacy_modules() -> None:
    edges = _module_level_edges()
    seen: set[str] = set()
    pending = [UI_ROOT]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(edges.get(current, ()))
    assert _forbidden(seen) == []


def test_ui_sources_never_reference_legacy_modules() -> None:
    ui_files = sorted((PKG / "ui").rglob("*.py"))
    assert ui_files
    offenders: dict[str, list[str]] = {}
    for path in ui_files:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(_targets(node))
        hits = sorted(name for name in names if _forbidden({name}))
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits
    assert offenders == {}


def test_product_entry_import_does_not_load_domain_or_legacy_stack() -> None:
    """Import the shipped entry point in a clean interpreter and inspect sys.modules."""
    script = (
        "import sys; import eduecon_radar.ui.app; "
        "mods = sorted(m for m in sys.modules if m.startswith('eduecon_radar')); "
        "print('\\n'.join(mods))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    loaded = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    assert loaded, "entry point imported nothing?"
    assert "eduecon_radar.ui.app" in loaded
    assert _forbidden(loaded) == []
