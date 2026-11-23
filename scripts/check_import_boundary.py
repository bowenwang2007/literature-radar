"""Assert the product startup closure contains only product-core modules.

The app once dragged the whole research/governance stack into every Streamlit
process through a single historical import (``ui/app.py`` ->
``services/personal_mode`` -> ``prototype_workflow`` -> ``pilot_runtime`` ->
``domain.*``).  Nothing in the UI called any of it.  This checker makes the
boundary mechanical: it builds the import graph with ``ast`` (no import of the
package, so it cannot touch the database or the network) and reports the
closure rooted at the shipped entry point.

Two closures are distinguished because they mean different things:

* ``startup``  - module-level imports only: what a plain ``streamlit run``
  actually loads before any user action.
* ``full``     - every import, including function-level ones: what the process
  can still load when a feature is explicitly triggered.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\check_import_boundary.py
    .\\.venv\\Scripts\\python.exe scripts\\check_import_boundary.py --json
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "eduecon_radar"
UI_ROOT = "eduecon_radar.ui.app"

#: Modules that must never be loaded by the ordinary product startup.  They are
#: kept in the repository (CLI / research / historical use) but are not part of
#: the product core.
FORBIDDEN_PREFIXES = (
    "eduecon_radar.domain",
    "eduecon_radar.services.pilot_runtime",
    "eduecon_radar.services.runtime_bundle",
    "eduecon_radar.services.prototype_workflow",
    "eduecon_radar.services.workflow",
    "eduecon_radar.services.workflow_resume",
    "eduecon_radar.services.task_lock",
    "eduecon_radar.services.task_review",
    "eduecon_radar.services.semantic_screening",
    "eduecon_radar.services.semantic_smoke",
    "eduecon_radar.services.gold_",
    "eduecon_radar.services.validation_",
    "eduecon_radar.services.migration_integrity",
    "eduecon_radar.services.schema_manifest",
    "eduecon_radar.services.runtime_doctor",
    "eduecon_radar.services.screening",  # any future name collision guard
    "eduecon_radar.services.ai_screening",
    "eduecon_radar.services.personal_mode",
    "eduecon_radar.services.archive",
)


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT / "src").with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _targets(node: ast.Import | ast.ImportFrom) -> list[str]:
    found: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith("eduecon_radar"):
                found.append(alias.name)
    elif not node.level and node.module and node.module.startswith("eduecon_radar"):
        found.append(node.module)
        for alias in node.names:
            found.append(f"{node.module}.{alias.name}")
    return found


def collect_graph() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """Return (module_level_edges, runtime_lazy_edges, typing_only_edges)."""
    modules = {module_name(path): path for path in PKG.rglob("*.py")}
    top: dict[str, set[str]] = {}
    every: dict[str, set[str]] = {}
    typing_only: dict[str, set[str]] = {}

    def resolve(target: str) -> str | None:
        candidate = target
        while candidate and candidate not in modules:
            if "." not in candidate:
                return None
            candidate = candidate.rsplit(".", 1)[0]
        return candidate or None

    def is_type_checking(node: ast.If) -> bool:
        test = node.test
        if isinstance(test, ast.Name):
            return test.id == "TYPE_CHECKING"
        if isinstance(test, ast.Attribute):
            return test.attr == "TYPE_CHECKING"
        return False

    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        top_edges: set[str] = set()
        typing_edges: set[str] = set()
        deferred_edges: set[str] = set()

        def add(
            node: ast.Import | ast.ImportFrom,
            bucket: set[str],
            *,
            owner: str = name,
        ) -> None:
            for target in _targets(node):
                resolved = resolve(target)
                if resolved and resolved != owner:
                    bucket.add(resolved)

        for node in tree.body:  # module level
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                add(node, top_edges)
            elif isinstance(node, ast.If) and is_type_checking(node):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        add(child, typing_edges)
            else:
                # Function bodies, lazy imports and guarded import blocks: not
                # executed by a plain module import, but reachable at runtime.
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        add(child, deferred_edges)
        top[name] = top_edges
        every[name] = top_edges | deferred_edges
        typing_only[name] = typing_edges - every[name]
    return top, every, typing_only


def closure(root: str, graph: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(graph.get(current, ()))
    return seen


def forbidden_hits(modules: set[str]) -> list[str]:
    return sorted(
        name
        for name in modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_PREFIXES
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    as_json = "--json" in args
    top_graph, all_graph, typing_graph = collect_graph()
    startup = closure(UI_ROOT, top_graph)
    # Runtime reachability ignores ``if TYPE_CHECKING`` imports, which never
    # execute.  What remains is what a triggered feature can still load.
    runtime_graph = {
        name: edges - typing_graph.get(name, set()) for name, edges in all_graph.items()
    }
    full = closure(UI_ROOT, runtime_graph)
    startup_bad = forbidden_hits(startup)
    lazy_bad = sorted(set(forbidden_hits(full)) - set(startup_bad))

    if as_json:
        print(
            json.dumps(
                {
                    "root": UI_ROOT,
                    "startup_modules": len(startup),
                    "runtime_modules": len(full),
                    "startup_forbidden": startup_bad,
                    "lazy_forbidden": lazy_bad,
                    "startup_closure": sorted(startup),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"root: {UI_ROOT}")
        print(f"startup closure: {len(startup)} modules")
        print(f"runtime closure: {len(full)} modules (TYPE_CHECKING imports excluded)")
        print(f"forbidden in startup: {len(startup_bad)}")
        for name in startup_bad:
            print(f"  ! {name}")
        print(f"forbidden reachable only behind an explicit feature: {len(lazy_bad)}")
        for name in lazy_bad:
            print(f"  ~ {name}")
    return 1 if startup_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
