"""Pure AST extractors for operator/template contract governance checks.

[Role] Inspect Python ASTs of governed operator packages and templates to
  extract ``Upstream-ABI:`` module headers, top-level string constants,
  declared ``TEMPLATE_ID``/``CATLASS_TEMPLATE_IDS``/``DSL_KERNEL_IDS``,
  ``SourceRef``/``TemplateSourceRef`` provenance declarations, symbol mentions,
  top-level exports, and unimplemented ``NotImplementedError`` stubs.
[Invariants] Strictly static AST inspection with zero code execution, zero
  filesystem writes, and deterministic extraction across all platforms.
[Timeouts & Locks] Pure in-memory AST walk; zero I/O, locks, or device contexts.
[Side-Effect Policy] Zero side effects.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
import re

_UPSTREAM_ABI_LINE_RE = re.compile(r"^\s*Upstream-ABI\s*:\s*([A-Za-z_][A-Za-z0-9_:.<>/-]*)\s*$")
_TEMPLATE_ID_CONST_NAMES = frozenset({"TEMPLATE_ID", "TEMPLATE_IDS", "CATLASS_TEMPLATE_IDS", "DSL_KERNEL_IDS"})


def extract_upstream_abi(tree: ast.AST) -> str:
    """Return the first ``Upstream-ABI: <symbol>`` declared in the module docstring."""
    doc = ast.get_docstring(tree) or ""
    for line in doc.splitlines():
        match = _UPSTREAM_ABI_LINE_RE.match(line)
        if match:
            return match.group(1).strip()
    return ""


def eval_str_expr(node: ast.AST | None, constants: Mapping[str, str]) -> str | None:
    """Statically resolve constant string literals, name refs, and ``+`` joins."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = eval_str_expr(node.left, constants), eval_str_expr(node.right, constants)
        if left is not None and right is not None:
            return left + right
    return None


def extract_module_str_constants(tree: ast.Module) -> dict[str, str]:
    """Extract top-level string assignments from an AST module."""
    constants: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            val = eval_str_expr(stmt.value, constants)
            if val is not None:
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = val
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            val = eval_str_expr(stmt.value, constants)
            if val is not None:
                constants[stmt.target.id] = val
    return constants


def extract_declared_template_ids(tree: ast.Module, constants: Mapping[str, str]) -> set[str]:
    """Extract declared template identifier strings from recognized top-level constants."""
    found: set[str] = set()
    for stmt in tree.body:
        names: list[str] = []
        val_node: ast.AST | None = None
        if isinstance(stmt, ast.Assign):
            names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            val_node = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names = [stmt.target.id]
            val_node = stmt.value
        if not any(n in _TEMPLATE_ID_CONST_NAMES for n in names) or val_node is None:
            continue
        single = eval_str_expr(val_node, constants)
        if single:
            found.add(single)
        elif isinstance(val_node, (ast.Tuple, ast.List, ast.Set)):
            for elt in val_node.elts:
                item = eval_str_expr(elt, constants)
                if item:
                    found.add(item)
        elif isinstance(val_node, ast.Dict):
            for key in val_node.keys:
                item = eval_str_expr(key, constants)
                if item:
                    found.add(item)
    return found


def extract_source_refs(tree: ast.Module, path: Path, constants: Mapping[str, str]) -> list[dict[str, str]]:
    """Extract ``SourceRef(...)`` and ``TemplateSourceRef(...)`` call records."""
    refs: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        fname = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        if fname not in ("SourceRef", "TemplateSourceRef"):
            continue
        pos = [eval_str_expr(arg, constants) for arg in node.args]
        kw = {str(k.arg): eval_str_expr(k.value, constants) for k in node.keywords if k.arg}
        refs.append({
            "file": path.name,
            "repository": kw.get("repository") or (pos[0] if len(pos) > 0 else None) or "",
            "revision": kw.get("revision") or (pos[1] if len(pos) > 1 else None) or "",
            "relative_path": kw.get("relative_path") or (pos[2] if len(pos) > 2 else None) or "",
            "source_sha256": kw.get("source_sha256") or (pos[3] if len(pos) > 3 else None) or "",
        })
    return refs


def extract_symbol_mentions(tree: ast.AST) -> set[str]:
    """Collect all identifier, attribute, and imported symbol names in ``tree``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
    return names


def find_unimplemented_stubs(tree: ast.Module, path: Path) -> list[str]:
    """Detect any function or method raising ``NotImplementedError`` in target sources."""
    stubs: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Raise) or inner.exc is None:
                continue
            exc = inner.exc
            name = (
                exc.id if isinstance(exc, ast.Name)
                else (exc.func.id if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) else "")
            )
            if name == "NotImplementedError":
                stubs.append(f"operator_target_unimplemented_stub:{path.name}:{node.name}")
                break
    return stubs


def extract_top_level_names(tree: ast.Module) -> set[str]:
    """Return all top-level assigned names, functions, and classes in ``tree``."""
    names: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
    return names


__all__ = [
    "eval_str_expr",
    "extract_declared_template_ids",
    "extract_module_str_constants",
    "extract_source_refs",
    "extract_symbol_mentions",
    "extract_top_level_names",
    "extract_upstream_abi",
    "find_unimplemented_stubs",
]
