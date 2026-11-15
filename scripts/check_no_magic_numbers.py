#!/usr/bin/env python3
"""Differential numeric-literal gate for ``mindie_sim``.

[Role]
  Compare the working tree against a base commit and report numeric literals
  introduced by changed or new code.  A literal is acceptable only when it is
  the value of a module-level UPPER_SNAKE_CASE assignment (or annotated
  assignment); pre-existing literals in untouched statements are grandfathered
  by structural statement identity rather than by line number.

[Invariants]
  * Booleans, ``None`` and strings are never numeric literals.
  * A unary +/- applied to a numeric constant is one literal occurrence.
  * A statement's numeric occurrences are matched by a value-masked AST dump;
    only additional occurrences/values in the same statement shape are new.
  * Nested defaults, subscripts, comprehensions, annotations and call
    arguments are inspected; only top-level named constants are exempt.
  * Untracked implementation files are part of the comparison.

[Timeouts & Locks]
  Bounded subprocess calls to git; no background work or locks.

[Side-Effect Policy]
  Read-only: never writes product files, index state or repository metadata.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import io
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

IMPLEMENTATION_ROOT = "mindie_sim"
PYTHON_SUFFIX = ".py"
NUMERIC_PLACEHOLDER = "__NUM__"
EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_USAGE = 2


@dataclass(frozen=True)
class Occurrence:
    value: object
    key: str
    line: int
    path: str
    context: str


def _is_numeric_node(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float, complex))
        and not isinstance(node.value, bool)
    )


def _statement_context(node: ast.stmt, path: str) -> str:
    snippet = ast.unparse(node).strip().splitlines()[0]
    return f"{path}: {snippet[:80]}"


class _Masker(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant):
        if _is_numeric_node(node):
            return ast.Constant(value=NUMERIC_PLACEHOLDER)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, (ast.USub, ast.UAdd)) and _is_numeric_node(node.operand):
            return ast.Constant(value=NUMERIC_PLACEHOLDER)
        return self.generic_visit(node)


def _masked_statement(node: ast.stmt) -> str:
    """Structural identity with numeric values and unary signs masked."""
    return ast.dump(_Masker().visit(node), include_attributes=False)


def _statement_is_allowed(node: ast.stmt) -> bool:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    if node.value is None:
        return False
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if not isinstance(target, ast.Name):
            return False
        names.append(target.id)
    if not names:
        return False
    return all(name and name.isupper() for name in names)


def _value_key(value: object) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    table: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            table[child] = parent
    return table


def _owner_statement(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.stmt | None:
    current: ast.AST | None = node
    while current is not None and not isinstance(current, ast.stmt):
        current = parents.get(current)
    return current if isinstance(current, ast.stmt) else None


def _numeric_occurrences(tree: ast.Module) -> list[tuple[ast.stmt, object, int]]:
    parents = _parents(tree)
    result: list[tuple[ast.stmt, object, int]] = []
    for node in ast.walk(tree):
        if not _is_numeric_node(node):
            continue
        owner = _owner_statement(node, parents)
        if owner is None:
            continue
        parent = parents.get(node)
        value = node.value
        line = node.lineno
        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, (ast.USub, ast.UAdd)):
            value = (-1 if isinstance(parent.op, ast.USub) else 1) * value
            line = parent.lineno
        result.append((owner, value, line))
    return result


def _collect(tree: ast.Module, path: str) -> tuple[list[Occurrence], set[str]]:
    occurrences: list[Occurrence] = []
    allowed: set[str] = set()
    for owner, value, line in _numeric_occurrences(tree):
        key = _masked_statement(owner)
        if _statement_is_allowed(owner):
            allowed.add(key)
            continue
        occurrences.append(Occurrence(value, key, line, path, _statement_context(owner, path)))
    return occurrences, allowed


def _grandfathered(tree: ast.Module) -> Counter:
    counts: Counter = Counter()
    for owner, value, _line in _numeric_occurrences(tree):
        if _statement_is_allowed(owner):
            continue
        counts[(_masked_statement(owner), _value_key(value))] += 1
    return counts


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "git " + " ".join(args) + " failed: " + completed.stderr.strip()
        )
    return completed.stdout


def _load_base_modules(root: Path, base: str) -> dict[str, ast.Module]:
    result: dict[str, ast.Module] = {}
    names = _git(root, "ls-tree", "-r", "--name-only", base, "--", IMPLEMENTATION_ROOT)
    for name in names.splitlines():
        if not name.endswith(PYTHON_SUFFIX):
            continue
        text = _git(root, "show", f"{base}:{name}")
        try:
            result[name] = ast.parse(text)
        except SyntaxError:
            continue
    return result


def _changed_files(root: Path, base: str) -> list[str]:
    diff = _git(root, "diff", "--name-only", base, "--", IMPLEMENTATION_ROOT)
    untracked = _git(
        root, "ls-files", "--others", "--exclude-standard", "--", IMPLEMENTATION_ROOT
    )
    names = {line for line in diff.splitlines() if line.endswith(PYTHON_SUFFIX)}
    names.update(line for line in untracked.splitlines() if line.endswith(PYTHON_SUFFIX))
    return sorted(names)


def check(root: Path, base: str) -> tuple[list[Occurrence], int]:
    base_modules = _load_base_modules(root, base)
    violations: list[Occurrence] = []
    checked = 0
    for name in _changed_files(root, base):
        path = root / name
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise RuntimeError(f"cannot parse {name}: {exc}") from exc
        checked += 1
        occurrences, _allowed = _collect(tree, name)
        remaining = _grandfathered(base_modules[name]) if name in base_modules else Counter()
        for item in occurrences:
            key = (item.key, _value_key(item.value))
            if remaining[key] > 0:
                remaining[key] -= 1
            else:
                violations.append(item)
    return violations, checked


def _render(violations: list[Occurrence]) -> str:
    buffer = io.StringIO()
    for item in violations:
        buffer.write(
            f"{item.path}:{item.line}: new numeric literal {item.value!r} "
            f"in {item.context.split(': ', 1)[-1]}\n"
        )
    buffer.write(f"new numeric literal violations: {len(violations)}\n")
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base commit to compare against")
    parser.add_argument("--root", default=".", help="repository root")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        violations, checked = check(root, args.base)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    sys.stdout.write(_render(violations))
    if checked:
        print(f"changed implementation files checked: {checked}")
    return EXIT_VIOLATIONS if violations else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
