#!/usr/bin/env python3
"""Differential numeric-literal gate for ``mindie_sim``.

[Role]
  Compare the working tree against a base commit and report numeric literals
  introduced by changed or new code.  Natural mathematics, structural indices
  (subscripts, slices, ``len`` comparisons and ``round`` digits) and genuine
  module/class-scope UPPER_SNAKE_CASE domain constants are accepted.  Numeric
  word aliases (``ZERO = 0``, ``_SIXTY_FOUR = 64``, ``THREE_FLOAT = 3.0``)
  are never accepted: they hide missing dynamic formulas behind a name.

[Invariants]
  * Booleans, ``None`` and strings are never numeric literals.
  * A unary +/- applied to a numeric constant is one literal occurrence.
  * Tautological number-word assignments are always reported, even when the
    assigned value is a natural literal, and even inside a function body.
  * An allowed UPPER_SNAKE_CASE constant must be a direct module/class-body
    assignment whose name carries a real domain noun.
  * Statement identity is value-masked AST plus enclosing scope, so a moved
    literal is new code while an untouched statement is grandfathered.
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
import re
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

# Mathematical identities, sign, parity / FMA factor, half, and the standard
# decimal / binary unit-conversion bases.  Floating-point tolerances in the
# 1e-6 .. 1e-12 decade are structural comparison aids, not physics constants.
NATURAL_INT_LITERALS = frozenset({
    -2, -1, 0, 1, 2, 100, 1000, 1024, 1_000_000, 1_000_000_000,
})
NATURAL_FLOAT_LITERALS = frozenset({
    -1.0, -0.5, 0.0, 0.5, 1.0, 2.0,
    100.0, 1000.0, 1024.0, 1_000_000.0, 1_000_000_000.0,
    1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12,
})

_TAUTOLOGICAL_TOKENS = frozenset({
    # Cardinal words
    "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
    "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN",
    "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY", "THIRTY",
    "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY", "HUNDRED",
    "THOUSAND", "MILLION", "BILLION",
    # Fractional / tuple-count words without a domain noun
    "HALF", "QUARTER", "THIRD", "THIRDS",
    # Ordinal / positional words without a domain noun
    "FIRST", "SECOND", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH",
    "NINTH", "TENTH", "LAST",
    # Sign / type / filler tokens
    "NEG", "NEGATIVE", "POS", "POSITIVE", "MINUS", "PLUS",
    "FLOAT", "INT", "INTEGER", "F", "I", "VAL", "VALUE",
    "NUM", "NUMBER", "NUMERIC", "CONST", "CONSTANT", "LITERAL", "LIT",
    "NONE",
})
_TAUTOLOGICAL_TRAILING_WORDS = frozenset({
    "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
    "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN",
    "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY", "THIRTY",
    "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY", "HUNDRED",
    "THOUSAND", "MILLION", "BILLION", "HALF", "QUARTER", "THIRD", "THIRDS",
    "FIRST", "SECOND", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH",
    "NINTH", "TENTH", "LAST",
})
_DIGIT_TOKEN_RE = re.compile(r"^[FNI]?\d+(?:P\d+)?$")
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class Occurrence:
    value: object
    key: str
    line: int
    path: str
    context: str
    kind: str = "literal"
    name: str = ""


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


def _is_tautological_name(name: str) -> bool:
    tokens = [tok for tok in name.strip("_").split("_") if tok]
    if not tokens:
        return False
    if all(
        tok in _TAUTOLOGICAL_TOKENS or bool(_DIGIT_TOKEN_RE.match(tok))
        for tok in tokens
    ):
        return True
    # A number word in final position after a bare feature abbreviation
    # (``_MTP_THREE``) carries no domain meaning.  Genuine unit nouns
    # (``MS_PER_SECOND``, ``PARTS_PER_MILLION``) and domain-qualified names
    # (``DEPTH_ONE``, ``FOUR_KIB``) remain legitimate.
    if tokens[-1] in _TAUTOLOGICAL_TRAILING_WORDS and "PER" not in tokens:
        return any(tok in {"MTP"} for tok in tokens[:-1])
    return False


def _is_natural_literal(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if type(value) is int:
        return value in NATURAL_INT_LITERALS
    if type(value) is float:
        return value in NATURAL_FLOAT_LITERALS
    return False


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


def _scope_key(node: ast.stmt, parents: dict[ast.AST, ast.AST]) -> str:
    names: list[str] = []
    current: ast.AST | None = parents.get(node)
    while current is not None and not isinstance(current, ast.Module):
        if isinstance(current, _SCOPE_NODES):
            names.append(f"{type(current).__name__}:{current.name}")
        current = parents.get(current)
    return "/".join(reversed(names))


def _assignment_target_names(node: ast.stmt) -> list[str] | None:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
        return None
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if not isinstance(target, ast.Name):
            return None
        names.append(target.id)
    return names or None


def _statement_is_tautological_const(node: ast.stmt) -> bool:
    names = _assignment_target_names(node)
    if not names:
        return False
    return any(name.isupper() and _is_tautological_name(name) for name in names)


def _statement_is_allowed(
    node: ast.stmt, parents: dict[ast.AST, ast.AST]
) -> bool:
    scope = parents.get(node)
    if not isinstance(scope, (ast.Module, ast.ClassDef)):
        return False
    names = _assignment_target_names(node)
    if not names:
        return False
    return all(
        name and name.isupper() and not _is_tautological_name(name)
        for name in names
    )


def _in_structural_ast_position(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    """Exempt slice/subscript indices, len(...) comparisons, and round(..., n)."""
    cur = node
    parent = parents.get(cur)
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, (ast.USub, ast.UAdd)):
        cur = parent
        parent = parents.get(cur)
    # 1) Subscript index or slice bound: x[0], x[-1], shape[-2:], hit[3]
    anc = parent
    while anc is not None and not isinstance(anc, ast.stmt):
        if isinstance(anc, ast.Subscript):
            probe: ast.AST | None = cur
            while probe is not None and probe is not anc:
                if probe is anc.slice:
                    return True
                probe = parents.get(probe)
        anc = parents.get(anc)
    # 2) Comparison against len(...), including chained comparisons.
    anc = parent
    while anc is not None and not isinstance(anc, ast.stmt):
        if isinstance(anc, ast.Compare):
            operands = [anc.left, *anc.comparators]
            if any(
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Name)
                and item.func.id == "len"
                for item in operands
            ):
                return True
        anc = parents.get(anc)
    # 3) round(value, ndigits)
    if (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and parent.func.id == "round"
        and len(parent.args) == 2
        and parent.args[1] is cur
    ):
        return True
    return False


def _value_key(value: object) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _numeric_occurrences(
    tree: ast.Module,
) -> list[tuple[ast.stmt, object, int, ast.AST, dict[ast.AST, ast.AST]]]:
    parents = _parents(tree)
    result: list[tuple[ast.stmt, object, int, ast.AST, dict[ast.AST, ast.AST]]] = []
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
        result.append((owner, value, line, node, parents))
    return result


def _occurrence_key(
    owner: ast.stmt, node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> str:
    key = _masked_statement(owner)
    if _assignment_target_names(owner):
        key = f"{key}@{_scope_key(owner, parents)}"
    return key


def _collect(tree: ast.Module, path: str) -> list[Occurrence]:
    occurrences: list[Occurrence] = []
    for owner, value, line, node, parents in _numeric_occurrences(tree):
        names = _assignment_target_names(owner) or []
        tautological = _statement_is_tautological_const(owner)
        if tautological:
            flagged = True
            kind = "tautological"
            name = next(n for n in names if n.isupper() and _is_tautological_name(n))
        elif _statement_is_allowed(owner, parents):
            flagged = False
            kind = "constant"
            name = ""
        else:
            flagged = not (
                _is_natural_literal(value)
                or _in_structural_ast_position(node, parents)
            )
            kind = "literal"
            name = ""
        if not flagged:
            continue
        occurrences.append(Occurrence(
            value,
            _occurrence_key(owner, node, parents),
            line,
            path,
            _statement_context(owner, path),
            kind,
            name,
        ))
    return occurrences


def _grandfathered(tree: ast.Module) -> Counter:
    counts: Counter = Counter()
    for owner, value, _line, node, parents in _numeric_occurrences(tree):
        if _statement_is_allowed(owner, parents):
            continue
        counts[(_occurrence_key(owner, node, parents), _value_key(value))] += 1
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
        occurrences = _collect(tree, name)
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
        if item.kind == "tautological":
            detail = f"tautological numeric constant {item.name} = {item.value!r}"
        else:
            detail = f"new numeric literal {item.value!r}"
        buffer.write(
            f"{item.path}:{item.line}: {detail} "
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
