#!/usr/bin/env python3
"""Architecture governance gate AG012: birth-certificate & subgraph closure.

[Role]
  Enforce 100% registry completeness across every AST-declared ``OpSpec``
  ``family``, every ``(family, recipe)`` birth-certificate entry in
  ``mindie_sim.structure_implementations``, and every ``declare_group``
  semantic subgraph role in ``mindie_sim.subgraph_math.ROLE_FAMILIES``.

[Invariants]
  * Every AST-declared ``family`` literal under ``mindie_sim/**/*.py`` MUST
    resolve in ``_FAMILY_REGISTRY`` with valid ``block``, non-banned
    ``submodule`` (``"Other"`` forbidden), valid ``backend_stack``, valid
    ``readiness``, non-empty ``source_ref``, and non-empty ``native_kinds``
    for all executed (non-``view_reshape``) operators.
  * Every ``declare_group(..., role=...)`` string literal MUST be registered
    in ``subgraph_math.ROLE_FAMILIES``.
  * ``mla_structure.py:declare_indexer`` MUST NOT gate subgraph emission on
    ``parallel.cp != 1``.

[Timeouts & Locks]
  Bounded local AST scan over ``mindie_sim/**/*.py`` (< 1s); no locks or I/O
  beyond read-only source inspection.

[Side-Effect Policy]
  Read-only governance scanner; never mutates repository files or git state.
"""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
from pathlib import Path, PurePosixPath
import sys

if __package__:
    from ._arch_governance_runtime import Violation
else:
    from _arch_governance_runtime import Violation

RULE_ID = "AG012"
FINGERPRINT_HEX_LEN = 16
EXIT_OK = 0
EXIT_VIOLATIONS = 1
_MIN_OPSPEC_POSITIONAL_ARGS = 2
_SIM_DIR = "mindie_sim"
_IMPL_REL = PurePosixPath("mindie_sim/structure_implementations.py")
_SUBGRAPH_MATH_REL = PurePosixPath("mindie_sim/subgraph_math.py")
_MLA_STRUCTURE_REL = PurePosixPath("mindie_sim/prediction/operators/mla/mla_structure.py")


def _fingerprint(tag: str) -> str:
    return sha256(f"{RULE_ID}:{tag}".encode("utf-8")).hexdigest()[:FINGERPRINT_HEX_LEN]


def _string_literals(expr: ast.AST) -> list[tuple[str, int]]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str) and expr.value:
        return [(expr.value, expr.lineno)]
    if isinstance(expr, ast.IfExp):
        return _string_literals(expr.body) + _string_literals(expr.orelse)
    return []


def _scan_ast_declarations(root: Path) -> tuple[dict[str, tuple[PurePosixPath, int]],
                                                dict[str, tuple[PurePosixPath, int]]]:
    """Return first ``(path, line)`` for every AST ``family`` and ``declare_group`` ``role``."""
    sim_root = root / _SIM_DIR
    families: dict[str, tuple[PurePosixPath, int]] = {}
    group_roles: dict[str, tuple[PurePosixPath, int]] = {}
    for path in sorted(sim_root.rglob("*.py")):
        rel = PurePosixPath(path.relative_to(root).as_posix())
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn_name = (node.func.id if isinstance(node.func, ast.Name)
                       else node.func.attr if isinstance(node.func, ast.Attribute) else "")
            if fn_name == "OpSpec" and len(node.args) >= _MIN_OPSPEC_POSITIONAL_ARGS:
                for val, line in _string_literals(node.args[1]):
                    families.setdefault(val, (rel, line))
            for kw in node.keywords:
                if kw.arg == "family" and fn_name != "declare_group":
                    for val, line in _string_literals(kw.value):
                        families.setdefault(val, (rel, line))
                elif kw.arg == "role" and fn_name == "declare_group":
                    for val, line in _string_literals(kw.value):
                        group_roles.setdefault(val, (rel, line))
    return families, group_roles


def _check_registry_specs(
    registry_name: str,
    entries: dict,
    *,
    valid_blocks: frozenset[str],
    valid_stacks: frozenset[str],
    valid_readiness: frozenset[str],
    banned_submodules: frozenset[str],
) -> list[Violation]:
    violations: list[Violation] = []
    for key, spec in sorted(entries.items(), key=lambda kv: str(kv[0])):
        label = f"{registry_name}[{key!r}]"
        if spec.block not in valid_blocks:
            violations.append(Violation(
                RULE_ID, _IMPL_REL, 1, _fingerprint(f"block:{label}"),
                f"{label} declares invalid block {spec.block!r}"))
        if str(spec.submodule or "").strip().lower() in banned_submodules:
            violations.append(Violation(
                RULE_ID, _IMPL_REL, 1, _fingerprint(f"submodule:{label}"),
                f"{label} declares banned submodule {spec.submodule!r}"))
        if spec.backend_stack not in valid_stacks:
            violations.append(Violation(
                RULE_ID, _IMPL_REL, 1, _fingerprint(f"stack:{label}"),
                f"{label} declares invalid backend_stack {spec.backend_stack!r}"))
        if spec.readiness not in valid_readiness:
            violations.append(Violation(
                RULE_ID, _IMPL_REL, 1, _fingerprint(f"readiness:{label}"),
                f"{label} declares invalid readiness {spec.readiness!r}"))
        if not str(spec.source_ref or "").strip():
            violations.append(Violation(
                RULE_ID, _IMPL_REL, 1, _fingerprint(f"source_ref:{label}"),
                f"{label} is missing source_ref"))
        if spec.backend_stack != "view_reshape" and not spec.native_kinds:
            violations.append(Violation(
                RULE_ID, _IMPL_REL, 1, _fingerprint(f"native_kinds:{label}"),
                f"{label} executed op has empty native_kinds"))
    return violations


def _check_unconditional_indexer_declaration(root: Path) -> list[Violation]:
    """Ensure ``mla_structure.py:declare_indexer`` never early-returns on ``parallel.cp``."""
    target = root / _MLA_STRUCTURE_REL
    if not target.is_file():
        return []
    tree = ast.parse(target.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "declare_indexer":
            continue
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.If):
                continue
            has_bare_return = any(isinstance(sub, ast.Return) and sub.value is None
                                  for sub in stmt.body)
            if has_bare_return and "parallel.cp" in ast.unparse(stmt.test):
                return [Violation(
                    RULE_ID, _MLA_STRUCTURE_REL, stmt.lineno,
                    _fingerprint("mla_structure:cp_early_return"),
                    "declare_indexer must not early-return on parallel.cp",
                )]
    return []


def check_ag012(root: Path) -> list[Violation]:
    """Run AG012 registry completeness and subgraph closure governance checks."""
    resolved = root.resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    from mindie_sim import structure_implementations as si
    from mindie_sim import subgraph_math as sm

    violations: list[Violation] = []
    families, group_roles = _scan_ast_declarations(resolved)
    for fam, (rel, lineno) in sorted(families.items()):
        if fam not in si._FAMILY_REGISTRY:
            violations.append(Violation(
                RULE_ID, rel, lineno, _fingerprint(f"unregistered_family:{fam}"),
                f"AST family {fam!r} is missing from structure_implementations._FAMILY_REGISTRY",
            ))
    for role, (rel, lineno) in sorted(group_roles.items()):
        if role not in sm.ROLE_FAMILIES:
            violations.append(Violation(
                RULE_ID, _SUBGRAPH_MATH_REL, lineno, _fingerprint(f"unregistered_role:{role}"),
                f"declare_group role {role!r} ({rel}:{lineno}) is missing from subgraph_math.ROLE_FAMILIES",
            ))
    violations.extend(_check_registry_specs(
        "_FAMILY_REGISTRY", si._FAMILY_REGISTRY,
        valid_blocks=si.VALID_BLOCKS, valid_stacks=si.VALID_BACKEND_STACKS,
        valid_readiness=si.VALID_READINESS, banned_submodules=si.BANNED_SUBMODULES,
    ))
    violations.extend(_check_registry_specs(
        "_RECIPE_REGISTRY", si._RECIPE_REGISTRY,
        valid_blocks=si.VALID_BLOCKS, valid_stacks=si.VALID_BACKEND_STACKS,
        valid_readiness=si.VALID_READINESS, banned_submodules=si.BANNED_SUBMODULES,
    ))
    violations.extend(_check_unconditional_indexer_declaration(resolved))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AG012 birth-certificate & subgraph closure gate")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    violations = check_ag012(args.root)
    for v in violations:
        print(f"{v.rule} {v.path}:{v.line} [{v.fingerprint}] {v.message}")
    print(f"AG012 governance violations: {len(violations)}")
    return EXIT_OK if not violations else EXIT_VIOLATIONS


if __name__ == "__main__":
    raise SystemExit(main())
