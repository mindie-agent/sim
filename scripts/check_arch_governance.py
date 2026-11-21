#!/usr/bin/env python3
"""Architecture governance gate (AG001..AG007) for ``mindie_sim``.

[Role] Enforce governed ``mindie_sim/prediction/operators/<family>/`` package
  boundaries, forbid flat ISA/CANN filename sprawl under ``mindie_sim/**/*.py``,
  block direct leaf imports, block ``service_modes`` / ``compute_peaks``
  capability sniffing, verify ISA provenance, and check dispatched recipe
  ownership and physical work conservation.
[Invariants] ``AG001`` scans strictly ``mindie_sim/**/*.py`` so model catalog
  JSON files (e.g. ``qwen3.5-35b-a3b.json``) are never flagged. Governed
  ``operators/<family>/**`` modules allow zero violations in every mode.
  Differential mode (``--base``) subtracts violations by multiset fingerprint.
[Timeouts & Locks] Bounded local AST scans and git ls-tree/show calls; no locks
  or network access.
[Side-Effect Policy] Read-only; never modifies repository files or git state.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import keyword
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

if __package__:
    from ._arch_governance_runtime import Violation, scan_runtime, verify_ag007_discrimination
else:
    from _arch_governance_runtime import Violation, scan_runtime, verify_ag007_discrimination

SIM_PACKAGE_ROOT = "mindie_sim"
GOVERNED_LEAF_NAMES = frozenset({"arch22.py", "arch35.py"})
GOVERNED_LEAF_STEMS = frozenset({"arch22", "arch35"})
AG001_STEM_RE = re.compile(
    r"(?:arch22|arch35|cann|(?:^|_)(?:a2|a3|a5)(?:$|_))",
    re.IGNORECASE,
)
UPSTREAM_ABI_RE = re.compile(r"^\s*Upstream-ABI:\s*(\S+)", re.MULTILINE)
HEURISTIC_IDENTIFIER_RE = re.compile(
    r"(?:^|_)(?:fudge|discount|overlap_decay|heuristic_factor|latency_multiplier)(?:_|$)",
    re.IGNORECASE,
)
CAMEL_CASE_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
HEX_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
DOCSTRING_HEADER_MAX_LINE = 30
FINGERPRINT_HEX_LEN = 16
PREVIEW_SHA_LEN = 12
GOVERNED_PATH_DEPTH = 5
FAMILY_PKG_DEPTH = 4
OPERATORS_PARTS = ("mindie_sim", "prediction", "operators")
SOURCE_REF_FIELDS = ("repository", "revision", "relative_path", "source_sha256")
EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_USAGE = 2

_INFRA_PREFIXES = ("mindie_sim/schema/", "mindie_sim/backends/")
_SM_INFRA_FILES = frozenset({
    "mindie_sim/custom_hardware.py",
    "mindie_sim/hardware_derivation.py",
    "mindie_sim/prediction/evaluation.py",
    "mindie_sim/prediction/execution.py",
    "mindie_sim/prediction/program.py",
})
_CP_INFRA_FILES = _SM_INFRA_FILES | frozenset({
    "mindie_sim/public/services.py",
    "mindie_sim/structure.py",
    "mindie_sim/prediction/cost.py",
    "mindie_sim/prediction/efficiency.py",
    "mindie_sim/prediction/matrix_precision.py",
})

# All flat ISA/CANN sidecars and AG003/AG004 legacy debt have been retired;
# strict mode requires zero allowlist entries.
LEGACY_ALLOWLIST: dict[str, frozenset[str]] = {}


def _is_governed_family_path(rel: PurePosixPath) -> tuple[bool, str | None, str]:
    parts = rel.parts
    if len(parts) >= GOVERNED_PATH_DEPTH and parts[: len(OPERATORS_PARTS)] == OPERATORS_PARTS:
        return True, parts[len(OPERATORS_PARTS)], rel.name
    return False, None, rel.name


def _is_allowed_isa_leaf(rel: PurePosixPath) -> bool:
    in_pkg, family, filename = _is_governed_family_path(rel)
    return (
        in_pkg and len(rel.parts) == GOVERNED_PATH_DEPTH
        and family is not None
        and filename in GOVERNED_LEAF_NAMES
        and family.isidentifier()
        and not keyword.iskeyword(family)
        and not AG001_STEM_RE.search(family)
    )


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _is_git_repo(root: Path) -> bool:
    done = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return done.returncode == 0 and done.stdout.strip() == "true"


def _enumerate_py_files(
    root: Path, *, revision: str | None = None
) -> list[tuple[PurePosixPath, str]]:
    entries: list[tuple[PurePosixPath, str]] = []
    if revision is not None:
        names = _git(root, "ls-tree", "-r", "--name-only", revision, "--", SIM_PACKAGE_ROOT)
        for line in sorted(names.splitlines()):
            if line.endswith(".py"):
                entries.append((PurePosixPath(line), _git(root, "show", f"{revision}:{line}")))
        return entries

    pkg_dir = root / SIM_PACKAGE_ROOT
    if not pkg_dir.is_dir():
        return entries
    if _is_git_repo(root):
        tracked = _git(root, "ls-files", "--", SIM_PACKAGE_ROOT).splitlines()
        untracked = _git(
            root, "ls-files", "--others", "--exclude-standard", "--", SIM_PACKAGE_ROOT
        ).splitlines()
        for name in sorted({n for n in (*tracked, *untracked) if n.endswith(".py")}):
            disk = root / name
            if disk.is_file():
                entries.append((PurePosixPath(name), disk.read_text(encoding="utf-8")))
        return entries

    for path in sorted(pkg_dir.rglob("*.py")):
        if "__pycache__" not in path.parts:
            rel = PurePosixPath(path.relative_to(root).as_posix())
            entries.append((rel, path.read_text(encoding="utf-8")))
    return entries


def _package_parts(rel: PurePosixPath) -> tuple[str, ...]:
    parts = list(rel.with_suffix("").parts)
    return tuple(parts[:-1]) if parts else ()


def _resolve_import_from(rel: PurePosixPath, module: str | None, level: int) -> tuple[str, ...]:
    if level == 0:
        return tuple(module.split(".")) if module else ()
    pkg = _package_parts(rel)
    up = level - 1
    base = pkg[: len(pkg) - up] if up <= len(pkg) else ()
    return base + (tuple(module.split(".")) if module else ())


def _is_leaf_parts(parts: tuple[str, ...]) -> tuple[bool, str | None]:
    idx = len(OPERATORS_PARTS)
    if len(parts) >= GOVERNED_PATH_DEPTH and parts[:idx] == OPERATORS_PARTS:
        if parts[idx + 1] in GOVERNED_LEAF_STEMS:
            return True, parts[idx]
    return False, None


def _family_pkg_name(parts: tuple[str, ...]) -> str | None:
    if len(parts) == FAMILY_PKG_DEPTH and parts[: len(OPERATORS_PARTS)] == OPERATORS_PARTS:
        return parts[len(OPERATORS_PARTS)]
    return None


def _fold_string(node: ast.AST, env: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold_string(node.left, env), _fold_string(node.right, env)
        return None if left is None or right is None else left + right
    return None


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for stmt in tree.body:
        target = None
        value = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        if isinstance(target, ast.Name) and value is not None:
            folded = _fold_string(value, constants)
            if folded is not None:
                constants[target.id] = folded
    return constants


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _enclosing_symbol(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names: list[str] = []
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(cur.name)
        cur = parents.get(cur)
    return ".".join(reversed(names)) or "<module>"


def _fingerprint(rule: str, rel: PurePosixPath, scope: str, node: ast.AST) -> str:
    raw = f"{rule}|{rel.as_posix()}|{scope}|{ast.dump(node, include_attributes=False)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:FINGERPRINT_HEX_LEN]


def _check_ag001(rel: PurePosixPath) -> list[Violation]:
    if not AG001_STEM_RE.search(rel.stem) or _is_allowed_isa_leaf(rel):
        return []
    msg = (
        f"unmanaged ISA/CANN filename {rel.name!r}; only "
        "mindie_sim/prediction/operators/<family>/{arch22,arch35}.py allowed"
    )
    return [Violation("AG001", rel, 1, f"AG001:{rel.as_posix()}", msg)]


def _check_ag002(rel: PurePosixPath, tree: ast.Module, parents: dict[ast.AST, ast.AST]) -> list[Violation]:
    violations: list[Violation] = []
    in_pkg, own_family, filename = _is_governed_family_path(rel)
    is_family_init = in_pkg and len(rel.parts) == GOVERNED_PATH_DEPTH and filename == "__init__.py"
    str_consts = _module_string_constants(tree)
    family_aliases: dict[str, str] = {}
    leaf_export_names: set[str] = set(GOVERNED_LEAF_STEMS)
    dynamic_imports: set[str] = {"__import__"}
    importlib_aliases: set[str] = set()

    def reject(target_family: str, leaf: str, node: ast.AST, reason: str) -> None:
        if is_family_init and own_family == target_family and reason in {"static", "direct_import"}:
            return
        fp = _fingerprint("AG002", rel, _enclosing_symbol(node, parents), node)
        msg = f"forbidden leaf access operators.{target_family}.{leaf} ({reason})"
        violations.append(Violation("AG002", rel, getattr(node, "lineno", 1), fp, msg))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = tuple(alias.name.split("."))
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or "importlib")
                is_leaf, fam = _is_leaf_parts(parts)
                if is_leaf and fam is not None:
                    if is_family_init and own_family == fam and alias.asname:
                        leaf_export_names.add(alias.asname)
                    reject(fam, parts[len(OPERATORS_PARTS) + 1], node, "direct_import")
                fam_pkg = _family_pkg_name(parts)
                if fam_pkg is not None:
                    family_aliases[alias.asname or alias.name] = fam_pkg
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from(rel, node.module, node.level)
            if base == ("importlib",):
                for alias in node.names:
                    if alias.name == "import_module":
                        dynamic_imports.add(alias.asname or "import_module")
            is_leaf, fam = _is_leaf_parts(base)
            if is_leaf and fam is not None:
                reject(fam, base[len(OPERATORS_PARTS) + 1], node, "static")
                continue
            fam_pkg = _family_pkg_name(base)
            for alias in node.names:
                if fam_pkg is not None and alias.name in GOVERNED_LEAF_STEMS:
                    if is_family_init and own_family == fam_pkg:
                        leaf_export_names.add(alias.asname or alias.name)
                    reject(fam_pkg, alias.name, node, "static")
                elif base == OPERATORS_PARTS and alias.name not in {"contracts", "registry"}:
                    family_aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Attribute):
            if (
                node.attr in GOVERNED_LEAF_STEMS
                and isinstance(node.value, ast.Name)
                and node.value.id in family_aliases
            ):
                reject(family_aliases[node.value.id], node.attr, node, "attribute")
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Name)
                and func.id == "getattr"
                and len(node.args) >= len(("obj", "attr"))
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in family_aliases
            ):
                attr = _fold_string(node.args[1], str_consts)
                if attr in GOVERNED_LEAF_STEMS:
                    reject(family_aliases[node.args[0].id], attr, node, "getattr")
            is_dyn = (
                (isinstance(func, ast.Name) and func.id in dynamic_imports)
                or (
                    isinstance(func, ast.Attribute)
                    and func.attr == "import_module"
                    and isinstance(func.value, ast.Name)
                    and func.value.id in importlib_aliases
                )
            )
            if is_dyn and node.args:
                target_str = _fold_string(node.args[0], str_consts)
                if target_str:
                    pkg_arg = _fold_string(node.args[1], str_consts) if len(node.args) > 1 else None
                    for kw in node.keywords:
                        if kw.arg == "package":
                            pkg_arg = _fold_string(kw.value, str_consts)
                    full = f"{pkg_arg}.{target_str.lstrip('.')}" if target_str.startswith(".") and pkg_arg else target_str
                    parts = tuple(p for p in full.split(".") if p)
                    is_leaf, fam = _is_leaf_parts(parts)
                    if is_leaf and fam is not None:
                        reject(fam, parts[len(OPERATORS_PARTS) + 1], node, "dynamic_import")

    if is_family_init and own_family is not None:
        for stmt in tree.body:
            seq_node: ast.AST | None = None
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in stmt.targets
            ):
                seq_node = stmt.value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "__all__":
                seq_node = stmt.value
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "__all__":
                seq_node = stmt.value
            if isinstance(seq_node, (ast.List, ast.Tuple, ast.Set)):
                for elt in seq_node.elts:
                    if isinstance(elt, ast.Constant) and elt.value in leaf_export_names:
                        fp = _fingerprint("AG002", rel, "<module>", stmt)
                        msg = f"family __init__.py must not re-export leaf {elt.value!r} in __all__"
                        violations.append(Violation("AG002", rel, stmt.lineno, fp, msg))
    return violations


def _is_attr_or_getattr(node: ast.AST, attr: str) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == attr:
        return True
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == attr
    ):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == attr
    )


def _is_tainted_map(node: ast.AST, attr: str, tainted: set[str]) -> bool:
    if _is_attr_or_getattr(node, attr) or (isinstance(node, ast.Name) and node.id in tainted):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "dict" and node.args:
            return _is_tainted_map(node.args[0], attr, tainted)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "keys":
            return _is_tainted_map(node.func.value, attr, tainted)
    return False


def _is_tainted_val(
    node: ast.AST, attr: str, maps: set[str], vals: set[str], *, peak_call: bool = False
) -> bool:
    if isinstance(node, ast.Name) and node.id in vals:
        return True
    if isinstance(node, ast.Subscript) and _is_tainted_map(node.value, attr, maps):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and _is_tainted_map(node.func.value, attr, maps):
                return True
            if peak_call and node.func.attr == "peak" and len(node.args) == 1:
                return True
        if isinstance(node.func, ast.Name) and node.func.id in {"float", "int"} and node.args:
            return _is_tainted_val(node.args[0], attr, maps, vals, peak_call=peak_call)
    return False


def _is_bool_condition(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    if isinstance(parent, (ast.If, ast.IfExp, ast.While)) and parent.test is node:
        return True
    if isinstance(parent, ast.BoolOp):
        return True
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
        return True
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name):
        return parent.func.id in {"bool", "all", "any"}
    if isinstance(parent, ast.Compare) and any(isinstance(op, (ast.Is, ast.IsNot)) for op in parent.ops):
        return any(isinstance(c, ast.Constant) and c.value is None for c in parent.comparators)
    return False


def _scope_nodes(stmts: list[ast.stmt]):
    stack: list[ast.AST] = list(reversed(stmts))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield node
        children = [
            c for c in ast.iter_child_nodes(node)
            if not isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        stack.extend(reversed(children))


def _scan_scope_taint(
    rel: PurePosixPath,
    scope: str,
    body: list[ast.stmt],
    parents: dict[ast.AST, ast.AST],
    *,
    check_sm: bool,
    check_cp: bool,
) -> list[Violation]:
    violations: list[Violation] = []
    sm_maps, sm_vals, cp_maps, cp_vals = set(), set(), set(), set()
    nodes = list(_scope_nodes(body))

    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if _is_tainted_map(node.value, "service_modes", sm_maps):
                    sm_maps.add(target.id)
                elif _is_tainted_val(node.value, "service_modes", sm_maps, sm_vals):
                    sm_vals.add(target.id)
                if _is_tainted_map(node.value, "compute_peaks", cp_maps):
                    cp_maps.add(target.id)
                elif _is_tainted_val(node.value, "compute_peaks", cp_maps, cp_vals, peak_call=True):
                    cp_vals.add(target.id)

    def add(rule: str, node: ast.AST, msg: str) -> None:
        fp = _fingerprint(rule, rel, scope, node)
        violations.append(Violation(rule, rel, getattr(node, "lineno", 1), fp, msg))

    for node in nodes:
        if check_sm:
            if isinstance(node, ast.Compare):
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.In, ast.NotIn)) and _is_tainted_map(comp, "service_modes", sm_maps):
                        add("AG003", node, "service_modes membership check is forbidden; use DatapathCapabilitySpec")
            if _is_tainted_val(node, "service_modes", sm_maps, sm_vals) and _is_bool_condition(node, parents):
                add("AG003", node, "service_modes .get() truthiness branch is forbidden; use DatapathCapabilitySpec or RecipeContext.service_rate")
        if check_cp:
            if isinstance(node, ast.Compare):
                if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops) and any(
                    _is_tainted_map(c, "compute_peaks", cp_maps) for c in node.comparators
                ):
                    add("AG004", node, "compute_peaks membership sniffing is forbidden; use ComputeArchSpec")
                if any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)) for op in node.ops) and any(
                    _is_tainted_val(x, "compute_peaks", cp_maps, cp_vals, peak_call=True)
                    for x in (node.left, *node.comparators)
                ):
                    add("AG004", node, "compute_peaks threshold comparison is forbidden; use ComputeArchSpec/DatapathCapabilitySpec")
            if (
                _is_tainted_val(node, "compute_peaks", cp_maps, cp_vals)
                or _is_tainted_map(node, "compute_peaks", cp_maps)
            ) and _is_bool_condition(node, parents):
                add("AG004", node, "compute_peaks truthiness sniffing is forbidden; use ComputeArchSpec")
    return violations


def _check_ag003_ag004(rel: PurePosixPath, tree: ast.Module, parents: dict[ast.AST, ast.AST]) -> list[Violation]:
    rel_str = rel.as_posix()
    check_sm = not (rel_str.startswith(_INFRA_PREFIXES) or rel_str in _SM_INFRA_FILES)
    check_cp = not (rel_str.startswith(_INFRA_PREFIXES) or rel_str in _CP_INFRA_FILES)
    if not (check_sm or check_cp):
        return []
    violations = _scan_scope_taint(rel, "<module>", tree.body, parents, check_sm=check_sm, check_cp=check_cp)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = _enclosing_symbol(node, parents)
            full_scope = f"{scope}.{node.name}" if scope != "<module>" else node.name
            violations.extend(_scan_scope_taint(rel, full_scope, node.body, parents, check_sm=check_sm, check_cp=check_cp))
    return violations


def _extract_source_ref_fields(call: ast.Call, str_consts: dict[str, str]) -> dict[str, str] | None:
    func_name = call.func.id if isinstance(call.func, ast.Name) else (
        call.func.attr if isinstance(call.func, ast.Attribute) else None
    )
    if func_name != "SourceRef":
        return None
    extracted: dict[str, str] = {}
    for idx, arg in enumerate(call.args):
        if idx >= len(SOURCE_REF_FIELDS):
            return None
        val = _fold_string(arg, str_consts)
        if val is None:
            return None
        extracted[SOURCE_REF_FIELDS[idx]] = val
    for kw in call.keywords:
        if kw.arg in SOURCE_REF_FIELDS:
            val = _fold_string(kw.value, str_consts)
            if val is None:
                return None
            extracted[kw.arg] = val
    return extracted


def _docstring_has_upstream_abi(tree: ast.Module) -> bool:
    if not tree.body:
        return False
    first = tree.body[0]
    if not (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return False
    start_line = int(getattr(first, "lineno", 1))
    if start_line > DOCSTRING_HEADER_MAX_LINE:
        return False
    allowed_lines = max(0, DOCSTRING_HEADER_MAX_LINE - start_line + 1)
    snippet = "\n".join(first.value.value.splitlines()[:allowed_lines])
    return bool(UPSTREAM_ABI_RE.search(snippet))


def _parse_repo_map(entries: Sequence[str] | None) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for raw in entries or ():
        name, sep, path_str = raw.partition("=")
        name, path_str = name.strip(), path_str.strip()
        if not sep or not name or not path_str:
            raise RuntimeError(f"invalid --repo-map entry {raw!r}; expected <name>=<path>")
        mapping[name] = Path(path_str).resolve()
    return mapping


def _resolve_repo_dir(
    repo: str, *, refs_root: Path | None, repo_map: Mapping[str, Path] | None
) -> Path | None:
    basename = repo.split("/")[-1]
    if repo_map:
        return repo_map.get(repo) or repo_map.get(basename) or (
            refs_root / basename if refs_root is not None else None
        )
    return refs_root / basename if refs_root is not None else None


def _check_ag005(
    rel: PurePosixPath,
    text: str,
    tree: ast.Module,
    *,
    refs_root: Path | None = None,
    repo_map: Mapping[str, Path] | None = None,
) -> list[Violation]:
    del text
    if not _is_allowed_isa_leaf(rel):
        return []
    violations: list[Violation] = []
    pos = rel.as_posix()
    def add_v(line: int, key: str, msg: str) -> None:
        violations.append(Violation("AG005", rel, line, f"AG005:{pos}:{key}", msg))

    if not _docstring_has_upstream_abi(tree):
        add_v(1, "missing_upstream_abi", "ISA leaf module docstring top 30 lines must declare 'Upstream-ABI: <symbol>'")
    str_consts = _module_string_constants(tree)
    sources_node: ast.AST | None = None
    has_strategy = False
    for stmt in tree.body:
        targets = (
            stmt.targets if isinstance(stmt, ast.Assign)
            else ([stmt.target] if isinstance(stmt, ast.AnnAssign) and stmt.value is not None else ())
        )
        for target in targets:
            if isinstance(target, ast.Name):
                if target.id == "SOURCES":
                    sources_node = stmt.value
                elif target.id == "STRATEGY":
                    has_strategy = True
    if not has_strategy:
        add_v(1, "missing_strategy", "ISA leaf must define module-level STRATEGY")
    if not isinstance(sources_node, ast.Tuple) or not sources_node.elts:
        add_v(getattr(sources_node, "lineno", 1), "missing_sources", "ISA leaf must define non-empty SOURCES: tuple[SourceRef, ...]")
        return violations
    for elt in sources_node.elts:
        if not isinstance(elt, ast.Call):
            add_v(getattr(elt, "lineno", 1), "invalid_source_entry", "SOURCES entries must be SourceRef(...) calls")
            continue
        fields = _extract_source_ref_fields(elt, str_consts)
        if fields is None or len(fields) < len(SOURCE_REF_FIELDS):
            add_v(elt.lineno, f"incomplete_source_ref:{elt.lineno}", "SourceRef must statically specify repository, revision, relative_path, source_sha256")
            continue
        repo, rev, rpath, sha = (fields[k].strip() for k in SOURCE_REF_FIELDS)
        if (
            not repo or not HEX_REVISION_RE.match(rev) or not rpath
            or rpath.startswith("/") or ".." in PurePosixPath(rpath).parts
            or not SHA256_HEX_RE.match(sha)
        ):
            add_v(elt.lineno, f"malformed_source_ref:{rpath}", f"malformed SourceRef(repo={repo!r}, rev={rev!r}, path={rpath!r}, sha256={sha[:PREVIEW_SHA_LEN]!r}...)")
            continue
        if refs_root is not None or repo_map:
            repo_dir = _resolve_repo_dir(repo, refs_root=refs_root, repo_map=repo_map)
            if repo_dir is None or not repo_dir.is_dir():
                add_v(elt.lineno, f"refs_repo_missing:{repo}", f"upstream refs repo directory not found: {repo_dir or repo}")
            else:
                rev_type = subprocess.run(
                    ["git", "-C", str(repo_dir), "cat-file", "-t", rev],
                    check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                obj_type = subprocess.run(
                    ["git", "-C", str(repo_dir), "cat-file", "-t", f"{rev}:{rpath}"],
                    check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                if rev_type.returncode != 0 or rev_type.stdout.strip() != "commit":
                    add_v(elt.lineno, f"invalid_commit:{rev}", f"upstream revision {repo}@{rev} is not a git commit")
                elif obj_type.returncode != 0 or obj_type.stdout.strip() != "blob":
                    add_v(elt.lineno, f"source_missing:{rpath}", f"upstream source {repo}@{rev}:{rpath} is not a git blob")
                else:
                    blob = subprocess.run(
                        ["git", "-C", str(repo_dir), "cat-file", "blob", f"{rev}:{rpath}"],
                        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    if blob.returncode != 0:
                        add_v(elt.lineno, f"source_missing:{rpath}", f"upstream source {repo}@{rev}:{rpath} not found")
                    else:
                        actual_sha = hashlib.sha256(blob.stdout).hexdigest()
                        if actual_sha != sha:
                            add_v(elt.lineno, f"sha_mismatch:{rpath}", f"SHA-256 mismatch for {rpath}: expected {sha}, got {actual_sha}")
    return violations


def _check_ag007(rel: PurePosixPath, tree: ast.Module) -> list[Violation]:
    if not _is_governed_family_path(rel)[0]:
        return []
    violations = []
    seen = set()
    for node in ast.walk(tree):
        for field, value in ast.iter_fields(node):
            if field not in {"id", "name", "asname", "attr", "arg", "rest", "value"}:
                continue
            if not isinstance(value, str) or not value.isidentifier():
                continue
            key = (node.lineno, value)
            spelling = CAMEL_CASE_BOUNDARY_RE.sub("_", value)
            if key not in seen and HEURISTIC_IDENTIFIER_RE.search(spelling):
                seen.add(key)
                violations.append(Violation(
                    "AG007", rel, node.lineno, f"AG007:{rel.as_posix()}:{value}",
                    f"ad hoc heuristic identifier {value!r} in an operator family",
                ))
    return violations


def scan_repository(
    root: Path, /, *, revision: str | None = None,
    refs_root: Path | None = None, repo_map: Mapping[str, Path] | None = None,
) -> tuple[Violation, ...]:
    """Return the sorted static findings under ``mindie_sim/**/*.py``."""
    violations: list[Violation] = []
    for rel, text in _enumerate_py_files(root, revision=revision):
        violations.extend(_check_ag001(rel))
        try:
            tree = ast.parse(text, filename=rel.as_posix())
        except SyntaxError as exc:
            violations.append(Violation(
                "AG005", rel, int(exc.lineno or 1),
                f"AG005:{rel.as_posix()}:syntax_error", f"syntax_error:{exc.msg}",
            ))
            continue
        parents = _parents(tree)
        violations.extend(_check_ag002(rel, tree, parents))
        violations.extend(_check_ag003_ag004(rel, tree, parents))
        violations.extend(_check_ag005(rel, text, tree, refs_root=refs_root, repo_map=repo_map))
        violations.extend(_check_ag007(rel, tree))
    violations.sort(key=lambda v: (v.path.as_posix(), v.line, v.rule, v.message))
    return tuple(violations)


def filter_allowlisted(violations: Sequence[Violation], /) -> tuple[Violation, ...]:
    """Drop only tracked pre-Stage-3 legacy violations; never waive governed operators."""
    kept: list[Violation] = []
    for item in violations:
        in_pkg, _family, _fname = _is_governed_family_path(item.path)
        if in_pkg or item.rule not in LEGACY_ALLOWLIST.get(item.path.as_posix(), frozenset()):
            kept.append(item)
    return tuple(kept)


def new_violations(
    current: Sequence[Violation], baseline: Sequence[Violation], /
) -> tuple[Violation, ...]:
    """Multiset differential subtraction preserving governed-package strictness."""
    remaining: Counter[tuple[str, str, str]] = Counter(
        (v.rule, v.path.as_posix(), v.fingerprint) for v in baseline
    )
    added: list[Violation] = []
    for item in current:
        in_pkg, _family, _fname = _is_governed_family_path(item.path)
        if in_pkg:
            added.append(item)
            continue
        key = (item.rule, item.path.as_posix(), item.fingerprint)
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            added.append(item)
    return tuple(added)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--base", default=None, help="git base ref for differential scan")
    parser.add_argument("--strict", action="store_true", help="run live checks and ignore legacy allowlist")
    parser.add_argument("--refs-root", default=None, help="local upstream repos root for SHA-256 check")
    parser.add_argument(
        "--repo-map",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="explicit upstream repo mapping overriding --refs-root (repeatable)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    refs_root = Path(args.refs_root).resolve() if args.refs_root else None
    try:
        repo_map = _parse_repo_map(args.repo_map)
        current = scan_repository(root, refs_root=refs_root, repo_map=repo_map or None)
        if args.base is not None:
            active = new_violations(current, scan_repository(root, revision=args.base))
        elif args.strict:
            active = current
        else:
            active = filter_allowlisted(current)
        if args.strict:
            active += scan_runtime(root)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    for item in active:
        print(f"{item.path.as_posix()}:{item.line}: {item.rule} {item.message}")
    print(f"arch governance violations: {len(active)}")
    return EXIT_VIOLATIONS if active else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
