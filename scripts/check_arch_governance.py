#!/usr/bin/env python3
"""Architecture governance gate (AG001..AG014) for ``mindie_sim``.

[Role] Enforce governed ``mindie_sim/prediction/operators/<family>/`` package
  boundaries, forbid flat ISA/CANN filename sprawl under ``mindie_sim/**/*.py``,
  block direct leaf imports, block ``service_modes`` / ``compute_peaks``
  capability sniffing, verify ISA provenance, check dispatched recipe ownership
  and physical work conservation, reject silent physical hardware defaults,
  enforce unidirectional layer imports, ensure operator Engine reachability,
  and detect per-model-name branching across executable ``mindie_sim/`` +
  ``manifest.json`` (AG014).
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
from collections.abc import Callable, Mapping, Sequence
import hashlib
import keyword
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from types import MappingProxyType

if __package__:
    from ._arch_governance_ag012 import check_ag012  # noqa: F401
    from ._arch_governance_runtime import (  # noqa: F401
        DOCSTRING_HEADER_MAX_LINE, FAMILY_PKG_DEPTH, GOVERNED_PATH_DEPTH,
        MAX_FLAT_PREDICTION_MODULES, MAX_ONBOARDING_FILES, MAX_ONBOARDING_LOC,
        MAX_OPERATOR_PACKAGE_FILES, MAX_TOPLEVEL_MODULES, OPERATORS_PARTS, Violation,
        _check_ag006_flat_prediction, _check_ag007, _check_ag008_module, _check_ag008_repo_bounds,
        _check_ag009_silent_physical_defaults, _check_ag010_layer_dag,
        _check_ag010_package_sccs, _check_ag011_operator_reachability,
        _check_ag013_anti_hardcoding as _base_ag013_anti_hardcoding,
        _check_ag014_zero_model_name_branches,
        _is_governed_family_path, check_ag014_manifest, scan_runtime,
        verify_ag007_discrimination, verify_ag011_engine_reachability,
    )
else:
    from _arch_governance_ag012 import check_ag012  # noqa: F401
    from _arch_governance_runtime import (  # noqa: F401
        DOCSTRING_HEADER_MAX_LINE, FAMILY_PKG_DEPTH, GOVERNED_PATH_DEPTH,
        MAX_FLAT_PREDICTION_MODULES, MAX_ONBOARDING_FILES, MAX_ONBOARDING_LOC,
        MAX_OPERATOR_PACKAGE_FILES, MAX_TOPLEVEL_MODULES, OPERATORS_PARTS, Violation,
        _check_ag006_flat_prediction, _check_ag007, _check_ag008_module, _check_ag008_repo_bounds,
        _check_ag009_silent_physical_defaults, _check_ag010_layer_dag,
        _check_ag010_package_sccs, _check_ag011_operator_reachability,
        _check_ag013_anti_hardcoding as _base_ag013_anti_hardcoding,
        _check_ag014_zero_model_name_branches,
        _is_governed_family_path, check_ag014_manifest, scan_runtime,
        verify_ag007_discrimination, verify_ag011_engine_reachability,
    )

SIM_PACKAGE_ROOT = "mindie_sim"
GOVERNED_LEAF_NAME_RE = re.compile(r"^arch\d+\.py$")
GOVERNED_LEAF_STEM_RE = re.compile(r"^arch\d+$")
AG001_STEM_RE = re.compile(
    r"(?:arch\d+|cann|(?:^|_)(?:a2|a3|a5)(?:$|_))",
    re.IGNORECASE,
)
UPSTREAM_ABI_RE = re.compile(r"^\s*Upstream-ABI:\s*(\S+)", re.MULTILINE)
HEX_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT_HEX_LEN = 16
PREVIEW_SHA_LEN = 12
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
    "mindie_sim/prediction/efficiency/cost.py",
    "mindie_sim/prediction/efficiency.py",
    "mindie_sim/prediction/operators/matrix/matrix_precision.py",
})

# All flat ISA/CANN sidecars and AG003/AG004 legacy debt have been retired;
# strict mode requires zero allowlist entries.
LEGACY_ALLOWLIST: dict[str, frozenset[str]] = {}

#: Canonical single-owner concept table enforced by AG008.
CANONICAL_CONCEPT_OWNERS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "RequestView": ("mindie_sim/schema/workload.py",),
    "IterationPlan": ("mindie_sim/schema/workload.py",),
    "bind_memory_access": ("mindie_sim/prediction/runtime/l2_residency.py",),
    "AdmissionMemoryView": ("mindie_sim/prediction/runtime/l2_residency.py",),
    "ByteLedger": ("mindie_sim/prediction/runtime/memory_physics.py",),
    "AddressRunPlan": ("mindie_sim/prediction/runtime/memory_physics.py",),
    "MemoryAccessPattern": ("mindie_sim/prediction/runtime/memory_physics.py",),
    "PrimitiveTable": ("mindie_sim/primitives/schema.py",),
    "CalibratedRateOverlay": ("mindie_sim/primitives/loader.py",),
    "FamilyRegistry": ("mindie_sim/prediction/operators/registry.py",),
    "certify_lower_bound": ("mindie_sim/optimization/dual_bound.py",),
    "OperatorPlanningContext": ("mindie_sim/prediction/operators/contracts.py",),
    "OptimalityCertificate": ("mindie_sim/optimization/dual_bound.py",),
    "Engine": ("mindie_sim/prediction/runtime/engine.py",),
    "ParallelSpec": ("mindie_sim/schema/parallel.py",),
    "MemoryHierarchySpec": ("mindie_sim/schema/physical_mechanisms.py",),
    "PhysicalMechanismSpec": ("mindie_sim/schema/physical_mechanisms.py",),
    "DatapathCapabilitySpec": ("mindie_sim/schema/datapath.py",),
    "ComputeArchSpec": ("mindie_sim/schema/compute_arch.py",),
    "CalibratedDispatchEnvelope": ("mindie_sim/primitives/loader.py",),
    "RecipeBook": ("mindie_sim/prediction/program/recipes.py",),
    "RecipeContext": ("mindie_sim/prediction/program/binding.py",),
    "PredictionSpine": ("mindie_sim/prediction/spine.py",),
    "CheckpointQuantMap": ("mindie_sim/schema/checkpoint.py",),
    "TensorPrecisionABI": ("mindie_sim/schema/operators.py",),
    "KernelBinding": ("mindie_sim/schema/operators.py",),
    "OwnerScheduleSpec": ("mindie_sim/schema/operators.py",),
    "CachePlaneABI": ("mindie_sim/schema/stack.py",),
    "CacheStateABI": ("mindie_sim/schema/stack.py",),
    "RuntimeQuantPlan": ("mindie_sim/schema/stack.py",),
    "StackLoweringPlan": ("mindie_sim/schema/stack.py",),
    "ManifestBinding": ("mindie_sim/schema/stack.py",),
    "RuntimeStackAdapter": ("mindie_sim/schema/stack.py",),
    "ScenarioContract": ("mindie_sim/schema/scenario.py",),
    "SLAContract": ("mindie_sim/schema/scenario.py",),
    "DeploymentOutcome": ("mindie_sim/schema/scenario.py",),
    "CoDesignOpportunity": ("mindie_sim/schema/scenario.py",),
    "AbiQueryKey": ("mindie_sim/schema/abi.py",),
    "ProbedAbiEntry": ("mindie_sim/schema/abi.py",),
    "AbiSupportVerdict": ("mindie_sim/schema/abi.py",),
    "AbiCatalogSnapshot": ("mindie_sim/schema/abi.py",),
    "StackTrackSpec": ("mindie_sim/schema/stack.py",),
    "SpeculativeWorkloadPolicy": ("mindie_sim/schema/workload.py",),
    "DSparkPreparePlan": ("mindie_sim/catalog/draft_spec.py",),
    "DraftStepSchedule": ("mindie_sim/catalog/draft_spec.py",),
    "SpeculativeExecutionContract": ("mindie_sim/catalog/draft_spec.py",),
    "UbAdmissionReceipt": ("mindie_sim/templates/contracts.py",),
    "TierLoweringOutcome": ("mindie_sim/templates/contracts.py",),
    "DualTierLoweringReceipt": ("mindie_sim/templates/contracts.py",),
    "StageReceipt": ("mindie_sim/prediction/results.py",),
    "EngineStepReceipt": ("mindie_sim/prediction/results.py",),
    "NativeCallBinding": ("mindie_sim/prediction/native_binding.py",),
})

def _is_allowed_isa_leaf(rel: PurePosixPath) -> bool:
    in_pkg, family, filename = _is_governed_family_path(rel)
    return (
        in_pkg and len(rel.parts) == GOVERNED_PATH_DEPTH
        and family is not None
        and bool(GOVERNED_LEAF_NAME_RE.match(filename))
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
        if GOVERNED_LEAF_STEM_RE.match(parts[idx + 1]):
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
        "mindie_sim/prediction/operators/<family>/arch<N>.py allowed"
    )
    return [Violation("AG001", rel, 1, f"AG001:{rel.as_posix()}", msg)]


def _check_ag002(rel: PurePosixPath, tree: ast.Module, parents: dict[ast.AST, ast.AST]) -> list[Violation]:
    violations: list[Violation] = []
    in_pkg, own_family, filename = _is_governed_family_path(rel)
    is_family_init = in_pkg and len(rel.parts) == GOVERNED_PATH_DEPTH and filename == "__init__.py"
    str_consts = _module_string_constants(tree)
    family_aliases: dict[str, str] = {}
    leaf_export_names: set[str] = set()
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
                if fam_pkg is not None and GOVERNED_LEAF_STEM_RE.match(alias.name):
                    if is_family_init and own_family == fam_pkg:
                        leaf_export_names.add(alias.asname or alias.name)
                    reject(fam_pkg, alias.name, node, "static")
                elif base == OPERATORS_PARTS and alias.name not in {"contracts", "registry"}:
                    family_aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Attribute):
            if (
                GOVERNED_LEAF_STEM_RE.match(node.attr)
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
                if attr is not None and GOVERNED_LEAF_STEM_RE.match(attr):
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
                    if (
                        isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)
                        and (elt.value in leaf_export_names or bool(GOVERNED_LEAF_STEM_RE.match(elt.value)))
                    ):
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


_AG013_HARDENED_ROOTS = frozenset({
    "rewrite", "optimization", "export", "solve", "sweep",
})
_AG013_ALLOWED_TIER_TOKENS = frozenset({"K3", "K3_dsl_pipeline"})
_AG013_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_AG013_MODEL_TOKEN_RE = re.compile(
    r"(?:^|[^a-z0-9])("
    r"kimi(?:[-_]?k\d*(?:\.\d+)?)?"
    r"|k3(?:\.\d+)?"
    r"|qwen(?:\d\w*)?"
    r"|deepseek\w*"
    r"|glm(?:[-_]?\d\w*)?"
    r"|minimax\w*"
    r"|llama(?:[-_]?\d\w*)?"
    r"|gemma(?:[-_]?\d\w*)?"
    r")(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)
_AG013_SUFFIX_TABLE_RE = re.compile(
    r"^_?(?:[A-Z0-9_]*_)?(?:SUFFIX(?:ES)?|OP_NAMES?|OP_IDS?|FUSION_SOURCES?)(?:_[A-Z0-9_]*)?$"
)
_AG013_MODULE_STR_TABLE_RE = re.compile(
    r"(?:^|_)(?:suffix(?:es)?|op_names?|op_ids?|fusion_sources?)(?:_|$)",
    re.IGNORECASE,
)
_AG013_TARGET_ETA_RE = re.compile(r"(?:^|_)target_eta(?:$|_)", re.IGNORECASE)
_AG013_EXPORT_PCT_HARD_RE = re.compile(r">=\s*(?:75|78|85)(?:\.0+)?\s*%")
_AG013_EXPORT_PCT_LSTAR_RE = re.compile(
    r"(?:目标|达成|>=|>)\s*(?:\d+(?:\.\d+)?|\{_\})\s*%\s*(?:[A-Za-z0-9_/\s]{0,16})?L\*"
)
_AG013_EXPORT_LSTAR_PCT_RE = re.compile(r"L\*\s*达成率\s*>=\s*(?:\d+(?:\.\d+)?|\{_\})\s*%")
_AG013_EXPORT_SOURCE_PATH_RE = re.compile(
    r"(?:^|[\s`'\"(])(?:vllm[-_]ascend|vllm|cann|mindie[-_]llm|sglang)/[A-Za-z0-9_./-]+\.py(?=$|[\s`'\"):,])",
    re.IGNORECASE,
)
_AG013_AIV_CCU_RE = re.compile(r"\b(?:AIV\s*/\s*CCU|CCU\s*/\s*AIV)\b", re.IGNORECASE)
_AG013_ACLGRAPH_CATCHALL = "避免 ACLGraph 图断裂"
_AG013_GUIDANCE_TARGETS = frozenset({
    "config_howto", "kernel_action", "framework_action",
    "action_note", "action_detail", "default_design_slot", "default_acceptance_slot",
})
_AG013_DISPATCH_KEYS = frozenset({
    "op_id", "suffix", "family", "recipe", "ascend_kernel",
    "submodule", "sku", "dominant_bound",
})
_AG013_HINT_TABLE_NAME_RE = re.compile(
    r"(?:^|_)(?:design_?hints?|op_?hints?|kernel_?hints?|guidance_?table|guidance_?map|action_?templates?|howto_?map|prose_?table)(?:$|_)",
    re.IGNORECASE,
)
_AG013_GUIDANCE_FUNC_RE = re.compile(
    r"(?:guidance|design_slot|acceptance_slot|action_howto|kernel_hint)",
    re.IGNORECASE,
)
_AG013_PLACEHOLDER_DEFAULTS = frozenset({"", "-", "—", "--", "N/A", "none", "None"})
_AG013_PRESCRIPTIVE_MIN_LEN = 6


def _ag013_fold_export_str(node: ast.AST, env: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _ag013_fold_export_str(node.left, env)
        right = _ag013_fold_export_str(node.right, env)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for val in node.values:
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                parts.append(val.value)
            else:
                parts.append("{_}")
        return "".join(parts)
    return None


def _ag013_norm_text(raw: str) -> str:
    return " ".join(raw.replace("\u2265", ">=").replace("\u2264", "<=").split())


def _ag013_is_source_ref_call(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func.id if isinstance(node.func, ast.Name) else (
        node.func.attr if isinstance(node.func, ast.Attribute) else ""
    )
    return fn == "SourceRef"


def _ag013_node_references_key(node: ast.AST, tainted: set[str]) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in tainted:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in _AG013_DISPATCH_KEYS:
            return True
    return False


def _ag013_unwrap_dict_expr(node: ast.AST) -> ast.Dict | None:
    if isinstance(node, ast.Call):
        fn = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else ""
        )
        if fn in {"MappingProxyType", "dict"} and len(node.args) == 1:
            return _ag013_unwrap_dict_expr(node.args[0])
        return None
    if isinstance(node, ast.Dict) and node.keys:
        if all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.keys):
            return node
    return None


def _ag013_is_prose_string(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < _AG013_PRESCRIPTIVE_MIN_LEN:
        return False
    return (" " in stripped) or ("\n" in stripped) or any(ord(ch) > 127 for ch in stripped)


def _ag013_dict_has_prose_values(node: ast.AST) -> bool:
    dict_node = _ag013_unwrap_dict_expr(node)
    if dict_node is None:
        return False
    for val in dict_node.values:
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            if _ag013_is_prose_string(val.value):
                return True
        elif isinstance(val, (ast.Tuple, ast.List)):
            for elt in val.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str) and _ag013_is_prose_string(elt.value):
                    return True
    return False


def _ag013_is_literal_map_expr(node: ast.AST) -> bool:
    return _ag013_unwrap_dict_expr(node) is not None


def _ag013_stmt_writes_guidance(stmts: Sequence[ast.stmt], *, is_guidance_fn: bool) -> bool:
    for st in stmts:
        for sub in ast.walk(st):
            if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                for tgt in targets:
                    if isinstance(tgt, ast.Name) and tgt.id in _AG013_GUIDANCE_TARGETS:
                        return True
                    if isinstance(tgt, ast.Attribute) and tgt.attr in _AG013_GUIDANCE_TARGETS:
                        return True
                    if isinstance(tgt, ast.Tuple):
                        for elt in tgt.elts:
                            if isinstance(elt, ast.Name) and elt.id in _AG013_GUIDANCE_TARGETS:
                                return True
            elif is_guidance_fn and isinstance(sub, ast.Return) and sub.value is not None:
                if isinstance(sub.value, (ast.Constant, ast.JoinedStr, ast.BinOp, ast.Tuple)):
                    return True
    return False


def _ag013_is_prescriptive_default(node: ast.AST, str_consts: Mapping[str, str]) -> str | None:
    folded = _ag013_fold_export_str(node, str_consts)
    if folded is None:
        return None
    stripped = folded.strip()
    if stripped in _AG013_PLACEHOLDER_DEFAULTS or len(stripped) < _AG013_PRESCRIPTIVE_MIN_LEN:
        return None
    return stripped


def _ag013_check_export_receipts(
    tree: ast.Module,
    docs: set[int],
    parents: Mapping[ast.AST, ast.AST],
    emit: Callable[[int, str, str, str], None],
) -> None:
    """Enforce zero canned prose, zero percentage L* targets, zero guessed .py paths, and zero AIV/CCU in export/."""
    str_consts = _module_string_constants(tree)
    literal_maps: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = (stmt.target,), stmt.value
        else:
            continue
        if not _ag013_is_literal_map_expr(value):
            continue
        has_prose = _ag013_dict_has_prose_values(value)
        for tgt in targets:
            if isinstance(tgt, ast.Name):
                literal_maps.add(tgt.id)
                if has_prose and _AG013_HINT_TABLE_NAME_RE.search(tgt.id):
                    emit(
                        int(getattr(stmt, "lineno", 1)),
                        "export_guidance_dispatch",
                        tgt.id,
                        f"canned prose hint table {tgt.id!r} forbidden in export/; emit typed receipts instead",
                    )

    for node in ast.walk(tree):
        lineno = int(getattr(node, "lineno", 1))
        if isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr)):
            if isinstance(node, ast.Constant) and (not isinstance(node.value, str) or id(node) in docs):
                continue
            parent = parents.get(node)
            if isinstance(parent, (ast.BinOp, ast.JoinedStr)):
                continue
            folded = _ag013_fold_export_str(node, str_consts)
            if folded is None:
                continue
            norm = _ag013_norm_text(folded)
            if (
                _AG013_EXPORT_PCT_HARD_RE.search(norm)
                or _AG013_EXPORT_PCT_LSTAR_RE.search(norm)
                or _AG013_EXPORT_LSTAR_PCT_RE.search(norm)
            ):
                emit(
                    lineno, "export_percent_target", norm[:48],
                    f"hardcoded percentage/L* efficiency target {folded!r} forbidden in export/",
                )
            in_source_ref = _ag013_is_source_ref_call(parent) or (
                isinstance(parent, ast.keyword) and _ag013_is_source_ref_call(parents.get(parent))
            )
            if not in_source_ref and _AG013_EXPORT_SOURCE_PATH_RE.search(folded):
                emit(
                    lineno, "export_source_path_guess", folded[:48],
                    f"guessed upstream .py source path {folded!r} forbidden in export/; use verified SourceRef",
                )
            if _AG013_ACLGRAPH_CATCHALL in folded:
                emit(
                    lineno, "export_unbound_guidance", _AG013_ACLGRAPH_CATCHALL,
                    "catch-all canned guidance '避免 ACLGraph 图断裂' forbidden in export/",
                )
            if _AG013_AIV_CCU_RE.search(folded):
                emit(
                    lineno, "export_unbound_guidance", "aiv_ccu_hybrid",
                    f"hybrid AIV/CCU label {folded!r} forbidden in export/; isolate A5 CCU vs A2/A3 AIV",
                )

        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and len(node.values) >= 2:
            head, tail = node.values[0], node.values[-1]
            head_is_src = (
                (isinstance(head, ast.Name) and head.id == "source_ref")
                or (isinstance(head, ast.Attribute) and head.attr == "source_ref")
            )
            if head_is_src:
                tail_str = _ag013_fold_export_str(tail, str_consts)
                if tail_str and ".py" in tail_str:
                    emit(
                        lineno, "export_source_path_guess", tail_str[:48],
                        f"fallback guessed source_ref path {tail_str!r} forbidden in export/",
                    )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "source_ref"
        ):
            def_str = _ag013_fold_export_str(node.args[1], str_consts)
            if def_str and ".py" in def_str:
                emit(
                    lineno, "export_source_path_guess", def_str[:48],
                    f"fallback .get('source_ref', {def_str!r}) forbidden in export/",
                )

    for fn_node in ast.walk(tree):
        if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_guidance_fn = bool(_AG013_GUIDANCE_FUNC_RE.search(fn_node.name))
        tainted: set[str] = {
            arg.arg for arg in (*fn_node.args.posonlyargs, *fn_node.args.args, *fn_node.args.kwonlyargs)
            if arg.arg in _AG013_DISPATCH_KEYS
        }
        body_nodes = list(_scope_nodes(fn_node.body))
        for sub in body_nodes:
            if isinstance(sub, (ast.Assign, ast.AnnAssign)) and sub.value is not None:
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                is_key_derived = _ag013_node_references_key(sub.value, tainted)
                for tgt in targets:
                    if isinstance(tgt, ast.Name):
                        if is_key_derived:
                            tainted.add(tgt.id)
                        if tgt.id in _AG013_GUIDANCE_TARGETS:
                            _ag013_check_guidance_rhs(
                                sub.value, int(getattr(sub, "lineno", 1)), tgt.id,
                                tainted, literal_maps, str_consts, emit,
                            )
                    elif isinstance(tgt, ast.Attribute) and tgt.attr in _AG013_GUIDANCE_TARGETS:
                        _ag013_check_guidance_rhs(
                            sub.value, int(getattr(sub, "lineno", 1)), tgt.attr,
                            tainted, literal_maps, str_consts, emit,
                        )
        for sub in body_nodes:
            lineno = int(getattr(sub, "lineno", 1))
            if isinstance(sub, ast.If):
                if _ag013_node_references_key(sub.test, tainted) and _ag013_stmt_writes_guidance(
                    (*sub.body, *sub.orelse), is_guidance_fn=is_guidance_fn,
                ):
                    emit(
                        lineno, "export_guidance_dispatch", fn_node.name,
                        f"op-key prose dispatch branch in {fn_node.name!r} forbidden in export/",
                    )
            elif isinstance(sub, ast.Match):
                if _ag013_node_references_key(sub.subject, tainted) and any(
                    _ag013_stmt_writes_guidance(case.body, is_guidance_fn=is_guidance_fn)
                    for case in sub.cases
                ):
                    emit(
                        lineno, "export_guidance_dispatch", fn_node.name,
                        f"op-key match prose dispatch in {fn_node.name!r} forbidden in export/",
                    )
            elif is_guidance_fn and isinstance(sub, ast.Return) and sub.value is not None:
                _ag013_check_guidance_rhs(
                    sub.value, lineno, fn_node.name, tainted, literal_maps, str_consts, emit,
                )


def _ag013_check_guidance_rhs(
    rhs: ast.AST,
    lineno: int,
    target_name: str,
    tainted: set[str],
    literal_maps: set[str],
    str_consts: Mapping[str, str],
    emit: Callable[[int, str, str, str], None],
) -> None:
    if isinstance(rhs, ast.Subscript) and isinstance(rhs.value, ast.Name) and rhs.value.id in literal_maps:
        if _ag013_node_references_key(rhs.slice, tainted):
            emit(
                lineno, "export_guidance_dispatch", f"{rhs.value.id}->{target_name}",
                f"literal prose map lookup {rhs.value.id}[...] feeding {target_name!r} forbidden in export/",
            )
    elif (
        isinstance(rhs, ast.Call)
        and isinstance(rhs.func, ast.Attribute)
        and rhs.func.attr == "get"
    ):
        if (
            isinstance(rhs.func.value, ast.Name)
            and rhs.func.value.id in literal_maps
            and rhs.args
            and _ag013_node_references_key(rhs.args[0], tainted)
        ):
            emit(
                lineno, "export_guidance_dispatch", f"{rhs.func.value.id}->{target_name}",
                f"literal prose map .get() on {rhs.func.value.id!r} feeding {target_name!r} forbidden in export/",
            )
        if len(rhs.args) >= 2:
            prescriptive = _ag013_is_prescriptive_default(rhs.args[1], str_consts)
            if prescriptive is not None:
                emit(
                    lineno, "export_unbound_guidance", f"{target_name}:get_default",
                    f"unbound prescriptive .get() default {prescriptive!r} on {target_name!r} forbidden in export/",
                )
    elif isinstance(rhs, ast.BoolOp) and isinstance(rhs.op, ast.Or) and len(rhs.values) >= 2:
        prescriptive = _ag013_is_prescriptive_default(rhs.values[-1], str_consts)
        if prescriptive is not None:
            emit(
                lineno, "export_unbound_guidance", f"{target_name}:or_default",
                f"unbound prescriptive fallback {prescriptive!r} on {target_name!r} forbidden in export/",
            )
    elif isinstance(rhs, ast.IfExp) and _ag013_node_references_key(rhs.test, tainted):
        if (
            _ag013_is_prescriptive_default(rhs.body, str_consts) is not None
            or _ag013_is_prescriptive_default(rhs.orelse, str_consts) is not None
        ):
            emit(
                lineno, "export_guidance_dispatch", f"ifexp->{target_name}",
                f"op-key conditional prose expression feeding {target_name!r} forbidden in export/",
            )
    elif isinstance(rhs, ast.Tuple):
        for elt in rhs.elts:
            prescriptive = _ag013_is_prescriptive_default(elt, str_consts)
            if prescriptive is not None:
                emit(
                    lineno, "export_unbound_guidance", f"{target_name}:tuple_literal",
                    f"canned guidance literal {prescriptive!r} returned by {target_name!r} forbidden in export/",
                )


def _ag013_is_string_table_expr(node: ast.AST) -> bool:
    """Return True when ``node`` is a constant container of string literals."""
    if isinstance(node, ast.Call):
        fn = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else ""
        )
        if fn in {"frozenset", "set", "tuple", "list", "dict", "MappingProxyType"} and len(node.args) == 1:
            return _ag013_is_string_table_expr(node.args[0])
        return False
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return bool(node.elts) and all(
            (isinstance(el, ast.Constant) and isinstance(el.value, str)) or _ag013_is_string_table_expr(el)
            for el in node.elts
        )
    if isinstance(node, ast.Dict):
        return bool(node.keys) and all(
            isinstance(k, ast.Constant) and isinstance(k.value, str)
            for k in node.keys
        )
    return False


def _ag013_check_module_string_tables(
    tree: ast.Module, emit: Callable[[int, str, str, str], None], top: str,
) -> None:
    """Flag any module-level string-matching constant table matching suffix/op-name patterns."""
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = (stmt.target,), stmt.value
        else:
            continue
        if not _ag013_is_string_table_expr(value):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and _AG013_MODULE_STR_TABLE_RE.search(target.id):
                emit(
                    int(getattr(stmt, "lineno", 1)),
                    "suffix_table",
                    target.id,
                    f"module-level string-matching table {target.id!r} forbidden in {top}; use typed OpSpec traits/metadata",
                )


def _ag013_docstring_node_ids(tree: ast.Module) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                ids.add(id(node.body[0].value))
    return ids


def _ag013_match_model_token(text: str) -> str | None:
    stripped = text.strip()
    if stripped in _AG013_ALLOWED_TIER_TOKENS:
        return None
    normalized = _AG013_CAMEL_BOUNDARY_RE.sub("_", stripped)
    match = _AG013_MODEL_TOKEN_RE.search(normalized)
    if match is None:
        return None
    token = match.group(1)
    if token == "K3" and stripped in _AG013_ALLOWED_TIER_TOKENS:
        return None
    return token


def _check_ag013_anti_hardcoding(rel: PurePosixPath, tree: ast.Module) -> list[Violation]:
    """Hardened AG013: ban model literals/identifiers, static op-suffix tables & TARGET_ETA multipliers."""
    if len(rel.parts) < 2 or rel.parts[0] != SIM_PACKAGE_ROOT:
        return []
    top = rel.parts[1][:-3] if rel.parts[1].endswith(".py") else rel.parts[1]
    if top not in _AG013_HARDENED_ROOTS:
        return list(_base_ag013_anti_hardcoding(rel, tree))

    pos = rel.as_posix()
    docs = _ag013_docstring_node_ids(tree)
    violations: list[Violation] = []
    seen: set[tuple[int, str, str]] = set()

    def _emit(line: int, kind: str, token: str, detail: str) -> None:
        key = (line, kind, token.lower())
        if key not in seen:
            seen.add(key)
            violations.append(Violation(
                "AG013", rel, line, f"AG013:{kind}:{pos}:{key[2]}", detail,
            ))

    _ag013_check_module_string_tables(tree, _emit, top)
    if top == "export":
        _ag013_check_export_receipts(tree, docs, _parents(tree), _emit)
    for node in ast.walk(tree):
        lineno = int(getattr(node, "lineno", 1))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docs:
                continue
            model_tok = _ag013_match_model_token(node.value)
            if model_tok is not None:
                _emit(
                    lineno, "model_literal", model_tok,
                    f"hardcoded model-specific string {node.value!r} ({model_tok}) forbidden in {top}",
                )
            elif _AG013_TARGET_ETA_RE.search(node.value):
                _emit(
                    lineno, "target_eta", "target_eta",
                    f"heuristic TARGET_ETA multiplier string {node.value!r} forbidden in {top}",
                )
        for field in ("id", "name", "attr", "arg"):
            val = getattr(node, field, None)
            if not isinstance(val, str) or not val:
                continue
            if _AG013_SUFFIX_TABLE_RE.match(val):
                _emit(
                    lineno, "suffix_table", val,
                    f"static op-suffix table {val!r} forbidden in {top}; use typed OpSpec traits/metadata",
                )
            elif _AG013_TARGET_ETA_RE.search(val):
                _emit(
                    lineno, "target_eta", val,
                    f"heuristic TARGET_ETA multiplier identifier {val!r} forbidden in {top}",
                )
            else:
                model_tok = _ag013_match_model_token(val)
                if model_tok is not None:
                    _emit(
                        lineno, "model_identifier", val,
                        f"hardcoded model-specific identifier {val!r} ({model_tok}) forbidden in {top}",
                    )
    return violations


def _ag014_manifest_for_revision(root: Path, revision: str | None) -> list[Violation]:
    manifest_text: str | None = None
    if revision is not None:
        manifest_rel = "mindie_sim/stacks/vllm_ascend/manifest.json"
        try:
            manifest_text = _git(root, "show", f"{revision}:{manifest_rel}")
        except RuntimeError:
            manifest_text = None
    if revision is None or manifest_text is not None:
        return check_ag014_manifest(root, manifest_text=manifest_text)
    return []


def scan_ag014(
    root: Path, /, *, revision: str | None = None,
) -> tuple[Violation, ...]:
    """Scan executable ``mindie_sim/**/*.py`` and ``manifest.json`` for AG014 model-name branches."""
    violations: list[Violation] = []
    for rel, text in _enumerate_py_files(root, revision=revision):
        try:
            tree = ast.parse(text, filename=rel.as_posix())
        except SyntaxError:
            continue
        violations.extend(_check_ag014_zero_model_name_branches(rel, tree))
    violations.extend(_ag014_manifest_for_revision(root, revision))
    violations.sort(key=lambda v: (v.path.as_posix(), v.line, v.rule, v.message))
    return tuple(violations)


def scan_repository(
    root: Path, /, *, revision: str | None = None,
    refs_root: Path | None = None, repo_map: Mapping[str, Path] | None = None,
    include_ag014: bool = True,
) -> tuple[Violation, ...]:
    """Return the sorted static findings under ``mindie_sim/**/*.py``."""
    violations: list[Violation] = []
    files = _enumerate_py_files(root, revision=revision)
    parsed_files: list[tuple[PurePosixPath, ast.Module]] = []
    texts: dict[PurePosixPath, str] = {}
    for rel, text in files:
        texts[rel] = text
        violations.extend(_check_ag001(rel))
        try:
            tree = ast.parse(text, filename=rel.as_posix())
        except SyntaxError as exc:
            violations.append(Violation(
                "AG005", rel, int(exc.lineno or 1),
                f"AG005:{rel.as_posix()}:syntax_error", f"syntax_error:{exc.msg}",
            ))
            continue
        parsed_files.append((rel, tree))
        parents = _parents(tree)
        violations.extend(_check_ag002(rel, tree, parents))
        violations.extend(_check_ag003_ag004(rel, tree, parents))
        violations.extend(_check_ag005(rel, text, tree, refs_root=refs_root, repo_map=repo_map))
        violations.extend(_check_ag006_flat_prediction(rel, tree))
        violations.extend(_check_ag007(rel, tree))
        violations.extend(_check_ag008_module(rel, text, tree, CANONICAL_CONCEPT_OWNERS))
        violations.extend(_check_ag009_silent_physical_defaults(rel, tree))
        violations.extend(_check_ag010_layer_dag(rel, tree))
        violations.extend(_check_ag013_anti_hardcoding(rel, tree))
        if include_ag014:
            violations.extend(_check_ag014_zero_model_name_branches(rel, tree))
    violations.extend(_check_ag008_repo_bounds([rel for rel, _ in files], texts=texts))
    violations.extend(_check_ag010_package_sccs(parsed_files))
    violations.extend(_check_ag011_operator_reachability(parsed_files))
    if include_ag014:
        violations.extend(_ag014_manifest_for_revision(root, revision))
    violations.sort(key=lambda v: (v.path.as_posix(), v.line, v.rule, v.message))
    return tuple(violations)


def filter_allowlisted(violations: Sequence[Violation], /) -> tuple[Violation, ...]:
    """Drop only tracked pre-Stage-3 legacy violations; never waive governed operators or AG014."""
    kept: list[Violation] = []
    for item in violations:
        in_pkg, _family, _fname = _is_governed_family_path(item.path)
        if in_pkg or item.rule == "AG014" or item.rule not in LEGACY_ALLOWLIST.get(item.path.as_posix(), frozenset()):
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
        if in_pkg or item.rule == "AG014":
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
    parser.add_argument(
        "--strict-ag014",
        action="store_true",
        help="compatibility no-op; AG014 zero model-name branching is enforced unconditionally by default",
    )
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
        all_current = scan_repository(
            root, refs_root=refs_root, repo_map=repo_map or None, include_ag014=True,
        )
        if args.base is not None:
            all_base = scan_repository(root, revision=args.base, include_ag014=True)
            active = new_violations(all_current, all_base)
        elif args.strict:
            active = all_current
        else:
            active = filter_allowlisted(all_current)
        if args.strict:
            active += scan_runtime(root) + tuple(check_ag012(root))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    for item in active:
        print(f"{item.path.as_posix()}:{item.line}: {item.rule} {item.message}")
    print(f"arch governance violations: {len(active)}")
    return EXIT_VIOLATIONS if active else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
