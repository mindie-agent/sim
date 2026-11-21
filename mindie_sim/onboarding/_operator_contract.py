"""Governed operator/template contract validator against real Sim & Profiler owners.

[Role] Validate an operator family package
  (``mindie_sim/prediction/operators/<family>/``) or template
  (``mindie_sim/templates/<name>.py``) against five concrete gates:
  (a) AST-grounded ``Upstream-ABI:`` declarations;
  (b) verified ``SourceRef`` / ``TemplateSourceRef`` provenance resolving
      module-level string constants and verifying Git blob SHA-256 hashes;
  (c) real Sim owner dispatch via ``builtin_family_registry().match(op)`` or
      exact target-matched ``TemplateCatalog.enumerate`` + ``.lower`` with
      executable-stage extraction & Lean 4 lower-bound certification derived
      strictly from the lowered executable program (rejecting zero-event
      recipes like ``program=lambda ctx: ()`` or empty ``KernelProgram``);
  (d) real Profiler owner dispatch via ``resolve_builder_strict`` cross-checked
      against target family, declared leaf ``Upstream-ABI:``, and Sim impl; and
  (e) ``tests/TAXONOMY.toml`` registration on disk + ``TILING_AXES`` check.
[Invariants] Strictly read-only over target files; signs contract reports with
  in-process HMAC; records explicit ``checked``/``lowered``/``verified`` flags
  so skipped checks are ineligible for release certification; ``TimeoutError``
  propagates before ``OSError``.
[Timeouts & Locks] Bounded local ``git cat-file`` and provenance subprocesses;
  zero locks or device contexts.
[Side-Effect Policy] Zero filesystem writes.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any

from mindie_sim.backends import ascend
from mindie_sim.observations.multilevel_stream import SHA256_HEX_LENGTH
from mindie_sim.optimization.dual_bound import (
    DagEdge,
    EventDagSpec,
    ResourceChoiceProgram,
    certify_lower_bound,
    check_optimality_certificate,
)
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.registry import FamilyRegistry, builtin_family_registry
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec
from mindie_sim.templates.registry import TemplateCatalog, builtin_template_catalog

from ._gate_runners import (
    resolve_profiler_builder_strict,
    resolve_repo_root,
    run_template_provenance_gate,
)
from ._lean_and_receipts import sign_stage_report
from .contracts import (
    HEX_COMMIT_LEN,
    NS_PER_MS,
    assert_sanitized_payload,
    subprocess_window_kwargs,
)

_UPSTREAM_ABI_LINE_RE = re.compile(r"^\s*Upstream-ABI\s*:\s*([A-Za-z_][A-Za-z0-9_:.<>/-]*)\s*$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_GIT_CAT_TIMEOUT_S = 15
_NS_PER_SECOND = 1_000_000_000.0
_REQUIRED_FAMILY_FILES = ("__init__.py", "common.py", "arch22.py", "arch35.py")
_DEFAULT_PROBE_SKU = "Ascend910B2"
_DEFAULT_PROBE_SKU_A5 = "Ascend950DT_9582"
_DEFAULT_PROBE_SHAPE = (1, 128)
_DEFAULT_TEMPLATE_DIM = 128
_TEMPLATE_ID_CONST_NAMES = frozenset({"TEMPLATE_ID", "TEMPLATE_IDS", "CATLASS_TEMPLATE_IDS", "DSL_KERNEL_IDS"})
_MAX_STAGE_WALK_DEPTH = 16


def _extract_upstream_abi(tree: ast.AST) -> str:
    doc = ast.get_docstring(tree) or ""
    for line in doc.splitlines():
        match = _UPSTREAM_ABI_LINE_RE.match(line)
        if match:
            return match.group(1).strip()
    return ""


def _eval_str_expr(node: ast.AST | None, constants: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _eval_str_expr(node.left, constants), _eval_str_expr(node.right, constants)
        if left is not None and right is not None:
            return left + right
    return None


def _extract_module_str_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            val = _eval_str_expr(stmt.value, constants)
            if val is not None:
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = val
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            val = _eval_str_expr(stmt.value, constants)
            if val is not None:
                constants[stmt.target.id] = val
    return constants


def _extract_declared_template_ids(tree: ast.Module, constants: Mapping[str, str]) -> set[str]:
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
        single = _eval_str_expr(val_node, constants)
        if single:
            found.add(single)
        elif isinstance(val_node, (ast.Tuple, ast.List, ast.Set)):
            for elt in val_node.elts:
                item = _eval_str_expr(elt, constants)
                if item:
                    found.add(item)
        elif isinstance(val_node, ast.Dict):
            for key in val_node.keys:
                item = _eval_str_expr(key, constants)
                if item:
                    found.add(item)
    return found


def _extract_source_refs(tree: ast.Module, path: Path, constants: Mapping[str, str]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        fname = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        if fname not in ("SourceRef", "TemplateSourceRef"):
            continue
        pos = [_eval_str_expr(arg, constants) for arg in node.args]
        kw = {str(k.arg): _eval_str_expr(k.value, constants) for k in node.keywords if k.arg}
        refs.append({
            "file": path.name,
            "repository": kw.get("repository") or (pos[0] if len(pos) > 0 else None) or "",
            "revision": kw.get("revision") or (pos[1] if len(pos) > 1 else None) or "",
            "relative_path": kw.get("relative_path") or (pos[2] if len(pos) > 2 else None) or "",
            "source_sha256": kw.get("source_sha256") or (pos[3] if len(pos) > 3 else None) or "",
        })
    return refs


def _extract_symbol_mentions(tree: ast.AST) -> set[str]:
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


def _extract_top_level_names(tree: ast.Module) -> set[str]:
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


def _verify_git_blob_sha256(
    ref: Mapping[str, str], *, refs_root: Path | None, repo_map: Mapping[str, Path], require_mounted_refs: bool,
) -> tuple[bool, bool, str]:
    repo, rev, rel, expected = ref["repository"], ref["revision"], ref["relative_path"], ref["source_sha256"].lower()
    repo_dir = repo_map.get(repo) or repo_map.get(repo.rsplit("/", 1)[-1])
    if repo_dir is None and refs_root is not None:
        for cand_name in (repo, repo.rsplit("/", 1)[-1]):
            candidate = refs_root / cand_name
            if candidate.exists():
                repo_dir = candidate
                break
    if repo_dir is None:
        if refs_root is not None or repo_map or require_mounted_refs:
            return False, False, f"source_repo_not_mounted:{repo}"
        return True, False, "unverified_no_refs_mounted"
    try:
        done = subprocess.run(
            ["git", "-C", str(repo_dir), "cat-file", "-p", f"{rev}:{rel}"],
            capture_output=True, timeout=_GIT_CAT_TIMEOUT_S, check=False, **subprocess_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"git_cat_file_timeout:{repo}:{rel}") from exc
    except TimeoutError:
        raise
    except OSError as exc:
        return False, False, f"git_cat_file_os_error:{repo}:{exc}"
    if done.returncode != 0:
        return False, False, f"git_blob_missing:{repo}:{rev}:{rel}"
    actual = hashlib.sha256(done.stdout).hexdigest().lower()
    if actual != expected:
        return False, False, f"source_sha256_mismatch:{repo}:{rel}:expected={expected[:12]}:actual={actual[:12]}"
    return True, True, "verified"


def _check_taxonomy_registration(repo_root: Path, target_rel: str, family: str) -> tuple[bool, str]:
    taxonomy = repo_root / "tests" / "TAXONOMY.toml"
    if not taxonomy.is_file():
        return False, "taxonomy_toml_missing"
    try:
        parsed = tomllib.loads(taxonomy.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return False, f"taxonomy_toml_invalid:{exc}"
    norm_target = target_rel.rstrip("/")
    for mod in parsed.get("module", ()):
        if not isinstance(mod, Mapping):
            continue
        mod_path = str(mod.get("path") or "")
        owns = [str(o).rstrip("/") for o in mod.get("owns", ())]
        if any(
            norm_target == owned or norm_target.startswith(owned + "/") or owned.startswith(norm_target + "/")
            for owned in owns
        ) or f"test_{family}.py" in mod_path:
            test_file = repo_root / "tests" / mod_path
            return (True, mod_path) if test_file.is_file() else (False, f"taxonomy_module_missing_on_disk:{mod_path}")
    return False, f"taxonomy_registration_missing:{norm_target}"


def _geometry_for_die(die: Any, dp: Any) -> Geometry:
    return Geometry(compute_units=die.compute_units, resource_units=die.resource_units, buffers=die.buffers, buffer_units=die.buffer_units, rates=die.rates, compute_peaks=die.compute_peaks, compute_arch=die.compute_arch, datapath=dp)


def _build_default_planning_context(op: OpSpec, *, sku: str = _DEFAULT_PROBE_SKU) -> OperatorPlanningContext:
    hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
    die = hw.dies[0]
    dp = datapath_for_die(die)
    return OperatorPlanningContext(occurrence=OpOccurrence(0, 0, 0, op.op_id), geometry=_geometry_for_die(die, dp), datapath=dp)


def _finite_rate(val: float) -> float:
    return val if math.isfinite(val) and val > 0.0 else 0.0


def _lookup_matrix_rate(geom: Geometry, dtype_key: str) -> float:
    arch = geom.compute_arch
    if arch is not None and arch.has_native_modes() and not arch.supports_native("matrix", dtype_key):
        return 0.0
    table = dict(geom.compute_peaks)
    try:
        return _finite_rate(float(table[dtype_key]))
    except (KeyError, TypeError, ValueError):
        return 0.0


def _resource_rate_per_s(resource_name: str, geom: Geometry) -> float:
    rates = dict(geom.rates)
    if resource_name in rates and float(rates[resource_name]) > 0.0:
        return float(rates[resource_name])
    aliases = {"hbm_bytes": "hbm", "mte2_bytes": "mte2_aiv", "mte2": "mte2_aiv", "mte3_bytes": "mte3", "vector_ops": "vector"}
    mapped = aliases.get(resource_name, "")
    if mapped and mapped in rates and float(rates[mapped]) > 0.0:
        return float(rates[mapped])
    direct_peak = _lookup_matrix_rate(geom, resource_name)
    if direct_peak > 0.0:
        return direct_peak
    if resource_name.startswith("cube"):
        suffix = resource_name.split(":", 1)[-1] if ":" in resource_name else "bf16"
        return _lookup_matrix_rate(geom, suffix) or _lookup_matrix_rate(geom, "bf16")
    return 0.0


def _collect_executable_stages(obj: Any, *, depth: int = 0) -> list[Any]:
    """Recursively extract executable Stage/NativeProgram entries from a lowered Program."""
    if obj is None or depth > _MAX_STAGE_WALK_DEPTH:
        return []
    if isinstance(obj, (tuple, list, dict, str, bytes, int, float, bool)):
        return []
    found: list[Any] = []
    if hasattr(obj, "work") and hasattr(obj, "name") and hasattr(obj, "after"):
        found.append(obj)
        nested = getattr(obj, "program", None)
        if nested is not None:
            found.extend(_collect_executable_stages(nested, depth=depth + 1))
        return found
    for attr in ("stages", "branches", "phases", "programs"):
        children = getattr(obj, attr, None)
        if isinstance(children, (tuple, list)):
            for child in children:
                found.extend(_collect_executable_stages(child, depth=depth + 1))
    for attr in ("placed", "program", "inner", "body", "template"):
        sub = getattr(obj, attr, None)
        if sub is not None and sub is not obj:
            found.extend(_collect_executable_stages(sub, depth=depth + 1))
    if not found and hasattr(obj, "work") and callable(getattr(obj, "start", None)):
        found.append(obj)
    return found


def _certify_stages_dual_bound(
    stages: Sequence[Any], geom: Geometry
) -> tuple[bool, int, str]:
    """Derive ``(ResourceChoiceProgram, EventDagSpec)`` strictly from lowered stages and certify."""
    res_totals_ns: dict[str, int] = {}
    stage_quanta: list[int] = []
    name_to_idx: dict[str, int] = {}
    edges: list[DagEdge] = []
    for st in stages:
        work_map = getattr(st, "work", None)
        latency_ms = float(getattr(st, "latency", 0.0) or 0.0)
        stage_ns = int(round(latency_ms * NS_PER_MS)) if latency_ms > 0.0 else 0
        if isinstance(work_map, Mapping):
            for res_name, amt_raw in work_map.items():
                amt = float(amt_raw)
                if amt <= 0.0:
                    continue
                rate = _resource_rate_per_s(str(res_name), geom)
                if rate <= 0.0:
                    return False, 0, f"resolved_stage_resource_rate_missing:{res_name}"
                dur_ns = max(1, math.ceil(amt / rate * _NS_PER_SECOND))
                res_totals_ns[str(res_name)] = res_totals_ns.get(str(res_name), 0) + dur_ns
                stage_ns = max(stage_ns, dur_ns)
        if stage_ns > 0:
            idx = len(stage_quanta)
            st_name = str(getattr(st, "name", f"s{idx}"))
            name_to_idx[st_name] = idx
            stage_quanta.append(stage_ns)
            for dep in getattr(st, "after", ()) or ():
                if str(dep) in name_to_idx:
                    edges.append(DagEdge(name_to_idx[str(dep)], idx, 0))
    if not stage_quanta:
        return False, 0, "resolved_operator_program_zero_executable_events"
    global_work = tuple(sorted(res_totals_ns.values())) if res_totals_ns else (max(stage_quanta),)
    prog = ResourceChoiceProgram(num_resources=len(global_work), global_work=global_work, site_modes=())
    dag = EventDagSpec(node_quantum_ns=tuple(stage_quanta), edges=tuple(edges))
    cert, _ = certify_lower_bound(prog, dag)
    if not check_optimality_certificate(prog, dag, cert):
        return False, int(cert.l_star_lean_ns), "lean_optimality_certificate_rejected"
    if cert.l_star_lean_ns <= 0:
        return False, int(cert.l_star_lean_ns), "lean_lower_bound_nonpositive_on_lowered_program"
    return True, int(cert.l_star_lean_ns), "verified"


def _verify_resolved_operator_dual_bound(
    resolved: Any, ctx: OperatorPlanningContext,
) -> tuple[bool, int, str]:
    """Lower ``resolved.recipe``, require executable stages, and certify Lean dual bound."""
    recipe = getattr(resolved, "recipe", None)
    if not isinstance(recipe, OperatorPlan):
        return False, 0, "resolved_operator_missing_operator_plan"
    if not callable(recipe.program):
        return False, 0, "resolved_operator_plan_missing_callable_program"
    if recipe.pipeline_spec is None and getattr(resolved, "descriptor", None) is None:
        return False, 0, "resolved_operator_missing_pipeline_spec_or_descriptor"

    arch = ctx.geometry.compute_arch
    if hasattr(recipe.work, "matrix"):
        for dtype_key, flops in dict(recipe.work.matrix).items():
            if float(flops) <= 0.0:
                continue
            dtype_str = str(dtype_key)
            if arch is not None and arch.has_native_modes() and not arch.supports_native("matrix", dtype_str):
                return False, 0, f"resolved_matrix_mode_unsupported:{dtype_str}"
            if _lookup_matrix_rate(ctx.geometry, dtype_str) <= 0.0:
                return False, 0, f"resolved_matrix_peak_missing:{dtype_str}"

    from mindie_sim.prediction.program.binding import RecipeContext

    r_ctx = RecipeContext(
        op=getattr(resolved, "op", None), work=dict(recipe.work.resources),
        resource_names={str(k): str(k) for k in dict(ctx.geometry.rates)},
        rates=dict(ctx.geometry.rates), core_count=max(1, int(ctx.geometry.compute_units)),
        resource_units=dict(ctx.geometry.resource_units), compute_peaks=dict(ctx.geometry.compute_peaks),
    )
    try:
        lowered = recipe.lower(r_ctx)
    except TimeoutError:
        raise
    except Exception as exc:
        return False, 0, f"resolved_operator_program_lower_failed:{type(exc).__name__}:{exc}"
    if lowered is None or not callable(getattr(lowered, "start", None)):
        return False, 0, "resolved_operator_program_zero_executable_events"
    stages = _collect_executable_stages(lowered)
    if not stages:
        return False, 0, "resolved_operator_program_zero_executable_events"
    return _certify_stages_dual_bound(stages, ctx.geometry)


def _lower_and_verify_template(cat: TemplateCatalog, tier: str, matched_id: str) -> tuple[bool, int, int, str]:
    """Enumerate and actually lower a registered K2/K3 template and certify its stages."""
    if matched_id == "fuse_region":
        return False, 0, 0, "template_fuse_region_requires_concrete_kernel_id"
    problem = {"m": _DEFAULT_TEMPLATE_DIM, "n": _DEFAULT_TEMPLATE_DIM, "k": _DEFAULT_TEMPLATE_DIM, "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "bf16"}
    last_err = "template_enumeration_empty"
    for sku in (_DEFAULT_PROBE_SKU, _DEFAULT_PROBE_SKU_A5):
        hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
        die = hw.dies[0]
        dp = datapath_for_die(die)
        geom = _geometry_for_die(die, dp)
        try:
            specs = tuple(cat.enumerate(tier, problem, dp, template_id=matched_id))  # type: ignore[arg-type]
        except Exception as exc:
            last_err = f"template_enumerate_failed:{type(exc).__name__}:{exc}"
            continue
        if not specs:
            continue
        try:
            lowered = cat.lower(tier, specs[0], problem, dp)  # type: ignore[arg-type]
        except Exception as exc:
            last_err = f"template_lower_failed:{type(exc).__name__}:{exc}"
            continue
        stages = _collect_executable_stages(getattr(lowered, "program", lowered))
        if not stages:
            return False, 0, 0, "template_lower_emitted_zero_stages"
        summary = getattr(lowered, "summary", None)
        has_work = bool(getattr(summary, "semantic_work", ()) or getattr(lowered, "semantic_work", ()))
        has_obls = bool(
            getattr(summary, "read_obligations", ())
            or getattr(summary, "write_obligations", ())
            or getattr(lowered, "obligations", ())
        )
        if not (has_work and has_obls):
            return False, 0, len(stages), "template_lower_missing_obligations_or_semantic_work"
        ok_dual, l_star, reason = _certify_stages_dual_bound(stages, geom)
        return ok_dual, l_star, len(stages), reason
    return False, 0, 0, last_err


def _verify_sim_owner_dispatch(
    *,
    is_family_pkg: bool,
    family: str,
    template_id: str,
    declared_template_ids: set[str],
    family_registry: FamilyRegistry | None,
    template_catalog: TemplateCatalog | None,
    sample_op: OpSpec | None,
    planning_context: OperatorPlanningContext | None,
) -> tuple[bool, dict[str, Any], list[str]]:
    violations: list[str] = []
    if is_family_pkg:
        reg = family_registry if family_registry is not None else builtin_family_registry()
        spec = reg.spec_for_family(family)
        if spec is None:
            for cand in reg.specs():
                if cand.name == family or family in cand.families:
                    spec = cand
                    break
        if sample_op is not None:
            matched = reg.match(sample_op)
        elif spec is not None:
            op_family = sorted(spec.families)[0] if spec.families else family
            op_recipe = sorted(spec.owned_recipe_ids)[0] if spec.owned_recipe_ids else ""
            sample_op = OpSpec(
                op_id=f"onboard_probe_{family}", family=op_family, phase="decode",
                stage=0, layer=0, requests=1,
                semantic_shape=_DEFAULT_PROBE_SHAPE, local_shape=_DEFAULT_PROBE_SHAPE,
                padded_shape=_DEFAULT_PROBE_SHAPE, launch_shape=_DEFAULT_PROBE_SHAPE,
                dtype="bf16", recipe=op_recipe,
            )
            matched = reg.match(sample_op)
        else:
            matched = None
        if matched is None or sample_op is None:
            violations.append(f"sim_registry_unmatched:{family}")
            return False, {"owner": "builtin_family_registry", "matched": False, "lowered": False}, violations
        ctx = planning_context if planning_context is not None else _build_default_planning_context(sample_op)
        detail: dict[str, Any] = {
            "owner": "builtin_family_registry", "matched": True, "spec_name": matched.name,
            "owned_recipe_count": len(matched.owned_recipe_ids),
            "resolved": False, "lowered": False, "lean_lower_bound_ns": 0,
        }
        try:
            resolved = matched.resolve(sample_op, ctx)
        except (ModelingError, NotImplementedError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            violations.append(f"sim_owner_resolve_failed:{family}:{exc}")
            return False, detail, violations
        lean_ok, l_star, lean_reason = _verify_resolved_operator_dual_bound(resolved, ctx)
        detail["resolved"] = True
        detail["lowered"] = lean_ok
        detail["resolved_implementation"] = str(getattr(resolved, "implementation", ""))
        detail["lean_lower_bound_ns"] = l_star
        detail["lean_status"] = lean_reason
        if not lean_ok:
            violations.append(f"sim_owner_dual_bound_failed:{family}:{lean_reason}")
            return False, detail, violations
        return True, detail, violations

    cat = template_catalog if template_catalog is not None else builtin_template_catalog()
    k2_ids, k3_ids = set(cat.ids("K2")), set(cat.ids("K3"))
    all_ids = k2_ids | k3_ids
    requested_id = template_id.strip() or family
    allowed_for_target = set(declared_template_ids) | {family}
    if requested_id not in allowed_for_target and requested_id.rsplit(".", 1)[-1] != family:
        violations.append(f"template_id_unrelated_to_target:requested={requested_id}:target={family}")
        return False, {"owner": "builtin_template_catalog", "matched": False, "lowered": False}, violations
    matched_id = requested_id if requested_id in all_ids else ""
    if not matched_id:
        violations.append(f"template_catalog_unregistered:{requested_id}")
        return False, {"owner": "builtin_template_catalog", "matched": False, "lowered": False}, violations
    tier = "K2" if matched_id in k2_ids else "K3"
    lower_ok, l_star, stage_cnt, reason = _lower_and_verify_template(cat, tier, matched_id)
    detail = {
        "owner": "builtin_template_catalog", "matched": True, "tier": tier,
        "template_id": matched_id, "lowered": lower_ok,
        "stage_count": stage_cnt, "lean_lower_bound_ns": l_star, "lean_status": reason,
    }
    if not lower_ok:
        violations.append(f"template_lower_verification_failed:{matched_id}:{reason}")
    return lower_ok, detail, violations


def validate_operator_contract(
    *,
    target_path: str | Path,
    repo_root: str | Path | None = None,
    refs_root: str | Path | None = None,
    repo_map: Mapping[str, str | Path] | None = None,
    profiler_root: str | Path | None = None,
    profiler_family: str = "",
    profiler_recipe: str = "",
    profiler_impl: str = "",
    profiler_dtypes: Sequence[str] = (),
    profiler_meta: Mapping[str, Any] | None = None,
    family_registry: FamilyRegistry | None = None,
    template_catalog: TemplateCatalog | None = None,
    template_id: str = "",
    sample_op: OpSpec | None = None,
    planning_context: OperatorPlanningContext | None = None,
    require_sim_owner: bool = True,
    require_taxonomy: bool = True,
    require_profiler_builder: bool = True,
    require_tiling_axes: bool = True,
    require_mounted_refs: bool = False,
) -> dict[str, Any]:
    """Stage 1B: Validate governed operator/template contract against real Sim & Profiler owners."""
    root = resolve_repo_root(repo_root)
    raw_target = Path(target_path)
    target = raw_target if raw_target.is_absolute() else (root / raw_target)
    if not target.exists():
        raise ModelingError(f"operator_contract_target_not_found:{target_path}")
    target = target.resolve()
    target_rel = target.relative_to(root).as_posix() if target.is_relative_to(root) else target.name

    is_family_pkg = target.is_dir()
    family = target.name if is_family_pkg else target.stem.removesuffix("_spec")
    violations: list[str] = []
    files: list[Path] = []
    if is_family_pkg:
        for req_name in _REQUIRED_FAMILY_FILES:
            cand = target / req_name
            if not cand.is_file():
                violations.append(f"missing_required_family_file:{req_name}")
            else:
                files.append(cand)
        for extra in sorted(target.glob("*.py")):
            if extra not in files:
                files.append(extra)
    else:
        files = [target]

    abis: dict[str, str] = {}
    all_refs: list[dict[str, str]] = []
    symbols: set[str] = set()
    declared_builder = ""
    declared_template_ids: set[str] = set()
    shared_constants: dict[str, str] = {}
    parsed_files: list[tuple[Path, ast.Module]] = []
    for py_file in files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        parsed_files.append((py_file, tree))
        shared_constants.update(_extract_module_str_constants(tree))
    for py_file, tree in parsed_files:
        top_names = _extract_top_level_names(tree)
        symbols |= _extract_symbol_mentions(tree)
        file_consts = {**shared_constants, **_extract_module_str_constants(tree)}
        declared_template_ids |= _extract_declared_template_ids(tree, file_consts)
        file_refs = _extract_source_refs(tree, py_file, file_consts)
        all_refs.extend(file_refs)
        abi = _extract_upstream_abi(tree)
        if py_file.name in ("arch22.py", "arch35.py") or not is_family_pkg:
            if not abi:
                violations.append(f"missing_upstream_abi_header:{py_file.name}")
            else:
                abis[py_file.name] = abi
            if is_family_pkg and "STRATEGY" not in top_names:
                violations.append(f"missing_strategy_export:{py_file.name}")
            if "SOURCES" not in top_names and not file_refs:
                violations.append(f"missing_sources_declaration:{py_file.name}")
        if not declared_builder and "PROFILER_BENCH_BUILDER" in file_consts:
            declared_builder = file_consts["PROFILER_BENCH_BUILDER"]

    if is_family_pkg and len(abis) >= 2 and len(set(abis.values())) < len(abis):
        violations.append("duplicate_upstream_abi_across_arch_leaves")

    resolved_refs_root = Path(refs_root).resolve() if refs_root is not None else None
    resolved_repo_map = {str(k): Path(v).resolve() for k, v in (repo_map or {}).items()}
    if not all_refs:
        violations.append("source_ref_missing")
    source_checks: list[dict[str, Any]] = []
    all_blobs_verified = bool(all_refs)
    for ref in all_refs:
        ref_errors: list[str] = []
        if not ref["repository"]:
            ref_errors.append("repository_empty")
        if len(ref["revision"]) < HEX_COMMIT_LEN or not _HEX_RE.fullmatch(ref["revision"]):
            ref_errors.append("revision_not_full_sha")
        if not ref["relative_path"] or ref["relative_path"].startswith("/") or ".." in Path(ref["relative_path"]).parts:
            ref_errors.append("relative_path_unsafe")
        if len(ref["source_sha256"]) != SHA256_HEX_LENGTH or not _HEX_RE.fullmatch(ref["source_sha256"]):
            ref_errors.append("source_sha256_invalid")
        blob_status, blob_verified = "skipped_invalid_ref", False
        if not ref_errors:
            ok_blob, blob_verified, blob_status = _verify_git_blob_sha256(
                ref, refs_root=resolved_refs_root, repo_map=resolved_repo_map, require_mounted_refs=require_mounted_refs,
            )
            if not ok_blob:
                ref_errors.append(blob_status)
        if not blob_verified:
            all_blobs_verified = False
        if ref_errors:
            violations.append(f"source_ref_invalid:{ref['file']}:{','.join(ref_errors)}")
        source_checks.append({**ref, "status": blob_status, "blob_verified": blob_verified, "errors": ref_errors})

    provenance_gate_outcome = None
    if not is_family_pkg and target_rel.startswith("mindie_sim/templates/") and (resolved_refs_root is not None or resolved_repo_map):
        provenance_gate_outcome = run_template_provenance_gate(
            repo_root=root, refs_root=resolved_refs_root, repo_map=resolved_repo_map,
        )
        if not provenance_gate_outcome.passed:
            violations.append(f"template_provenance_gate_failed:{provenance_gate_outcome.detail}")

    has_semantic = "SemanticWorkItem" in symbols
    has_obligations = "TensorRangeObligation" in symbols
    has_tiling_axes = "TilingAxes" in symbols or "TILING_AXES" in symbols
    if not has_semantic:
        violations.append("semantic_work_item_declaration_missing")
    if not has_obligations:
        violations.append("tensor_range_obligation_declaration_missing")
    if require_tiling_axes and not has_tiling_axes:
        violations.append("tiling_axes_declaration_missing")

    sim_owner_detail: dict[str, Any] = {"checked": False, "matched": False, "lowered": False}
    if require_sim_owner:
        _, sim_owner_detail, sim_violations = _verify_sim_owner_dispatch(
            is_family_pkg=is_family_pkg, family=family, template_id=template_id,
            declared_template_ids=declared_template_ids,
            family_registry=family_registry, template_catalog=template_catalog,
            sample_op=sample_op, planning_context=planning_context,
        )
        sim_owner_detail["checked"] = True
        violations.extend(sim_violations)

    taxonomy_ok, taxonomy_detail = _check_taxonomy_registration(root, target_rel, family)
    if require_taxonomy and not taxonomy_ok:
        violations.append(taxonomy_detail)

    allowed_prof_families = {x for x in (family, declared_builder) if x}
    req_prof_family = profiler_family.strip()
    eff_prof_family = (req_prof_family or declared_builder or family).strip()
    eff_prof_recipe = (profiler_recipe or eff_prof_family).strip()
    eff_prof_impl = (profiler_impl or next(iter(abis.values()), "")).strip()
    profiler_check: dict[str, Any] = {
        "checked": bool(require_profiler_builder), "verified": False,
        "family": eff_prof_family, "recipe": eff_prof_recipe, "impl": eff_prof_impl,
        "resolved_builder": "", "resolved_abi": "",
    }
    if require_profiler_builder:
        if req_prof_family and req_prof_family not in allowed_prof_families:
            err = f"profiler_family_unrelated_to_target:requested={req_prof_family}:allowed={sorted(allowed_prof_families)}"
            profiler_check["error"] = err
            violations.append(err)
        else:
            try:
                b_name, b_abi = resolve_profiler_builder_strict(
                    eff_prof_family, eff_prof_recipe, eff_prof_impl,
                    dtypes=profiler_dtypes, meta=profiler_meta, profiler_root=profiler_root,
                )
            except ModelingError as exc:
                profiler_check["error"] = str(exc)
                violations.append(str(exc))
            else:
                sim_impl = str(sim_owner_detail.get("resolved_implementation") or "").strip()
                declared_abis = set(abis.values())
                allowed_abis = (
                    {sim_impl}
                    if sim_impl and (sim_impl in declared_abis or sim_impl.startswith("aclnn"))
                    else (declared_abis | ({sim_impl} if sim_impl else set()))
                )
                if allowed_abis and b_abi not in allowed_abis:
                    err = f"profiler_abi_mismatch:resolved={b_abi}:declared={sorted(allowed_abis)}"
                    profiler_check.update(resolved_builder=b_name, resolved_abi=b_abi, error=err)
                    violations.append(err)
                else:
                    profiler_check.update(verified=True, resolved_builder=b_name, resolved_abi=b_abi)

    result: dict[str, Any] = {
        "stage": "validate-contract",
        "ok": not violations,
        "status": "pass" if not violations else "fail",
        "target": target_rel,
        "family": family,
        "upstream_abis": abis,
        "source_refs_checked": len(source_checks),
        "source_blobs_verified": all_blobs_verified,
        "source_refs": source_checks,
        "template_provenance_gate": provenance_gate_outcome.to_dict() if provenance_gate_outcome is not None else None,
        "semantic_work_declared": has_semantic,
        "tensor_obligations_declared": has_obligations,
        "tiling_axes_declared": has_tiling_axes,
        "sim_owner_dispatch": sim_owner_detail,
        "taxonomy": {"checked": bool(require_taxonomy), "registered": taxonomy_ok, "detail": taxonomy_detail},
        "profiler_builder": profiler_check,
        "violations": violations,
    }
    result["report_hmac"] = sign_stage_report("operator_contract", result)
    assert_sanitized_payload(result, field_name="operator_contract_validation")
    return result


__all__ = ["validate_operator_contract"]
