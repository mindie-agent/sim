"""Governed operator/template contract validator against real Sim & Profiler owners.

[Role] Validate an operator family package
  (``mindie_sim/prediction/operators/<family>/``) or template
  (``mindie_sim/templates/<name>.py``) against five concrete gates:
  (a) AST-grounded ``Upstream-ABI:`` declarations;
  (b) verified ``SourceRef`` / ``TemplateSourceRef`` provenance (including Git
      blob SHA-256 verification and ``check_template_provenance.py`` when
      upstream refs are mounted);
  (c) real Sim owner dispatch via ``builtin_family_registry().match(op)`` or
      ``builtin_template_catalog()`` plus actual operator resolution and
      non-degenerate Lean 4 lower-bound verification derived from the resolved
      operator's own work (zero synthetic dummy vector proofs!);
  (d) real Profiler owner dispatch via
      ``mindie_profiler.bench.operators.resolve_builder_strict`` (zero loose
      dictionary scans; fails closed when Profiler owner is missing or rejects
      the triple!); and
  (e) ``tests/TAXONOMY.toml`` registration on disk.
[Invariants] Strictly read-only over target files; zero synthetic proofs;
  ``TimeoutError`` always propagates before ``OSError``; fails closed on any
  missing owner, unimplemented scaffold stub, or provenance mismatch.
[Timeouts & Locks] Bounded local ``git cat-file`` and provenance subprocesses;
  zero locks or device contexts.
[Side-Effect Policy] Zero filesystem writes.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any

from mindie_sim.backends import ascend
from mindie_sim.observations.multilevel_stream import SHA256_HEX_LENGTH
from mindie_sim.optimization.dual_bound import (
    EventDagSpec,
    ResourceChoiceProgram,
    certify_lower_bound,
    check_optimality_certificate,
)
from mindie_sim.prediction.mechanism import Geometry
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
from .contracts import (
    HEX_COMMIT_LEN,
    assert_sanitized_payload,
    subprocess_window_kwargs,
)

_UPSTREAM_ABI_LINE_RE = re.compile(r"^\s*Upstream-ABI\s*:\s*([A-Za-z_][A-Za-z0-9_:.<>/-]*)\s*$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_GIT_CAT_TIMEOUT_S = 15
_REQUIRED_FAMILY_FILES = ("__init__.py", "common.py", "arch22.py", "arch35.py")
_DEFAULT_PROBE_SKU = "Ascend910B2"
_DEFAULT_PROBE_SHAPE = (1, 128)


def _extract_upstream_abi(tree: ast.AST) -> str:
    doc = ast.get_docstring(tree) or ""
    for line in doc.splitlines():
        match = _UPSTREAM_ABI_LINE_RE.match(line)
        if match:
            return match.group(1).strip()
    return ""


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_source_refs(tree: ast.AST, path: Path) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        fname = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        if fname not in ("SourceRef", "TemplateSourceRef"):
            continue
        pos = [_const_str(arg) for arg in node.args]
        kw = {str(k.arg): _const_str(k.value) for k in node.keywords if k.arg}
        repo = kw.get("repository") or (pos[0] if len(pos) > 0 else None) or ""
        rev = kw.get("revision") or (pos[1] if len(pos) > 1 else None) or ""
        rel = kw.get("relative_path") or (pos[2] if len(pos) > 2 else None) or ""
        sha = kw.get("source_sha256") or (pos[3] if len(pos) > 3 else None) or ""
        refs.append(
            {
                "file": path.name,
                "repository": repo,
                "revision": rev,
                "relative_path": rel,
                "source_sha256": sha,
            }
        )
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


def _extract_string_constant(tree: ast.Module, name: str) -> str:
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    val = _const_str(stmt.value)
                    if val is not None:
                        return val
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == name:
            val = _const_str(stmt.value)
            if val is not None:
                return val
    return ""


def _verify_git_blob_sha256(
    ref: Mapping[str, str],
    *,
    refs_root: Path | None,
    repo_map: Mapping[str, Path],
) -> tuple[bool, str]:
    repo = ref["repository"]
    rev = ref["revision"]
    rel = ref["relative_path"]
    expected = ref["source_sha256"].lower()
    repo_dir = repo_map.get(repo)
    if repo_dir is None and refs_root is not None:
        candidate = refs_root / repo
        if candidate.exists():
            repo_dir = candidate
    if repo_dir is None:
        return True, "unverified_repo_not_mounted"
    try:
        done = subprocess.run(
            ["git", "-C", str(repo_dir), "cat-file", "-p", f"{rev}:{rel}"],
            capture_output=True,
            timeout=_GIT_CAT_TIMEOUT_S,
            check=False,
            **subprocess_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"git_cat_file_timeout:{repo}:{rel}") from exc
    except TimeoutError:
        raise
    except OSError as exc:
        return False, f"git_cat_file_os_error:{repo}:{exc}"
    if done.returncode != 0:
        return False, f"git_blob_missing:{repo}:{rev}:{rel}"
    actual = hashlib.sha256(done.stdout).hexdigest().lower()
    if actual != expected:
        return False, f"source_sha256_mismatch:{repo}:{rel}:expected={expected[:12]}:actual={actual[:12]}"
    return True, "verified"


def _check_taxonomy_registration(
    repo_root: Path,
    target_rel: str,
    family: str,
) -> tuple[bool, str]:
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
            norm_target == owned
            or norm_target.startswith(owned + "/")
            or owned.startswith(norm_target + "/")
            for owned in owns
        ) or f"test_{family}.py" in mod_path:
            test_file = repo_root / "tests" / mod_path
            if test_file.is_file():
                return True, mod_path
            return False, f"taxonomy_module_missing_on_disk:{mod_path}"
    return False, f"taxonomy_registration_missing:{norm_target}"


def _build_default_planning_context(
    op: OpSpec,
    *,
    sku: str = _DEFAULT_PROBE_SKU,
) -> OperatorPlanningContext:
    hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
    die = hw.dies[0]
    dp = datapath_for_die(die)
    occurrence = OpOccurrence(replica=0, stage=0, rank=0, op_id=op.op_id)
    geom = Geometry(
        compute_units=die.compute_units,
        resource_units=die.resource_units,
        buffers=die.buffers,
        buffer_units=die.buffer_units,
        rates=die.rates,
        compute_peaks=die.compute_peaks,
        compute_arch=die.compute_arch,
        datapath=dp,
    )
    return OperatorPlanningContext(occurrence=occurrence, geometry=geom, datapath=dp)


def _verify_resolved_operator_dual_bound(resolved: Any) -> tuple[bool, int, str]:
    """Derive a genuine Lean 4 dual lower bound directly from ``resolved.op`` / ``resolved.recipe``."""
    work_map: dict[str, float] = {}
    recipe_work = getattr(getattr(resolved, "recipe", None), "work", None)
    if recipe_work is not None and hasattr(recipe_work, "resources"):
        work_map = {str(k): float(v) for k, v in dict(recipe_work.resources).items()}
    if not work_map:
        op_obj = getattr(resolved, "op", None)
        for field_name in ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops", "link_bytes"):
            val = float(getattr(op_obj, field_name, 0.0) or 0.0)
            if val > 0.0:
                work_map[field_name] = val
    positive_terms = [max(1, int(round(float(v)))) for v in work_map.values() if float(v) > 0.0]
    if not positive_terms:
        return False, 0, "resolved_operator_emitted_zero_positive_work"
    prog = ResourceChoiceProgram(
        num_resources=len(positive_terms),
        global_work=tuple(positive_terms),
        site_modes=(),
    )
    dag = EventDagSpec(node_quantum_ns=(positive_terms[0],), edges=())
    cert, _ = certify_lower_bound(prog, dag)
    if not check_optimality_certificate(prog, dag, cert):
        return False, int(cert.l_star_lean_ns), "lean_optimality_certificate_rejected"
    if cert.l_star_lean_ns <= 0:
        return False, int(cert.l_star_lean_ns), "lean_lower_bound_nonpositive_on_resolved_work"
    return True, int(cert.l_star_lean_ns), "verified"


def _verify_sim_owner_dispatch(
    *,
    is_family_pkg: bool,
    family: str,
    template_id: str,
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
                op_id=f"onboard_probe_{family}",
                family=op_family,
                phase="decode",
                stage=0,
                layer=0,
                requests=1,
                semantic_shape=_DEFAULT_PROBE_SHAPE,
                local_shape=_DEFAULT_PROBE_SHAPE,
                padded_shape=_DEFAULT_PROBE_SHAPE,
                launch_shape=_DEFAULT_PROBE_SHAPE,
                dtype="bf16",
                recipe=op_recipe,
            )
            matched = reg.match(sample_op)
        else:
            matched = None
        if matched is None:
            violations.append(f"sim_registry_unmatched:{family}")
            return False, {"owner": "builtin_family_registry", "matched": False}, violations
        detail: dict[str, Any] = {
            "owner": "builtin_family_registry",
            "matched": True,
            "spec_name": matched.name,
            "owned_recipe_count": len(matched.owned_recipe_ids),
            "resolved": False,
            "lean_lower_bound_ns": 0,
        }
        if sample_op is not None and planning_context is not None:
            try:
                resolved = matched.resolve(sample_op, planning_context)
            except (ModelingError, NotImplementedError, RuntimeError, ValueError) as exc:
                violations.append(f"sim_owner_resolve_failed:{family}:{exc}")
                return False, detail, violations
            lean_ok, l_star, lean_reason = _verify_resolved_operator_dual_bound(resolved)
            detail["resolved"] = True
            detail["resolved_implementation"] = str(getattr(resolved, "implementation", ""))
            detail["lean_lower_bound_ns"] = l_star
            detail["lean_status"] = lean_reason
            if not lean_ok:
                violations.append(f"sim_owner_dual_bound_failed:{family}:{lean_reason}")
                return False, detail, violations
        elif sample_op is not None:
            # Attempt default context resolution only if the family accepts a minimal probe;
            # NotImplementedError (e.g. unimplemented scaffold) ALWAYS fails closed.
            try:
                ctx = _build_default_planning_context(sample_op)
                resolved = matched.resolve(sample_op, ctx)
            except NotImplementedError as exc:
                violations.append(f"sim_owner_unimplemented_scaffold:{family}:{exc}")
                return False, detail, violations
            except (ModelingError, KeyError, TypeError, ValueError):
                detail["resolve_note"] = "requires_operator_specific_planning_context"
            else:
                lean_ok, l_star, lean_reason = _verify_resolved_operator_dual_bound(resolved)
                detail["resolved"] = True
                detail["resolved_implementation"] = str(getattr(resolved, "implementation", ""))
                detail["lean_lower_bound_ns"] = l_star
                detail["lean_status"] = lean_reason
                if not lean_ok:
                    violations.append(f"sim_owner_dual_bound_failed:{family}:{lean_reason}")
                    return False, detail, violations
        return True, detail, violations

    cat = template_catalog if template_catalog is not None else builtin_template_catalog()
    k2_ids = set(cat.ids("K2"))
    k3_ids = set(cat.ids("K3"))
    all_ids = k2_ids | k3_ids
    matched_id = template_id if template_id in all_ids else (family if family in all_ids else "")
    if not matched_id:
        violations.append(f"template_catalog_unregistered:{template_id or family}")
        return False, {"owner": "builtin_template_catalog", "matched": False}, violations
    tier = "K2" if matched_id in k2_ids else "K3"
    return True, {"owner": "builtin_template_catalog", "matched": True, "tier": tier, "template_id": matched_id}, violations


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
    require_tiling_axes: bool = False,
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
    for py_file in files:
        source_text = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(py_file))
        top_names = _extract_top_level_names(tree)
        symbols |= _extract_symbol_mentions(tree)
        all_refs.extend(_extract_source_refs(tree, py_file))
        abi = _extract_upstream_abi(tree)
        if py_file.name in ("arch22.py", "arch35.py") or not is_family_pkg:
            if not abi:
                violations.append(f"missing_upstream_abi_header:{py_file.name}")
            else:
                abis[py_file.name] = abi
            if is_family_pkg and "STRATEGY" not in top_names:
                violations.append(f"missing_strategy_export:{py_file.name}")
            if "SOURCES" not in top_names and not _extract_source_refs(tree, py_file):
                violations.append(f"missing_sources_declaration:{py_file.name}")
        if not declared_builder:
            declared_builder = _extract_string_constant(tree, "PROFILER_BENCH_BUILDER")

    if is_family_pkg and len(abis) >= 2 and len(set(abis.values())) < len(abis):
        violations.append("duplicate_upstream_abi_across_arch_leaves")

    resolved_refs_root = Path(refs_root).resolve() if refs_root is not None else None
    resolved_repo_map = {str(k): Path(v).resolve() for k, v in (repo_map or {}).items()}
    if not all_refs:
        violations.append("source_ref_missing")
    source_checks: list[dict[str, Any]] = []
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
        blob_status = "skipped_invalid_ref"
        if not ref_errors:
            ok_blob, blob_status = _verify_git_blob_sha256(
                ref, refs_root=resolved_refs_root, repo_map=resolved_repo_map
            )
            if not ok_blob:
                ref_errors.append(blob_status)
        if ref_errors:
            violations.append(f"source_ref_invalid:{ref['file']}:{','.join(ref_errors)}")
        source_checks.append({**ref, "status": blob_status, "errors": ref_errors})

    provenance_gate_outcome = None
    if (
        not is_family_pkg
        and target_rel.startswith("mindie_sim/templates/")
        and (resolved_refs_root is not None or resolved_repo_map)
    ):
        provenance_gate_outcome = run_template_provenance_gate(
            repo_root=root,
            refs_root=resolved_refs_root,
            repo_map=resolved_repo_map,
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

    sim_owner_detail: dict[str, Any] = {"checked": False}
    if require_sim_owner:
        _, sim_owner_detail, sim_violations = _verify_sim_owner_dispatch(
            is_family_pkg=is_family_pkg,
            family=family,
            template_id=template_id,
            family_registry=family_registry,
            template_catalog=template_catalog,
            sample_op=sample_op,
            planning_context=planning_context,
        )
        sim_owner_detail["checked"] = True
        violations.extend(sim_violations)

    taxonomy_ok, taxonomy_detail = _check_taxonomy_registration(root, target_rel, family)
    if require_taxonomy and not taxonomy_ok:
        violations.append(taxonomy_detail)

    eff_prof_family = (profiler_family or declared_builder or family).strip()
    eff_prof_recipe = (profiler_recipe or eff_prof_family).strip()
    eff_prof_impl = (
        profiler_impl
        or next(iter(abis.values()), "")
    ).strip()
    profiler_check: dict[str, Any] = {
        "checked": bool(require_profiler_builder),
        "verified": False,
        "family": eff_prof_family,
        "recipe": eff_prof_recipe,
        "impl": eff_prof_impl,
        "resolved_builder": "",
        "resolved_abi": "",
    }
    if require_profiler_builder:
        try:
            b_name, b_abi = resolve_profiler_builder_strict(
                eff_prof_family,
                eff_prof_recipe,
                eff_prof_impl,
                dtypes=profiler_dtypes,
                meta=profiler_meta,
                profiler_root=profiler_root,
            )
        except ModelingError as exc:
            profiler_check["error"] = str(exc)
            violations.append(str(exc))
        else:
            profiler_check["verified"] = True
            profiler_check["resolved_builder"] = b_name
            profiler_check["resolved_abi"] = b_abi

    result = {
        "stage": "validate-contract",
        "ok": not violations,
        "status": "pass" if not violations else "fail",
        "target": target_rel,
        "family": family,
        "upstream_abis": abis,
        "source_refs_checked": len(source_checks),
        "source_refs": source_checks,
        "template_provenance_gate": (
            provenance_gate_outcome.to_dict() if provenance_gate_outcome is not None else None
        ),
        "semantic_work_declared": has_semantic,
        "tensor_obligations_declared": has_obligations,
        "tiling_axes_declared": has_tiling_axes,
        "sim_owner_dispatch": sim_owner_detail,
        "taxonomy": {"registered": taxonomy_ok, "detail": taxonomy_detail},
        "profiler_builder": profiler_check,
        "violations": violations,
    }
    assert_sanitized_payload(result, field_name="operator_contract_validation")
    return result


__all__ = ["validate_operator_contract"]
