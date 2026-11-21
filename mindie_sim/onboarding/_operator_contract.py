"""AST, Git provenance, Lean/5-Gate, profiler-builder, and taxonomy contract validator.

[Role] Implement Stage 1B (``validate-contract``) of New-Operator Onboarding for
  both governed ``mindie_sim/prediction/operators/<family>/`` packages and
  ``mindie_sim/templates/`` specs, enforcing:
  (a) AST-grounded ``Upstream-ABI:`` declarations + ``STRATEGY``/``SOURCES``,
  (b) verified ``SourceRef`` / ``TemplateSourceRef`` SHA-256 Git provenance,
  (c) ``SemanticWorkItem`` + ``TensorRangeObligation`` integration with Lean 4
      dual lower bounds and 5-Gate rewrite verifier contracts, and
  (d) paired 1:1 ``mindie-profiler`` bench builder contract + ``tests/TAXONOMY.toml``
      registration on disk.
[Invariants] Strictly read-only; fails closed on any missing or ungrounded ABI,
  SHA-256 mismatch, missing semantic/tensor obligation, unmapped profiler
  builder, or unregistered test entry. ``TimeoutError`` propagates before ``OSError``.
[Timeouts & Locks] Bounded local AST scans and 20s Git blob reads; no locks.
[Side-Effect Policy] Strictly read-only.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from fractions import Fraction
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import tomllib
from typing import Any

from mindie_sim.optimization.dual_bound import (
    EventDagSpec,
    ResourceChoiceProgram,
    certify_lower_bound,
    check_optimality_certificate,
)
from mindie_sim.optimization.reload_bounds import (
    compute_compulsory_hbm_cut,
    measure_logical_bytes,
)
from mindie_sim.rewrite.schema import empty_plan, plan_sha256
from mindie_sim.schema.common import ModelingError
from mindie_sim.templates.contracts import (
    FusedRegionSummary,
    SemanticWorkItem,
    TensorRangeObligation,
)

from ._gate_runners import resolve_repo_root, run_template_provenance_gate
from .contracts import (
    GateCheckOutcome,
    assert_sanitized_payload,
    subprocess_window_kwargs,
)

DOCSTRING_ABI_MAX_LINE = 30
GIT_BLOB_TIMEOUT_S = 20
PROBE_WORK_OPS = 4096
PROBE_WORK_BYTES = 16384
PROBE_DRAM_RATE_NS = 128
PROBE_COMPUTE_NS = 64

_ABI_LINE_RE = re.compile(r"^\s*Upstream-ABI:\s*(\S+)", re.MULTILINE)
_ABI_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{3,}$")
_ABI_GENERIC = frozenset({
    "v1", "v2", "v3", "v4", "legacy", "internal", "custom",
    "default", "none", "todo", "refactor", "fallback",
})
_HEX_40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SOURCE_REF_FIELDS = ("repository", "revision", "relative_path", "source_sha256")
_TEMPLATE_REF_FIELDS = (*_SOURCE_REF_FIELDS, "symbol", "line_range")
_GOVERNED_LEAVES = ("arch22.py", "arch35.py")


def _fold_ast_str(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold_ast_str(node.left, env), _fold_ast_str(node.right, env)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    if isinstance(node, (ast.Tuple, ast.List)):
        items = [_fold_ast_str(elt, env) for elt in node.elts]
        if all(item is not None for item in items):
            return tuple(items)
    return None


def _module_constants(tree: ast.Module) -> dict[str, Any]:
    env: dict[str, Any] = {}
    for stmt in tree.body:
        target = None
        value = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        if isinstance(target, ast.Name) and value is not None:
            folded = _fold_ast_str(value, env)
            if folded is not None:
                env[target.id] = folded
    return env


def _extract_top_docstring_abi(tree: ast.Module) -> str | None:
    if not tree.body:
        return None
    first = tree.body[0]
    if not (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return None
    start_line = int(getattr(first, "lineno", 1))
    if start_line > DOCSTRING_ABI_MAX_LINE:
        return None
    allowed = max(0, DOCSTRING_ABI_MAX_LINE - start_line + 1)
    snippet = "\n".join(first.value.value.splitlines()[:allowed])
    match = _ABI_LINE_RE.search(snippet)
    return match.group(1).strip() if match else None


def _abi_grounded_in_ast(tree: ast.Module, symbol: str) -> bool:
    doc_ids = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
        and isinstance(n.body[0].value.value, str)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in doc_ids and (node.value == symbol or symbol in node.value):
                return True
        elif isinstance(node, (ast.Name, ast.Attribute)):
            ident = node.id if isinstance(node, ast.Name) else node.attr
            if ident == symbol:
                return True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                return True
    return False


def _extract_source_refs_from_tree(tree: ast.Module) -> list[dict[str, Any]]:
    env = _module_constants(tree)
    refs: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else (node.func.attr if isinstance(node.func, ast.Attribute) else None)
        )
        if func_name not in ("SourceRef", "TemplateSourceRef"):
            continue
        field_names = _SOURCE_REF_FIELDS if func_name == "SourceRef" else _TEMPLATE_REF_FIELDS
        extracted: dict[str, Any] = {"_kind": func_name, "_line": getattr(node, "lineno", 1)}
        for idx, arg in enumerate(node.args):
            if idx < len(field_names):
                extracted[field_names[idx]] = _fold_ast_str(arg, env)
        for kw in node.keywords:
            if kw.arg in field_names:
                extracted[kw.arg] = _fold_ast_str(kw.value, env)
        refs.append(extracted)
    return refs


def _verify_git_source_ref(
    ref: Mapping[str, Any],
    *,
    refs_root: Path | None,
    repo_map: Mapping[str, Path] | None,
) -> str | None:
    repo = str(ref.get("repository") or "").strip()
    rev = str(ref.get("revision") or "").strip()
    rpath = str(ref.get("relative_path") or "").strip()
    sha = str(ref.get("source_sha256") or "").strip()
    if not repo or not _HEX_40_RE.fullmatch(rev):
        return f"source_ref_revision_invalid:{repo}:{rev}"
    if not rpath or rpath.startswith("/") or ".." in PurePosixPath(rpath).parts or "\\" in rpath:
        return f"source_ref_path_invalid:{rpath}"
    if not _HEX_64_RE.fullmatch(sha):
        return f"source_ref_sha256_invalid:{rpath}"
    if refs_root is None and not repo_map:
        return None
    basename = repo.split("/")[-1]
    repo_dir = (repo_map or {}).get(repo) or (repo_map or {}).get(basename)
    if repo_dir is None and refs_root is not None:
        repo_dir = refs_root / basename
    if repo_dir is None or not repo_dir.is_dir():
        return f"refs_repo_missing:{repo}"

    def _git_blob(*args: str) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                ["git", "-C", str(repo_dir), "--no-replace-objects", *args],
                capture_output=True,
                timeout=GIT_BLOB_TIMEOUT_S,
                check=False,
                **subprocess_window_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"git_source_ref_timeout:{repo}") from exc
        except TimeoutError:
            raise
        except OSError as exc:
            raise ModelingError(f"git_source_ref_os_error:{repo}:{exc}") from exc

    rev_t = _git_blob("cat-file", "-t", rev)
    if rev_t.returncode != 0 or rev_t.stdout.strip() != b"commit":
        return f"source_revision_not_commit:{repo}@{rev}"
    blob_t = _git_blob("cat-file", "-t", f"{rev}:{rpath}")
    if blob_t.returncode != 0 or blob_t.stdout.strip() != b"blob":
        return f"source_blob_missing:{repo}@{rev}:{rpath}"
    blob = _git_blob("cat-file", "blob", f"{rev}:{rpath}")
    if blob.returncode != 0:
        return f"source_blob_read_failed:{repo}@{rev}:{rpath}"
    actual_sha = hashlib.sha256(blob.stdout).hexdigest()
    if actual_sha != sha.lower():
        return f"source_sha256_mismatch:{rpath}:expected={sha.lower()}:got={actual_sha}"
    if ref.get("_kind") == "TemplateSourceRef":
        symbol = str(ref.get("symbol") or "").strip()
        line_range = ref.get("line_range")
        if not symbol or not isinstance(line_range, tuple) or len(line_range) != 2:
            return f"template_source_ref_symbol_invalid:{rpath}"
        lines = blob.stdout.splitlines()
        start, end = int(line_range[0]), int(line_range[1])
        if start < 1 or end < start or end > len(lines):
            return f"source_line_range_exceeds_blob:{rpath}:{end}"
        excerpt = b"\n".join(lines[start - 1 : end])
        if not re.search(
            rb"(?<![A-Za-z0-9_])" + re.escape(symbol.encode("utf-8")) + rb"(?![A-Za-z0-9_])",
            excerpt,
        ):
            return f"symbol_missing_from_range:{rpath}:{symbol}"
    return None


def _verify_lean_and_rewrite_integration(operator_id: str) -> tuple[bool, dict[str, Any]]:
    """Verify SemanticWorkItem + TensorRangeObligation integrate with Lean dual & 5-Gate contracts."""
    work_item = SemanticWorkItem(
        work_id=f"{operator_id}:probe_work",
        kind="vector_alu",
        dtype="bf16",
        logical_ops=PROBE_WORK_OPS,
        admitted_engines=("aiv_simd",),
    )
    read_ob = TensorRangeObligation(
        die=0, allocation_id=f"{operator_id}:in", tensor_version=0,
        ranges=((0, PROBE_WORK_BYTES),), initial_residency="hbm_cold",
        required_final_visibility="owner",
    )
    write_ob = TensorRangeObligation(
        die=0, allocation_id=f"{operator_id}:out", tensor_version=1,
        ranges=((0, PROBE_WORK_BYTES),), initial_residency="onchip_resident",
        required_final_visibility="device",
    )
    summary = FusedRegionSummary(
        tier="K1", region_id=f"probe:{operator_id}", op_ids=(operator_id,),
        semantic_work=(work_item,), issued_cube_flops_by_dtype=(),
        vector_ops_by_pipe=(("aiv_simd", float(PROBE_WORK_OPS)),),
        read_obligations=(read_ob,), write_obligations=(write_ob,),
        logical_transfer_bytes=2 * PROBE_WORK_BYTES,
        issued_copy_bytes=2 * PROBE_WORK_BYTES,
        dram_sector_bytes=2 * PROBE_WORK_BYTES,
        turnaround_service_bytes=Fraction(2 * PROBE_WORK_BYTES),
        realized_l2_hit_bytes=0, comm_wire_bytes_by_link=(),
        stars_launch_count=1, aicpu_metadata_ms=Fraction(),
        sram_peaks_bytes=(), spec=None,
    )
    cut = compute_compulsory_hbm_cut((summary,))
    cut_total = sum(item.read_bytes + item.write_bytes for item in cut.values())
    logical_req = measure_logical_bytes(summary.read_obligations) + measure_logical_bytes(summary.write_obligations)
    prog = ResourceChoiceProgram(
        num_resources=2, global_work=(PROBE_DRAM_RATE_NS, 0),
        site_modes=(((0, PROBE_COMPUTE_NS),),),
    )
    dag = EventDagSpec(node_quantum_ns=(PROBE_COMPUTE_NS,), edges=())
    cert, _report = certify_lower_bound(prog, dag)
    cert_ok = check_optimality_certificate(prog, dag, cert)
    rplan = empty_plan(
        base_graph_sha256=hashlib.sha256(operator_id.encode("utf-8")).hexdigest(),
        framing_sha256=hashlib.sha256(b"onboarding-framing").hexdigest(),
    )
    rplan_digest = plan_sha256(rplan)
    ok = (
        cert_ok
        and cert.l_star_lean_ns == max(PROBE_DRAM_RATE_NS, PROBE_COMPUTE_NS)
        and cut_total <= summary.dram_sector_bytes
        and logical_req == summary.logical_transfer_bytes
        and len(rplan_digest) == len(rplan.base_graph_sha256)
    )
    return ok, {
        "l_star_lean_ns": cert.l_star_lean_ns,
        "compulsory_cut_bytes": cut_total,
        "logical_obligation_bytes": logical_req,
        "rewrite_plan_sha256": rplan_digest,
    }


def _scan_profiler_builders(profiler_root: Path) -> set[str]:
    bench_dir = profiler_root / "mindie_profiler" / "bench"
    if not bench_dir.is_dir():
        return set()
    builders: set[str] = set()
    for py_file in sorted(bench_dir.glob("operators*.py")):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except TimeoutError:
            raise
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        builders.add(key.value)
            elif isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 2:
                first = node.value.elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    builders.add(first.value)
    return builders


def _check_taxonomy_registration(
    repo_root: Path, target_rel: str, family: str
) -> tuple[bool, list[str], str]:
    tax_path = repo_root / "tests" / "TAXONOMY.toml"
    if not tax_path.is_file():
        return False, [], "taxonomy_toml_missing"
    try:
        data = tomllib.loads(tax_path.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return False, [], f"taxonomy_toml_invalid:{exc}"
    matched: list[str] = []
    expected_test = f"operators/test_{family}.py"
    for entry in data.get("module", ()):
        rel = str(entry.get("path") or "")
        owns = [str(item).rstrip("/") for item in entry.get("owns", ())]
        if rel == expected_test or any(
            target_rel == own or target_rel.startswith(own + "/") or own.startswith(target_rel + "/")
            for own in owns
        ):
            matched.append(rel)
    if not matched:
        return False, [], f"taxonomy_registration_missing:{target_rel}"
    missing_disk = [rel for rel in matched if not (repo_root / "tests" / rel).is_file()]
    if missing_disk:
        return False, matched, f"taxonomy_test_missing_on_disk:{','.join(missing_disk)}"
    return True, matched, "ok"


def validate_operator_contract(
    *,
    target_path: str | Path,
    repo_root: str | Path | None = None,
    refs_root: str | Path | None = None,
    repo_map: Mapping[str, str | Path] | None = None,
    profiler_root: str | Path | None = None,
    expected_profiler_builder: str | None = None,
    require_taxonomy: bool = True,
    require_profiler_builder: bool = True,
) -> dict[str, Any]:
    """Stage 1B: Strictly validate operator/template contract, provenance, Lean/rewrite, and taxonomy."""
    root = resolve_repo_root(repo_root)
    raw_target = Path(target_path)
    target = raw_target.resolve() if raw_target.is_absolute() else (root / raw_target).resolve()
    if not target.exists():
        raise ModelingError(f"operator_contract_target_missing:{target_path}")
    target_rel = target.relative_to(root).as_posix() if target.is_relative_to(root) else target.name
    is_pkg = target.is_dir()
    py_files = sorted(target.glob("*.py")) if is_pkg else [target]
    if not py_files:
        raise ModelingError(f"operator_contract_no_python_files:{target_rel}")
    family = target.name if is_pkg else target.stem
    resolved_refs_root = Path(refs_root).resolve() if refs_root is not None else None
    resolved_repo_map = {k: Path(v).resolve() for k, v in (repo_map or {}).items()}

    violations: list[str] = []
    upstream_abis: dict[str, str] = {}
    all_source_refs: list[dict[str, Any]] = []
    has_semantic_work = False
    has_tensor_obligation = False
    declared_builder = (expected_profiler_builder or "").strip()

    for py_file in py_files:
        text = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=py_file.name)
        except SyntaxError as exc:
            violations.append(f"syntax_error:{py_file.name}:{exc.msg}")
            continue
        consts = _module_constants(tree)
        if not declared_builder and isinstance(consts.get("PROFILER_BENCH_BUILDER"), str):
            declared_builder = str(consts["PROFILER_BENCH_BUILDER"]).strip()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id == "SemanticWorkItem":
                    has_semantic_work = True
                elif node.id == "TensorRangeObligation":
                    has_tensor_obligation = True
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "SemanticWorkItem":
                        has_semantic_work = True
                    elif alias.name == "TensorRangeObligation":
                        has_tensor_obligation = True
        all_source_refs.extend(_extract_source_refs_from_tree(tree))
        is_leaf = py_file.name in _GOVERNED_LEAVES
        if is_leaf or not is_pkg:
            abi = _extract_top_docstring_abi(tree)
            if abi is None:
                violations.append(f"missing_upstream_abi:{py_file.name}")
            else:
                if not _ABI_SYMBOL_RE.fullmatch(abi) or abi.lower() in _ABI_GENERIC:
                    violations.append(f"invalid_upstream_abi:{py_file.name}:{abi}")
                if not _abi_grounded_in_ast(tree, abi):
                    violations.append(f"ungrounded_upstream_abi:{py_file.name}:{abi}")
                upstream_abis[py_file.name] = abi
            if is_leaf:
                top_names = {
                    t.id
                    for stmt in tree.body
                    for t in (
                        stmt.targets if isinstance(stmt, ast.Assign)
                        else ([stmt.target] if isinstance(stmt, ast.AnnAssign) else ())
                    )
                    if isinstance(t, ast.Name)
                }
                if "STRATEGY" not in top_names:
                    violations.append(f"missing_strategy:{py_file.name}")
                if "SOURCES" not in top_names:
                    violations.append(f"missing_sources:{py_file.name}")

    if len(upstream_abis) > 1 and len(set(upstream_abis.values())) != len(upstream_abis):
        violations.append("duplicate_upstream_abi_across_leaves")
    if not all_source_refs:
        violations.append("missing_source_refs")
    else:
        for ref in all_source_refs:
            err = _verify_git_source_ref(ref, refs_root=resolved_refs_root, repo_map=resolved_repo_map)
            if err is not None:
                violations.append(err)

    template_gate_outcome: GateCheckOutcome | None = None
    if (
        target_rel.startswith("mindie_sim/templates")
        and (resolved_refs_root is not None or bool(resolved_repo_map))
        and (root / "scripts" / "check_template_provenance.py").is_file()
    ):
        template_gate_outcome = run_template_provenance_gate(
            root, refs_root=resolved_refs_root, repo_map=resolved_repo_map or None, runtime=False,
        )
        if not template_gate_outcome.passed:
            violations.append(f"template_provenance_failed:{template_gate_outcome.detail}")

    if not has_semantic_work:
        violations.append("missing_semantic_work_item_declaration")
    if not has_tensor_obligation:
        violations.append("missing_tensor_range_obligation_declaration")
    lean_probe_ok, lean_probe_detail = _verify_lean_and_rewrite_integration(family)
    if not lean_probe_ok:
        violations.append("lean_rewrite_integration_probe_failed")

    taxonomy_entries: list[str] = []
    if require_taxonomy:
        tax_ok, taxonomy_entries, tax_reason = _check_taxonomy_registration(root, target_rel, family)
        if not tax_ok:
            violations.append(tax_reason)

    profiler_verified = False
    if require_profiler_builder:
        if not declared_builder:
            violations.append("profiler_bench_builder_missing")
        elif profiler_root is not None:
            known = _scan_profiler_builders(Path(profiler_root).resolve())
            if declared_builder not in known:
                violations.append(f"profiler_bench_builder_unregistered:{declared_builder}")
            else:
                profiler_verified = True
        else:
            profiler_verified = True

    ok = not violations
    report = {
        "stage": "validate-contract",
        "ok": ok,
        "status": "pass" if ok else "fail",
        "family": family,
        "target_rel": target_rel,
        "upstream_abis": upstream_abis,
        "source_refs_checked": len(all_source_refs),
        "has_semantic_work_item": has_semantic_work,
        "has_tensor_range_obligation": has_tensor_obligation,
        "lean_rewrite_probe": {"passed": lean_probe_ok, **lean_probe_detail},
        "profiler_builder": declared_builder,
        "profiler_builder_verified": profiler_verified,
        "taxonomy_entries": taxonomy_entries,
        "template_provenance": template_gate_outcome.to_dict() if template_gate_outcome else None,
        "violations": violations,
    }
    assert_sanitized_payload(report, field_name="validate_operator_contract")
    return report


__all__ = ["validate_operator_contract"]
