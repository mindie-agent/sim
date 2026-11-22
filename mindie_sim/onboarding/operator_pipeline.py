"""Turnkey 4-stage New-Operator onboarding pipeline (scaffold/contract -> cliff -> score -> certify).

[Role] Orchestrate standardized onboarding of a new Ascend operator family or
  template across four stages:
  * Stage 1A (``scaffold_operator``): scaffold governed operator packages under
    ``mindie_sim/prediction/operators/<family>/{__init__,common,arch22,arch35}.py``
    or templates under ``mindie_sim/templates/<family>_spec.py`` with AST
    ``Upstream-ABI:``, pinned ``SourceRef``, ``TILING_AXES``,
    ``SemanticWorkItem`` + ``TensorRangeObligation``, paired profiler builder,
    and orthogonal Tier-0 test + ``TAXONOMY.toml`` snippet.
  * Stage 1B (``validate_operator_contract``): validate AST ``Upstream-ABI:``,
    verified Git blob ``SourceRef`` provenance, real Sim ``FamilyRegistry`` /
    ``TemplateCatalog`` lowering + Lean 4 dual-bound certification, paired 1:1
    ``mindie-profiler`` ``resolve_builder_strict``, and ``TAXONOMY.toml``.
  * Stage 2 (``generate_operator_cliff_plan``): synthesize disjoint Anchor +
    Off-Grid Cliff + ABI-rejection bench plan.
  * Stage 3 (``bench_and_score_operator``): dispatch via ``mindie-profiler`` or
    score captured silicon observations against pure white-box ``mindie-sim``
    (zero ``service_regimes``).
  * Stage 4 (``certify_operator`` & ``run_operator_onboarding``): enforce
    pointwise ``max(10%, 2.0 us)`` cliff tolerance, ``0`` Lean inversions,
    ``0`` strict governance/orthogonality violations, auto-trigger
    ``diagnose_tail_cluster.py`` on failure, and seal a SHA-256 receipt.
[Invariants] Strictly read-only on validation; zero hardware tokens in family
  stems; zero ``service_regimes``; ``TimeoutError`` propagates before ``OSError``.
[Timeouts & Locks] Bounded ``git cat-file`` (15s) and repo-gate subprocesses
  (120s); zero local locks or device contexts.
[Side-Effect Policy] Writes only explicitly requested scaffold/output paths.
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
import hashlib
import keyword
import math
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any, Literal

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

from .contracts import (
    CONTEXT_MAX_TOKENS, CONTEXT_MIN_TOKENS, DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT, DEFAULT_PHYSICAL_ATOL_NS, DEFAULT_TILE_DIM, HEX_COMMIT_LEN,
    NS_PER_MS, OPERATOR_ONBOARDING_RECEIPT_SCHEMA, GateCheckOutcome, OperatorCliffParameters,
    assert_sanitized_payload, collect_receipt_warnings, evaluate_abi_rejections,
    finalize_stage4_certification, read_bounded_json, resolve_repo_root,
    run_template_provenance_gate, subprocess_window_kwargs, synthesize_operator_cliff_plan,
    validate_strict_atol_ns, validate_strict_threshold_pct, verify_cliff_plan_integrity,
    write_atomic_bytes, write_atomic_json,
)
from .profiler_normalizer import (
    acquire_stage3_observations, assign_plan_point_level, resolve_profiler_builder_strict,
)
from .whitebox_runner import score_sealed_gate_cases

_FORBIDDEN_FAMILY_RE = re.compile(r"(?:arch22|arch35|cann|(?:^|_)(?:a2|a3|a5)(?:$|_))", re.IGNORECASE)
_ABI_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{3,}$")
_UPSTREAM_ABI_LINE_RE = re.compile(r"^\s*Upstream-ABI\s*:\s*([A-Za-z_][A-Za-z0-9_:.<>/-]*)\s*$")
_HEX_40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ABI_GENERIC = frozenset({"v1", "v2", "v3", "v4", "legacy", "internal", "custom", "default", "none", "todo", "refactor", "fallback"})
_TEMPLATE_ID_NAMES = frozenset({"TEMPLATE_ID", "TEMPLATE_IDS", "CATLASS_TEMPLATE_IDS", "DSL_KERNEL_IDS"})
_REQUIRED_FAMILY_FILES = ("__init__.py", "common.py", "arch22.py", "arch35.py")
_GIT_CAT_TIMEOUT_S = 15
_NS_PER_SECOND = 1_000_000_000.0
_DEFAULT_PROBE_SKU = "Ascend910B2"
_DEFAULT_PROBE_SKU_A5 = "Ascend950DT_9582"
_DEFAULT_PROBE_SHAPE = (1, 128)
_DEFAULT_TEMPLATE_DIM = 128
_MAX_STAGE_DEPTH = 16
_SOURCE_REF_POS_ARGS = 4


def _render_source_refs_code(source_refs: Sequence[Mapping[str, Any]]) -> str:
    if not source_refs:
        raise ModelingError("operator_scaffold_requires_source_refs")
    lines = ["SOURCES: tuple[SourceRef, ...] = ("]
    for ref in source_refs:
        repo, rev = str(ref.get("repository") or "").strip(), str(ref.get("revision") or "").strip()
        rpath, sha = str(ref.get("relative_path") or "").strip(), str(ref.get("source_sha256") or "").strip()
        if not repo or not _HEX_40_RE.fullmatch(rev) or not rpath or not _HEX_64_RE.fullmatch(sha):
            raise ModelingError(f"operator_scaffold_source_ref_invalid:{repo}:{rpath}")
        lines.extend(("    SourceRef(", f"        repository={repo!r},", f"        revision={rev!r},", f"        relative_path={rpath!r},", f"        source_sha256={sha.lower()!r},", "    ),"))
    lines.append(")")
    return "\n".join(lines)


def scaffold_operator(
    *, family: str, upstream_abi_arch22: str, upstream_abi_arch35: str,
    source_refs: Sequence[Mapping[str, Any]], target_root: str | Path,
    target_kind: Literal["operator_family", "template"] = "operator_family",
    profiler_builder: str = "", tile_dim: int = DEFAULT_TILE_DIM,
    overwrite: bool = False, scaffold_test: bool = True,
) -> dict[str, Any]:
    """Stage 1A: Scaffold a governed operator family package or template skeleton."""
    if not family or not family.isidentifier() or keyword.iskeyword(family):
        raise ModelingError(f"operator_family_identifier_invalid:{family}")
    if _FORBIDDEN_FAMILY_RE.search(family):
        raise ModelingError(f"operator_family_forbidden_hardware_token:{family}")
    for lbl, abi in (("arch22", upstream_abi_arch22), ("arch35", upstream_abi_arch35)):
        if not abi or not _ABI_SYMBOL_RE.fullmatch(abi) or abi.lower() in _ABI_GENERIC:
            raise ModelingError(f"operator_upstream_abi_invalid:{lbl}:{abi}")
    if target_kind == "operator_family" and upstream_abi_arch22 == upstream_abi_arch35:
        raise ModelingError("operator_scaffold_duplicate_leaf_abi")
    builder_name = (profiler_builder or family).strip()
    root = Path(target_root).resolve()
    src_block = _render_source_refs_code(source_refs)

    if target_kind == "template":
        rel_target = f"mindie_sim/templates/{family}_spec.py"
        files_to_write: dict[Path, str] = {
            root / rel_target: (
                f'"""Governed template specification for ``{family}``.\n\nUpstream-ABI: {upstream_abi_arch35}\n\n'
                f"[Role] Declare pinned provenance, ``TILING_AXES``, ``SemanticWorkItem``, and ``TensorRangeObligation`` for ``{family}``.\n"
                "[Invariants] Verified SourceRef provenance.\n[Timeouts & Locks] Deterministic; no locks.\n[Side-Effect Policy] Zero side effects.\n\"\"\"\n"
                "from __future__ import annotations\n\nfrom mindie_sim.onboarding import TilingAxes\n"
                "from mindie_sim.schema.datapath import SourceRef\nfrom mindie_sim.templates.contracts import SemanticWorkItem, TensorRangeObligation\n\n"
                f"UPSTREAM_ABI = {upstream_abi_arch35!r}\nPROFILER_BENCH_BUILDER = {builder_name!r}\nTILING_AXES = TilingAxes(tile_dims=({int(tile_dim)},))\n{src_block}\n\n\n"
                f"def build_template_obligations(problem: object) -> tuple[tuple[SemanticWorkItem, ...], tuple[TensorRangeObligation, ...]]:\n    del problem\n    raise NotImplementedError('implement_{family}_template_obligations')\n"
            )
        }
    else:
        rel_target = f"mindie_sim/prediction/operators/{family}"
        pkg = root / rel_target

        def _leaf(lbl: str, aid: str, abi: str) -> str:
            return (
                f'"""{lbl} ISA lowering strategy for ``{family}``.\n\nUpstream-ABI: {abi}\n\n'
                f"[Role] Lower ``{family}`` via ``{abi}`` on ``{aid}``.\n[Invariants] Verified SourceRef.\n[Timeouts & Locks] O(1); no locks.\n[Side-Effect Policy] Pure output.\n\"\"\"\n"
                "from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom mindie_sim.schema.datapath import ArchId, SourceRef\n"
                "from mindie_sim.schema.operators import OpSpec\nfrom ...mechanism import OperatorPlan\nfrom ..contracts import OperatorPlanningContext\nfrom .common import build_plan\n\n"
                f"ARCH_ID = ArchId({aid!r})\nUPSTREAM_ABI = {abi!r}\n{src_block}\n\n\n@dataclass(frozen=True, slots=True)\nclass _Strategy:\n"
                "    arch_id: ArchId = ARCH_ID\n    abi: str = UPSTREAM_ABI\n    sources: tuple[SourceRef, ...] = SOURCES\n"
                "    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:\n        del op\n        return ctx.compute_arch.arch_id == self.arch_id\n"
                "    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:\n        return build_plan(op, ctx, self.abi, self.sources)\n\n\nSTRATEGY = _Strategy()\n"
            )

        files_to_write = {
            pkg / "__init__.py": (
                f'"""Single public entrypoint for governed ``{family}`` operator family.\n\n[Role] Register ``{family}`` in ``FamilyRegistry``.\n'
                "[Invariants] Never re-exports internal leaves.\n[Timeouts & Locks] O(1); no locks.\n[Side-Effect Policy] Pure output.\n\"\"\"\n"
                "from __future__ import annotations\n\nfrom mindie_sim.schema.common import ModelingError\nfrom mindie_sim.schema.operators import OpSpec\n"
                "from ..contracts import OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator\nfrom . import arch22 as _arch22, arch35 as _arch35\n"
                "from .common import FAMILY, PROFILER_BENCH_BUILDER, describe\n\n_STRATEGIES = {_arch22.ARCH_ID: _arch22.STRATEGY, _arch35.ARCH_ID: _arch35.STRATEGY}\n\n\n"
                "def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext, /) -> ResolvedOperator:\n"
                "    strategy = _STRATEGIES.get(ctx.compute_arch.arch_id)\n    if strategy is None:\n        raise ModelingError(f'unsupported_arch:{FAMILY}:{ctx.compute_arch.arch_id}')\n"
                "    plan = strategy.build(op, ctx)\n    return ResolvedOperator(occurrence=ctx.occurrence, op=op, recipe=plan, implementation=strategy.abi, sources=strategy.sources, descriptor=describe(op, ctx, plan))\n\n\n"
                "def register_family() -> OperatorFamilySpec:\n    return OperatorFamilySpec(name=FAMILY, families=frozenset({FAMILY}), owned_recipe_ids=frozenset({FAMILY}), resolve=resolve_operator_plan)\n\n\n"
                '__all__ = ["PROFILER_BENCH_BUILDER", "register_family", "resolve_operator_plan"]\n'
            ),
            pkg / "common.py": (
                f'"""Shared work conservation, obligations, and TilingAxes for ``{family}``.\n\n[Role] Declare ``TILING_AXES``, ``SemanticWorkItem``, ``TensorRangeObligation``.\n'
                "[Invariants] Zero SKU-name sniffing.\n[Timeouts & Locks] O(1); no locks.\n[Side-Effect Policy] Pure output.\n\"\"\"\n"
                "from __future__ import annotations\n\nfrom mindie_sim.onboarding import TilingAxes\nfrom mindie_sim.schema.operators import OpSpec\n"
                "from mindie_sim.templates.contracts import SemanticWorkItem, TensorRangeObligation\nfrom ...mechanism import OperatorPlan\n"
                "from ...results import FusedPipelineDescriptor, FusedPipelineSpec\nfrom ..contracts import OperatorPlanningContext\n\n"
                f"FAMILY = {family!r}\nDEFAULT_TILE_DIM = {int(tile_dim)}\nPROFILER_BENCH_BUILDER = {builder_name!r}\n"
                'TILING_AXES = TilingAxes(tile_dims=(DEFAULT_TILE_DIM,), core_product_axes=("tokens", "batch_x_kv_heads"))\n\n\n'
                f"def build_semantic_work(op: OpSpec, *, die: int = 0) -> tuple[SemanticWorkItem, ...]:\n    del op, die\n    raise NotImplementedError('implement_{family}_semantic_work')\n\n\n"
                f"def build_tensor_obligations(op: OpSpec, *, die: int = 0) -> tuple[tuple[TensorRangeObligation, ...], tuple[TensorRangeObligation, ...]]:\n    del op, die\n    raise NotImplementedError('implement_{family}_obligations')\n\n\n"
                f"def build_plan(op: OpSpec, ctx: OperatorPlanningContext, abi: str, sources) -> OperatorPlan:\n    del op, ctx, abi, sources\n    raise NotImplementedError('implement_{family}_plan')\n\n\n"
                "def describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:\n"
                "    spec = plan.pipeline_spec\n    assert isinstance(spec, FusedPipelineSpec)\n"
                "    return FusedPipelineDescriptor(owner=ctx.occurrence, family=spec.family, implementation=spec.implementation, member_op_ids=(op.op_id,), absorbed_source_op_ids=(), phases=spec.phases, sources=spec.sources, repeats=spec.repeats)\n"
            ),
            pkg / "arch22.py": _leaf("Arch22", "npu_arch_2201", upstream_abi_arch22),
            pkg / "arch35.py": _leaf("Arch35", "npu_arch_3510", upstream_abi_arch35),
        }
    test_rel = f"tests/operators/test_{family}.py"
    if scaffold_test:
        files_to_write[root / test_rel] = f'"""Orthogonal Tier-0 tests for ``{family}``.\n\n[Role] Verify ``{family}`` invariants.\n[Invariants] Isolated.\n[Timeouts & Locks] Fast.\n[Side-Effect Policy] None.\n"""\nimport pytest\npytestmark = pytest.mark.tier0\n\ndef test_{family}_placeholder() -> None:\n    assert True\n'
    if not overwrite:
        existing = [str(p.relative_to(root)) for p in files_to_write if p.exists()]
        if existing:
            raise ModelingError(f"operator_scaffold_files_exist:{','.join(existing)}")
    written = []
    for path, content in files_to_write.items():
        write_atomic_bytes(path, content.encode("utf-8"))
        written.append(str(path.relative_to(root)))
    return {"stage": "scaffold", "ok": True, "status": "scaffolded", "family": family, "target_kind": target_kind, "package_rel": rel_target, "files_written": sorted(written), "taxonomy_snippet": f'[[module]]\npath = "operators/test_{family}.py"\ntier = "tier0"\nowns = ["{rel_target}"]\n'}


def _eval_str(node: ast.AST | None, consts: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        lo, hi = _eval_str(node.left, consts), _eval_str(node.right, consts)
        return (lo + hi) if (lo is not None and hi is not None) else None
    return None


def _has_not_implemented_stub(fn_node: ast.AST) -> bool:
    for inner in ast.walk(fn_node):
        if isinstance(inner, ast.Raise) and inner.exc is not None:
            ex = inner.exc
            ename = ex.id if isinstance(ex, ast.Name) else (ex.func.id if isinstance(ex, ast.Call) and isinstance(ex.func, ast.Name) else "")
            if ename == "NotImplementedError":
                return True
    return False


def _scan_ast_files(files: Sequence[Path], *, is_pkg: bool) -> tuple[dict[str, str], list[dict[str, str]], set[str], str, set[str], list[str]]:
    abis, refs, symbols, decl_builder, decl_tids, shared, viols = {}, [], set(), "", set(), {}, []
    trees = [(f, ast.parse(f.read_text(encoding="utf-8"), filename=str(f))) for f in files]
    for _f, tree in trees:
        for st in tree.body:
            if isinstance(st, ast.Assign) and (val := _eval_str(st.value, shared)) is not None:
                for t in st.targets:
                    if isinstance(t, ast.Name):
                        shared[t.id] = val
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value and (val := _eval_str(st.value, shared)) is not None:
                shared[st.target.id] = val
    for f, tree in trees:
        top = {t.id for st in tree.body if isinstance(st, ast.Assign) for t in st.targets if isinstance(t, ast.Name)} | {st.target.id for st in tree.body if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)} | {st.name for st in tree.body if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                symbols.add(n.id)
            elif isinstance(n, ast.Attribute):
                symbols.add(n.attr)
            elif isinstance(n, ast.ImportFrom):
                symbols.update(a.name for a in n.names)
            elif isinstance(n, ast.Call):
                fn = n.func.id if isinstance(n.func, ast.Name) else (n.func.attr if isinstance(n.func, ast.Attribute) else "")
                if fn in {"SourceRef", "TemplateSourceRef"}:
                    pos = [_eval_str(a, shared) for a in n.args]
                    kw = {str(k.arg): _eval_str(k.value, shared) for k in n.keywords if k.arg}
                    refs.append({"file": f.name, "repository": kw.get("repository") or (pos[0] if len(pos) > 0 else "") or "", "revision": kw.get("revision") or (pos[1] if len(pos) > 1 else "") or "", "relative_path": kw.get("relative_path") or (pos[2] if len(pos) > 2 else "") or "", "source_sha256": kw.get("source_sha256") or (pos[3] if len(pos) >= _SOURCE_REF_POS_ARGS else "") or ""})
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _has_not_implemented_stub(n):
                viols.append(f"operator_target_unimplemented_stub:{f.name}:{n.name}")
        for st in tree.body:
            tname = [t.id for t in st.targets if isinstance(t, ast.Name)] if isinstance(st, ast.Assign) else ([st.target.id] if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) else [])
            vnode = st.value if isinstance(st, (ast.Assign, ast.AnnAssign)) else None
            if any(x in _TEMPLATE_ID_NAMES for x in tname) and vnode is not None:
                if (s1 := _eval_str(vnode, shared)):
                    decl_tids.add(s1)
                elif isinstance(vnode, (ast.Tuple, ast.List, ast.Set)):
                    decl_tids.update(s for e in vnode.elts if (s := _eval_str(e, shared)))
        abi = next((m.group(1).strip() for ln in (ast.get_docstring(tree) or "").splitlines() if (m := _UPSTREAM_ABI_LINE_RE.match(ln))), "")
        if f.name in {"arch22.py", "arch35.py"} or not is_pkg:
            if not abi:
                viols.append(f"missing_upstream_abi_header:{f.name}")
            else:
                abis[f.name] = abi
            if is_pkg and "STRATEGY" not in top:
                viols.append(f"missing_strategy_export:{f.name}")
            if "SOURCES" not in top and not any(r["file"] == f.name for r in refs):
                viols.append(f"missing_sources_declaration:{f.name}")
        if not decl_builder and "PROFILER_BENCH_BUILDER" in shared:
            decl_builder = shared["PROFILER_BENCH_BUILDER"]
    if is_pkg and len(abis) >= 2 and len(set(abis.values())) < len(abis):
        viols.append("duplicate_upstream_abi_across_arch_leaves")
    return abis, refs, symbols, decl_builder, decl_tids, viols


def _build_default_planning_context(op: OpSpec, *, sku: str = _DEFAULT_PROBE_SKU) -> OperatorPlanningContext:
    hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
    die = hw.dies[0]
    dp = datapath_for_die(die)
    geom = Geometry(compute_units=die.compute_units, resource_units=die.resource_units, buffers=die.buffers, buffer_units=die.buffer_units, rates=die.rates, compute_peaks=die.compute_peaks, compute_arch=die.compute_arch, datapath=dp)
    return OperatorPlanningContext(occurrence=OpOccurrence(0, 0, 0, op.op_id), geometry=geom, datapath=dp)


def _collect_stages(obj: Any, depth: int = 0) -> list[Any]:
    if obj is None or depth > _MAX_STAGE_DEPTH or isinstance(obj, (tuple, list, dict, str, bytes, int, float, bool)):
        return []
    found: list[Any] = []
    if hasattr(obj, "work") and hasattr(obj, "name") and hasattr(obj, "after"):
        found.append(obj)
        if getattr(obj, "program", None) is not None:
            found.extend(_collect_stages(obj.program, depth + 1))
        return found
    for attr in ("stages", "branches", "phases", "programs"):
        ch = getattr(obj, attr, None)
        if isinstance(ch, (tuple, list)):
            for c in ch:
                found.extend(_collect_stages(c, depth + 1))
    for attr in ("placed", "program", "inner", "body", "template"):
        sub = getattr(obj, attr, None)
        if sub is not None and sub is not obj:
            found.extend(_collect_stages(sub, depth + 1))
    if not found and hasattr(obj, "work") and callable(getattr(obj, "start", None)):
        found.append(obj)
    return found


def _certify_stages_dual_bound(stages: Sequence[Any], geom: Geometry) -> tuple[bool, int, str]:
    rates, peaks = dict(geom.rates), dict(geom.compute_peaks)
    aliases = {"hbm_bytes": "hbm", "mte2_bytes": "mte2_aiv", "mte2": "mte2_aiv", "mte3_bytes": "mte3", "vector_ops": "vector"}
    res_ns: dict[str, int] = {}
    quanta, name_idx, edges = [], {}, []
    for st in stages:
        st_ns = int(round(float(getattr(st, "latency", 0.0) or 0.0) * NS_PER_MS))
        wmap = getattr(st, "work", None)
        if isinstance(wmap, Mapping):
            for rn, amt_raw in wmap.items():
                amt = float(amt_raw)
                if amt <= 0.0:
                    continue
                rk = str(rn)
                rate = float(rates.get(rk) or rates.get(aliases.get(rk, "")) or peaks.get(rk) or (peaks.get(rk.split(":", 1)[-1], peaks.get("bf16", 0.0)) if rk.startswith("cube") else 0.0) or 0.0)
                if rate <= 0.0 or not math.isfinite(rate):
                    return False, 0, f"resolved_stage_resource_rate_missing:{rk}"
                dur = max(1, math.ceil(amt / rate * _NS_PER_SECOND))
                res_ns[rk] = res_ns.get(rk, 0) + dur
                st_ns = max(st_ns, dur)
        if st_ns > 0:
            idx = len(quanta)
            name_idx[str(getattr(st, "name", f"s{idx}"))] = idx
            quanta.append(st_ns)
            edges.extend(DagEdge(name_idx[str(d)], idx, 0) for d in (getattr(st, "after", ()) or ()) if str(d) in name_idx)
    if not quanta:
        return False, 0, "resolved_operator_program_zero_executable_events"
    gwork = tuple(sorted(res_ns.values())) if res_ns else (max(quanta),)
    prog, dag = ResourceChoiceProgram(len(gwork), gwork, ()), EventDagSpec(tuple(quanta), tuple(edges))
    cert, _ = certify_lower_bound(prog, dag)
    if not check_optimality_certificate(prog, dag, cert) or cert.l_star_lean_ns <= 0:
        return False, int(cert.l_star_lean_ns), "lean_optimality_certificate_rejected"
    return True, int(cert.l_star_lean_ns), "verified"


def _verify_resolved_operator_dual_bound(resolved: Any, ctx: OperatorPlanningContext) -> tuple[bool, int, str]:
    recipe = getattr(resolved, "recipe", None)
    if not isinstance(recipe, OperatorPlan) or not callable(recipe.program):
        return False, 0, "resolved_operator_missing_operator_plan"
    from mindie_sim.prediction.program.binding import RecipeContext

    r_ctx = RecipeContext(op=getattr(resolved, "op", None), work=dict(recipe.work.resources), resource_names={str(k): str(k) for k in dict(ctx.geometry.rates)}, rates=dict(ctx.geometry.rates), core_count=max(1, int(ctx.geometry.compute_units)), resource_units=dict(ctx.geometry.resource_units), compute_peaks=dict(ctx.geometry.compute_peaks))
    try:
        lowered = recipe.lower(r_ctx)
    except TimeoutError:
        raise
    except Exception as exc:
        return False, 0, f"resolved_operator_program_lower_failed:{exc}"
    stages = _collect_stages(lowered) if (lowered is not None and callable(getattr(lowered, "start", None))) else []
    return _certify_stages_dual_bound(stages, ctx.geometry) if stages else (False, 0, "resolved_operator_program_zero_executable_events")


def _verify_sim_owner(is_pkg: bool, target_path: Path, family: str, template_id: str, family_registry: FamilyRegistry | None, template_catalog: TemplateCatalog | None) -> tuple[bool, dict[str, Any], list[str]]:
    if is_pkg:
        reg = family_registry or builtin_family_registry()
        spec = reg.spec_for_family(family) or next((c for c in reg.specs() if c.name == family or family in c.families), None)
        if spec is None:
            return False, {"owner": "builtin_family_registry", "checked": True, "matched": False, "lowered": False}, [f"sim_registry_unmatched:{family}"]
        op = OpSpec(f"probe_{family}", sorted(spec.families)[0] if spec.families else family, sorted(spec.owned_recipe_ids)[0] if spec.owned_recipe_ids else "", "decode", "bf16", (), _DEFAULT_PROBE_SHAPE, _DEFAULT_PROBE_SHAPE, _DEFAULT_PROBE_SHAPE, _DEFAULT_PROBE_SHAPE)
        matched = reg.match(op)
        if matched is None:
            return False, {"owner": "builtin_family_registry", "checked": True, "matched": False, "lowered": False}, [f"sim_registry_unmatched:{family}"]
        ctx = _build_default_planning_context(op)
        try:
            resolved = matched.resolve(op, ctx)
        except Exception as exc:
            return False, {"owner": "builtin_family_registry", "checked": True, "matched": True, "lowered": False}, [f"sim_owner_resolve_failed:{family}:{exc}"]
        ok_l, l_star, reason = _verify_resolved_operator_dual_bound(resolved, ctx)
        det = {"owner": "builtin_family_registry", "checked": True, "matched": True, "lowered": ok_l, "owned_recipe_ids": sorted(str(x) for x in matched.owned_recipe_ids), "resolved_implementation": str(getattr(resolved, "implementation", "")), "lean_lower_bound_ns": l_star, "lean_status": reason}
        return ok_l, det, ([] if ok_l else [f"sim_owner_dual_bound_failed:{family}:{reason}"])

    cat = template_catalog or builtin_template_catalog()
    k2, k3 = set(cat.ids("K2")), set(cat.ids("K3"))
    req_id = template_id.strip() or family
    if req_id not in (k2 | k3):
        return False, {"owner": "builtin_template_catalog", "checked": True, "matched": False, "lowered": False}, [f"template_catalog_unregistered:{req_id}"]
    tier = "K2" if req_id in k2 else "K3"
    if template_catalog is None:
        from mindie_sim.templates import catlass_catalog, catlass_lower, dsl_catalog, dsl_lower
        owners = {Path(m.__file__).resolve() for m in ((catlass_catalog, catlass_lower) if tier == "K2" else (dsl_catalog, dsl_lower)) if getattr(m, "__file__", None)}
        if target_path not in owners:
            return False, {"owner": "builtin_template_catalog", "checked": True, "matched": False, "lowered": False}, [f"template_id_not_owned_by_target:{req_id}:{target_path.name}"]
    problem = {"m": _DEFAULT_TEMPLATE_DIM, "n": _DEFAULT_TEMPLATE_DIM, "k": _DEFAULT_TEMPLATE_DIM, "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "bf16"}
    for sku in (_DEFAULT_PROBE_SKU, _DEFAULT_PROBE_SKU_A5):
        hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
        die = hw.dies[0]
        dp = datapath_for_die(die)
        geom = Geometry(compute_units=die.compute_units, resource_units=die.resource_units, buffers=die.buffers, buffer_units=die.buffer_units, rates=die.rates, compute_peaks=die.compute_peaks, compute_arch=die.compute_arch, datapath=dp)
        try:
            specs = tuple(cat.enumerate(tier, problem, dp, template_id=req_id))  # type: ignore[arg-type]
            if not specs:
                continue
            low = cat.lower(tier, specs[0], problem, dp)  # type: ignore[arg-type]
            ok_l, l_star, reason = _certify_stages_dual_bound(_collect_stages(getattr(low, "program", low)), geom)
            return ok_l, {"owner": "builtin_template_catalog", "checked": True, "matched": True, "tier": tier, "template_id": req_id, "lowered": ok_l, "lean_lower_bound_ns": l_star, "lean_status": reason}, ([] if ok_l else [f"template_lower_verification_failed:{req_id}:{reason}"])
        except Exception:
            continue
    return False, {"owner": "builtin_template_catalog", "checked": True, "matched": True, "lowered": False}, [f"template_enumeration_empty:{req_id}"]


def validate_operator_contract(
    *, target_path: str | Path, repo_root: str | Path | None = None, refs_root: str | Path | None = None,
    repo_map: Mapping[str, str | Path] | None = None, profiler_root: str | Path | None = None,
    profiler_family: str = "", profiler_recipe: str = "", profiler_impl: str = "", template_id: str = "",
    family_registry: FamilyRegistry | None = None, template_catalog: TemplateCatalog | None = None,
    require_sim_owner: bool = True, require_taxonomy: bool = True,
    require_profiler_builder: bool = True, require_tiling_axes: bool = True, require_mounted_refs: bool = False,
) -> dict[str, Any]:
    """Stage 1B: Validate ABI, SourceRef Git blob SHAs, Sim/Profiler owners, obligations, and taxonomy."""
    root = resolve_repo_root(repo_root)
    raw_t = Path(target_path)
    target = (raw_t if raw_t.is_absolute() else (root / raw_t)).resolve()
    if not target.exists():
        raise ModelingError(f"operator_target_not_found:{target}")
    is_pkg = target.is_dir()
    family = target.name if is_pkg else target.stem.removesuffix("_spec")
    viols: list[str] = []
    if is_pkg:
        files = [target / fn for fn in _REQUIRED_FAMILY_FILES if (target / fn).is_file()]
        viols.extend(f"missing_required_family_file:{fn}" for fn in _REQUIRED_FAMILY_FILES if not (target / fn).is_file())
    else:
        files = [target]
    abis, refs, symbols, decl_builder, _decl_tids, ast_v = _scan_ast_files(files, is_pkg=is_pkg)
    viols.extend(ast_v)
    has_sem, has_obl, has_axes = "SemanticWorkItem" in symbols, "TensorRangeObligation" in symbols, "TilingAxes" in symbols or "TILING_AXES" in symbols
    if not has_sem:
        viols.append("missing_semantic_work_item_declaration")
    if not has_obl:
        viols.append("missing_tensor_range_obligation_declaration")
    if require_tiling_axes and not has_axes:
        viols.append("missing_tiling_axes_declaration")

    r_root = Path(refs_root).resolve() if refs_root is not None else None
    r_map = {str(k): Path(v).resolve() for k, v in (repo_map or {}).items()}
    if not refs:
        viols.append("source_ref_missing")
    all_blobs_ok = bool(refs)
    for ref in refs:
        repo, rev, rel, exp_sha = ref["repository"], ref["revision"], ref["relative_path"], ref["source_sha256"].lower()
        if not repo or len(rev) < HEX_COMMIT_LEN or not rel or rel.startswith("/") or len(exp_sha) != SHA256_HEX_LENGTH:
            viols.append(f"source_ref_invalid:{ref['file']}")
            all_blobs_ok = False
            continue
        rdir = r_map.get(repo) or r_map.get(repo.rsplit("/", 1)[-1]) or (next((r_root / c for c in (repo, repo.rsplit("/", 1)[-1]) if (r_root / c).exists()), None) if r_root else None)
        if rdir is None:
            all_blobs_ok = False
            if r_root is not None or r_map or require_mounted_refs:
                viols.append(f"source_repo_not_mounted:{repo}")
            continue
        try:
            done = subprocess.run(["git", "-C", str(rdir), "cat-file", "-p", f"{rev}:{rel}"], capture_output=True, timeout=_GIT_CAT_TIMEOUT_S, check=False, **subprocess_window_kwargs())
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"git_cat_file_timeout:{repo}:{rel}") from exc
        except TimeoutError:
            raise
        except OSError as exc:
            viols.append(f"git_cat_file_error:{repo}:{exc}")
            all_blobs_ok = False
            continue
        if done.returncode != 0 or hashlib.sha256(done.stdout).hexdigest().lower() != exp_sha:
            viols.append(f"source_sha256_mismatch:{repo}:{rel}")
            all_blobs_ok = False

    sim_disp: dict[str, Any] = {"checked": False, "matched": False, "lowered": False, "lean_status": "skipped"}
    if require_sim_owner:
        _ok_s, sim_disp, s_v = _verify_sim_owner(is_pkg, target, family, template_id, family_registry, template_catalog)
        viols.extend(s_v)

    prof_disp: dict[str, Any] = {"checked": False, "verified": False}
    if require_profiler_builder:
        pfam, prec = profiler_family or family, profiler_recipe or decl_builder or family
        pimpl = profiler_impl or str(sim_disp.get("resolved_implementation") or next(iter(abis.values()), ""))
        try:
            bname, r_abi = resolve_profiler_builder_strict(pfam, prec, pimpl, profiler_root=profiler_root)
            prof_disp = {"checked": True, "verified": True, "builder": bname, "resolved_abi": r_abi}
            if abis and r_abi and r_abi not in set(abis.values()):
                viols.append(f"profiler_abi_mismatch:resolved={r_abi}:declared={sorted(abis.values())}")
        except ModelingError as exc:
            prof_disp = {"checked": True, "verified": False, "error": str(exc)}
            viols.append(str(exc))

    tax_info: dict[str, Any] = {"checked": False, "registered": False}
    if require_taxonomy:
        t_rel = str(target.relative_to(root)) if target.is_relative_to(root) else str(target_path)
        tax_f = root / "tests" / "TAXONOMY.toml"
        reg_ok, reg_det = False, "taxonomy_toml_missing"
        if tax_f.is_file():
            parsed = tomllib.loads(tax_f.read_text(encoding="utf-8"))
            for mod in parsed.get("module", ()):
                if isinstance(mod, Mapping):
                    mpath = str(mod.get("path") or "")
                    owns = [str(o).rstrip("/") for o in mod.get("owns", ())]
                    if any(t_rel.rstrip("/") == o or t_rel.rstrip("/").startswith(o + "/") for o in owns) or f"test_{family}.py" in mpath:
                        reg_ok = (root / "tests" / mpath).is_file()
                        reg_det = mpath if reg_ok else f"taxonomy_module_missing_on_disk:{mpath}"
                        break
            else:
                reg_det = f"taxonomy_registration_missing:{t_rel}"
        tax_info = {"checked": True, "registered": reg_ok, "detail": reg_det}
        if not reg_ok:
            viols.append(reg_det)

    if r_root is not None and (root / "scripts" / "check_template_provenance.py").is_file():
        prov_gate = run_template_provenance_gate(repo_root=root, refs_root=r_root, repo_map=r_map or None)
        if not prov_gate.passed:
            viols.append(f"template_provenance_gate_failed:{prov_gate.detail}")

    ok = not viols
    out = {
        "stage": "validate-contract", "ok": ok, "status": "pass" if ok else "fail", "family": family,
        "upstream_abis": abis, "source_blobs_verified": all_blobs_ok,
        "semantic_work_declared": has_sem, "tensor_obligations_declared": has_obl, "tiling_axes_declared": has_axes,
        "sim_owner_dispatch": sim_disp, "profiler_builder": prof_disp, "taxonomy": tax_info, "violations": viols,
    }
    assert_sanitized_payload(out, field_name="operator_contract_report")
    return out


def generate_operator_cliff_plan(
    *, operator_id: str, family: str, sku: str, tile_dim: int, core_count: int, l2_bytes: int,
    bytes_per_token: int = 0, topk_threshold: int | None = None,
    context_min: int = CONTEXT_MIN_TOKENS, context_max: int = CONTEXT_MAX_TOKENS,
    campaign_id: str = "onboard-op", plan_out: str | Path | None = None,
) -> dict[str, Any]:
    """Stage 2: Synthesize disjoint Anchor + Off-Grid Cliff + ABI-rejection plan for an operator."""
    plan = synthesize_operator_cliff_plan(
        OperatorCliffParameters(operator_id, family, int(tile_dim), int(core_count), int(l2_bytes), int(bytes_per_token), topk_threshold, int(context_min), int(context_max)),
        sku=sku, campaign_id=campaign_id,
    ).to_dict()
    if plan_out is not None:
        write_atomic_json(plan_out, plan)
    return {"stage": "generate-cliff-plan", "ok": True, "status": "pass", "plan": plan}


def bench_and_score_operator(
    *, cliff_plan: str | Path | Mapping[str, Any], sku: str,
    captured_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    live_profiler: bool = False, endpoint_alias: str = "", profiler_root: str | Path | None = None,
    coordinator_bin: str = "", threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    atol_ns: int = DEFAULT_PHYSICAL_ATOL_NS, out_score: str | Path | None = None,
    capture_root: str | Path | None = None, require_capture_files: bool = False,
    runner: Any = None, bench_dispatcher: Callable[[Mapping[str, Any], int], Sequence[Mapping[str, Any]]] | None = None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Stage 3: Dispatch operator bench plan or score captured silicon rows against white-box ``mindie-sim``."""
    cap_pct, eff_atol = validate_strict_threshold_pct(threshold_pct), validate_strict_atol_ns(atol_ns)
    plan_doc = dict(read_bounded_json(cliff_plan) if isinstance(cliff_plan, (str, Path)) else cliff_plan)
    plan_sha = verify_cliff_plan_integrity(plan_doc)
    plan_family = str(plan_doc.get("family") or "").strip()
    custom_run = (lambda p: bench_dispatcher(p, DEFAULT_GATE_SUBPROCESS_TIMEOUT_S)) if bench_dispatcher is not None else None
    rows, abi_obs, lease_pub, req_npus = acquire_stage3_observations(
        plan_doc, sku=sku, raw_source=captured_json, custom_runner=custom_run,
        live_profiler=live_profiler, endpoint_alias=endpoint_alias, profiler_root=profiler_root,
        coordinator_bin=coordinator_bin, capture_root=capture_root,
        abi_rejection_results=abi_rejection_results, lease_acquirer=lease_acquirer,
    )
    exp_levels = {str(p["point_id"]): assign_plan_point_level(plan_doc, p) for sk in ("train_points", "heldout_points") for p in (plan_doc.get(sk) or ()) if isinstance(p, Mapping) and p.get("point_id")}
    scored = score_sealed_gate_cases(
        rows, sku=sku, runner=runner, threshold_pct=cap_pct, atol_ns=eff_atol,
        external_train_roots=[str(plan_doc["train_lineage_root"])] if plan_doc.get("train_lineage_root") else [],
        expected_points=plan_doc.get("heldout_points") or (), expected_train_points=plan_doc.get("train_points") or (),
        expected_levels=exp_levels, expected_family=plan_family, require_disjoint_level_roots=False,
        capture_root=capture_root, require_capture_files=require_capture_files,
    )
    abi_chk = evaluate_abi_rejections(plan_doc.get("abi_rejection_points") or (), abi_obs)
    ok = bool(scored["ok"]) and bool(abi_chk["ok"])
    status = "insufficient_evidence" if scored["status"] == "insufficient_evidence" else ("pass" if ok else "fail")
    report: dict[str, Any] = {
        "stage": "bench-and-score", "ok": ok, "status": status, "sku": sku,
        "threshold_pct": cap_pct, "atol_ns": eff_atol, "target_id": str(plan_doc.get("target_id") or ""),
        "family": plan_family, "plan_sha256": plan_sha, "required_ranks": req_npus,
        "acquisition_mode": "live_silicon" if live_profiler else "offline_reverify",
        "capture_files_verified": bool(scored["capture_files_verified"]), "lease": lease_pub,
        "missing_heldout_points": scored["missing_plan_points"], "missing_train_points": scored["missing_train_points"],
        "over_budget_points": scored["over_budget_points"], "failing_cases": scored["failing_cases"],
        "abi_rejection_check": abi_chk, "lineage": scored["lineage"],
        "heldout_summary": scored["heldout_summary"], "train_summary": scored["train_summary"], "residuals": scored["residuals"],
    }
    rec_u, rec_w = collect_receipt_warnings(lease_pub)
    if rec_u or rec_w:
        report["receipt_unconfirmed"], report["receipt_warning"] = rec_u, rec_w
    assert_sanitized_payload(report, field_name="bench_and_score_report")
    if out_score is not None:
        write_atomic_json(out_score, report)
    return report


def certify_operator(
    *, operator_id: str, sku: str, contract_report: str | Path | Mapping[str, Any],
    score_report: str | Path | Mapping[str, Any], lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    capture_root: str | Path | None = None, require_capture_files: bool = False,
    repo_root: str | Path | None = None, threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    atol_ns: int = DEFAULT_PHYSICAL_ATOL_NS, run_repo_gates: bool = True, orthogonality_base: str | None = None,
    out_receipt: str | Path | None = None, diagnosis_out: str | Path | None = None,
) -> dict[str, Any]:
    """Stage 4: Certify operator contract, pointwise cliff APE, Lean 4 non-inversion, and repo gates."""
    del capture_root, require_capture_files
    cap_pct, eff_atol = validate_strict_threshold_pct(threshold_pct), validate_strict_atol_ns(atol_ns)
    c_doc = dict(read_bounded_json(contract_report) if isinstance(contract_report, (str, Path)) else contract_report)
    s_doc = dict(read_bounded_json(score_report) if isinstance(score_report, (str, Path)) else score_report)
    id_ok = str(s_doc.get("target_id") or "") == operator_id and bool(c_doc.get("ok"))
    pre = (GateCheckOutcome("operator_contract", id_ok, "pass" if id_ok else "fail", "contract valid" if id_ok else f"contract_or_id_invalid:{operator_id}"),)
    all_ok, status, receipt, diagnosis = finalize_stage4_certification(
        schema=OPERATOR_ONBOARDING_RECEIPT_SCHEMA, target_kind="operator", target_id=operator_id, sku=sku,
        score_doc=s_doc, lean_bounds_json=lean_bounds_json, cap_pct=cap_pct, eff_atol=eff_atol,
        pre_gates=pre, ape_gate_name="cliff_heldout_ape", artifacts={"plan_sha256": str(s_doc.get("plan_sha256") or "")},
        repo_root=repo_root, run_repo_gates=run_repo_gates, orthogonality_base=orthogonality_base,
        out_receipt=out_receipt, diagnosis_out=diagnosis_out,
    )
    out: dict[str, Any] = {"stage": "certify", "ok": all_ok, "status": status, "operator_id": operator_id, "sku": sku, "receipt": receipt}
    if diagnosis is not None:
        out["diagnosis"] = diagnosis
    return out


def run_operator_onboarding(
    *, operator_id: str, family: str, target_path: str | Path, sku: str, tile_dim: int, core_count: int, l2_bytes: int,
    bytes_per_token: int = 0, topk_threshold: int | None = None,
    captured_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    repo_root: str | Path | None = None, refs_root: str | Path | None = None,
    repo_map: Mapping[str, str | Path] | None = None, profiler_root: str | Path | None = None,
    profiler_family: str = "", profiler_recipe: str = "", profiler_impl: str = "", template_id: str = "",
    family_registry: FamilyRegistry | None = None, template_catalog: TemplateCatalog | None = None,
    require_sim_owner: bool = True, require_taxonomy: bool = True, require_profiler_builder: bool = True,
    require_tiling_axes: bool = True, require_mounted_refs: bool = False,
    out_dir: str | Path | None = None, run_repo_gates: bool = True, orthogonality_base: str | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT, live_profiler: bool = False,
    endpoint_alias: str = "", coordinator_bin: str = "", capture_root: str | Path | None = None,
    require_capture_files: bool = False, runner: Any = None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute all 4 New-Operator onboarding stages end-to-end."""
    base_out = Path(out_dir) if out_dir is not None else None
    s1 = validate_operator_contract(target_path=target_path, repo_root=repo_root, refs_root=refs_root, repo_map=repo_map, profiler_root=profiler_root, profiler_family=profiler_family, profiler_recipe=profiler_recipe, profiler_impl=profiler_impl, template_id=template_id, family_registry=family_registry, template_catalog=template_catalog, require_sim_owner=require_sim_owner, require_taxonomy=require_taxonomy, require_profiler_builder=require_profiler_builder, require_tiling_axes=require_tiling_axes, require_mounted_refs=require_mounted_refs)
    s2 = generate_operator_cliff_plan(operator_id=operator_id, family=family, sku=sku, tile_dim=tile_dim, core_count=core_count, l2_bytes=l2_bytes, bytes_per_token=bytes_per_token, topk_threshold=topk_threshold, plan_out=(base_out / "cliff_plan.json") if base_out else None)
    s3 = bench_and_score_operator(cliff_plan=s2["plan"], sku=sku, captured_json=captured_json, live_profiler=live_profiler, endpoint_alias=endpoint_alias, profiler_root=profiler_root, coordinator_bin=coordinator_bin, threshold_pct=threshold_pct, out_score=(base_out / "score_report.json") if base_out else None, capture_root=capture_root, require_capture_files=require_capture_files, runner=runner, abi_rejection_results=abi_rejection_results)
    s4 = certify_operator(operator_id=operator_id, sku=sku, contract_report=s1, score_report=s3, lean_bounds_json=lean_bounds_json, capture_root=capture_root, require_capture_files=require_capture_files, repo_root=repo_root, threshold_pct=threshold_pct, run_repo_gates=run_repo_gates, orthogonality_base=orthogonality_base, out_receipt=(base_out / "operator_onboarding_receipt.json") if base_out else None, diagnosis_out=(base_out / "tail_diagnosis.json") if base_out else None)
    ok = bool(s1["ok"] and s2["ok"] and s3["ok"] and s4["ok"])
    return {"stage": "run-all", "ok": ok, "status": s4["status"] if ok else "failed", "operator_id": operator_id, "sku": sku, "stages": {"validate-contract": s1, "generate-cliff-plan": s2, "bench-and-score": s3, "certify": s4}}
