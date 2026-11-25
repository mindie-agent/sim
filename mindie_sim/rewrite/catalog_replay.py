"""Pure CATLASS and DSL template replay binding and canonical summary derivation.

[Role] Resolve CATLASS (K2) and DSL/FuseRegion (K3) bindings into executable
  ``PlannedRecipe`` + ``FusedRegionSummary`` pairs for rewrite replay and
  verification without importing ``mindie_sim.optimization``.
[Invariants] Replay specialization, obligation derivation, and canonical summary
  checking share one deterministic template lowering path; zero hardcoded model
  geometry defaults (no 96/7168).
[Timeouts & Locks] Local finite template enumeration and lowering; no locks or I/O.
[Side-Effect Policy] Pure read-only capability queries and recipe construction.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial
import hashlib
from math import prod

from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import Program
from mindie_sim.prediction.runtime.memory_physics import _merge_byte_intervals
from mindie_sim.prediction.streams import MAIN_STREAM, bind_stream_owners
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import DatapathCapabilitySpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.templates.catlass_catalog import CatlassProblem, matrix_problem, streamk_partition
from mindie_sim.templates.contracts import CatlassKernelSpec, DslKernelSpec, _union_obligations
from mindie_sim.templates.dsl_catalog import (
    ATTN_RES, COMMIT_KDA, DSL_KERNEL_IDS, FLASH_KDA, MEGA_KDA, MLA_PROLOG, MQSMLA, QSLI,
    AttnResProblem, CommitRecurrentKdaProblem, FlashKdaProblem, MegaRecurrentKdaProblem,
    MixedQuantSparseMlaProblem, MlaPrologProblem, QsliProblem,
)
from mindie_sim.templates.registry import builtin_template_catalog

from .schema import FuseRegion, KernelBoundary, OpRef, SelectTemplate

MATRIX_RANK = 3
_RECIPE_DIGEST_LEN = 16
_PAIR_LEN = 2
_GROUPED_FAMILIES = frozenset({"grouped_gemm", "grouped_matmul", "grouped_moe", "mc2"})
_MX_MODES = frozenset({"mxfp8", "mxfp4", "a8w4_mx"})
_FP8_FORMATS = frozenset({"fp8", "fp8_e4m3", "fp8_e4m3fn", "fp8_e5m2"})
_FP4_FORMATS = frozenset({"fp4", "fp4_e2m1"})
_WEIGHT_ID_KEYS = ("weight_allocation_id", "allocation_id")
_DSL_RECIPE_ALIAS = {
    "block_attn_res_update_rms_norm": ATTN_RES,
    "quant_lightning_indexer": QSLI,
    "quant_sparse_lightning_indexer": QSLI,
    "mla_prolog": MLA_PROLOG,
    "mla_prolog_v4": MLA_PROLOG,
    "mixed_quant_sparse_flash_mla": MQSMLA,
    "flash_kda": FLASH_KDA,
    "mega_recurrent_kda": MEGA_KDA,
    "commit_recurrent_kda": COMMIT_KDA,
}
_CATLASS_FIELDS = (
    "dispatch_policy", "scheduler", "splitk_factor", "swizzle_offset", "swizzle_direction",
    "l1_tile_shape", "l0_tile_shape", "stages", "epilogue_kind", "simt_thread_block_dim",
)
_PIPE_ROLE_MAP = {
    "mte2_aic": "mte2",
    "mte2_aiv": "mte2",
    "vec_alu": "vector",
    "vec_reduce": "vector",
    "vec_sfu": "vector",
    "sort_topk": "vector",
    "aiv_simt": "vector",
}

__all__ = (
    "MATRIX_RANK",
    "streamk_crossing_partials",
    "select_catlass_candidates",
    "select_dsl_candidates",
    "_catlass_proposals",
    "_dsl_proposals",
    "_dsl_declarations",
    "_declared_regions",
    "_convex_members",
    "_matrix_problem",
    "_region_lowerer",
    "_encode_spec_value",
    "_replay_ops",
    "bind_catalog_recipe",
    "replay_catalog_binding",
    "canonical_region_summaries",
    "summary_matches_canonical",
    "catalog_recipe_id",
)


def streamk_crossing_partials(b_tot: int, g_k: int, c_aic: int) -> int:
    """Return exact StreamK cross-tile partial count for ``(b_tot, g_k, c_aic)``."""
    if b_tot < 0 or g_k < 1 or c_aic < 0:
        raise ValueError("invalid StreamK geometry")
    return streamk_partition(b_tot, g_k, c_aic).partial_count if c_aic else 0


def _weight_allocation_id(meta: Mapping[str, object]) -> str | None:
    return next((
        value for key in _WEIGHT_ID_KEYS
        if isinstance(value := meta.get(key), str) and value.strip()
    ), None)


_CATLASS_DTYPE_ALIASES: Mapping[str, str] = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float8_e4m3fn": "fp8_e4m3",
    "float8_e4m3": "fp8_e4m3",
    "fp8_e4m3fn": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
    "float8_e5m2": "fp8_e5m2",
    "float4_e2m1": "fp4_e2m1",
    "fp4": "fp4_e2m1",
    "ue8m0": "fp8_e8m0",
}
_ABI_METHOD_MX_MODES: Mapping[str, str] = {
    "mx_fp8": "mxfp8",
    "mx_fp4": "mxfp4",
}


def _norm_catlass_dtype(raw: object) -> str:
    tok = str(raw or "").strip().lower()
    return _CATLASS_DTYPE_ALIASES.get(tok, tok)


def _weight_backing_offset(meta: Mapping[str, object]) -> int:
    ranges = meta.get("weight_ranges", meta.get("ranges"))
    if not isinstance(ranges, Sequence) or isinstance(ranges, (str, bytes)) or not ranges:
        return 0
    spans = []
    for item in ranges:
        if not isinstance(item, Sequence) or len(item) != _PAIR_LEN:
            raise ModelingError("rewrite_catlass_invalid_weight_ranges")
        lo, hi = item
        if type(lo) is not int or type(hi) is not int or lo < 0 or hi <= lo:
            raise ModelingError("rewrite_catlass_invalid_weight_ranges")
        spans.append((lo, hi))
    merged = _merge_byte_intervals(spans)
    if len(merged) > 1:
        raise ModelingError("rewrite_catlass_disjoint_weight_ranges")
    return merged[0][0]


def _weight_backing_version(meta: Mapping[str, object]) -> int:
    version = meta.get("weight_tensor_version", meta.get("tensor_version", 0))
    return version if type(version) is int and version >= 0 else 0


def _resolve_catlass_precisions(op: OpSpec, meta: Mapping[str, object]) -> tuple[str, str, str, str, str | None]:
    """Derive ``(dtype_a, dtype_b, dtype_c, dtype_acc, mx_mode)`` from ``precision_abi`` + validated meta."""
    p_abi = getattr(op, "precision_abi", None)
    if p_abi is not None:
        dtype_a = _norm_catlass_dtype(p_abi.input_dtype or op.dtype)
        dtype_b = _norm_catlass_dtype(p_abi.weight_dtype or p_abi.input_dtype or op.dtype)
        dtype_c = _norm_catlass_dtype(p_abi.output_dtype or op.dtype)
        dtype_acc = _norm_catlass_dtype(
            p_abi.accum_dtype or ("int32" if dtype_a == "int8" else "fp32")
        )
        for field_name, expected, keys in (
            ("dtype_a", dtype_a, ("dtype_a", "activation_format")),
            ("dtype_b", dtype_b, ("dtype_b", "weight_format")),
            ("dtype_c", dtype_c, ("dtype_c", "output_format")),
            ("dtype_acc", dtype_acc, ("dtype_acc",)),
        ):
            for key in keys:
                raw_val = meta.get(key)
                if raw_val is not None and str(raw_val).strip():
                    if _norm_catlass_dtype(raw_val) != expected:
                        raise ModelingError(
                            f"rewrite_catlass_precision_abi_conflict:{op.op_id}:{field_name}"
                        )
        scale_fmt = _norm_catlass_dtype(
            p_abi.weight_scale_dtype or meta.get("weight_scale_format") or ""
        )
        mode = meta.get("native_matrix_mode")
        abi_method_mode = _ABI_METHOD_MX_MODES.get(p_abi.linear_method)
        mx_mode = mode if mode in _MX_MODES else None
        needs_mx = mx_mode is None and (scale_fmt == "fp8_e8m0" or abi_method_mode is not None)
    else:
        dtype_a = _norm_catlass_dtype(meta.get("dtype_a", meta.get("activation_format", op.dtype)))
        dtype_b = _norm_catlass_dtype(meta.get("dtype_b", meta.get("weight_format", op.dtype)))
        dtype_c = _norm_catlass_dtype(meta.get("dtype_c", meta.get("output_format", op.dtype)))
        dtype_acc = _norm_catlass_dtype(
            meta.get("dtype_acc", "int32" if dtype_a == "int8" else "fp32")
        )
        scale_fmt = _norm_catlass_dtype(meta.get("weight_scale_format") or "")
        mode = meta.get("native_matrix_mode", op.dtype)
        abi_method_mode = None
        mx_mode = mode if mode in _MX_MODES else None
        needs_mx = mx_mode is None and scale_fmt == "fp8_e8m0"
    if needs_mx:
        if dtype_a in _FP8_FORMATS and dtype_b in _FP8_FORMATS:
            mx_mode = "mxfp8"
        elif dtype_a in _FP4_FORMATS and dtype_b in _FP4_FORMATS:
            mx_mode = "mxfp4"
        elif dtype_a in (_FP8_FORMATS | {"int8"}) and dtype_b in (_FP4_FORMATS | {"int4"}):
            mx_mode = "a8w4_mx"
        elif abi_method_mode is not None:
            mx_mode = abi_method_mode
    return dtype_a, dtype_b, dtype_c, dtype_acc, mx_mode


def _matrix_problem(op: OpSpec, meta: Mapping[str, object]) -> CatlassProblem | None:
    if op.source_boundary == "call":
        return None
    weight_id = _weight_allocation_id(meta)
    b_base_byte = _weight_backing_offset(meta) if weight_id is not None else 0
    b_version = _weight_backing_version(meta) if weight_id is not None else 0
    declared = meta.get("catlass_problem")
    if declared is not None:
        problem = matrix_problem(declared)
        if weight_id is not None:
            updates: dict[str, object] = {"b_id": weight_id}
            if problem.b_base_byte == 0 and b_base_byte:
                updates["b_base_byte"] = b_base_byte
            if problem.b_version == 0 and b_version:
                updates["b_version"] = b_version
            problem = replace(problem, **updates)
        return problem
    shape = tuple(meta.get("matrix_shape", op.local_shape))
    if len(shape) != MATRIX_RANK or min(shape, default=0) <= 0 or op.link_bytes or op.vector_ops:
        return None
    if sum(flops for _, flops in op.matrix_flops or ((op.dtype, op.flops),)) != _PAIR_LEN * prod(shape):
        return None
    groups = tuple(meta.get("group_rows", ()))
    if op.family in _GROUPED_FAMILIES and not groups:
        return None
    dtype_a, dtype_b, dtype_c, dtype_acc, mx_mode = _resolve_catlass_precisions(op, meta)
    p_abi = getattr(op, "precision_abi", None)
    has_weight_scale = bool(
        meta.get("weight_scale_format")
        or (p_abi is not None and p_abi.weight_scale_dtype)
    )
    if (dtype_a != dtype_b and not mx_mode) or (has_weight_scale and not mx_mode):
        return None
    return CatlassProblem(
        *shape, dtype_a, dtype_b, dtype_c, dtype_acc,
        groups=groups, a_id=f"{op.op_id}:a", b_id=weight_id or f"{op.op_id}:b",
        c_id=f"{op.op_id}:c", has_bias=bool(meta.get("has_bias")), mx_mode=mx_mode,
        b_base_byte=b_base_byte, b_version=b_version,
    )


def _catlass_specs(op, problem, datapath, aic, aiv, template_id=None):
    meta = dict(op.metadata)
    options: dict[str, object] = {}
    for key, parameter in (
        ("l1_tile_shape", "l1_tiles"),
        ("l0_tile_shape", "l0_tiles"),
        ("mutex_stages", "stage_options"),
        ("simt_thread_block_dim", "simt_thread_blocks"),
    ):
        if key in meta:
            options[parameter] = (tuple(meta[key]),)
    if "swizzle_offsets" in meta:
        options["swizzle_offsets"] = tuple(meta["swizzle_offsets"])
    return builtin_template_catalog().enumerate(
        "K2", problem, datapath, template_id=template_id, aic_budget=aic, aiv_budget=aiv,
        evg_tree=tuple(meta.get("evg_tree", ())), enable_unit_flag=bool(meta.get("enable_unit_flag")),
        enable_l1_resident=bool(meta.get("l1_resident_slot_match")), **options,
    )


def select_catlass_candidates(
    op: OpSpec, datapath: DatapathCapabilitySpec, aic_budget: int, aiv_budget: int,
) -> tuple[CatlassKernelSpec, ...]:
    if not aic_budget:
        return ()
    try:
        problem = _matrix_problem(op, dict(op.metadata))
        return _catlass_specs(op, problem, datapath, aic_budget, aiv_budget) if problem else ()
    except ModelingError:
        return ()


def _freeze_dsl_field_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_freeze_dsl_field_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_dsl_field_value(item) for item in value)
    return value


def _dsl_problem_kwargs(
    problem_cls: type, meta: Mapping[str, object], *, exclude: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Extract every declared dataclass field present in ``meta`` with frozen tuple sequences."""
    kw: dict[str, object] = {}
    for field_spec in fields(problem_cls):
        name = field_spec.name
        if name in exclude or name not in meta:
            continue
        raw = meta[name]
        if raw is None and field_spec.default is not None:
            continue
        kw[name] = _freeze_dsl_field_value(raw)
    return kw


def _infer_dsl_problem(op: OpSpec, meta: Mapping[str, object], kernel: str):
    if kernel == ATTN_RES and len(op.local_shape) >= _PAIR_LEN and op.layout == "contiguous":
        tokens = int(meta.get("tokens") or meta.get("rows") or prod(op.local_shape[:-1]))
        hidden_size = int(meta.get("hidden_size") or meta.get("hidden") or op.local_shape[-1])
        kw = _dsl_problem_kwargs(AttnResProblem, meta, exclude=frozenset({"tokens", "hidden_size"}))
        kw.setdefault("dtype", op.dtype)
        return AttnResProblem(tokens, hidden_size, **kw)
    if kernel == QSLI:
        q_rows = int(meta.get("query_rows") or (op.local_shape[0] if op.local_shape else 0))
        topk = int(meta.get("topk") or meta.get("selected_token_rows") or 0)
        cands = int(meta.get("candidates") or meta.get("kv_len") or 0)
        if not cands and len(op.local_shape) == MATRIX_RANK and op.flops > 0:
            denom = _PAIR_LEN * prod(op.local_shape)
            est = round(op.flops / denom) if denom > 0 else 0
            if est > 0 and est * denom == int(op.flops):
                cands = est
        if q_rows > 0 and cands > 0 and topk > 0:
            kw = _dsl_problem_kwargs(
                QsliProblem, meta, exclude=frozenset({"query_rows", "candidates", "topk"}),
            )
            if "split_parts" not in kw and meta.get("index_context_partitions") is not None:
                kw["split_parts"] = max(1, int(meta["index_context_partitions"]))
            return QsliProblem(q_rows, cands, topk, **kw)
    if kernel == MLA_PROLOG:
        tokens = int(meta.get("tokens") or (op.local_shape[0] if op.local_shape else 0))
        dim = int(meta.get("dim") or (op.local_shape[-1] if len(op.local_shape) >= _PAIR_LEN else 0))
        if tokens > 0 and dim > 0:
            kw = _dsl_problem_kwargs(MlaPrologProblem, meta, exclude=frozenset({"tokens", "dim"}))
            return MlaPrologProblem(tokens, dim=dim, **kw)
    if kernel == MQSMLA and ("n_ori" in meta or "n_cmp" in meta):
        n_ori, n_cmp = int(meta.get("n_ori", 0)), int(meta.get("n_cmp", 0))
        queries = int(meta.get("queries") or (op.local_shape[0] if op.local_shape else 1))
        if n_ori > 0 or n_cmp > 0:
            kw = _dsl_problem_kwargs(
                MixedQuantSparseMlaProblem, meta, exclude=frozenset({"n_ori", "n_cmp", "queries"}),
            )
            return MixedQuantSparseMlaProblem(n_ori, n_cmp, queries=queries, **kw)
    if kernel == FLASH_KDA and meta.get("lengths"):
        lengths = tuple(int(v) for v in meta["lengths"])
        kw = _dsl_problem_kwargs(FlashKdaProblem, meta, exclude=frozenset({"lengths"}))
        return FlashKdaProblem(lengths, **kw)
    if kernel == MEGA_KDA and {"batch", "tp", "sequence", "num_heads", "hidden_size"} <= meta.keys():
        kw = _dsl_problem_kwargs(
            MegaRecurrentKdaProblem, meta,
            exclude=frozenset({"batch", "tp", "sequence", "num_heads", "hidden_size"}),
        )
        return MegaRecurrentKdaProblem(
            int(meta["batch"]), int(meta["tp"]), int(meta["sequence"]),
            int(meta["num_heads"]), int(meta["hidden_size"]),
            **kw,
        )
    if kernel == COMMIT_KDA and meta.get("accepted"):
        accepted = tuple(int(v) for v in meta["accepted"])
        kw = _dsl_problem_kwargs(CommitRecurrentKdaProblem, meta, exclude=frozenset({"accepted"}))
        return CommitRecurrentKdaProblem(accepted, **kw)
    return None


def _dsl_declarations(op: OpSpec):
    base = dict(op.metadata)
    for index, override in enumerate(base.get("dsl_variants", ({},))):
        meta = {**base, **override}
        raw_id = meta.get("dsl_kernel")
        if raw_id is None:
            raw_id = _DSL_RECIPE_ALIAS.get(op.family) or _DSL_RECIPE_ALIAS.get(op.recipe) or (
                op.family if f"dsl.{op.family}" in DSL_KERNEL_IDS else op.recipe
            )
        identifier = "dsl." + str(raw_id).removeprefix("dsl.")
        kernel = "dsl." + identifier.removeprefix("dsl.").partition(".")[0]
        problem = meta.get("dsl_problem")
        if problem is None and kernel in DSL_KERNEL_IDS:
            problem = _infer_dsl_problem(op, meta, kernel)
        if problem is not None and kernel in DSL_KERNEL_IDS:
            yield index, identifier, problem, tuple(meta.get("dsl_members", (op.op_id,)))


def _convex_members(ops: Sequence[OpSpec], ids: tuple[str, ...]) -> tuple[OpSpec, ...]:
    if len(ids) != len(set(ids)) or not ids:
        return ()
    members = set(ids)
    by_id = {op.op_id: op for op in ops}
    ordered = tuple(op for op in ops if op.op_id in members)
    if tuple(op.op_id for op in ordered) != ids or any(
        op.link_bytes or dict(op.metadata).get("collective_barrier") for op in ordered
    ):
        return ()
    internal: set[str] = set()
    outside: list[str] = []
    for op in ordered:
        if any(dep in members and dep not in internal for dep in op.deps):
            return ()
        internal.add(op.op_id)
        outside.extend(dep for dep in op.deps if dep not in members)
    seen: set[str] = set()
    while outside:
        dep = outside.pop()
        if dep in members:
            return ()
        if dep in by_id and dep not in seen:
            seen.add(dep)
            outside.extend(by_id[dep].deps)
    return ordered


def _dsl_candidates(ops, datapath, aic, aiv):
    if not aic or not aiv:
        return
    for op in ops:
        for index, identifier, problem, members in _dsl_declarations(op):
            if op.op_id not in members or not _convex_members(ops, members):
                continue
            try:
                specs = builtin_template_catalog().enumerate(
                    "K3", problem, datapath, template_id=identifier, all_variants=True,
                    aic_budget=aic, aiv_budget=aiv,
                )
            except ModelingError:
                continue
            for spec in specs:
                yield spec, members, op.op_id, index


def select_dsl_candidates(
    ops: Sequence[OpSpec], datapath: DatapathCapabilitySpec, aic_budget: int, aiv_budget: int,
) -> tuple[tuple[DslKernelSpec, tuple[str, ...]], ...]:
    return tuple(dict.fromkeys(
        (spec, ids) for spec, ids, _, _ in _dsl_candidates(ops, datapath, aic_budget, aiv_budget)
    ))


def _declared_regions(ops: Sequence[OpSpec]):
    for op in ops:
        for index, declaration in enumerate(dict(op.metadata).get("fuse_regions", ())):
            if declaration.get("numerically_equivalent") and "compiler_options" in declaration:
                yield op.op_id, index, declaration


def _replay_ops(ops: Sequence[OpSpec]) -> tuple[OpSpec, ...]:
    return tuple(
        replace(op, metadata=tuple((key, value) for key, value in op.metadata if key != "fuse_regions"))
        if any(key == "fuse_regions" for key, _ in op.metadata) else op
        for op in ops
    )


def _region_lowerer(declaration, members, graph_ops):
    options = dict(declaration["compiler_options"])
    options.update(
        dependencies={op.op_id: op.deps for op in graph_ops},
        edge_routes=tuple(declaration["edge_routes"]),
        live_in_obligations=tuple(declaration.get("live_in_obligations", ())),
        live_out_obligations=tuple(declaration.get("live_out_obligations", ())),
        backend=declaration["backend"],
    )

    def lower(datapath, *, region_id, **binding):
        _, result = builtin_template_catalog().compile_region(
            region_id, tuple(op.op_id for op in members), datapath, **{**options, **binding},
        )
        return result

    return lower


def _encode_spec_value(value):
    return ",".join(map(str, value)) if isinstance(value, tuple) else "" if value is None else value


def _dsl_proposals(ops, datapath, aic, aiv):
    for spec, ids, declaring_op, index in _dsl_candidates(ops, datapath, aic, aiv):
        refs = tuple(OpRef("atom", op_id) for op_id in ids)
        meta = next(dict(op.metadata) for op in ops if op.op_id == ids[-1])
        params = (("aic_budget", spec.aic_budget), ("aiv_budget", spec.aiv_budget), ("declaration", index))
        if len(refs) == 1:
            yield (SelectTemplate(refs[0], "K3", spec.kernel_id, params),), spec
        elif "dsl_preserve_ports" in meta:
            params += (("declaring_op", declaring_op), ("variant", spec.template_variant))
            boundary = KernelBoundary(spec.kernel_id, refs, params)
            yield (FuseRegion(
                refs, tuple(meta["dsl_preserve_ports"]), (boundary,),
                fusion_tier="K3_dsl_pipeline",
            ),), spec
    for declaring_op, index, declaration in _declared_regions(ops):
        ids = tuple(declaration["op_ids"])
        members = _convex_members(ops, ids)
        if len(members) < _PAIR_LEN:
            continue
        identifier = declaration["region_id"]
        try:
            result = _region_lowerer(declaration, members, ops)(datapath, region_id=identifier)
        except ModelingError:
            continue
        refs = tuple(OpRef("atom", op_id) for op_id in ids)
        preserve = tuple(declaration.get("preserve_ports", ()))
        params = (("declaration", index), ("declaring_op", declaring_op))
        yield (FuseRegion(
            refs, preserve, (KernelBoundary(identifier, refs, params),),
            fusion_tier="K3_dsl_pipeline",
        ),), result.summary.spec


def _catlass_proposals(ops, datapath, aic, aiv):
    for op in ops:
        candidates = select_catlass_candidates(op, datapath, aic, aiv)
        first_by_template = {spec.template_id: spec for spec in reversed(candidates)}
        for spec in sorted(candidates, key=lambda item: item != first_by_template[item.template_id]):
            params = tuple((key, _encode_spec_value(getattr(spec, key))) for key in _CATLASS_FIELDS)
            params += (("aic_budget", spec.aic_budget), ("aiv_budget", spec.aiv_budget))
            yield (SelectTemplate(OpRef("atom", op.op_id), "K2", spec.template_id, params),), spec


def _catalog_lowerer(op, members, graph_ops, datapath, tier, identifier, params, limits):
    catalog = builtin_template_catalog()
    options = dict(params)
    if len(options) != len(params):
        raise ModelingError(f"rewrite_template_params_invalid:{op.op_id}")
    aic = options.pop("aic_budget", limits[0])
    aiv = options.pop("aiv_budget", limits[1])
    if any(type(budget) is not int or not 0 <= budget <= limit for budget, limit in zip((aic, aiv), limits)):
        raise ModelingError(f"rewrite_template_core_budget_invalid:{op.op_id}")
    if tier == "K2":
        problem = _matrix_problem(op, dict(op.metadata))
        if problem is not None and options.keys() <= set(_CATLASS_FIELDS):
            for spec in _catlass_specs(op, problem, datapath, aic, aiv, identifier):
                if all(_encode_spec_value(getattr(spec, key)) == value for key, value in options.items()):
                    return partial(catalog.lower, tier, spec, problem)
    elif tier == "K3":
        index = options.pop("declaration", None)
        declaring_op = options.pop("declaring_op", None)
        if not options and op.family == "rewrite_fused" and (index is None or type(index) is int):
            ids = tuple(member.op_id for member in members)
            sources = members if (declaring_op is None or any(m.op_id == declaring_op for m in members)) else graph_ops
            matches = [
                entry for source_op, ordinal, entry in _declared_regions(sources)
                if entry["region_id"] == identifier and tuple(entry["op_ids"]) == ids
                and (declaring_op is None or source_op == declaring_op)
                and (index is None or ordinal == index)
            ]
            if len(matches) == 1:
                return _region_lowerer(matches[0], members, graph_ops)
            if len(matches) > 1:
                raise ModelingError(f"rewrite_template_recipe_unbound:{op.op_id}:{identifier}")
        kernel = "dsl." + identifier.removeprefix("dsl.").partition(".")[0]
        if options.keys() <= {"variant"} and (index is None or type(index) is int):
            choices = []
            member_ids = tuple(member.op_id for member in members)
            for source in members:
                if declaring_op is not None and source.op_id != declaring_op:
                    continue
                for ordinal, declared, problem, ids in _dsl_declarations(source):
                    if (
                        ids == member_ids and declared in {kernel, identifier}
                        and (index is None or index == ordinal)
                    ):
                        choices.append(problem)
            if len(choices) == 1:
                spec, = catalog.enumerate(
                    tier, choices[0], datapath, template_id=identifier,
                    aic_budget=aic, aiv_budget=aiv, **options,
                )
                return partial(catalog.lower, tier, spec, choices[0])
    raise ModelingError(f"rewrite_template_recipe_unbound:{op.op_id}:{identifier}")


@dataclass(frozen=True, slots=True)
class _CatalogReplayProgram:
    """Keep directional row registration intact through static DAG coalescing."""

    program: Program

    def start(self, engine, name, ready, *, owner=""):
        return self.program.start(engine, name, ready, owner=owner)


def bind_catalog_recipe(
    op, members, datapath, *, tier, identifier, params, limits, die, replica, graph_ops=None,
):
    """Resolve the edit against its problem and use one specialization for audit and execution."""
    lower = _catalog_lowerer(
        op, members, graph_ops if graph_ops is not None else members,
        datapath, tier, identifier, params, limits,
    )
    result = lower(datapath, die=die, region_id=op.op_id)
    raw = result.summary
    summary = replace(
        raw,
        op_ids=tuple(member.op_id for member in members),
        semantic_work=tuple(
            replace(item, die=die, replica=replica, pp_stage=op.stage)
            for item in raw.semantic_work
        ),
    )
    if isinstance(raw.spec, DslKernelSpec):
        summary = replace(summary, **{
            field_name: tuple(
                replace(item, allocation_id=f"{op.op_id}:{item.allocation_id}")
                for item in getattr(raw, field_name)
            )
            for field_name in ("read_obligations", "write_obligations")
        })
    resources: Counter = Counter()
    if hasattr(result.program, "resource_work"):
        for role, amount in result.program.resource_work:
            if role != "cube":
                resources[_PIPE_ROLE_MAP.get(role, role)] += amount
    else:
        for stage in result.program.stages:
            for role, amount in stage.work.items():
                if role != "cube":
                    resources[_PIPE_ROLE_MAP.get(role, role)] += amount
    work = Work(dict(summary.issued_cube_flops_by_dtype), resources)

    def build(_op, _geometry):
        def program(context):
            dp = context.datapath or datapath
            if hasattr(result.program, "with_context"):
                bound = result.program.with_context(dp, context)
            else:
                bound = lower(dp, context=context, die=die, region_id=op.op_id).program
            return _CatalogReplayProgram(bound) if getattr(bound, "hbm_bus", None) is not None else bound

        return OperatorPlan(work, program)

    return PlannedRecipe(build), summary


def replay_catalog_binding(scenario, entry, op, binding, applied, stream):
    from .apply import _die, _path

    if binding and binding.tier in {"K2", "K3"} and binding.template_id.startswith(("catlass.", "dsl.")):
        tier, identifier, params = binding.tier, binding.template_id, binding.params
        members = (op,)
    elif op.family == "rewrite_fused" and dict(op.metadata).get("fusion_tier") == "K3_dsl_pipeline":
        edit = next(edit for edit in applied.plan.edits if op.op_id == f"rewrite:{edit.edit_id}")
        operation = edit.operation
        if (
            not isinstance(operation, FuseRegion)
            or len(operation.kernel_boundaries) != 1
            or any(ref.kind != "atom" for ref in operation.members)
        ):
            return None
        ids = {ref.id for ref in operation.members}
        members = tuple(member for member in entry.ops if member.op_id in ids)
        boundary, = operation.kernel_boundaries
        tier, identifier, params = "K3", boundary.abi_id, boundary.params
        if not identifier.startswith("dsl.") and not any(
            decl["region_id"] == identifier for _, _, decl in _declared_regions(entry.ops)
        ):
            return None
    else:
        return None
    path = _path(scenario, entry.replica, op.stage)
    limits = path.aic_count, path.aiv_count
    if stream is not None:
        owners = bind_stream_owners(stream, path)[dict(stream.assignments).get(op.op_id, MAIN_STREAM)]
        limits = owners.aic_limit, owners.aiv_limit
    recipe, summary = bind_catalog_recipe(
        op, members, path, tier=tier, identifier=identifier, params=params, limits=limits,
        die=_die(scenario, entry.replica, op.stage).rank, replica=entry.replica, graph_ops=entry.ops,
    )
    if binding and op.family == "rewrite_fused":
        summary = replace(summary, op_ids=dict(applied.absorbed_atoms)[op.op_id])
    return recipe, summary


def canonical_region_summaries(scenario, applied):
    """Derive canonical per-site summaries for all catalog-bound regions in ``applied``."""
    templates = {
        (getattr(binding, "replica", 0), binding.target_op): binding
        for binding in applied.template_bindings
    }
    atomic = {entry.replica: entry for entry in scenario.atomic_ops}
    streams = {stream.rank: stream for stream in applied.streams}
    canonical = {}
    for entry in applied.executable:
        rank_atomic = atomic.get(entry.replica)
        if rank_atomic is None:
            continue
        stream = streams.get(entry.replica)
        for op in entry.ops:
            bound = replay_catalog_binding(
                scenario, rank_atomic, op, templates.get((entry.replica, op.op_id)), applied, stream,
            )
            if bound is not None:
                canonical[entry.replica, op.stage, op.op_id] = bound[1]
    return canonical


def summary_matches_canonical(summary, canonical) -> bool:
    """Verify caller-supplied obligations, work and spec match the canonical recipe."""
    return (
        summary.tier == canonical.tier
        and summary.region_id == canonical.region_id
        and tuple(summary.op_ids) == tuple(canonical.op_ids)
        and summary.spec == canonical.spec
        and tuple(summary.semantic_work) == tuple(canonical.semantic_work)
        and _union_obligations(summary.read_obligations) == _union_obligations(canonical.read_obligations)
        and _union_obligations(summary.write_obligations) == _union_obligations(canonical.write_obligations)
    )


def _canon_digest_value(value):
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value).__qualname__, tuple(
            (name, _canon_digest_value(getattr(value, name)))
            for name in sorted(field.name for field in fields(value))
        ))
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _canon_digest_value(v)) for k, v in value.items()))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_canon_digest_value(v) for v in value))
    if isinstance(value, (list, tuple)):
        return tuple(_canon_digest_value(v) for v in value)
    return value


def catalog_recipe_id(replica: int, op: OpSpec, template, summary) -> str:
    """Content-address bound catalog recipes so isolated caches never collide."""
    payload = _canon_digest_value((summary, template))
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:_RECIPE_DIGEST_LEN]
    return f"rewrite_template:{replica}:{op.op_id}:{digest}"
