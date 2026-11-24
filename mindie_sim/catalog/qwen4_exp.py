"""Structural PLE, low-rank gated HC, and QSA shared-slab adapters.

[Role]
  Normalize per-layer n-gram conv PLE tables (1-based ``ple_layer_ids``, prime
  head layouts, conv kernel), low-rank gated hyper-connection matrices, QSA
  indexer weight terms, pure L1 PLE ``OpSpec`` lowering, and shared-slab pool
  breakdown delegation to canonical ``catalog.cache_pools.evaluate_hybrid_pool_receipt``.

[Invariants]
  * Zero imports from ``mindie_sim.prediction.*`` (AG010 clean).
  * Zero AG014 model-name comparisons; PLE detection uses structural
    ``storage_kind == 'ple_ngram_conv'`` / ``ple_conv_kernel_size``.
  * Zero hardcoded ``arch_family='A5'``, ``readiness='shipping'``, ``64``-owner
    caps, or silent ``bf16``/``int8`` fallbacks.
  * ``ple_layer_ids`` are one-based; a table's ``layer_id`` is ``ple_id - 1``.

[Timeouts & Locks]
  Pure integer/float arithmetic bounded by layer and head counts; no I/O or locks.

[Side-Effect Policy]
  No filesystem, network, device, or global mutation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from mindie_sim.catalog.cache_pools import (
    REGIME_SHARED_SLAB_PRIVATE_AUX,
    _align_up,
    _ceil_div,
    evaluate_hybrid_pool_receipt,
    exact_element_bytes,
)
from mindie_sim.catalog.config import _int
from mindie_sim.catalog.draft_weights import (
    lowrank_hc_draft_head_terms,
    lowrank_hc_draft_terms,
)
from mindie_sim.catalog.hf_value_formats import native_dtype
from mindie_sim.catalog.ops import comm_op, gemm_op, vector_op
from mindie_sim.catalog.speculative_state import (
    PHASE_DECODE,
    resolve_speculative_query_tokens,
)
from mindie_sim.catalog.types import EngramSpec, EngramTableSpec, merge_source
from mindie_sim.catalog.weight_terms import WeightTerm, _t
from mindie_sim.catalog.weights import (
    PLE_STORAGE_KIND,
    is_ple_engram,
    lowrank_hc_head_terms,
    ple_gate_terms_by_layer,
    ple_table_terms,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import (
    OpSpec,
    OwnerMemoryResidency,
    OwnerScheduleSpec,
    SerialPhaseSpec,
)
from mindie_sim.schema.parallel import ParallelSpec

MX_GROUP = 32
INT8_SCALE_BYTES = 4
MXFP8_SCALE_BYTES = 1
PLE_HBM_SECTOR_BYTES = 128
PLE_UVA_RECIPE = "ple_host_uva_lookup"
PLE_QUANTIZED_FORMATS = frozenset({"int8", "mxfp8"})
PLE_FLOAT_FORMATS = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})
DEFAULT_PLE_CONV_KERNEL = 4
DEFAULT_PLE_DIVISOR = 128
PLE_GATE_OPS_PER_ELEM = 6.0
PLE_LOOKUP_OPS_PER_ELEM = 2.0
PLE_ALLREDUCE_TRAFFIC_FACTOR = 2
PLE_RESIDENCY_DOUBLE_BUFFER = 2
PLE_GATE_READ_HC_FACTOR = 2
PLE_GATE_WRITE_HC_FACTOR = 2
PLE_CONV_OPS_FACTOR = 2
PLE_CONV_BIAS_OPS = 2
PLE_GATE_NORM_OPS_PER_ELEM = 4.0
PLE_GATE_BLEND_OPS_PER_ELEM = 2.0

QWEN4_EXP_PLE_SOURCE: dict[str, str] = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/models/qwen4_exp/ple.py",
    "source_sha256": "1327d2f777c8d262696865f0d5ffad1b061826e6f54b8d711a3af14e30f38033",
}
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_MILLER_RABIN_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in _SMALL_PRIMES:
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in _MILLER_RABIN_BASES:
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


def build_ple_spec(c: Mapping[str, Any]) -> EngramSpec:
    """Build structural ``EngramSpec`` for per-layer n-gram conv PLE tables."""
    ple_layers = [int(v) for v in c.get("ple_layer_ids", ()) or ()]
    max_ngram = _int(c, "ngram_size")
    heads = _int(c, "heads_per_ngram")
    embed = _int(c, "ple_embed_dim")
    base_rows = _int(c, "ngram_vocab_size_base")
    ngram_heads = (max_ngram - 1) * heads
    if max_ngram < 2 or heads <= 0 or embed <= 0 or ngram_heads <= 0:
        raise ModelingError("model_config_ple_geometry_missing")
    if embed % ngram_heads:
        raise ModelingError("model_config_ple_geometry_missing")
    width = embed // ngram_heads
    divisor = _int(c, "make_ngram_vocab_size_divisible_by", DEFAULT_PLE_DIVISOR)
    conv_k = _int(c, "ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL)
    elem_bytes = exact_element_bytes(1, native_dtype(c))
    tables: list[EngramTableSpec] = []
    for index, ple_id in enumerate(ple_layers):
        if ple_id <= 0:
            raise ModelingError("model_config_ple_layer_id_invalid")
        total = 0
        for local_head in range(ngram_heads):
            global_head = index * ngram_heads + local_head
            total += _nth_prime_after(base_rows - 1, global_head + 1)
        tables.append(EngramTableSpec(
            layer_id=ple_id - 1,
            table_id=index,
            rows=_align_up(total, divisor),
            width=width,
        ))
    return EngramSpec(
        tables=tuple(tables),
        max_ngram=max_ngram,
        heads=heads,
        width=width,
        pad_token_id=_int(c, "engram_pad_token_id", 0),
        host_response_bytes=width * elem_bytes,
        source=merge_source(
            QWEN4_EXP_PLE_SOURCE,
            {
                "storage_kind": PLE_STORAGE_KIND,
                "ple_layers": str(len(tables)),
                "ple_conv_kernel_size": str(conv_k),
                "ple_embed_dim": str(embed),
            },
        ),
    )


def is_qwen4_exp_ple(engram: Any) -> bool:
    """Structural predicate for per-layer n-gram conv PLE (zero model-name checks)."""
    return is_ple_engram(engram)


def _normalize_float_format(raw: str) -> str:
    low = str(raw or "").strip().lower()
    if low in {"bfloat16", "bf16"}:
        return "bf16"
    if low in {"float16", "fp16"}:
        return "fp16"
    if low in {"float32", "fp32"}:
        return "fp32"
    return low


def _table_marker_format(quant: Any) -> str | None:
    """Resolve declared PLE table format from checkpoint markers, or ``None``."""
    if quant is None:
        return None
    overrides = dict(getattr(quant, "category_overrides", ()) or ())
    for role in ("ple_table", "engram_table"):
        if role in overrides:
            fields = dict(overrides[role])
            wf = str(fields.get("weight_format", "") or "").lower().replace("-", "_")
            if wf in PLE_FLOAT_FORMATS:
                return _normalize_float_format(wf)
            if wf in {"mxfp8", "mx_fp8", "fp8_e4m3", "fp8"}:
                sf = str(fields.get("scale_format", "") or getattr(quant, "scale_format", "") or "").lower()
                return "mxfp8" if sf in {"fp8_e8m0", "ue8m0"} or "mx" in wf else "int8"
            if wf in {"int8", "w8a8"}:
                return "int8"
    exclusions = set(getattr(quant, "exclusions", ()) or ())
    if exclusions & {"ple_table", "engram_table", "ple"}:
        native = _normalize_float_format(str(getattr(quant, "native_format", "") or ""))
        if not native:
            raise ModelingError("ple_lookup_format_undeclared")
        return native
    table_fmt = str(getattr(quant, "engram_table_format", "") or "").lower().replace("-", "_")
    if table_fmt in PLE_FLOAT_FORMATS:
        return _normalize_float_format(table_fmt)
    if table_fmt in {"mxfp8", "mx_fp8", "fp8_e4m3", "fp8"}:
        sf = str(getattr(quant, "scale_format", "") or "").lower()
        return "mxfp8" if sf in {"fp8_e8m0", "ue8m0"} else "int8"
    if table_fmt == "int8":
        return "int8"
    return None


def ple_lookup_format(model: Any, runtime: Any = None) -> str:
    """Resolve declared PLE table precision strictly from checkpoint/runtime contracts."""
    quant = getattr(model, "quant", None)
    marker_fmt = _table_marker_format(quant)
    if marker_fmt in PLE_FLOAT_FORMATS:
        return marker_fmt
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    declared = str(metadata.get("ple_lookup_format", "") or "").strip().lower()
    pass_active = "ple_quantized_lookup" in tuple(getattr(runtime, "lowering_passes", ()) or ())
    if declared:
        if declared in PLE_QUANTIZED_FORMATS or declared in PLE_FLOAT_FORMATS:
            return _normalize_float_format(declared)
        raise ModelingError(f"ple_lookup_format_unsupported:{declared}")
    if marker_fmt in PLE_QUANTIZED_FORMATS:
        return marker_fmt
    if pass_active:
        raise ModelingError("ple_lookup_format_undeclared")
    model_native = _normalize_float_format(str(getattr(model, "activation_dtype", "") or ""))
    if model_native in PLE_FLOAT_FORMATS:
        return model_native
    raise ModelingError("ple_lookup_format_undeclared")


def _ple_owned_hits(lookups: int, tp: int) -> int:
    lookups_i, tp_i = int(lookups), max(1, int(tp))
    if lookups_i <= 0:
        return 0
    return _ceil_div(lookups_i, tp_i)


def _ple_row_bytes(width: int, lookup_format: str) -> int:
    width_i = int(width)
    if lookup_format in PLE_FLOAT_FORMATS:
        return exact_element_bytes(width_i, lookup_format)
    scale_bytes = MXFP8_SCALE_BYTES if lookup_format == "mxfp8" else INT8_SCALE_BYTES
    return width_i + _ceil_div(width_i, MX_GROUP) * scale_bytes


def _ple_read_bytes(
    hits_local: int, width: int, lookup_format: str, *, sector_aligned: bool = False,
) -> float:
    row_bytes = _ple_row_bytes(width, lookup_format)
    if sector_aligned and row_bytes > 0:
        row_bytes = _align_up(row_bytes, PLE_HBM_SECTOR_BYTES)
    return float(int(hits_local) * row_bytes)


def _runtime_aiv_capacity(runtime: Any) -> int:
    """Optional hardware AIV core ceiling from runtime metadata; 0 means hardware-unbound."""
    if runtime is None:
        return 0
    meta = dict(getattr(runtime, "metadata", ()) or ())
    raw = meta.get("aiv_cores") or meta.get("die_aiv_cores") or 0
    return max(0, int(raw or 0))


def _ple_lookup_owner_schedule(
    hits_local: int,
    read_bytes: float,
    write_bytes: float,
    index_ops: float,
    dequant_ops: float,
    row_bytes: int,
    *,
    aiv_capacity: int = 0,
) -> OwnerScheduleSpec:
    tiles = max(1, int(hits_local))
    owners = min(aiv_capacity, tiles) if aiv_capacity > 0 else tiles
    max_tiles = _ceil_div(tiles, owners)
    per_owner_read = read_bytes / float(owners)
    per_owner_write = write_bytes / float(owners)
    phases = [
        SerialPhaseSpec("index_address", "vec_alu", index_ops / float(owners), "op"),
        SerialPhaseSpec("table_gather", "mte2", per_owner_read, "bytes"),
    ]
    if dequant_ops > 0.0:
        phases.append(SerialPhaseSpec("table_dequant", "vec_alu", dequant_ops / float(owners), "op"))
    phases.append(SerialPhaseSpec("activation_publish", "mte3", per_owner_write, "bytes"))
    return OwnerScheduleSpec(
        owner_kind="aiv",
        tiling_strategy="lookup_row_owners",
        parallel_axis="lookup_rows",
        work_tiles=tiles,
        launched_owners=owners,
        active_owners=owners,
        total_owners=aiv_capacity,
        max_tiles_per_owner=max_tiles,
        residency=OwnerMemoryResidency(
            ub_live_bytes=max(
                PLE_HBM_SECTOR_BYTES,
                max_tiles * max(1, row_bytes) * PLE_RESIDENCY_DOUBLE_BUFFER,
            ),
            ub_capacity_bytes=0,
            owner_mte2_bytes=per_owner_read,
            owner_mte3_bytes=per_owner_write,
            shared_hbm_read_bytes=read_bytes,
            shared_hbm_write_bytes=write_bytes,
            l2_admission_mode="bypass",
        ),
        serial_phases=tuple(phases),
        dispatch_envelope_source="L1_structural_ple",
    )


def _ple_gate_owner_schedule(
    padded_tokens: int,
    hc_mult: int,
    hidden: int,
    act_bytes: int,
    read_bytes: float,
    write_bytes: float,
    *,
    aiv_capacity: int = 0,
) -> OwnerScheduleSpec:
    rows = max(1, int(padded_tokens))
    owners = min(aiv_capacity, rows) if aiv_capacity > 0 else rows
    max_tiles = _ceil_div(rows, owners)
    per_owner_read = read_bytes / float(owners)
    per_owner_write = write_bytes / float(owners)
    elems_per_owner = float(max_tiles * hc_mult * hidden)
    return OwnerScheduleSpec(
        owner_kind="aiv",
        tiling_strategy="token_row_reduction",
        parallel_axis="token_rows",
        work_tiles=rows,
        launched_owners=owners,
        active_owners=owners,
        total_owners=aiv_capacity,
        max_tiles_per_owner=max_tiles,
        residency=OwnerMemoryResidency(
            ub_live_bytes=max(PLE_HBM_SECTOR_BYTES, int(hc_mult * hidden * act_bytes)),
            ub_capacity_bytes=0,
            owner_mte2_bytes=per_owner_read,
            owner_mte3_bytes=per_owner_write,
            shared_hbm_read_bytes=read_bytes,
            shared_hbm_write_bytes=write_bytes,
            l2_admission_mode="reuse",
        ),
        serial_phases=(
            SerialPhaseSpec("kv_query_load", "mte2", per_owner_read, "bytes"),
            SerialPhaseSpec("rms_norm_dot", "vec_alu", elems_per_owner * PLE_GATE_NORM_OPS_PER_ELEM, "op"),
            SerialPhaseSpec("gate_blend", "vec_alu", elems_per_owner * PLE_GATE_BLEND_OPS_PER_ELEM, "op"),
            SerialPhaseSpec("gated_store", "mte3", per_owner_write, "bytes"),
        ),
        dispatch_envelope_source="L1_structural_ple",
    )


def _build_ple_table_lookup_op(
    *, prefix: str, table: Any, query_id: str, stage: int, layer: int,
    requests: tuple[Any, ...], mode: str, padded_tokens: int, embed: int,
    width: int, lookups: int, hits_local: int, lookup_format: str,
    act_dtype: str, read_bytes: float, activation_bytes: float, dequant_ops: float,
    lookup_meta: tuple[tuple[str, Any], ...], has_host: bool, aiv_capacity: int,
) -> OpSpec:
    if has_host:
        return vector_op(
            op_id=f"{prefix}.ple_host_uva_t{table.table_id}", deps=(query_id,),
            stage=stage, layer=layer, requests=requests,
            elements=0, ops_per_element=0.0,
            dtype=act_dtype, family="engram_rh2d", phase=mode,
            recipe=PLE_UVA_RECIPE, shape=(padded_tokens, embed),
            read_bytes_override=0.0,
            write_bytes_override=0.0,
            metadata=(("resource_work", {
                "host_memory": read_bytes,
                "host_vector": float(lookups * PLE_LOOKUP_OPS_PER_ELEM),
                "h2d": read_bytes,
            }),) + lookup_meta + (
                ("table_id", table.table_id),
                ("activation_output_bytes", activation_bytes),
                ("ple_dequant_ops", dequant_ops),
                ("device_table_read_bytes", 0.0),
            ),
        )
    sector_read = _ple_read_bytes(hits_local, width, lookup_format, sector_aligned=True)
    index_pack_ops = float(lookups * PLE_LOOKUP_OPS_PER_ELEM + hits_local * width)
    sched = _ple_lookup_owner_schedule(
        hits_local, sector_read, activation_bytes, index_pack_ops, dequant_ops,
        _ple_row_bytes(width, lookup_format), aiv_capacity=aiv_capacity,
    )
    lookup = vector_op(
        op_id=f"{prefix}.ple_device_lookup_t{table.table_id}", deps=(query_id,),
        stage=stage, layer=layer, requests=requests,
        elements=max(1, padded_tokens * embed),
        ops_per_element=PLE_LOOKUP_OPS_PER_ELEM, dtype=act_dtype,
        family="engram_gather", phase=mode, recipe="ple_device_lookup",
        shape=(padded_tokens, embed),
        read_bytes_override=sector_read,
        write_bytes_override=activation_bytes,
        vector_ops_override=index_pack_ops + dequant_ops,
        metadata=lookup_meta + (
            ("table_id", table.table_id),
            ("activation_output_bytes", activation_bytes),
            ("logical_table_read_bytes", read_bytes),
            ("sector_aligned_read_bytes", sector_read),
            ("ple_dequant_ops", dequant_ops),
            ("quantized_lookup_charged", lookup_format in PLE_QUANTIZED_FORMATS),
            ("dequantize_on_npu", lookup_format in PLE_QUANTIZED_FORMATS),
            ("invocation_class", "vector"),
            ("vector_submode", "outer_axis_reduction"),
            ("launched_blocks", sched.launched_owners),
            ("active_aivs", sched.active_owners),
        ),
    )
    return replace(lookup, owner_schedule=sched)


def lower_ple(
    *, prefix: str, plan: Any, stage: int, deps: str, model: ModelSpec,
    parallel: ParallelSpec, quant: Any, mode: str, padded_tokens: int,
    views: Any, runtime: Any,
) -> tuple[list[OpSpec], str]:
    """Emit pure L1 PLE ``OpSpec``s (hardware-agnostic; zero L1->L3 imports)."""
    engram = model.engram
    if not is_ple_engram(engram):
        raise ModelingError(f"catalog_ple_missing:layer{plan.index}")
    engram.validate()
    requests = tuple(view.request_id for view in views)
    h = int(model.hidden_size)
    hc_mult = int(plan.hc_mult or 1)
    columns = int(engram.columns_per_table)
    width = int(engram.width)
    embed = columns * width
    act_dtype = str(getattr(model, "activation_dtype", "") or "").strip()
    act_bytes = exact_element_bytes(1, act_dtype)
    lookups = max(1, int(padded_tokens) * int(engram.lookups_per_token_per_table))
    lookup_format = ple_lookup_format(model, runtime)
    hits_local = _ple_owned_hits(lookups, parallel.tp)
    read_bytes = _ple_read_bytes(hits_local, width, lookup_format)
    aiv_capacity = _runtime_aiv_capacity(runtime)
    lookup_meta = (
        ("lookups", lookups), ("row_width", width),
        ("engram_width", width), ("columns_per_table", columns),
        ("padded_tokens", padded_tokens),
        ("tp", parallel.tp), ("etp", parallel.etp),
        ("lookup_format", lookup_format),
        ("ple_lookup_format", lookup_format),
    )
    conv_k = int(dict(engram.source).get("ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL))
    state_bytes = ple_conv_state_bytes(model, runtime) * max(1, len(views))
    conv_input = float(padded_tokens * hc_mult * h * act_bytes)
    gated_output = float(padded_tokens * hc_mult * h * act_bytes)
    outer_residual = float(padded_tokens * hc_mult * h * act_bytes)
    activation_bytes = float(int(padded_tokens) * columns * width * act_bytes)
    dequant_ops = float(hits_local * width) if lookup_format in PLE_QUANTIZED_FORMATS else 0.0
    has_host = bool(getattr(runtime, "engram_cpu_offload", False))
    ops: list[OpSpec] = []
    last = ""
    for table in engram.tables:
        if table.layer_id != plan.index:
            continue
        query = vector_op(
            op_id=f"{prefix}.ple_ngram_ids_t{table.table_id}", deps=(deps,),
            stage=stage, layer=plan.index, requests=requests,
            elements=lookups, ops_per_element=PLE_LOOKUP_OPS_PER_ELEM,
            dtype="fp32", family="engram_query", phase=mode, recipe="ple_ngram_ids",
            shape=(padded_tokens, engram.lookups_per_token_per_table),
            metadata=(
                ("table_id", table.table_id), ("lookups", lookups),
                ("columns_per_table", columns), ("row_width", width),
            ),
        )
        ops.append(query)
        lookup = _build_ple_table_lookup_op(
            prefix=prefix, table=table, query_id=query.op_id, stage=stage,
            layer=plan.index, requests=requests, mode=mode,
            padded_tokens=padded_tokens, embed=embed, width=width,
            lookups=lookups, hits_local=hits_local, lookup_format=lookup_format,
            act_dtype=act_dtype, read_bytes=read_bytes, activation_bytes=activation_bytes,
            dequant_ops=dequant_ops, lookup_meta=lookup_meta, has_host=has_host,
            aiv_capacity=aiv_capacity,
        )
        ops.append(lookup)
        gathered = lookup.op_id
        if parallel.tp > 1:
            reduce = comm_op(
                op_id=f"{prefix}.ple_allreduce_t{table.table_id}", deps=(gathered,),
                stage=stage, layer=plan.index, requests=requests,
                family="tp_allreduce", phase=mode, recipe="ple_tp_allreduce",
                link_bytes=float(
                    PLE_ALLREDUCE_TRAFFIC_FACTOR * padded_tokens * embed * act_bytes
                    * (parallel.tp - 1) / parallel.tp
                ),
                shape=(padded_tokens, embed),
            )
            ops.append(reduce)
            gathered = reduce.op_id
        kv = gemm_op(
            op_id=f"{prefix}.ple_kv_proj_t{table.table_id}", deps=(gathered,),
            stage=stage, layer=plan.index, requests=requests, m=padded_tokens,
            n=(hc_mult + 1) * h, k=embed, quant=quant, category="ple_proj",
            phase=mode, family="gemm", recipe="ple_kv_proj",
            metadata=(("table_id", table.table_id), ("replicated_projection", True)),
        )
        gate_read = float((PLE_GATE_READ_HC_FACTOR * hc_mult + 1) * h * padded_tokens * act_bytes)
        gate_write = float(PLE_GATE_WRITE_HC_FACTOR * hc_mult * h * padded_tokens * act_bytes)
        gate_sched = _ple_gate_owner_schedule(
            padded_tokens, hc_mult, h, act_bytes, gate_read, gate_write,
            aiv_capacity=aiv_capacity,
        )
        gate = vector_op(
            op_id=f"{prefix}.ple_gate_t{table.table_id}", deps=(kv.op_id,),
            stage=stage, layer=plan.index, requests=requests,
            elements=max(1, padded_tokens * hc_mult * h),
            ops_per_element=PLE_GATE_OPS_PER_ELEM, dtype=act_dtype,
            family="engram_apply", phase=mode, recipe="ple_gate",
            shape=(padded_tokens, PLE_GATE_WRITE_HC_FACTOR * hc_mult * h),
            read_bytes_override=gate_read,
            write_bytes_override=gate_write,
            metadata=(
                ("table_id", table.table_id),
                ("invocation_class", "vector"),
                ("vector_submode", "outer_axis_reduction"),
                ("launched_blocks", gate_sched.launched_owners),
                ("active_aivs", gate_sched.active_owners),
            ),
        )
        gate = replace(gate, owner_schedule=gate_sched)
        conv = vector_op(
            op_id=f"{prefix}.ple_short_conv_t{table.table_id}",
            deps=(gate.op_id, deps), stage=stage, layer=plan.index,
            requests=requests, elements=max(1, padded_tokens * hc_mult * h),
            ops_per_element=float(PLE_CONV_OPS_FACTOR * conv_k + PLE_CONV_BIAS_OPS),
            dtype=act_dtype, family="conv1d", phase=mode,
            recipe="ple_short_conv", shape=(padded_tokens, hc_mult * h),
            read_bytes_override=conv_input + outer_residual + state_bytes,
            write_bytes_override=gated_output + state_bytes,
        )
        ops.extend([kv, gate, conv])
        last = conv.op_id
        if parallel.pcp > 1 and conv_k > 1 and any(v.phase == "prefill" for v in views):
            prefill_reqs = sum(1 for v in views if v.phase == "prefill")
            pcp_conv_bytes = float(
                prefill_reqs * (conv_k - 1) * hc_mult * h * act_bytes
                * (2 * parallel.pcp - 1) / parallel.pcp
            )
            pcp_conv = comm_op(
                op_id=f"{prefix}.pcp_ple_conv_state_t{table.table_id}",
                deps=(last,), stage=stage, layer=plan.index, requests=requests,
                family="allgather", phase=mode, recipe="pcp_ple_conv_state_step",
                link_bytes=pcp_conv_bytes,
                shape=(prefill_reqs * (conv_k - 1), hc_mult * h),
                metadata=(("group_axis", "pcp"),),
            )
            ops.append(pcp_conv)
            last = pcp_conv.op_id
    if not last:
        raise ModelingError(f"catalog_engram_table_not_at_layer:{plan.index}")
    return ops, last


def lowrank_hc_layer_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    hc_dim = plan.hc_mult * model.hidden_size
    terms: list[WeightTerm] = []
    for part in ("attn", "ffn"):
        terms.extend([
            _t(f"hc_{part}_norm", hc_dim, 1, "norm"),
            _t(f"hc_{part}_down_inject", plan.hc_lowrank + plan.hc_mult, hc_dim, "linear_param_bf16"),
            _t(f"hc_{part}_up", hc_dim, plan.hc_lowrank, "linear_param_bf16"),
        ])
    return terms


def qsa_indexer_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    if not getattr(plan, "indexer", False) or plan.index_heads <= 0 or plan.index_dim <= 0:
        return []
    return [
        _t("index_qk_proj", (plan.index_heads + 1) * plan.index_dim, model.hidden_size, "indexer"),
        _t("indexer_q_norm", plan.index_dim, 1, "norm"),
        _t("indexer_k_norm", plan.index_dim, 1, "norm"),
    ]


def ple_conv_state_bytes(
    model: Any, runtime: Any = None, *, query_tokens: int | None = None,
) -> float:
    """Structural per-request PLE conv state bytes using ``model.activation_dtype``."""
    engram = getattr(model, "engram", None)
    if not is_ple_engram(engram):
        return 0.0
    conv_k = int(dict(engram.source).get("ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL))
    conv_state_len = (conv_k - 1) * int(engram.max_ngram)
    if query_tokens is not None:
        num_spec = max(0, int(query_tokens) - 1)
    elif runtime is not None:
        num_spec = max(0, resolve_speculative_query_tokens(runtime, phase=PHASE_DECODE) - 1)
    else:
        num_spec = 0
    hc = max((int(p.hc_mult) for p in model.layer_plans), default=1) or 1
    elem_bytes = exact_element_bytes(1, str(getattr(model, "activation_dtype", "") or ""))
    return float((conv_state_len + num_spec) * hc * int(model.hidden_size) * elem_bytes)


def qwen4_exp_pool_breakdown(
    model: Any, parallel: ParallelSpec, sequence_tokens: int, *,
    runtime: Any = None, query_tokens: int | None = None,
) -> dict[str, Any] | None:
    """Delegate shared-slab pool breakdown uniformly to canonical ``evaluate_hybrid_pool_receipt``."""
    if not is_ple_engram(getattr(model, "engram", None)):
        return None
    if int(getattr(parallel, "cp", 1) or 1) > 1 or int(getattr(parallel, "pcp", 1) or 1) > 1:
        return None
    receipt = evaluate_hybrid_pool_receipt(
        model, parallel, sequence_tokens,
        num_requests=1, runtime=runtime, query_tokens=query_tokens, phase=PHASE_DECODE,
    )
    contract = receipt.contract
    if contract.pool_regime != REGIME_SHARED_SLAB_PRIVATE_AUX:
        return None
    total_pool = float(sum(receipt.total_pool_bytes_by_stage))
    total_reserve = float(sum(receipt.stride_reserve_by_stage))
    total_replay = float(sum(receipt.outside_replay_bytes_by_stage))
    return {
        "slot_count": int(sum(contract.local_slot_counts)),
        "manager_block_size": int(contract.normalized_block_size),
        "state_stride": int(contract.main_page_bytes),
        "pool_bytes_per_block": int(sum(receipt.pool_unit_bytes_by_stage)),
        "qsa_blocks": int(receipt.qsa_blocks_per_request),
        "gdn_state_groups": int(sum(contract.mamba_group_counts)),
        "raw_blocks": int(receipt.raw_blocks_per_request),
        "ple_blocks": int(receipt.ple_blocks_per_request),
        "total_pool_blocks": int(receipt.blocks_per_request),
        "private_ple_bytes": float(receipt.per_request_ple_page_bytes if sequence_tokens > 0 else 0),
        "total_pool_bytes": total_pool,
        "stride_reserve_bytes": total_reserve,
        "outside_replay_bytes": total_replay,
    }


__all__ = [
    "build_ple_spec",
    "is_qwen4_exp_ple",
    "lower_ple",
    "ple_lookup_format",
    "ple_table_terms",
    "ple_gate_terms_by_layer",
    "lowrank_hc_layer_terms",
    "lowrank_hc_head_terms",
    "lowrank_hc_draft_terms",
    "lowrank_hc_draft_head_terms",
    "qsa_indexer_terms",
    "ple_conv_state_bytes",
    "qwen4_exp_pool_breakdown",
]
