"""Qwen4Exp PLE, low-rank gated HC and QSA shared-pool geometry.

[Role]
  Normalize the Qwen4Exp n-gram PLE tables (1-based ``ple_layer_ids``, prime
  head layouts, conv kernel), the low-rank gated hyper-connection matrices,
  and the weight-free QSA indexer into existing weight/op/cache records.

[Invariants]
  * ``ple_layer_ids`` are one-based; a table's ``layer_id`` is ``ple_id - 1``.
  * Table rows are the exact sum of per-head primes rounded to the declared
    divisor; table width is ``ple_embed_dim / ((ngram-1)*heads)``.
  * PLE projections are TP-replicated; table rows shard over TP/ETP.
  * ``hc_lowrank > 0`` selects GatedResidual terms and suppresses Sinkhorn.
  * The QSA shared pool is 12 slot-backed pages, 3 GDN state groups and one
    raw circular page; PLE conv state is private, per table and TP-replicated.

[Timeouts & Locks]
  Pure integer/float arithmetic bounded by layer and head counts; no I/O.

[Side-Effect Policy]
  No filesystem, network, device or global mutation.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping

from mindie_sim.catalog.cache_pools import gbsa_cache_layout
from mindie_sim.catalog.config import _int
from mindie_sim.catalog.ops import comm_op, gemm_op, vector_op
from mindie_sim.catalog.speculative_state import (
    PHASE_DECODE, resolve_speculative_query_tokens, runtime_recover_ssm,
    speculative_state_plan,
)
from mindie_sim.catalog.types import (
    DraftSpec, EngramSpec, EngramTableSpec, merge_source,
)
from mindie_sim.catalog.weight_terms import WeightTerm, _ceil_div, _t
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import (
    KernelBinding,
    OpSpec,
    OwnerMemoryResidency,
    OwnerScheduleSpec,
    SerialPhaseSpec,
)
from mindie_sim.schema.parallel import ParallelSpec

ZERO = 0
ONE = 1
TWO = 2
ONE_FLOAT = 1.0
ZERO_FLOAT = 0.0
BF16_BYTES = 2
FP32_BYTES = 4
MX_GROUP = 32
INT8_SCALE_BYTES = 4
MXFP8_SCALE_BYTES = 1
PLE_HBM_SECTOR_BYTES = 128
PLE_OWNER_CAPACITY_DEFAULT = 64
PLE_UVA_RECIPE = "ple_host_uva_lookup"
PLE_QUANTIZED_FORMATS = frozenset({"int8", "mxfp8"})
PLE_FLOAT_FORMATS = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})
KERNEL_BLOCK_TOKENS = 128
DEFAULT_PLE_CONV_KERNEL = 4
DEFAULT_PLE_DIVISOR = 128
PLE_GATE_OPS_PER_ELEM = 6.0
PLE_LOOKUP_OPS_PER_ELEM = 2.0
DRAFT_HC_HEAD_TERMS = 3
QWEN4_EXP_PLE_SOURCE: dict[str, str] = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/models/qwen4_exp/ple.py",
    "source_sha256": "1327d2f777c8d262696865f0d5ffad1b061826e6f54b8d711a3af14e30f38033",
}
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_MILLER_RABIN_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _round_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - ONE) // int(alignment)) * int(alignment)


def _is_prime_64(value: int) -> bool:
    if value < TWO:
        return False
    for prime in _SMALL_PRIMES:
        if value % prime == ZERO:
            return value == prime
    exponent = value - ONE
    shifts = ZERO
    while exponent % TWO == ZERO:
        exponent //= TWO
        shifts += ONE
    for base in _MILLER_RABIN_BASES:
        if base % value == ZERO:
            continue
        witness = pow(base, exponent, value)
        if witness in (ONE, value - ONE):
            continue
        for _ in range(shifts - ONE):
            witness = pow(witness, TWO, value)
            if witness == value - ONE:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + ONE
        if candidate <= TWO:
            prime = TWO
            continue
        if candidate % TWO == ZERO:
            candidate += ONE
        while not _is_prime_64(candidate):
            candidate += TWO
        prime = candidate
    return prime


def build_ple_spec(c: Mapping[str, Any]) -> EngramSpec:
    ple_layers = [int(v) for v in c.get("ple_layer_ids", ()) or ()]
    max_ngram = _int(c, "ngram_size")
    heads = _int(c, "heads_per_ngram")
    embed = _int(c, "ple_embed_dim")
    base_rows = _int(c, "ngram_vocab_size_base")
    ngram_heads = (max_ngram - ONE) * heads
    if max_ngram < TWO or heads <= ZERO or embed <= ZERO or ngram_heads <= ZERO:
        raise ModelingError("model_config_ple_geometry_missing")
    if embed % ngram_heads:
        raise ModelingError("model_config_ple_geometry_missing")
    width = embed // ngram_heads
    divisor = _int(c, "make_ngram_vocab_size_divisible_by", DEFAULT_PLE_DIVISOR)
    conv_k = _int(c, "ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL)
    tables = []
    for index, ple_id in enumerate(ple_layers):
        if ple_id <= ZERO:
            raise ModelingError("model_config_ple_layer_id_invalid")
        total = ZERO
        for local_head in range(ngram_heads):
            global_head = index * ngram_heads + local_head
            total += _nth_prime_after(base_rows - ONE, global_head + ONE)
        tables.append(EngramTableSpec(
            layer_id=ple_id - ONE, table_id=index,
            rows=_round_up(total, divisor), width=width))
    return EngramSpec(
        tables=tuple(tables), max_ngram=max_ngram, heads=heads, width=width,
        pad_token_id=_int(c, "engram_pad_token_id", ZERO),
        host_response_bytes=width * BF16_BYTES,
        source=merge_source(
            QWEN4_EXP_PLE_SOURCE,
            {
                "kind": "qwen4_exp_ple", "ple_layers": str(len(tables)),
                "ple_conv_kernel_size": str(conv_k), "ple_embed_dim": str(embed),
            },
        ),
    )


def is_qwen4_exp_ple(engram: Any) -> bool:
    return engram is not None and dict(
        getattr(engram, "source", ()) or ()).get("kind") == "qwen4_exp_ple"


def ple_table_terms(model: ModelSpec) -> list[WeightTerm]:
    engram = model.engram
    if engram is None:
        return []
    return [WeightTerm(f"ple_table_{table.table_id}", table.rows, table.width,
                       "engram_table", "out_tp", ONE) for table in engram.tables]


def ple_gate_terms_by_layer(model: ModelSpec) -> dict[int, list[WeightTerm]]:
    engram = model.engram
    if engram is None:
        return {}
    hc = max((p.hc_mult for p in model.layer_plans), default=ONE)
    h = model.hidden_size
    hc_h = hc * h
    embed = engram.columns_per_table * engram.width
    conv_k = int(dict(engram.source).get("ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL))
    by_layer: dict[int, list[WeightTerm]] = {}
    for table in engram.tables:
        by_layer.setdefault(table.layer_id, []).extend([
            _t("ple_kv_proj", hc_h + h, embed, "ple_proj", count=ONE),
            _t("ple_norm_key", hc_h, ONE, "norm", count=ONE),
            _t("ple_norm_query", hc_h, ONE, "norm", count=ONE),
            _t("ple_norm_conv", hc_h, ONE, "norm", count=ONE),
            _t("ple_conv1d", hc_h, conv_k, "conv_weight_bf16", count=ONE),
        ])
    return by_layer


def _table_marker_format(quant: Any) -> str | None:
    """Resolve PLE table precision from checkpoint table markers, or None if undeclared."""
    if quant is None:
        return None
    overrides = dict(getattr(quant, "category_overrides", ()) or ())
    for role in ("ple_table", "engram_table"):
        if role in overrides:
            fields = dict(overrides[role])
            wf = str(fields.get("weight_format", "") or "").lower().replace("-", "_")
            if wf in PLE_FLOAT_FORMATS:
                return "bf16"
            if wf in {"mxfp8", "mx_fp8", "fp8_e4m3", "fp8"}:
                sf = str(fields.get("scale_format", "") or getattr(quant, "scale_format", "") or "").lower()
                return "mxfp8" if sf in {"fp8_e8m0", "ue8m0"} or "mx" in wf else "int8"
            if wf in {"int8", "w8a8"}:
                return "int8"
    exclusions = set(getattr(quant, "exclusions", ()) or ())
    if exclusions & {"ple_table", "engram_table", "ple"}:
        return "bf16"
    table_fmt = str(getattr(quant, "engram_table_format", "") or "").lower().replace("-", "_")
    if table_fmt in PLE_FLOAT_FORMATS:
        return "bf16"
    if table_fmt in {"mxfp8", "mx_fp8", "fp8_e4m3", "fp8"}:
        sf = str(getattr(quant, "scale_format", "") or "").lower()
        return "mxfp8" if sf in {"fp8_e8m0", "ue8m0"} else "int8"
    if table_fmt == "int8":
        return "int8"
    return None


def ple_lookup_format(model: Any, runtime: Any = None) -> str:
    """Resolve the declared PLE table representation from table checkpoint markers.

    A FLOAT table has two bytes per element and zero dequant work, including on
    host offload.  Table markers take precedence over model-wide MX names.
    """
    quant = getattr(model, "quant", None)
    marker_fmt = _table_marker_format(quant)
    if marker_fmt == "bf16":
        return "bf16"
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    declared = str(metadata.get("ple_lookup_format", "") or "")
    active = (declared in PLE_QUANTIZED_FORMATS
              or "ple_quantized_lookup" in tuple(
                  getattr(runtime, "lowering_passes", ()) or ()))
    if not active:
        return marker_fmt if marker_fmt in PLE_QUANTIZED_FORMATS else "bf16"
    if marker_fmt in PLE_QUANTIZED_FORMATS:
        return marker_fmt
    if declared in PLE_QUANTIZED_FORMATS:
        return declared
    return "int8"


def _ple_owned_hits(lookups: int, tp: int) -> int:
    """Actual table rows owned by one TP rank; zero only with no lookups.

    A short lookup row (``lookups < tp``) still lands on the front ranks, so
    the analytical owner reports the ceiling instead of dropping every hit to
    zero.
    """
    lookups, tp = int(lookups), max(1, int(tp))
    if lookups <= 0:
        return 0
    quotient, remainder = divmod(lookups, tp)
    return quotient + (1 if remainder > 0 else 0)


def _ple_scale_bytes(lookup_format: str) -> int:
    return MXFP8_SCALE_BYTES if lookup_format == "mxfp8" else INT8_SCALE_BYTES


def _ple_row_bytes(width: int, lookup_format: str) -> int:
    width = int(width)
    if lookup_format == "bf16":
        return width * BF16_BYTES
    scales = _ceil_div(width, MX_GROUP) * _ple_scale_bytes(lookup_format)
    return width + scales


def _ple_read_bytes(hits_local: int, width: int, lookup_format: str, *, sector_aligned: bool = False) -> float:
    """Owner read for one table: BF16 rows or quantized codes plus group scales."""
    row_bytes = _ple_row_bytes(width, lookup_format)
    if sector_aligned and row_bytes > ZERO:
        row_bytes = _round_up(row_bytes, PLE_HBM_SECTOR_BYTES)
    return float(int(hits_local) * row_bytes)


def _ple_lookup_owner_schedule(
    hits_local: int,
    read_bytes: float,
    write_bytes: float,
    index_ops: float,
    dequant_ops: float,
    row_bytes: int,
) -> OwnerScheduleSpec:
    owners = max(ONE, min(PLE_OWNER_CAPACITY_DEFAULT, max(ONE, int(hits_local))))
    max_tiles = _ceil_div(max(ONE, int(hits_local)), owners)
    per_owner_read = read_bytes / float(owners)
    per_owner_write = write_bytes / float(owners)
    phases = [
        SerialPhaseSpec("index_address", "vec_alu", index_ops / float(owners), "op"),
        SerialPhaseSpec("table_gather", "mte2", per_owner_read, "bytes"),
    ]
    if dequant_ops > ZERO_FLOAT:
        phases.append(SerialPhaseSpec("table_dequant", "vec_alu", dequant_ops / float(owners), "op"))
    phases.append(SerialPhaseSpec("activation_publish", "mte3", per_owner_write, "bytes"))
    return OwnerScheduleSpec(
        owner_kind="aiv",
        tiling_strategy="lookup_row_owners",
        parallel_axis="lookup_rows",
        work_tiles=max(ONE, int(hits_local)),
        launched_owners=owners,
        active_owners=owners,
        total_owners=PLE_OWNER_CAPACITY_DEFAULT,
        max_tiles_per_owner=max_tiles,
        residency=OwnerMemoryResidency(
            ub_live_bytes=max(PLE_HBM_SECTOR_BYTES, max_tiles * max(ONE, row_bytes) * TWO),
            ub_capacity_bytes=ZERO,
            owner_mte2_bytes=per_owner_read,
            owner_mte3_bytes=per_owner_write,
            shared_hbm_read_bytes=read_bytes,
            shared_hbm_write_bytes=write_bytes,
            l2_admission_mode="bypass",
        ),
        serial_phases=tuple(phases),
        dispatch_envelope_source="P15_dispatch",
    )


def _ple_gate_owner_schedule(
    padded_tokens: int,
    hc_mult: int,
    hidden: int,
    read_bytes: float,
    write_bytes: float,
) -> OwnerScheduleSpec:
    rows = max(ONE, int(padded_tokens))
    owners = max(ONE, min(PLE_OWNER_CAPACITY_DEFAULT, rows))
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
        total_owners=PLE_OWNER_CAPACITY_DEFAULT,
        max_tiles_per_owner=max_tiles,
        residency=OwnerMemoryResidency(
            ub_live_bytes=max(PLE_HBM_SECTOR_BYTES, int(hc_mult * hidden * BF16_BYTES)),
            ub_capacity_bytes=ZERO,
            owner_mte2_bytes=per_owner_read,
            owner_mte3_bytes=per_owner_write,
            shared_hbm_read_bytes=read_bytes,
            shared_hbm_write_bytes=write_bytes,
            l2_admission_mode="reuse",
        ),
        serial_phases=(
            SerialPhaseSpec("kv_query_load", "mte2", per_owner_read, "bytes"),
            SerialPhaseSpec("key_query_rmsnorm", "vec_reduce", elems_per_owner * TWO, "op"),
            SerialPhaseSpec("dot_sigmoid_gate", "vec_sfu", elems_per_owner * TWO, "op"),
            SerialPhaseSpec("gated_combine_store", "mte3", per_owner_write, "bytes"),
        ),
        dispatch_envelope_source="P15_dispatch",
    )


def _ple_activation_bytes(padded_tokens: int, columns: int, width: int,
                          act_bytes: int) -> float:
    """Publish all ``columns_per_table * width`` activation lanes exactly once."""
    return float(int(padded_tokens) * int(columns) * int(width) * int(act_bytes))


def _build_ple_table_lookup_op(
    *, prefix: str, table: Any, query_id: str, stage: int, layer: int,
    requests: tuple[Any, ...], mode: str, padded_tokens: int, embed: int,
    width: int, lookups: int, hits_local: int, lookup_format: str,
    read_bytes: float, activation_bytes: float, dequant_ops: float,
    lookup_meta: tuple[tuple[str, Any], ...], has_host: bool,
) -> OpSpec:
    if has_host:
        # The baseline UVA caller keeps the host lookup, one host-vector
        # decode pass and the H2D payload; it never charges device work.
        # ``ple_uva_lookup`` (when active) annotates ``uva_dequant_device``
        # and the registered scoped recipe moves the decode onto the NPU.
        return vector_op(
            op_id=f"{prefix}.ple_host_uva_t{table.table_id}", deps=(query_id,),
            stage=stage, layer=layer, requests=requests,
            elements=ZERO, ops_per_element=ZERO_FLOAT,
            dtype="bf16", family="engram_rh2d", phase=mode,
            recipe=PLE_UVA_RECIPE, shape=(padded_tokens, embed),
            read_bytes_override=ZERO_FLOAT,
            write_bytes_override=ZERO_FLOAT,
            metadata=((("resource_work", {
                "host_memory": read_bytes,
                "host_vector": float(lookups * PLE_LOOKUP_OPS_PER_ELEM),
                "h2d": read_bytes,
            }),) + lookup_meta + (
                ("table_id", table.table_id),
                ("activation_output_bytes", activation_bytes),
                ("ple_dequant_ops", dequant_ops),
                ("device_table_read_bytes", ZERO_FLOAT),
            )))
    sector_read = _ple_read_bytes(hits_local, width, lookup_format, sector_aligned=True)
    index_pack_ops = float(lookups * PLE_LOOKUP_OPS_PER_ELEM + hits_local * width)
    lookup_owners = max(ONE, min(PLE_OWNER_CAPACITY_DEFAULT, max(ONE, int(hits_local))))
    lookup = vector_op(
        op_id=f"{prefix}.ple_device_lookup_t{table.table_id}", deps=(query_id,),
        stage=stage, layer=layer, requests=requests,
        elements=max(ONE, padded_tokens * embed),
        ops_per_element=PLE_LOOKUP_OPS_PER_ELEM, dtype="bf16",
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
            ("launched_blocks", lookup_owners),
            ("active_aivs", lookup_owners),
        ))
    return replace(
        lookup,
        owner_schedule=_ple_lookup_owner_schedule(
            hits_local, sector_read, activation_bytes,
            index_pack_ops, dequant_ops, _ple_row_bytes(width, lookup_format),
        ),
        kernel_binding=KernelBinding(
            kernel_id="PLEDeviceLookup",
            backend="Custom-Ops",
            arch_family="A5",
            submodule="ple_lookup",
            macro_module="engram",
            readiness="shipping",
            recipe_id="ple_device_lookup",
        ),
    )


def lower_ple(
    *, prefix: str, plan: Any, stage: int, deps: str, model: ModelSpec,
    parallel: ParallelSpec, quant: Any, mode: str, padded_tokens: int,
    views: Any, runtime: Any,
) -> tuple[list[OpSpec], str]:
    engram = model.engram
    if not is_qwen4_exp_ple(engram):
        raise ModelingError(f"catalog_qwen4_exp_ple_missing:layer{plan.index}")
    engram.validate()
    requests = tuple(view.request_id for view in views)
    h = model.hidden_size
    hc_mult = plan.hc_mult or ONE
    columns = engram.columns_per_table
    width = engram.width
    embed = columns * width
    act_bytes = dtype_bytes(model.activation_dtype)
    lookups = max(ONE, padded_tokens * engram.lookups_per_token_per_table)
    lookup_format = ple_lookup_format(model, runtime)
    hits_local = _ple_owned_hits(lookups, parallel.tp)
    read_bytes = _ple_read_bytes(hits_local, width, lookup_format)
    lookup_meta = (
        ("lookups", lookups), ("row_width", width),
        ("engram_width", width), ("columns_per_table", columns),
        ("padded_tokens", padded_tokens),
        ("tp", parallel.tp), ("etp", parallel.etp),
        ("lookup_format", lookup_format),
        ("ple_lookup_format", lookup_format),
    )
    conv_k = int(dict(engram.source).get("ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL))
    state_bytes = ple_conv_state_bytes(model, runtime) * max(ONE, len(views))
    conv_input = float(padded_tokens * hc_mult * h * act_bytes)
    gated_output = float(padded_tokens * hc_mult * h * act_bytes)
    outer_residual = float(padded_tokens * hc_mult * h * act_bytes)
    activation_bytes = _ple_activation_bytes(
        padded_tokens, columns, width, act_bytes)
    dequant_ops = (float(hits_local * width)
                   if lookup_format in PLE_QUANTIZED_FORMATS else ZERO_FLOAT)
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
            metadata=(("table_id", table.table_id), ("lookups", lookups),
                      ("columns_per_table", columns), ("row_width", width)))
        ops.append(query)
        lookup = _build_ple_table_lookup_op(
            prefix=prefix, table=table, query_id=query.op_id, stage=stage,
            layer=plan.index, requests=requests, mode=mode,
            padded_tokens=padded_tokens, embed=embed, width=width,
            lookups=lookups, hits_local=hits_local, lookup_format=lookup_format,
            read_bytes=read_bytes, activation_bytes=activation_bytes,
            dequant_ops=dequant_ops, lookup_meta=lookup_meta, has_host=has_host,
        )
        ops.append(lookup)
        gathered = lookup.op_id
        if parallel.tp > ONE:
            reduce = comm_op(
                op_id=f"{prefix}.ple_allreduce_t{table.table_id}", deps=(gathered,),
                stage=stage, layer=plan.index, requests=requests,
                family="tp_allreduce", phase=mode, recipe="ple_tp_allreduce",
                link_bytes=float(TWO * padded_tokens * embed * BF16_BYTES
                                 * (parallel.tp - ONE) / parallel.tp),
                shape=(padded_tokens, embed))
            ops.append(reduce)
            gathered = reduce.op_id
        kv = gemm_op(
            op_id=f"{prefix}.ple_kv_proj_t{table.table_id}", deps=(gathered,),
            stage=stage, layer=plan.index, requests=requests, m=padded_tokens,
            n=(hc_mult + ONE) * h, k=embed, quant=quant, category="ple_proj",
            phase=mode, family="gemm", recipe="ple_kv_proj",
            metadata=(("table_id", table.table_id),
                      ("replicated_projection", True)))
        gate_read = float((TWO * hc_mult + ONE) * h * padded_tokens * act_bytes)
        gate_write = float(TWO * hc_mult * h * padded_tokens * act_bytes)
        gate_owners = max(ONE, min(PLE_OWNER_CAPACITY_DEFAULT, max(ONE, int(padded_tokens))))
        gate = vector_op(
            op_id=f"{prefix}.ple_gate_t{table.table_id}", deps=(kv.op_id,),
            stage=stage, layer=plan.index, requests=requests,
            elements=max(ONE, padded_tokens * hc_mult * h),
            ops_per_element=PLE_GATE_OPS_PER_ELEM, dtype=model.activation_dtype,
            family="engram_apply", phase=mode, recipe="ple_gate",
            shape=(padded_tokens, TWO * hc_mult * h),
            read_bytes_override=gate_read,
            write_bytes_override=gate_write,
            metadata=(
                ("table_id", table.table_id),
                ("invocation_class", "vector"),
                ("vector_submode", "outer_axis_reduction"),
                ("launched_blocks", gate_owners),
                ("active_aivs", gate_owners),
            ))
        gate = replace(
            gate,
            owner_schedule=_ple_gate_owner_schedule(
                padded_tokens, hc_mult, h, gate_read, gate_write,
            ),
            kernel_binding=KernelBinding(
                kernel_id="PLEGateResidual",
                backend="Triton",
                arch_family="A5",
                submodule="ple_gate",
                macro_module="engram",
                readiness="shipping",
                recipe_id="ple_gate",
            ),
        )
        conv = vector_op(
            op_id=f"{prefix}.ple_short_conv_t{table.table_id}",
            deps=(gate.op_id, deps), stage=stage, layer=plan.index,
            requests=requests, elements=max(ONE, padded_tokens * hc_mult * h),
            ops_per_element=float(TWO * conv_k + TWO),
            dtype=model.activation_dtype, family="conv1d", phase=mode,
            recipe="ple_short_conv", shape=(padded_tokens, hc_mult * h),
            read_bytes_override=conv_input + outer_residual + state_bytes,
            write_bytes_override=gated_output + state_bytes)
        ops.extend([kv, gate, conv])
        last = conv.op_id
        if parallel.pcp > 1 and conv_k > 1 and any(v.phase == "prefill" for v in views):
            prefill_reqs = sum(1 for v in views if v.phase == "prefill")
            pcp_conv_bytes = float(
                prefill_reqs * (conv_k - 1) * hc_mult * h * act_bytes
                * (2 * parallel.pcp - 1) / parallel.pcp)
            pcp_conv = comm_op(
                op_id=f"{prefix}.pcp_ple_conv_state_t{table.table_id}",
                deps=(last,), stage=stage, layer=plan.index, requests=requests,
                family="allgather", phase=mode, recipe="pcp_ple_conv_state_step",
                link_bytes=pcp_conv_bytes,
                shape=(prefill_reqs * (conv_k - 1), hc_mult * h),
                metadata=(("group_axis", "pcp"),))
            ops.append(pcp_conv)
            last = pcp_conv.op_id
    if not last:
        raise ModelingError(f"catalog_engram_table_not_at_layer:{plan.index}")
    return ops, last


def _lower_ple_uva_npu_dequant(context: Any) -> Any:
    """Three declared phases: host lookup, H2D transfer and NPU dequant publish."""
    from mindie_sim.prediction.program import KernelProgram, Stage

    names = dict(getattr(context, "resource_names", {}) or ())
    work = {str(key): float(value) for key, value in dict(context.work).items()}

    def take(role: str) -> dict[str, float]:
        name = names.get(role)
        if name is None or name not in work:
            return {}
        return {name: work.pop(name)}

    lookup = take("host_memory")
    transfer = take("h2d")
    publish: dict[str, float] = {}
    for role in ("vector", "mte3", "hbm"):
        publish.update(take(role))
    for name, value in tuple(work.items()):
        # Conserve any residual declared resource on the publish phase.
        publish[name] = publish.get(name, 0.0) + value
    return KernelProgram((
        Stage("lookup", lookup),
        Stage("transfer", transfer, after=("lookup",)),
        Stage("dequant_publish", publish, after=("transfer",)),
    ), context.minimum_ms)


def build_ple_uva_npu_dequant(op: OpSpec, geometry: Any) -> Any:
    """Planned recipe for the UVA PLE lookup with NPU-side dequantization.

    The payload crosses H2D; the device table is never read (``mte2_aiv`` is a
    declared zero).  The dequant lane work and the activation output write are
    each paid exactly once and replace the host-vector decode stage.
    """
    from mindie_sim.prediction.mechanism import OperatorPlan, Work

    del geometry
    meta = dict(getattr(op, "metadata", ()) or ())
    declared = dict(meta.get("resource_work", {}) or {})
    payload = float(declared.get("host_memory", 0.0) or 0.0)
    transfer = float(declared.get("h2d", payload) or 0.0)
    publish = float(meta.get("activation_output_bytes", 0.0) or 0.0)
    dequant = float(meta.get("ple_dequant_ops", 0.0) or 0.0)
    return OperatorPlan(
        Work(resources={
            "host_memory": payload,
            "h2d": transfer,
            "vector": dequant,
            "mte3": publish,
            "hbm": publish,
            "mte2_aiv": 0.0,
        }), _lower_ple_uva_npu_dequant,
        metadata={
            "physical_class": "PleUvaNpuDequant",
            "stage_basis": "declared_host_lookup_transfer_npu_dequant_phases",
        })


def build_ple_host_staging(op: OpSpec, geometry: Any) -> Any:
    """Renderer for callers that keep the host-vector decode (pass excluded)."""
    from mindie_sim.prediction.mechanism import OperatorPlan
    from mindie_sim.prediction.physical_recipe_base import work_from_op
    from mindie_sim.prediction.program import HostStaging

    del geometry
    return OperatorPlan(
        work_from_op(op), HostStaging().lower,
        metadata={
            "physical_class": "PleHostStaging",
            "stage_basis": "host_lookup_host_vector_decode_h2d_transfer",
        })


def lowrank_hc_layer_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    hc_dim = plan.hc_mult * model.hidden_size
    terms: list[WeightTerm] = []
    for part in ("attn", "ffn"):
        terms.extend([
            _t(f"hc_{part}_norm", hc_dim, ONE, "norm"),
            _t(f"hc_{part}_down_inject", plan.hc_lowrank + plan.hc_mult, hc_dim, "linear_param_bf16"),
            _t(f"hc_{part}_up", hc_dim, plan.hc_lowrank, "linear_param_bf16"),
        ])
    return terms


def lowrank_hc_head_terms(model: ModelSpec) -> list[WeightTerm]:
    hc = max((p.hc_mult for p in model.layer_plans), default=ZERO)
    lr = max((p.hc_lowrank for p in model.layer_plans), default=ZERO)
    hc_dim = hc * model.hidden_size
    return [_t("hc_head_norm", hc_dim, ONE, "norm"),
            _t("hc_head_down", lr, hc_dim, "linear_param_bf16"),
            _t("hc_head_up", hc_dim, lr, "linear_param_bf16")]


def lowrank_hc_draft_terms(draft: DraftSpec) -> list[WeightTerm]:
    if draft.mhc_mult <= ZERO or draft.mhc_lowrank <= ZERO:
        return []
    hc_dim = draft.mhc_mult * draft.hidden_size
    terms: list[WeightTerm] = []
    for part in ("attn", "ffn"):
        terms.extend([
            _t(f"draft_hc_{part}_norm", hc_dim, ONE, "norm"),
            _t(f"draft_hc_{part}_down_inject",
               draft.mhc_lowrank + draft.mhc_mult, hc_dim, "linear_param_bf16"),
            _t(f"draft_hc_{part}_up", hc_dim, draft.mhc_lowrank, "linear_param_bf16"),
        ])
    return terms


def lowrank_hc_draft_head_terms(draft: DraftSpec) -> list[WeightTerm]:
    if draft.mhc_mult <= ZERO or draft.mhc_lowrank <= ZERO:
        return []
    hc_dim = draft.mhc_mult * draft.hidden_size
    return [
        _t("draft_hc_head_norm", hc_dim, ONE, "norm"),
        _t("draft_hc_head_down", draft.mhc_lowrank, hc_dim, "linear_param_bf16"),
        _t("draft_hc_head_up", hc_dim, draft.mhc_lowrank, "linear_param_bf16"),
    ]


def qsa_indexer_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    if not getattr(plan, "indexer", False) or plan.index_heads <= ZERO or plan.index_dim <= ZERO:
        return []
    return [_t("index_qk_proj", (plan.index_heads + ONE) * plan.index_dim,
               model.hidden_size, "indexer"),
            _t("indexer_q_norm", plan.index_dim, ONE, "norm"),
            _t("indexer_k_norm", plan.index_dim, ONE, "norm")]


def ple_conv_state_bytes(model: Any, runtime: Any = None) -> float:
    if not is_qwen4_exp_ple(getattr(model, "engram", None)):
        return ZERO_FLOAT
    engram = model.engram
    conv_k = int(dict(engram.source).get("ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL))
    conv_state_len = (conv_k - ONE) * engram.max_ngram
    num_spec = (int(getattr(runtime, "mtp_proposal_tokens", ZERO) or ZERO)
                if runtime is not None else ZERO)
    hc = max((p.hc_mult for p in model.layer_plans), default=ONE)
    return float((conv_state_len + num_spec) * hc * model.hidden_size * BF16_BYTES)


def qwen4_exp_pool_breakdown(
    model: Any, parallel: ParallelSpec, sequence_tokens: int, *, runtime: Any = None,
) -> dict[str, Any] | None:
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    if facts.get("model_type") != "qwen4_exp_text" and not is_qwen4_exp_ple(
            getattr(model, "engram", None)):
        return None
    qsa = [p for p in model.layer_plans if p.attention == "gqa" and p.indexer]
    gdn = [p for p in model.layer_plans if p.attention == "gdn"]
    if not qsa or not gdn:
        return None
    tp = max(ONE, parallel.tp)
    seq = int(sequence_tokens)
    slot_count = len(qsa)
    local_kv = max(1, (qsa[0].num_kv_heads + tp - 1) // tp)
    int8_fa3 = gbsa_cache_layout(runtime) == "int8_fa3"
    if int8_fa3:
        # C8 reserves ``h*(D+4)`` for each K/V region: INT8 codes, the dynamic
        # K scale and the reserved static V scale slot.
        kv_token_bytes = int(2 * local_kv * (qsa[0].head_dim + FP32_BYTES))
    else:
        kv_elem = dtype_bytes(str(getattr(getattr(model, "quant", None), "kv_dtype", "")
                                  or getattr(model, "kv_dtype", "bf16")))
        kv_token_bytes = int(2 * local_kv * qsa[0].head_dim * kv_elem)
    # A verify window owns ``q`` recurrent checkpoint matrices (or the replay
    # record set); the pool must reserve the full active multiplicity, not one
    # slot per request.
    state_plans = tuple(
        speculative_state_plan(
            model, plan, parallel,
            query_tokens=resolve_speculative_query_tokens(
                runtime, phase=PHASE_DECODE),
            recover_ssm=runtime_recover_ssm(runtime),
            capacity_padded=True,
            state_contract=getattr(plan, "state_contract", None),
            runtime=runtime)
        for plan in gdn)
    max_ssm_bytes = max(
        int(math.ceil(state.recurrent_bytes + state.replay_bytes))
        for state in state_plans)
    conv_bytes = max(
        int(math.ceil(state.conv_bytes)) for state in state_plans)
    subpage_alignment = KERNEL_BLOCK_TOKENS * kv_token_bytes
    min_block_tokens = _round_up(_ceil_div(max_ssm_bytes, kv_token_bytes), KERNEL_BLOCK_TOKENS)
    cfg_block = int(getattr(runtime, "kv_block_size", KERNEL_BLOCK_TOKENS)
                    or KERNEL_BLOCK_TOKENS)
    manager_block_size = max(min_block_tokens, _round_up(cfg_block, KERNEL_BLOCK_TOKENS))
    main_page_bytes = manager_block_size * kv_token_bytes
    state_stride = _round_up(max(conv_bytes + max_ssm_bytes, main_page_bytes), subpage_alignment)
    manager_block_size = state_stride // kv_token_bytes
    ratio = max(1, qsa[0].index_kpool)
    idx_elem = dtype_bytes(str(getattr(getattr(model, "quant", None), "index_dtype", "")
                               or "bf16"))
    raw_page_bytes = int(2 * ratio * qsa[0].index_dim * BF16_BYTES)
    if int8_fa3:
        # INT8 compressed-index codes plus one FP16 row scale written into the
        # declared ``Di + 4`` aligned pitch (132 B at Di=128).
        compressed_page_bytes = int(
            (manager_block_size // ratio) * (qsa[0].index_dim + FP32_BYTES))
    else:
        compressed_page_bytes = int(
            (manager_block_size // ratio) * qsa[0].index_dim * idx_elem)
    slot_backing_bytes = state_stride + raw_page_bytes + compressed_page_bytes
    pool_bytes_per_block = slot_count * slot_backing_bytes
    qsa_blocks = _ceil_div(max(ONE, seq), manager_block_size) if seq > ZERO else ZERO
    gdn_state_groups = _ceil_div(len(gdn), slot_count)
    raw_blocks = ONE if seq > ZERO else ZERO
    total_pool_blocks = qsa_blocks + raw_blocks + (gdn_state_groups if seq > ZERO else ZERO)
    private_ple_bytes = ple_conv_state_bytes(model, runtime) if seq > ZERO else ZERO_FLOAT
    return {
        "slot_count": slot_count, "manager_block_size": manager_block_size,
        "state_stride": state_stride, "pool_bytes_per_block": pool_bytes_per_block,
        "qsa_blocks": qsa_blocks, "gdn_state_groups": gdn_state_groups,
        "raw_blocks": raw_blocks, "total_pool_blocks": total_pool_blocks,
        "private_ple_bytes": private_ple_bytes,
        "total_pool_bytes": float(total_pool_blocks * pool_bytes_per_block) + private_ple_bytes,
    }


__all__ = [
    "build_ple_spec", "is_qwen4_exp_ple", "lower_ple", "ple_lookup_format",
    "build_ple_uva_npu_dequant", "build_ple_host_staging", "ple_table_terms",
    "ple_gate_terms_by_layer", "lowrank_hc_layer_terms", "lowrank_hc_head_terms",
    "lowrank_hc_draft_terms", "lowrank_hc_draft_head_terms", "qsa_indexer_terms",
    "ple_conv_state_bytes", "qwen4_exp_pool_breakdown",
]
