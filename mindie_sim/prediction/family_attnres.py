"""Source-selected Kimi AttnRes cross-layer prefix mechanism.

[Role] Build one native AttnResFwd point per source residual point and price the
  retained AIV-only kernel on the common engine.
[Invariants] The retained caller owns the state transition: a block boundary
  writes slot ``layer // B`` and clears the live prefix before attention; every
  FFN output stays pending until the next point or final point. Bank
  capacity and physical token stride stay distinct. Resident versus reload uses
  the retained tiling formula over declared UB, never a device or model name.
  No Cube GEMM or FP32-v user workspace is created. Selection is explicit and
  an unselected graph is untouched.
[Timeouts & Locks] Finite local arithmetic and per-core programs; no I/O, locks,
  device contexts or measurement reads.
[Side-Effect Policy] Returns immutable plans/records; source files are cited by
  hash, never read.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from .mechanism import OperatorPlan, PlannedRecipe, Work
from .ownership import read_role, unit_group
from .program import KernelProgram, ParallelProgram, Stage
from .structure_groups import declare_group


IMPLEMENTATION = "vllm_ascend_attn_res_prefix"
RECIPE = "attn_res_fwd"
NATIVE_KIND = "AttnResFwd"
GROUP_ROLE = "attn_res_prefix_point"

# Retained image identity. This is a source-compatible declaration; it does not
# prove the captured events loaded this library or these scalar flags.
IMAGE_SHA256 = "883670fcc91cbd75a7070b00501be0ecd81a266b81366a21bf02139d8e6a33e1"
CALLER_SHA256 = "be9adc792774f919f443caf483eb2e04e4453617465381f47a482dd4736ec63f"
SOURCE = {
    "repository": "vllm-ascend",
    "image_id": "sha256:" + IMAGE_SHA256,
    "caller_sha256": CALLER_SHA256,
    "caller": {
        "path": "vllm_ascend/models/kimi_k3.py",
        "lines": "650-830,1030-1150",
        "sha256": CALLER_SHA256,
    },
    "operator": {
        "common_sha256": "f8abf2be43c0e0a70a19df4d4d1cdab518dc939fee6612aabcaf1f6e090e53a2",
        "tiling_sha256": "3249e4f252af0c68187b8caa7a00f63592a42c231fefecc28b7b1e7609b60669",
        "resident_sha256": "bb5a40a14e3d5637ab9deff26b77e8e3a305500f9c582a3274d7bf0b23840d94",
        "reload_sha256": "e346cd0db6eeedcfffaf911a07f6ddf4e34475c69677f73cb9e279f39641f070",
        "path": "vllm_ascend/csrc/attention/attn_res_fwd",
    },
    "scope": "source_compatible_declaration; not_capture_binary_identity",
    "capture_flags_proven": False,
}
LIMITATIONS = (
    "capture_library_and_scalar_flags_not_proven",
    "strict_descriptor_contract_not_derived_from_csv",
    "per_owner_load_compute_store_phases_are_serial",
    "cross_core_overlap_is_shared_rate_service_not_a_compiler_schedule",
    "pp1_source_caller_domain",
    "sequence_parallel_ag_rs_and_tokens_local_are_caller_supplied",
    "semantic_vector_lanes_not_isa_latency_or_special_function_throughput",
    "aux_materialization_is_explicit_not_inferred",
    "no_measured_duration_or_bandwidth_fit_is_used",
)

UB_OVERHEAD_BYTES = 32 * 1024           # TPipe/alignment/queue allowance
SYSTEM_WORKSPACE_BYTES = 16 * 1024 * 1024  # system reservation, not user tensor
ELEM_BF16_BLOCK = 16
ELEM_FP32_BLOCK = 8
SCALAR_LOCAL_ELEMS = 8


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _align(value, multiple):
    return _ceil_div(value, multiple) * multiple


def selection(runtime):
    """Return explicit options for our selector, or None when unselected.

    A binding of another registered implementation is deliberately not
    reinterpreted as this mechanism.
    """
    bindings = runtime.kernels.get("lowerings", {})
    if "attention_residual" not in bindings:
        return None
    from .implementations import select

    selected = select(bindings, "attention_residual")
    if selected.name != IMPLEMENTATION:
        return None
    # Source `_use_attn_res_prefill_cache` is false outside pure prefill A5.
    # This planner never infers a hardware name, so the cache is opt-in.
    value = selected.options.get("optimize_prefill", False)
    if not isinstance(value, bool):
        raise ModelingError("attn_res_optimize_prefill_must_be_boolean")
    return {"optimize_prefill": value}


def bank_capacity(model):
    block = next((plan.residual_block_size for plan in model.layer_plans
                  if plan.residual_block_size), 0)
    if block < 1:
        raise ModelingError("attn_res_requires_residual_block_size")
    end = max((plan.index for plan in model.layer_plans), default=-1) + 1
    return _ceil_div(end, block)


def block_token_stride(model, capacity):
    """The source caller allocates a contiguous [T, capacity, H] bank."""
    return capacity * model.hidden_size


def max_resident_rows(hidden, ub_bytes, fuse_add=True):
    """Exact retained CalcMaxResidentRows over a declared UB byte capacity."""
    if hidden < 1 or ub_bytes < 1:
        return 0
    work = (4 * hidden * 4 + 2 * hidden * 2 + UB_OVERHEAD_BYTES
            + (hidden * 2 if fuse_add else 0))
    if work >= ub_bytes:
        return 0
    return (ub_bytes - work) // (hidden * 2 + 4)


def estimate_ub_bytes(block_count, hidden, resident, fuse_add=True):
    """Retained EstimateUbComputeBytes for the active buffer geometry."""
    h_bf16, h_fp32 = _align(hidden, ELEM_BF16_BLOCK), _align(hidden, ELEM_FP32_BLOCK)
    meta = _align(block_count, ELEM_FP32_BLOCK)
    total = 0
    if resident:
        total += 2 * h_bf16 * 2 + block_count * h_bf16 * 2
    else:
        total += 3 * h_bf16 * 2
    total += 3 * h_fp32 * 4
    total += 2 * meta * 4
    total += meta * ELEM_FP32_BLOCK * 4
    total += SCALAR_LOCAL_ELEMS * 4
    if fuse_add:
        total += h_bf16 * 2
    return total + UB_OVERHEAD_BYTES


def _cache_output_norm(view, ub_bytes):
    if not (view["optimize_prefill"] and view["output_norm"]):
        return False
    if view["tokens_per_core"] <= 1:
        return False
    fit = estimate_ub_bytes(view["valid_blocks"] + 1, view["hidden"],
                            view["resident"], True)
    return fit + _align(view["hidden"], ELEM_BF16_BLOCK) * 2 <= ub_bytes


def _validate_bank(bank_ids, capacity, valid, write_slot, mix):
    if len(bank_ids) != capacity:
        raise ModelingError("attn_res_bank_ids_require_capacity_slots")
    if valid < 0 or valid > capacity:
        raise ModelingError("attn_res_valid_blocks_out_of_capacity")
    if write_slot is not None:
        if isinstance(write_slot, bool) or not isinstance(write_slot, int):
            raise ModelingError("attn_res_write_slot_invalid")
        if write_slot < 0 or write_slot >= capacity or write_slot < valid:
            raise ModelingError("attn_res_write_slot_requires_next_slot")
    if mix and valid:
        if any(bank_ids[index] is None for index in range(valid)):
            raise ModelingError("attn_res_mixture_requires_written_valid_slots")


def _vector_work(hidden, valid, *, mixing, fuse_add, output_norm):
    """Semantic arithmetic/cast lanes, not instruction or cycle estimates.

    Reductions count their logical additions. Exp/sqrt/div each count one
    operation; their compiled expansions and lane issue costs are not inferred.
    """
    # Two input casts, add, then BF16 round before scoring or bank storage.
    add = 4 * hidden if fuse_add else 0
    if mixing:
        rows = valid + 1
        # Cast, square/reduce/inverse RMS, normalize, dot/reduce; then each
        # BF16 value is cast again for its weighted multiply-add.
        score = rows * (6 * hidden + 2)
        weighted = rows * 3 * hidden
        # max, subtraction, exp, sum, reciprocal, normalization.
        softmax = 5 * rows - 1
        mixture = score + weighted + softmax
        initialize = 3 * hidden  # two weight casts and proj*norm, once/owner
    else:
        mixture, initialize = hidden, 0  # bypass BF16 -> FP32
    output_round = hidden
    # Rounded mixture cast + RMS + gamma cast + two multiplies + final cast.
    normalization = 7 * hidden + 3 if output_norm else 0
    return {
        "per_token": add + mixture + output_round + normalization,
        "per_owner": initialize,
        "fused_add": add, "mixture": mixture, "output_round": output_round,
        "output_norm": normalization,
    }


def point_metrics(view, ub_bytes, aiv):
    """Independent traffic/vector arithmetic for one native point.

    ``T`` rows use ``used=min(T,AIV)`` contiguous owners and ``ceil(T/used)``
    rows per owner; owners past the last real row are absent. Resident history
    is read once (2 bytes per element); reload reads it twice (4 bytes).
    """
    tokens, hidden = view["tokens_local"], view["hidden"]
    valid = view["valid_blocks"]
    mixing = view["mix"] and valid > 0
    used = min(tokens, aiv) if tokens and aiv else 0
    per_core = _ceil_div(tokens, used) if used else 0
    ranges = tuple((start, min(per_core, tokens - start))
                   for start in range(0, tokens, per_core)) if per_core else ()
    resident = max_resident_rows(hidden, ub_bytes, True)
    resident = valid + 1 <= resident
    ready = dict(view, resident=resident, used_cores=len(ranges),
                 tokens_per_core=per_core, core_ranges=ranges)
    ready["output_norm_cached"] = _cache_output_norm(ready, ub_bytes)
    footprint = estimate_ub_bytes(valid + 1, hidden, resident)
    if ready["output_norm_cached"]:
        footprint += _align(hidden, ELEM_BF16_BLOCK) * 2
    if footprint > ub_bytes:
        raise ModelingError("attn_res_source_ub_footprint_exceeds_capacity")
    ready["ub_bytes_per_owner"] = footprint
    byte2, byte4 = dtype_bytes("bf16"), dtype_bytes("float32")
    bank_read = ((byte2 if resident else byte4) * tokens * valid * hidden
                 if mixing else 0)
    score_weights = (byte2 * hidden + byte2 * hidden) * len(ranges) if mixing else 0
    gamma = (byte2 * hidden * (len(ranges) if ready["output_norm_cached"] else tokens)
             if view["output_norm"] else 0)
    prefix_read = byte2 * tokens * hidden
    addend_read = byte2 * tokens * hidden if view["fuse_add"] else 0
    writes = byte2 * tokens * hidden * (
        1 + int(view["fuse_add"]) + int(view["block_write_idx"] >= 0)
        + int(view["save_materialized"]))
    vector = _vector_work(hidden, valid, mixing=mixing,
                          fuse_add=view["fuse_add"], output_norm=view["output_norm"])
    ready["vector_work_basis"] = vector
    cores = []
    for start, count in ranges:
        read = (2 * count * hidden
                + (2 * count * hidden if view["fuse_add"] else 0)
                + ((2 if resident else 4) * count * valid * hidden if mixing else 0)
                + (4 * hidden if mixing else 0)
                + (2 * hidden if ready["output_norm_cached"]
                   else (2 * count * hidden if view["output_norm"] else 0)))
        write = 2 * count * hidden * (
            1 + int(view["fuse_add"]) + int(view["block_write_idx"] >= 0)
            + int(view["save_materialized"]))
        cores.append({
            "token_start": start, "token_count": count,
            "read_bytes": read, "write_bytes": write,
            "vector_ops": count * vector["per_token"] + vector["per_owner"],
        })
    resources = {
        "hbm": prefix_read + addend_read + bank_read + score_weights + gamma + writes,
        "mte2": prefix_read + addend_read + bank_read + score_weights + gamma,
        "mte3": writes,
        "vector": tokens * vector["per_token"] + len(ranges) * vector["per_owner"],
    }
    return {
        "resources": {key: float(value) for key, value in resources.items() if value},
        "view": ready,
        "cores": tuple(cores),
        "bank_read": bank_read,
        "score_weight_read": score_weights,
        "output_gamma_read": gamma,
        "write_bytes": writes,
    }


class PointResult(NamedTuple):
    ops: tuple
    hidden_id: str
    raw_prefix_id: str
    materialized_id: str
    bank_ids: tuple


def _point_view(context, *, prefix_id, addend_id, bank_ids, valid_blocks,
                write_slot, output_norm, mix, materialize, tokens_local_value,
                capacity, stride, optimize_prefill):
    model = context["model"]
    if model.activation_dtype != "bf16":
        raise ModelingError("attn_res_source_requires_bf16")
    hidden = model.hidden_size
    if tokens_local_value < 0 or hidden < 1 or capacity < 1:
        raise ModelingError("attn_res_point_geometry_invalid")
    if stride < capacity * hidden:
        raise ModelingError("attn_res_token_stride_smaller_than_capacity_rows")
    if not prefix_id:
        raise ModelingError("attn_res_point_requires_prefix")
    bank_ids = tuple(bank_ids)
    _validate_bank(bank_ids, capacity, valid_blocks, write_slot, mix)
    view = {
        "tokens": context.get("tokens", tokens_local_value),
        "tokens_local": tokens_local_value,
        "hidden": hidden,
        "valid_blocks": valid_blocks,
        "bank_capacity": capacity,
        "block_token_stride": stride,
        "fuse_add": addend_id is not None,
        "block_write_idx": -1 if write_slot is None else write_slot,
        "mix": bool(mix),
        "output_norm": bool(output_norm),
        "save_materialized": bool(materialize),
        "optimize_prefill": bool(optimize_prefill),
        "phase": context.get("mode", "prefill"),
        "requested_optimize_prefill": bool(optimize_prefill),
        "system_workspace_bytes": SYSTEM_WORKSPACE_BYTES,
        "user_fp32_workspace_bytes": 0,
        "stored_dtype": model.activation_dtype,
        "working_dtype": "fp32",
        "rounding": "bf16_after_add_and_mix",
        "dtype": model.activation_dtype,
    }
    return view, bank_ids


def attn_res_point(context, *, prefix_id, addend_id, bank_ids, valid_blocks,
                   write_slot, output_norm, mix, materialize, tokens_local,
                   bank_capacity, block_token_stride, optimize_prefill=False):
    """Build one native AttnResFwd call and its explicit bank transition.

    ``bank_ids`` is one entry per physical capacity slot. Only ``valid_blocks``
    older rows feed the mixture; the current prefix is the extra row. The
    returned raw-prefix/materialized ids alias input/output exactly when the
    source does (no addend, or no save_materialized).
    """
    view, banks = _point_view(
        context, prefix_id=prefix_id, addend_id=addend_id, bank_ids=bank_ids,
        valid_blocks=valid_blocks, write_slot=write_slot,
        output_norm=output_norm, mix=mix, materialize=materialize,
        tokens_local_value=tokens_local, capacity=bank_capacity,
        stride=block_token_stride, optimize_prefill=optimize_prefill)
    bank_rows = tuple(banks[index] for index in range(valid_blocks)) if mix else ()
    deps = tuple(dict.fromkeys(
        (prefix_id, *((addend_id,) if addend_id is not None else ()), *bank_rows)))
    op_id = context["prefix"] + ".attn_res_fwd"
    raw_id = op_id if addend_id is not None else prefix_id
    materialized_id = op_id
    op = OpSpec(
        op_id=op_id, family="attn_res_prefix",
        phase=context.get("mode", "prefill"), stage=context["stage"],
        layer=context["layer"], requests=tuple(context.get("requests", ())),
        semantic_shape=(tokens_local, context["model"].hidden_size),
        local_shape=(tokens_local, context["model"].hidden_size),
        padded_shape=(tokens_local, context["model"].hidden_size),
        launch_shape=(tokens_local, context["model"].hidden_size),
        deps=deps, recipe=RECIPE, source_boundary="kernel",
        dtype=context["model"].activation_dtype,
        layout="contiguous",
        coverage="complete_source_call",
        returns=("hidden_states", "prefix_out", "materialized"),
        metadata=tuple(sorted({
            "native_kind": NATIVE_KIND,
            "attn_res_view": view,
            "attn_res_bank_ids": banks,
            "attn_res_prefix_id": prefix_id,
            "attn_res_addend_id": addend_id,
            "attn_res_bank_rows": bank_rows,
            "attn_res_raw_prefix_id": raw_id,
            "attn_res_materialized_id": materialized_id,
            "attn_res_prefix_out_aliases_prefix": addend_id is None,
            "attn_res_materialized_aliases_hidden": not materialize,
            "attn_res_source": SOURCE,
            "attn_res_limitations": LIMITATIONS,
            "source_boundary_kind": "kernel",
            "submitted_kernels": 1,
            "implementation": {"native_kind": NATIVE_KIND,
                               "scope": "full_source_call"},
        }.items())),
        notes=(
            "source: retained AttnResFwd caller and AIV-only arch35 kernel",
            "one native point launch; no cube GEMM and no FP32-v user workspace",
            "serial per-owner load/compute/store phases, shared-rate contention",
        ),
    )
    ops = [op]
    declare_group(
        ops, key=op_id + ".group", role=GROUP_ROLE, family="attn_res_prefix",
        members=(op_id,), entries=(op_id,), anchor=op_id, view=view,
        implementation={"native_kind": NATIVE_KIND,
                        "conditions": {"scope": "full_source_call"}},
        source=SOURCE, block=context.get("structure_block"),
        scope_role="residual",
        coverage={"kind": "complete_subgraph", "limitations": list(LIMITATIONS)},
    )
    bank_outputs = list(banks)
    if write_slot is not None:
        bank_outputs[write_slot] = op_id
    return PointResult(tuple(ops), op_id, raw_id, materialized_id,
                       tuple(bank_outputs))


@dataclass
class AttnResPrefixState:
    """The caller's cross-layer prefix/pending-MLP state machine."""

    model: object
    runtime: object
    tokens: int
    tokens_local: int
    mode: str
    requests: tuple
    optimize_prefill: bool
    capacity: int
    stride: int
    block_size: int
    prefix_id: str
    bank_ids: tuple
    pending_mlp_id: str | None = None
    clear_before_attention: bool = False

    @classmethod
    def start(cls, *, model, runtime, ready, tokens,
              mode, requests, optimize_prefill=False):
        if runtime.parallel.pp != 1:
            raise ModelingError("attn_res_source_caller_requires_pp1")
        block = next((plan.residual_block_size for plan in model.layer_plans
                      if plan.residual_block_size), 0)
        if block < 1:
            raise ModelingError("attn_res_requires_residual_block_size")
        if any((plan.residual_block_size != block or plan.engram
                or plan.hyper_connections) for plan in model.layer_plans):
            raise ModelingError("attn_res_selected_semantics_unsupported_for_layer")
        capacity = bank_capacity(model)
        return cls(model, runtime, tokens, tokens, mode,
                   tuple(requests), bool(optimize_prefill and mode == "prefill"), capacity,
                   block_token_stride(model, capacity), block, ready,
                   (None,) * capacity)

    def _context(self, prefix, layer, stage, structure_block):
        return {
            "prefix": prefix, "stage": stage, "layer": layer.index,
            "requests": self.requests, "model": self.model,
            "mode": self.mode,
            "tokens": self.tokens, "structure_block": structure_block,
        }

    def prepare(self, *, prefix, layer, stage, materialize=False):
        block = self.block_size
        boundary = layer.index % block == 0
        valid = _ceil_div(layer.index, block)
        slot = layer.index // block if boundary else None
        result = attn_res_point(
            self._context(prefix, layer, stage, "attention"),
            prefix_id=self.prefix_id, addend_id=self.pending_mlp_id,
            bank_ids=self.bank_ids, valid_blocks=valid, write_slot=slot,
            output_norm=True, mix=True, materialize=materialize,
            tokens_local=self.tokens_local, bank_capacity=self.capacity,
            block_token_stride=self.stride,
            optimize_prefill=self.optimize_prefill)
        self.prefix_id = result.raw_prefix_id
        self.bank_ids = result.bank_ids
        self.pending_mlp_id = None
        self.clear_before_attention = boundary
        return result

    def attention_added(self, *, prefix, layer, stage, attention_id, materialize=False):
        block = self.block_size
        boundary = self.clear_before_attention
        valid = _ceil_div(layer.index, block) + int(boundary)
        result = attn_res_point(
            self._context(prefix, layer, stage, "ffn"),
            prefix_id=attention_id if boundary else self.prefix_id,
            addend_id=None if boundary else attention_id,
            bank_ids=self.bank_ids, valid_blocks=valid, write_slot=None,
            output_norm=True, mix=True, materialize=materialize,
            tokens_local=self.tokens_local, bank_capacity=self.capacity,
            block_token_stride=self.stride,
            optimize_prefill=self.optimize_prefill)
        self.prefix_id = result.raw_prefix_id
        self.bank_ids = result.bank_ids
        self.clear_before_attention = False
        return result

    def defer_mlp(self, mlp_id):
        if self.pending_mlp_id is not None:
            raise ModelingError("attn_res_pending_mlp_must_be_absorbed_once")
        self.pending_mlp_id = mlp_id

    def final(self, *, prefix, last_layer, stage):
        return attn_res_point(
            self._context(prefix, last_layer, stage, "ffn"),
            prefix_id=self.prefix_id, addend_id=self.pending_mlp_id,
            bank_ids=self.bank_ids,
            valid_blocks=_ceil_div(last_layer.index + 1, self.block_size),
            write_slot=None, output_norm=False, mix=True, materialize=False,
            tokens_local=self.tokens_local, bank_capacity=self.capacity,
            block_token_stride=self.stride,
            optimize_prefill=self.optimize_prefill)

def _physical_metrics(op, geometry):
    if op.recipe != RECIPE or op.source_boundary != "kernel":
        raise ModelingError("attn_res_plan_requires_selected_kernel_source")
    if op.dtype != "bf16":
        raise ModelingError("attn_res_source_requires_bf16")
    metadata = dict(op.metadata)
    view = dict(metadata["attn_res_view"])
    ub = dict(geometry.buffers).get("ub")
    if not ub:
        raise ModelingError("attn_res_requires_declared_ub_buffer")
    aiv = geometry.resource_units.get("vector", geometry.compute_units)
    role = read_role(geometry.resource_units, 0)
    metrics = point_metrics(view, ub, aiv)
    return metrics, role, aiv


def _plan(metrics, role, aiv):
    resources = dict(metrics["resources"])
    resources[role] = resources.pop("mte2", 0)
    total = Work({}, resources)

    def program(context):
        cores = []
        for owner, core in enumerate(metrics["cores"]):
            placement = context.owned(owner, aiv, (role, "vector", "mte3"))
            allocation = (context.allocate(
                "attn_res_ub", "ub", metrics["view"]["ub_bytes_per_owner"],
                unit_group(owner, aiv, context.pool_units.get("ub", aiv))),)
            load = Work({}, {"hbm": core["read_bytes"], role: core["read_bytes"]})
            compute = Work({}, {"vector": core["vector_ops"]})
            store = Work({}, {"hbm": core["write_bytes"], "mte3": core["write_bytes"]})
            cores.append(KernelProgram((
                Stage("load", context.bind(load), placement=placement, acquire=allocation),
                Stage("compute", context.bind(compute), after=("load",), placement=placement),
                Stage("store", context.bind(store), after=("compute",),
                      placement=placement, free=allocation),
            )))
        return ParallelProgram(tuple(cores))

    return OperatorPlan(total, program)


def build(op, geometry):
    """Bind physical ownership at pricing, independently of semantic lowering."""
    return _plan(*_physical_metrics(op, geometry))


class AttnResRecipe(PlannedRecipe):
    def prepare(self, op, geometry):
        metrics, role, aiv = _physical_metrics(op, geometry)
        plan = _plan(metrics, role, aiv)
        metadata = dict(op.metadata)
        metadata.update(attn_res_view=metrics["view"],
                        attn_res_core_work=metrics["cores"],
                        attn_res_read_role=role)
        metadata["structure_subgraphs"] = tuple(
            {**group, "tensor_view": metrics["view"]}
            for group in metadata.get("structure_subgraphs", ()))
        return plan.work.apply(replace(op, metadata=tuple(metadata.items()))), plan


def recipes():
    return {RECIPE: AttnResRecipe(build)}


__all__ = [
    "AttnResPrefixState", "IMPLEMENTATION", "LIMITATIONS", "NATIVE_KIND",
    "PointResult", "RECIPE", "SOURCE", "attn_res_point", "bank_capacity",
    "block_token_stride", "build", "estimate_ub_bytes", "max_resident_rows",
    "point_metrics", "recipes", "selection",
]
