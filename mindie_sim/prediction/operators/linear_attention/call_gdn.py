"""Complete device work for selected standalone GDN public calls.

[Role] Build a call DAG from explicit arguments, separate from a serving block.
[Invariants] Inputs and state are prepared before the call. Gating FP32 output
  is not the served BF16 scalar-store path. No observation enters the planner.
[Timeouts & Locks] Lazy bounded programs on the common engine; no I/O or locks.
[Side-Effect Policy] Returns operators, recipes and a boundary ledger only.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import math

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from mindie_sim.prediction.operators.linear_attention import gdn_composite
from .gdn_domain import GDN_CHUNK64_GQA_RATIOS, GDN_CHUNK64_TOKENS, GDN_HEAD_DIM_K, GDN_CHUNK32_HEAD_DIM_V
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.operators.causal_conv import native_causal_conv_prefill
from mindie_sim.prediction.operators.linear_attention import gdn_recurrent
from mindie_sim.prediction.operators.vector.mixed_core import read_role
from mindie_sim.prediction.program import ParallelProgram, PipelineProgram, RecipeBook, Stage
from mindie_sim.prediction.tensors import TensorLowering

GATING_TILE_ROWS = 64
GATING_TILE_HEADS = 8
GATING_ELEMENT_OPS = 13
GATING_FP32_BYTES = 4
GATING_SOFTPLUS_THRESHOLD = 20.0
CONV_WRAPPER_SOURCE_LINES = (1018, 1078)
SCAN_WRAPPER_SOURCE_LINES = (1472, 1516)
SCAN_CALL_CONV_TAPS = 3


GATING_RECIPE = "fused_gdn_gating_fp32_call"
GATING_SOURCE = {
    "path": "data/observations/sources/gdn-image-c3d00a196905/fused_gdn_gating.py.txt",
    "sha256": "5587bf3440e64eda125899f5de2d1ab609061e2b5336f847af101ada51d81058",
    "kernel_lines": [14, 56], "wrapper_lines": [59, 101],
}
WRAPPER_SOURCE = {
    "archive": "data/observations/sources/gdn-prefill-image-c3d00a196905/sources.tar.gz",
    "member": "ops/ascendc/_stable.py",
    "sha256": "d6be8820bb54e0177cf92e396099011402f97066b0974a2f1a8a90187f08aae0",
}
PREFILL_SOURCE = {
    "archive": "data/observations/sources/gdn-prefill-image-c3d00a196905/sources.tar.gz",
    "sha256": "97bba4058df7809c5d67ee0196e291ae3f4afc5b3ae16081c01333d6b2dffe65",
    "members": (
        "chunk_gated_delta_rule_fwd_prepare/chunk_gated_delta_rule_fwd_prepare.cpp",
        "chunk_fwd_h/chunk_fwd_h.cpp",
        "chunk_fwd_o/chunk_fwd_o.cpp",
    ),
}
def _normalize_partition(params, role):
    """One partition source for lengths/qsl/total/sequence count.

    Explicit ``lengths`` or ``cu_seqlens/query_start_loc`` wins over the legacy
    equal-length ``batch``/``tokens`` pair.  Supplying both explicit forms must
    agree.  The legacy ``tokens`` remains per-sequence only when no explicit
    partition exists.
    """
    raw_lengths = params.get("lengths")
    raw_qsl = params.get("cu_seqlens", params.get("query_start_loc"))
    supplied_qsl = None
    if raw_qsl is not None:
        if (isinstance(raw_qsl, (str, bytes))
                or not isinstance(raw_qsl, (list, tuple)) or len(raw_qsl) < 2):
            raise ModelingError(f"gdn_{role}_cu_seqlens_must_be_nonempty_sequence")
        supplied_qsl = tuple(raw_qsl)
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in supplied_qsl):
            raise ModelingError(f"gdn_{role}_cu_seqlens_must_be_integers")
        if supplied_qsl[0] != 0 or any(
                left > right for left, right in zip(supplied_qsl, supplied_qsl[1:])):
            raise ModelingError(f"gdn_{role}_cu_seqlens_must_be_monotonic_from_zero")
    if raw_lengths is not None:
        if (isinstance(raw_lengths, (str, bytes))
                or not isinstance(raw_lengths, (list, tuple)) or not raw_lengths):
            raise ModelingError(f"gdn_{role}_lengths_must_be_nonempty_sequence")
        lengths = tuple(raw_lengths)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in lengths):
            raise ModelingError(f"gdn_{role}_lengths_must_be_nonnegative_ints")
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        if supplied_qsl is not None and supplied_qsl != tuple(cu):
            raise ModelingError(f"gdn_{role}_lengths_cu_seqlens_mismatch")
    elif supplied_qsl is not None:
        lengths = tuple(right - left for left, right in
                        zip(supplied_qsl, supplied_qsl[1:]))
        cu = list(supplied_qsl)
    else:
        batch, tokens = _positive(params, "batch"), _positive(params, "tokens")
        lengths = (tokens,) * batch
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
    if params.get("batch") is not None and params["batch"] != len(lengths):
        raise ModelingError(f"gdn_{role}_partition_batch_mismatch")
    total = sum(lengths)
    if total <= 0:
        raise ModelingError(f"gdn_{role}_partition_total_must_be_positive")
    return lengths, len(lengths), total, tuple(cu)


_CALL_PLAN = SimpleNamespace(index=0)
_CALL_MODEL = SimpleNamespace(activation_dtype="bf16", compute_dtype="bf16", quant={})


@dataclass(frozen=True)
class GDNCall:
    ops: tuple[OpSpec, ...]
    recipes: RecipeBook
    ledger: dict
    limitations: tuple[str, ...]

    @property
    def device_program(self) -> dict:
        """The selected call's actual arguments and tensor descriptors.

        This is the existing ledger view: it is prepare-only metadata and never
        allocates a tensor, imports a runtime or launches a kernel.
        """
        ledger = self.ledger or {}
        return {
            "call": ledger.get("call"),
            "source": dict(ledger.get("source") or {}),
            "arguments": dict(ledger.get("arguments") or {}),
            "inputs": dict(ledger.get("inputs") or {}),
            "outputs": dict(ledger.get("outputs") or {}),
            "exports": dict(ledger.get("exports") or {}),
            "state": dict(ledger.get("state") or {}),
            "device_kernels": list(ledger.get("device_kernels") or ()),
            "wrapper_source": dict(ledger.get("wrapper_source") or {}),
            "materializations": self._materialization_records(),
            "effects": [
                {"op_id": op.op_id,
                 "state_contract": dict(op.metadata).get("gdn_state_contract"),
                 "state_effects": dict(op.metadata).get("gdn_state_effects", ())}
                for op in self.ops
                if "gdn_state_effects" in dict(op.metadata)
            ],
        }

    def _materialization_records(self) -> list[dict]:
        by_id = {op.op_id: op for op in self.ops}
        records = []
        for item in self.ledger.get("materializations") or ():
            op = by_id.get(item.get("op_id"))
            record = dict(item)
            if op is not None:
                meta = dict(op.metadata)
                record.update({
                    "recipe": op.recipe, "dtype": op.dtype,
                    "semantic_shape": list(op.semantic_shape or ()),
                    "padded_shape": list(op.padded_shape or ()),
                    "source_layout": meta.get("source_layout"),
                    "target_layout": meta.get("target_layout"),
                })
            records.append(record)
        return records


def _positive(params, name):
    value = params[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ModelingError(f"gdn_call_requires_positive_integer:{name}")
    return value


def _gating_plan(op, geometry):
    rows, heads = op.local_shape
    units = geometry.resource_units.get("vector", geometry.compute_units)
    row_iterations = math.ceil(rows / (GATING_TILE_ROWS * units))
    col_iterations = math.ceil(heads / GATING_TILE_HEADS)
    reads = 2 * GATING_FP32_BYTES * (rows * heads + units * row_iterations * heads)
    writes = 2 * GATING_FP32_BYTES * rows * heads
    # One elementary lane-op envelope: x/add, beta*x, compare, exp/add/log/
    # scale/select, g product, sigmoid neg/exp/add/div. A_log exp/neg is
    # column-local. Masks suppress memory, not these vector expressions.
    per_tile_ops = GATING_TILE_HEADS * (GATING_ELEMENT_OPS * GATING_TILE_ROWS + 2)
    vector = units * row_iterations * col_iterations * per_tile_ops
    read = read_role(geometry.resource_units, 0)
    work = Work(resources={"hbm": reads + writes, read: reads,
                           "mte3": writes, "vector": vector})

    def program(context):
        def tile(index, unit):
            row_index, col_index = divmod(index, col_iterations)
            first = (unit * row_iterations + row_index) * GATING_TILE_ROWS
            valid_rows = max(0, min(GATING_TILE_ROWS, rows - first))
            valid_cols = min(GATING_TILE_HEADS, heads - col_index * GATING_TILE_HEADS)
            count = valid_rows * valid_cols
            incoming = 2 * GATING_FP32_BYTES * (count + valid_cols)
            owned = context.owned(unit, units, ("vector", read, "mte3"))
            return (
                Stage("load", context.bind(Work(resources={"hbm": incoming, read: incoming})),
                      placement=owned),
                Stage("elementwise", context.bind(Work(resources={"vector": per_tile_ops})),
                      after=("load",), placement=owned),
                Stage("store_g", context.bind(Work(resources={"hbm": count * GATING_FP32_BYTES, "mte3": count * GATING_FP32_BYTES})),
                      after=("elementwise",), placement=owned),
                Stage("store_beta", context.bind(Work(resources={"hbm": count * GATING_FP32_BYTES, "mte3": count * GATING_FP32_BYTES})),
                      after=("store_g",), placement=owned),
            )
        return ParallelProgram(tuple(
            PipelineProgram(row_iterations * col_iterations, 1,
                            lambda index, unit=unit: tile(index, unit))
            for unit in range(units)))
    return OperatorPlan(work, program)


def _gating(params):
    rows, heads = _positive(params, "rows"), _positive(params, "heads")
    if any(params.get(name, "fp32") != "fp32"
           for name in ("a_dtype", "b_dtype", "a_log_dtype", "dt_bias_dtype")):
        raise ModelingError("gdn_gating_call_requires_fp32_arguments")
    if params.get("layout", "contiguous") != "contiguous":
        raise ModelingError("gdn_gating_call_requires_contiguous_arguments")
    if params.get("beta", 1.0) != 1.0 or params.get("threshold", GATING_SOFTPLUS_THRESHOLD) != GATING_SOFTPLUS_THRESHOLD:
        raise ModelingError("gdn_gating_call_requires_default_softplus_arguments")
    floor_reads = float(2 * GATING_FP32_BYTES * (rows * heads + heads))
    floor_writes = float(2 * GATING_FP32_BYTES * rows * heads)
    floor_vector = float(rows * heads * GATING_ELEMENT_OPS)
    op = OpSpec(
        "gating", "gdn_gating", params.get("phase", "prefill"), 0, 0, (),
        (rows, heads), (rows, heads), (rows, heads), (rows, heads),
        vector_ops=floor_vector, hbm_bytes=floor_reads + floor_writes,
        mte2_bytes=floor_reads, mte3_bytes=floor_writes,
        dtype="fp32", layout="contiguous", recipe=GATING_RECIPE,
        source_boundary="kernel",
        metadata=(("implementation", GATING_RECIPE), ("source", GATING_SOURCE),
                  ("output_dtypes", {"g": "fp32", "beta": "fp32"})),
    )
    ledger = {
        "call": "vllm_ascend.ops.triton.fused_gdn_gating.fused_gdn_gating_patch",
        "arguments": {"rows": rows, "heads": heads, "a_dtype": "fp32", "b_dtype": "fp32",
                      "a_log_dtype": "fp32", "dt_bias_dtype": "fp32", "layout": "contiguous",
                      "beta": 1.0, "threshold": GATING_SOFTPLUS_THRESHOLD},
        "device_kernels": [{"op_id": op.op_id, "component": "fused_gdn_gating_kernel",
                            "source": GATING_SOURCE}],
        "host_only": ["allocate uninitialized g and beta tensors", "compute launch geometry"],
        "outside_call": ["create/copy a, b, A_log and dt_bias", "warmup and event bookkeeping"],
        "outputs": {"g": {"shape": [1, rows, heads], "dtype": "fp32"},
                    "beta": {"shape": [1, rows, heads], "dtype": "fp32"}},
        "boundary": "device_work_of_complete_call",
    }
    return GDNCall((op,), RecipeBook({GATING_RECIPE: PlannedRecipe(_gating_plan)}), ledger, (
        "Elementwise expressions use declared vector service; this is not a compiled instruction schedule.",
        "FP32 loads/stores use the declared copy resources; no BF16 scalar-store fit is borrowed.",
        "Host allocation/dispatch and event-enqueue gaps require separate submission assumptions.",
    ))


def _conv_prefill(params):
    lengths, batch, tokens, cu_seqlens = _normalize_partition(params, "conv_call")
    dim, width = _positive(params, "conv_width"), _positive(params, "kernel")
    if params.get("input_dtype", "bf16") != "bf16":
        raise ModelingError("gdn_conv_call_requires_bf16_arguments")
    if params.get("layout", "contiguous") != "contiguous":
        raise ModelingError("gdn_conv_call_requires_contiguous_arguments")
    bias = params.get("bias", False)
    if not isinstance(bias, bool):
        raise ModelingError("gdn_conv_call_bias_must_be_bool")
    if params.get("activation", "silu") != "silu":
        raise ModelingError("gdn_conv_call_requires_silu_activation")
    if params.get("initial_state", False) is not False:
        raise ModelingError("gdn_conv_call_requires_no_initial_state")
    absent_index = "cache_indices" in params and params["cache_indices"] is None
    raw_indices = params.get("cache_indices")
    if absent_index:
        cache_indices = None
        index_source = "absent_identity"
    elif raw_indices is None:
        cache_indices = tuple(range(1, batch + 1))
        index_source = "uniform_fixture"
    else:
        if (isinstance(raw_indices, (str, bytes))
                or not isinstance(raw_indices, (list, tuple))
                or len(raw_indices) != batch):
            raise ModelingError("gdn_conv_call_cache_indices_length_mismatch")
        cache_indices = tuple(raw_indices)
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0
               for index in cache_indices):
            raise ModelingError("gdn_conv_call_cache_indices_must_be_nonnegative")
        index_source = "caller_metadata"
    state_rows = params.get("state_rows")
    rows_decl = params.get("state_cache_rows")
    if state_rows is None and rows_decl is not None:
        if rows_decl == "batch+1":
            state_rows = batch + 1
        elif isinstance(rows_decl, int) and not isinstance(rows_decl, bool):
            state_rows = rows_decl
        else:
            raise ModelingError("gdn_conv_call_state_cache_rows_invalid")
    if state_rows is not None and (
            isinstance(state_rows, bool) or not isinstance(state_rows, int)
            or (cache_indices is not None
                and state_rows <= max(cache_indices, default=0))):
        raise ModelingError("gdn_conv_call_state_rows_below_cache_index")
    arguments = {
        "lengths": lengths, "query_start_loc": list(cu_seqlens),
        "dim": dim, "width": width, "dtype": "bf16",
        "activation": "silu", "bias": bias, "metadata_placement": "device",
        "has_initial_state": (False,) * batch, "cache_indices": cache_indices,
        "state_history_slots": width - 1, "null_block_id": 0,
        "pad_slot_id": -1,
    }
    if state_rows is not None:
        arguments["state_rows"] = state_rows
    op = replace(native_causal_conv_prefill.make_op(arguments), op_id="convolution")
    state_views = tuple(SimpleNamespace(
        request_id=f"seq{index}", q_len=length, context=0,
        padding=bool(cache_indices is not None and cache_indices[index] == 0),
        phase="prefill", draft_accepted_tokens=0, draft_advanced_tokens=0)
        for index, length in enumerate(lengths))
    conv_contract, conv_effects = gdn_recurrent.state_contract(
        views=state_views, phase="prefill", state_dtype=None,
        conv_state_dtype="bf16", conv_kernel=width, target_prefix="conv_call",
        state_layout="slots_history_channels",
        initial_state=(False,) * batch, direction="conv", target_kind="conv",
        global_lengths=lengths, slot_indices=cache_indices,
        state_capacity=state_rows, slot_source=index_source)
    conv_input_port = gdn_recurrent.tensor_port(TensorValue(
        "", "bf16", TensorLayout.contiguous((tokens, dim)), output="caller:x",
        storage="caller:x", binding="external", binding_name="caller:x"),
        ready="caller:x", role="conv_input", owner="caller")
    conv_contract, conv_outputs = gdn_recurrent.bind_state_ports(
        conv_contract, "convolution", head_shards=1)
    conv_contract = {**conv_contract, "value_inputs": (conv_input_port,)}
    op = replace(op, metadata=tuple(sorted({
        **dict(op.metadata),
        "padding_slots": (),
        "state_dtype": "bf16", "state_layout": "slots_history_channels",
        "state_mutation": "inplace_history_writeback",
        "state_write_bytes": sum(
            0 if view.padding else (width - 1) * dim * 2
            for view in state_views),
        "cache_indices": cache_indices,
        "cache_index_present": cache_indices is not None,
        "cache_index_bound": cache_indices is not None,
        "state_rows": state_rows,
        "state_rows_bound": state_rows is not None,
        "state_index_source": index_source,
        "state_ready": "convolution",
        "state_value_inputs": (conv_input_port,),
        "state_value_outputs": conv_outputs,
        "gdn_state_contract": conv_contract,
        "gdn_state_effects": conv_effects,
    }.items())))
    ledger = {
        "call": "fla_npu.ops.ascendc.causal_conv1d_fn",
        "arguments": arguments,
        "device_kernels": [{"op_id": op.op_id, "component": "CausalConv1d FN runMode=0",
                            "source_revision": "a4a795857868729fbdbb0118c1e06c8655a7b31b"}],
        "wrapper_source": {**WRAPPER_SOURCE, "lines": list(CONV_WRAPPER_SOURCE_LINES)},
        "host_only": ["resolve cached stable adapter/current stream", "allocate output/workspace descriptors",
                      "increment state tensor version"],
        "outside_call": ["create x, weight, bias, qsl, cache indices and all-false flags",
                         "initialize conv state tensor", "warmup and event bookkeeping"],
        "outputs": {"out": {"shape": [tokens, dim], "dtype": "bf16"},
                    "state": {"shape": ([state_rows, width - 1, dim]
                                        if state_rows is not None else
                                        [None, width - 1, dim]),
                              "dtype": "bf16",
                              "mutation": "selected cache rows receive final history"}},
        "boundary": "device_work_of_complete_call",
    }
    return GDNCall((op,), RecipeBook({
        op.recipe: PlannedRecipe(native_causal_conv_prefill.build)}), ledger, (
        "Device-visible all-false flags retain snapshot workspace and synchronization, without initial-state payload.",
        "Host adapter/allocation/event gaps require separate submission assumptions.",
        "Source/image identity does not establish the historical compiled binary identity.",
    ))



def _validate_scan_prefill_params(params):
    lengths, batch, tokens, cu_seqlens = _normalize_partition(params, "scan_prefill")
    heads, keys = _positive(params, "v_heads"), _positive(params, "k_heads")
    dk, dv = _positive(params, "k_dim"), _positive(params, "v_dim")
    chunk = _positive(params, "chunk")
    if (chunk, dk, dv) != (GDN_CHUNK64_TOKENS, GDN_HEAD_DIM_K, GDN_CHUNK32_HEAD_DIM_V):
        raise ModelingError("gdn_scan_prefill_requires_chunk64_k128_v128")
    for name, expected in (("phase", "prefill"), ("scale", dk ** -0.5),
                           ("chunk_size", chunk), ("a_log", None), ("dt_bias", None)):
        if params.get(name, expected) != expected:
            raise ModelingError(f"gdn_scan_prefill_argument_not_selected:{name}")
    if heads % keys or heads // keys not in GDN_CHUNK64_GQA_RATIOS:
        raise ModelingError("gdn_scan_prefill_requires_integral_qk_v_head_grouping")
    for key, expected, err in (
        ("input_dtype", "bf16", "gdn_scan_prefill_requires_bf16_qkv"),
        ("gate_dtype", "fp32", "gdn_scan_prefill_requires_fp32_gates"),
        ("beta_dtype", "bf16", "gdn_scan_prefill_requires_bf16_beta_argument"),
        ("state_dtype", "fp32", "gdn_scan_prefill_requires_fp32_state"),
        ("layout", "BSND", "gdn_scan_prefill_requires_BSND_inputs"),
    ):
        if params.get(key, expected) != expected:
            raise ModelingError(err)
    for name, expected in (("use_exp2", True), ("use_qk_l2norm_in_kernel", True),
                           ("allow_neg_eigval", False), ("disable_recompute", True),
                           ("state_v_first", True), ("output_final_state", True),
                           ("use_gate_in_kernel", False),
                           ("use_beta_sigmoid_in_kernel", False),
                           ("return_intermediate_states", False),
                           ("initial_state", True)):
        if params.get(name, expected) is not expected:
            raise ModelingError(f"gdn_scan_prefill_argument_not_selected:{name}")
    return lengths, batch, tokens, cu_seqlens, heads, keys, dk, dv, chunk


def _scan_prefill(params):
    lengths, batch, tokens, cu_seqlens, heads, keys, dk, dv, chunk = _validate_scan_prefill_params(params)
    total = tokens
    chunks = sum((length + chunk - 1) // chunk for length in lengths)
    chunk_indices = []
    for sequence, length in enumerate(lengths):
        for local in range((length + chunk - 1) // chunk):
            chunk_indices.extend((sequence, local))
    builder = Builder("gdn_scan_prefill", _CALL_PLAN, 0, (), _CALL_MODEL, "prefill")
    tensors = TensorLowering(builder)
    values = {
        name: TensorValue("", dtype, TensorLayout.contiguous(shape), output="caller:" + name)
        for name, dtype, shape in (
            ("q", "bf16", (1, total, keys, dk)),
            ("k", "bf16", (1, total, keys, dk)),
            ("v", "bf16", (1, total, heads, dv)),
            ("g", "fp32", (1, total, heads)),
            ("beta", "bf16", (1, total, heads)),
        )
    }
    initial_state = {"shape": [batch, heads, dv, dk], "dtype": "fp32",
                     "layout": "V,K", "source": "outside_timed_callback"}
    state_views = tuple(SimpleNamespace(
        request_id=f"seq{index}", q_len=length, context=0, padding=False,
        phase="prefill", draft_accepted_tokens=0, draft_advanced_tokens=0)
        for index, length in enumerate(lengths))
    initial_port = gdn_recurrent.tensor_port(TensorValue(
        "", "fp32", TensorLayout.contiguous((batch, heads, dv, dk)),
        output="caller:initial_state", storage="caller:initial_state",
        binding="external", binding_name="caller:initial_state"),
        ready="caller:initial_state", role="state_input", owner="caller")
    final_port = gdn_recurrent.tensor_port(TensorValue(
        "", "fp32", TensorLayout.contiguous((batch, heads, dv, dk)),
        output="caller:final_state", storage="caller:final_state",
        binding="external", binding_name="caller:final_state"),
        ready="call_return", role="state_output", owner="caller")
    state_contract, state_effects = gdn_recurrent.state_contract(
        views=state_views, phase="prefill", state_dtype="fp32",
        conv_state_dtype="bf16", conv_kernel=SCAN_CALL_CONV_TAPS, target_prefix="scan_call",
        chunk=chunk, initial_state=(True,) * batch,
        direction="call", target_kind="call",
        global_lengths=tuple(lengths), state_capacity=batch,
        slot_source="call_external", value_inputs=(initial_port,),
        value_outputs=(final_port,), heads=heads)
    composite = gdn_composite.prefill(
        builder=builder, tensors=tensors, q=values["q"], k=values["k"], v=values["v"],
        gate=values["g"], beta=values["beta"], lengths=lengths, heads=heads,
        key_heads=keys, dk=dk, dv=dv, chunk=chunk, initial_state_dependency=None,
        external={name: name for name in values},
        state_contract=state_contract, state_effects=state_effects)
    ops = list(composite.ops)
    for index, op in enumerate(ops):
        if op.family == "gdn_state":
            meta = dict(op.metadata)
            contract = meta.get("gdn_state_contract")
            if contract is not None:
                contract, outputs = gdn_recurrent.bind_state_ports(
                    contract, op.op_id, heads=heads, head_shards=1,
                    value_outputs=(final_port,))
                meta["gdn_state_contract"] = contract
                meta["state_ready"] = op.op_id
                meta["state_value_inputs"] = (initial_port,)
                meta["state_value_outputs"] = outputs
            ops[index] = replace(op, metadata=tuple(sorted({
                **meta,
                "state_external_initial": "caller:initial_state",
                "state_external_final": "caller:final_state",
                "state_external_initial_port": initial_port,
                "state_external_final_port": final_port,
                "state_external_final_version": None}.items())))
    kernels = (
        {"op_id": composite.prepare, "component": "chunk_gated_delta_rule_fwd_prepare",
         "source_member": PREFILL_SOURCE["members"][0]},
        {"op_id": composite.state, "component": "chunk_fwd_h",
         "source_member": PREFILL_SOURCE["members"][1]},
        {"op_id": composite.output, "component": "chunk_fwd_o",
         "source_member": PREFILL_SOURCE["members"][2]},
    )
    ledger = {
        "call": "fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd",
        "arguments": {
            "lengths": list(lengths), "batch": batch, "tokens": total,
            "chunks": chunks, "cu_seqlens": cu_seqlens,
            "chunk_size": chunk,
            "chunk_indices": {"count": chunks, "entries_per_chunk": 2,
                              "values": chunk_indices,
                              "shape": [len(chunk_indices)],
                              "construction": "host list from normalized sequence lengths"},
            "layout": "BSND", "scale": dk ** -0.5, "use_exp2": True,
            "use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": False,
            "use_beta_sigmoid_in_kernel": False, "allow_neg_eigval": False,
            "disable_recompute": True, "state_v_first": True,
            "output_final_state": True,
        },
        "device_kernels": list(kernels),
        "source": PREFILL_SOURCE,
        "wrapper_source": {**WRAPPER_SOURCE, "lines": list(SCAN_WRAPPER_SOURCE_LINES)},
        "host_only": ["marshal prebuilt cu_seqlens/chunk_indices lists", "resolve cached adapter/current stream",
                      "allocate requested intermediate export slots"],
        "outside_call": ["create q/k/v/g/beta and the FP32 initial state",
                         "build cu_seqlens/chunk_indices host lists",
                         "projection, gating, convolution and cache lookup",
                         "warmup and event bookkeeping"],
        "inputs": {
            "q": {"shape": [1, total, keys, dk], "dtype": "bf16", "layout": "BSND"},
            "k": {"shape": [1, total, keys, dk], "dtype": "bf16", "layout": "BSND"},
            "v": {"shape": [1, total, heads, dv], "dtype": "bf16", "layout": "BSND"},
            "g": {"shape": [1, total, heads], "dtype": "fp32", "layout": "BSN"},
            "beta": {"shape": [1, total, heads], "dtype": "bf16", "layout": "BSN"},
            "initial_state": initial_state,
        },
        "state": {
            "initial": initial_state,
            "final": {"shape": [batch, heads, dv, dk], "dtype": "fp32", "layout": "V,K",
                      "mutation": "H writes the selected sequence/head final state"},
            "use_initial_state": True, "output_final_state": True,
            "contract": state_contract,
            "effects": list(state_effects),
            "initial_port": initial_port,
            "final_port": final_port,
        },
        "exports": {
            "q_hat": {"shape": [1, total, keys, dk], "dtype": "bf16", "layout": "BSND"},
            "k_hat": {"shape": [1, total, keys, dk], "dtype": "bf16", "layout": "BSND"},
            "q_rstd": {"shape": [1, keys, total], "dtype": "fp32", "layout": "BNS"},
            "k_rstd": {"shape": [1, keys, total], "dtype": "fp32", "layout": "BNS"},
            "A": {"shape": [1, heads, total, chunk], "dtype": "bf16", "layout": "BNSC"},
            "g_cumsum": {"shape": [1, total, heads], "dtype": "fp32", "layout": "BSN"},
            "note": "the public wrapper returns these even when Python ignores them",
        },
        "materializations": [
            {"op_id": op.op_id, "reason": dict(op.metadata).get("materialization_reason"),
             "source_input": dict(op.metadata).get("source_input"),
             "source_output": dict(op.metadata).get("source_output"),
             "target_output": dict(op.metadata).get("target_output"),
             "source_stream": dict(op.metadata).get("source_stream"),
             "read_bytes": op.mte2_bytes, "write_bytes": op.mte3_bytes}
            for op in ops if op.family == "tensor_copy"],
        "outputs": {"o": {"shape": [1, total, heads, dv], "dtype": "bf16", "layout": "BSND"},
                    "final_state": {"shape": [batch, heads, dv, dk], "dtype": "fp32",
                                    "layout": "V,K"}},
        "boundary": "device_work_of_complete_call",
    }
    return GDNCall(tuple(ops), RecipeBook(gdn_composite.recipes()), ledger, (
        "Includes caller Q/K/V/g/beta materializations, Prepare/H/O and requested exports.",
        "Declared transpose/copy work does not assert the compiled dispatch count or absence of fusion.",
        "Excludes serving projections, gating, convolution, cache lookup and input initialization.",
        "Host launch/allocation/event gaps require separate submission assumptions.",
    ))


_BUILDERS = {"gating": _gating, "conv_prefill": _conv_prefill, "scan_prefill": _scan_prefill}


def make_call(role, params):
    """Build supported calls without reading historical files or measurements."""
    if role not in _BUILDERS:
        raise ModelingError(f"gdn_call_branch_not_implemented:{role}")
    return _BUILDERS[role](params)
