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
from . import gdn_composite
from .lowering import Builder
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .operators import native_causal_conv_prefill
from .operators.mixed_core import read_role
from .program import ParallelProgram, PipelineProgram, RecipeBook, Stage
from .tensors import TensorLowering


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
_CALL_PLAN = SimpleNamespace(index=0)
_CALL_MODEL = SimpleNamespace(activation_dtype="bf16", compute_dtype="bf16", quant={})


@dataclass(frozen=True)
class GDNCall:
    ops: tuple[OpSpec, ...]
    recipes: RecipeBook
    ledger: dict
    limitations: tuple[str, ...]


def _positive(params, name):
    value = params[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ModelingError(f"gdn_call_requires_positive_integer:{name}")
    return value


def _gating_plan(op, geometry):
    rows, heads = op.local_shape
    units = geometry.resource_units.get("vector", geometry.compute_units)
    row_iterations = math.ceil(rows / (64 * units))
    col_iterations = math.ceil(heads / 8)
    reads = 8 * rows * heads + 8 * units * row_iterations * heads
    writes = 8 * rows * heads
    # One elementary lane-op envelope: x/add, beta*x, compare, exp/add/log/
    # scale/select, g product, sigmoid neg/exp/add/div. A_log exp/neg is
    # column-local. Masks suppress memory, not these vector expressions.
    per_tile_ops = 13 * 64 * 8 + 2 * 8
    vector = units * row_iterations * col_iterations * per_tile_ops
    read = read_role(geometry.resource_units, 0)
    work = Work(resources={"hbm": reads + writes, read: reads,
                           "mte3": writes, "vector": vector})

    def program(context):
        def tile(index, unit):
            row_index, col_index = divmod(index, col_iterations)
            first = (unit * row_iterations + row_index) * 64
            valid_rows = max(0, min(64, rows - first))
            valid_cols = min(8, heads - col_index * 8)
            count = valid_rows * valid_cols
            incoming = count * 8 + valid_cols * 8
            owned = context.owned(unit, units, ("vector", read, "mte3"))
            return (
                Stage("load", context.bind(Work(resources={"hbm": incoming, read: incoming})),
                      placement=owned),
                Stage("elementwise", context.bind(Work(resources={"vector": per_tile_ops})),
                      after=("load",), placement=owned),
                Stage("store_g", context.bind(Work(resources={"hbm": count * 4, "mte3": count * 4})),
                      after=("elementwise",), placement=owned),
                Stage("store_beta", context.bind(Work(resources={"hbm": count * 4, "mte3": count * 4})),
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
    if params.get("beta", 1.0) != 1.0 or params.get("threshold", 20.0) != 20.0:
        raise ModelingError("gdn_gating_call_requires_default_softplus_arguments")
    op = OpSpec(
        "gating", "gdn_gating", params.get("phase", "prefill"), 0, 0, (),
        (rows, heads), (rows, heads), (rows, heads), (rows, heads),
        dtype="fp32", layout="contiguous", recipe=GATING_RECIPE,
        source_boundary="kernel",
        metadata=(("implementation", GATING_RECIPE), ("source", GATING_SOURCE),
                  ("output_dtypes", {"g": "fp32", "beta": "fp32"})),
    )
    ledger = {
        "call": "vllm_ascend.ops.triton.fused_gdn_gating.fused_gdn_gating_patch",
        "arguments": {"rows": rows, "heads": heads, "a_dtype": "fp32", "b_dtype": "fp32",
                      "a_log_dtype": "fp32", "dt_bias_dtype": "fp32", "layout": "contiguous",
                      "beta": 1.0, "threshold": 20.0},
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
    batch, tokens = _positive(params, "batch"), _positive(params, "tokens")
    dim, width = _positive(params, "conv_width"), _positive(params, "kernel")
    if params.get("input_dtype", "bf16") != "bf16":
        raise ModelingError("gdn_conv_call_requires_bf16_arguments")
    if params.get("layout", "contiguous") != "contiguous":
        raise ModelingError("gdn_conv_call_requires_contiguous_arguments")
    if params.get("activation", "silu") != "silu" or params.get("bias", True) is not True:
        raise ModelingError("gdn_conv_call_requires_silu_and_bias")
    if params.get("initial_state", False) is not False:
        raise ModelingError("gdn_conv_call_requires_no_initial_state")
    arguments = {
        "lengths": (tokens,) * batch, "dim": dim, "width": width, "dtype": "bf16",
        "activation": "silu", "bias": True, "metadata_placement": "device",
        "has_initial_state": (False,) * batch, "cache_indices": tuple(range(1, batch + 1)),
        "state_rows": batch + 1, "state_history_slots": width - 1,
        "null_block_id": 0, "pad_slot_id": -1,
    }
    op = replace(native_causal_conv_prefill.make_op(arguments), op_id="convolution")
    ledger = {
        "call": "fla_npu.ops.ascendc.causal_conv1d_fn",
        "arguments": arguments,
        "device_kernels": [{"op_id": op.op_id, "component": "CausalConv1d FN runMode=0",
                            "source_revision": "a4a795857868729fbdbb0118c1e06c8655a7b31b"}],
        "wrapper_source": {**WRAPPER_SOURCE, "lines": [1018, 1078]},
        "host_only": ["resolve cached stable adapter/current stream", "allocate output/workspace descriptors",
                      "increment state tensor version"],
        "outside_call": ["create x, weight, bias, qsl, cache indices and all-false flags",
                         "initialize conv state tensor", "warmup and event bookkeeping"],
        "outputs": {"out": {"shape": [batch * tokens, dim], "dtype": "bf16"},
                    "state": {"shape": [batch + 1, width - 1, dim], "dtype": "bf16",
                              "mutation": "selected cache rows receive final history"}},
        "boundary": "device_work_of_complete_call",
    }
    return GDNCall((op,), RecipeBook({
        op.recipe: PlannedRecipe(native_causal_conv_prefill.build)}), ledger, (
        "Device-visible all-false flags retain snapshot workspace and synchronization, without initial-state payload.",
        "Host adapter/allocation/event gaps require separate submission assumptions.",
        "Source/image identity does not establish the historical compiled binary identity.",
    ))



def _scan_prefill(params):
    batch, tokens = _positive(params, "batch"), _positive(params, "tokens")
    heads, keys = _positive(params, "v_heads"), _positive(params, "k_heads")
    dk, dv = _positive(params, "k_dim"), _positive(params, "v_dim")
    chunk = _positive(params, "chunk")
    if (chunk, dk, dv) != (64, 128, 128):
        raise ModelingError("gdn_scan_prefill_requires_chunk64_k128_v128")
    for name, expected in (("phase", "prefill"), ("scale", dk ** -0.5),
                           ("chunk_size", chunk), ("a_log", None), ("dt_bias", None)):
        if params.get(name, expected) != expected:
            raise ModelingError(f"gdn_scan_prefill_argument_not_selected:{name}")
    if heads % keys or heads // keys not in (1, 2, 3, 4):
        raise ModelingError("gdn_scan_prefill_requires_integral_qk_v_head_grouping")
    if params.get("input_dtype", "bf16") != "bf16":
        raise ModelingError("gdn_scan_prefill_requires_bf16_qkv")
    if params.get("gate_dtype", "fp32") != "fp32":
        raise ModelingError("gdn_scan_prefill_requires_fp32_gates")
    if params.get("beta_dtype", "bf16") != "bf16":
        raise ModelingError("gdn_scan_prefill_requires_bf16_beta_argument")
    if params.get("state_dtype", "fp32") != "fp32":
        raise ModelingError("gdn_scan_prefill_requires_fp32_state")
    if params.get("layout", "BSND") != "BSND":
        raise ModelingError("gdn_scan_prefill_requires_BSND_inputs")
    for name, expected in (("use_exp2", True), ("use_qk_l2norm_in_kernel", True),
                           ("allow_neg_eigval", False), ("disable_recompute", True),
                           ("state_v_first", True), ("output_final_state", True),
                           ("use_gate_in_kernel", False),
                           ("use_beta_sigmoid_in_kernel", False),
                           ("return_intermediate_states", False),
                           ("initial_state", True)):
        if params.get(name, expected) is not expected:
            raise ModelingError(f"gdn_scan_prefill_argument_not_selected:{name}")
    lengths = (tokens,) * batch
    total = batch * tokens
    chunks = sum((length + chunk - 1) // chunk for length in lengths)
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
    composite = gdn_composite.prefill(
        builder=builder, tensors=tensors, q=values["q"], k=values["k"], v=values["v"],
        gate=values["g"], beta=values["beta"], lengths=lengths, heads=heads,
        key_heads=keys, dk=dk, dv=dv, chunk=chunk, initial_state_dependency=None,
        external={name: name for name in values})
    ops = composite.ops
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
            "lengths": list(lengths), "batch": batch, "tokens": tokens,
            "chunks": chunks, "cu_seqlens": [tokens * index for index in range(batch + 1)],
            "chunk_size": chunk,
            "chunk_indices": {"count": chunks, "entries_per_chunk": 2,
                              "construction": "host list from canonical equal-length chunks"},
            "layout": "BSND", "scale": dk ** -0.5, "use_exp2": True,
            "use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": False,
            "use_beta_sigmoid_in_kernel": False, "allow_neg_eigval": False,
            "disable_recompute": True, "state_v_first": True,
            "output_final_state": True,
        },
        "device_kernels": list(kernels),
        "source": PREFILL_SOURCE,
        "wrapper_source": {**WRAPPER_SOURCE, "lines": [1472, 1510]},
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
    return GDNCall(ops, RecipeBook(gdn_composite.recipes()), ledger, (
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
