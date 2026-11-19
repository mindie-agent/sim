"""Arch35 GDN ``Chunk32`` (PR #22147) Prepare, State, and Output source planners.

[Role]
  Model the ``fla-npu`` ``chunk_gated_delta_rule_fwd`` ``BT=32`` path (``PR #22147``,
  ``use_exp2=False``, ``use_beta_sigmoid_in_kernel=True``) across all three
  mixed AIC+2AIV kernels (Prepare/WY, ChunkFwdH, ChunkFwdO), admitting arbitrary
  integral GQA head ratios ``Hv % Hk == 0`` (including ``Hv/Hk = 8``).

[Invariants]
  * ``BT=32`` fits the ``32x32`` unit-triangular WY inverse in a single UB VCS
    tile, eliminating the ``Chunk64`` ``4x 64^3`` FP32 Cube block products and
    ``y_dump``/``y_reload`` GM workspace round-trip.
  * Every ``OperatorPlan`` strictly conserves ``Work`` across ``hbm``,
    ``mte2_aic``, ``mte2_aiv``, ``mte1``, ``cube``, ``vector``, ``fixpipe``, and
    ``mte3`` via ``mixed_core`` ownership and dynamic ``geometry.buffers`` checks.
  * Zero model-name branches or ``ResourceEnvelope`` fallbacks participate.

[Timeouts & Locks]
  O(sequences + cores) plan construction and bounded per-core pipelines; no I/O
  or locks.

[Side-Effect Policy]
  Pure immutable ``OpSpec`` and ``OperatorPlan`` construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from .gdn_domain import (
    FLA_GDN_CHUNK_SOURCE,
    GDN_CHUNK32_HEAD_DIM_V,
    GDN_CHUNK32_TOKENS,
    GDN_HEAD_DIM_K,
    can_run_gdn_chunk32,
)
from .mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from .operators.gdn_prepare import PackedChunks
from .operators.mixed_core import phases, require_pair, resident, split_reads
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage

PREPARE_RECIPE = "gdn_prepare_arch35_chunk32"
STATE_RECIPE = "gdn_state_arch35_chunk32"
OUTPUT_RECIPE = "gdn_output_arch35_chunk32"

PR22147_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "57de212f6a",
    "relative_path": "vllm_ascend/ops/gdn.py",
    "source_sha256": "b170b30626cf46bc4ec1e47080f9487c1e5443e51add42bb8c67bfd20a79fac4",
}

BF16_BYTES = 2
FP32_BYTES = 4
MAX_TASK_GROUP_HEADS = 4
PIPELINE_WINDOW_SLOTS = 2
L2NORM_OPS_PER_ELEM = 5
SIGMOID_SCALE_OPS_PER_ELEM = 6
GATE_CUMSUM_OPS_PER_ROW = 4
CAUSAL_DECAY_OPS_PER_PAIR = 7
VCS32_TRIANGULAR_OPS = 66464
STATE_DECAY_OPS_PER_ELEM = 4
OUTPUT_FUSE_OPS_PER_ELEM = 5
CHUNK32_BUFFERS = {
    "ub": 160 * 1024, "l1": 192 * 1024, "l0a": 16 * 1024,
    "l0b": 32 * 1024, "l0c": 32 * 1024,
}


def _check_buffers(geometry: Geometry) -> dict[str, int]:
    available = dict(geometry.buffers or {})
    for pool, required in CHUNK32_BUFFERS.items():
        cap = int(available.get(pool, 0))
        if cap > 0 and required > cap:
            raise ModelingError(f"gdn_chunk32_buffer_capacity_exceeded:{pool}")
    return {p: sz for p, sz in CHUNK32_BUFFERS.items() if available.get(p, 0) > 0}


def _validate_chunk32_params(
    params: Mapping[str, Any],
) -> tuple[tuple[int, ...], int, int, int, int]:
    lengths = tuple(int(v) for v in (params.get("lengths") or ()))
    if not lengths or min(lengths) < 1:
        raise ValueError("gdn_chunk32_requires_positive_lengths")
    heads, keys = int(params.get("heads", 0)), int(params.get("key_heads", 0))
    dk = int(params.get("dk", GDN_HEAD_DIM_K))
    dv = int(params.get("dv", GDN_CHUNK32_HEAD_DIM_V))
    chunk = int(params.get("chunk", GDN_CHUNK32_TOKENS))
    dtype = str(params.get("dtype", "bf16")).lower()
    if chunk != GDN_CHUNK32_TOKENS:
        raise ValueError("gdn_chunk32_requires_chunk_32")
    if not can_run_gdn_chunk32(keys, heads, dk, dv, activation_dtype=dtype, state_dtype="fp32"):
        raise ValueError("gdn_chunk32_domain_unsupported")
    return lengths, heads, keys, dk, dv


def _build_mixed_plan(
    geometry: Geometry,
    tasks: tuple[Any, ...],
    work_fn: Callable[[Any], tuple[Work, Work]],
    *,
    aiv_first: bool,
    slots: int,
    first_label: str,
    second_label: str,
    extra_meta: Mapping[str, Any],
) -> OperatorPlan:
    require_pair(geometry)
    buf_sizes = _check_buffers(geometry)
    cores = max(1, int(geometry.compute_units))
    owners: list[list[Any]] = [[] for _ in range(cores)]
    for idx, item in enumerate(tasks):
        owners[idx % cores].append(item)

    total_work = Work()
    for items in owners:
        for item in items:
            aiv_w, aic_w = work_fn(item)
            total_work += split_reads(aiv_w, 0, geometry.resource_units)
            total_work += split_reads(aic_w, aic_w.resources["mte2"], geometry.resource_units)

    def program(context):
        def per_core(core: int, items: list[Any]):
            if not items:
                return KernelProgram(())

            def tile(idx: int):
                aiv_w, aic_w = work_fn(items[idx])
                if aiv_first:
                    return (
                        phases(first_label, aiv_w, context, core, aiv=idx % 2),
                        phases(second_label, aic_w, context, core, aiv=None, after=(first_label,)),
                    )
                return (
                    phases(first_label, aic_w, context, core, aiv=None),
                    phases(second_label, aiv_w, context, core, aiv=idx % 2, after=(first_label,)),
                )

            body = KernelProgram((Stage("tiles", program=PipelineProgram(
                len(items), slots, tile)),))
            return resident(context, core, buf_sizes, body)

        return ParallelProgram(tuple(per_core(c, items) for c, items in enumerate(owners)))

    return OperatorPlan(total_work, program, {
        "execution_class": "mixed_aic_aiv",
        "chunk_size": GDN_CHUNK32_TOKENS,
        "source_program": FLA_GDN_CHUNK_SOURCE,
        **dict(extra_meta),
    })


def make_prepare_op(params: Mapping[str, Any]) -> OpSpec:
    lengths, heads, keys, dk, dv = _validate_chunk32_params(params)
    chunks = PackedChunks.from_lengths(lengths, GDN_CHUNK32_TOKENS)
    tokens, gqa = sum(lengths), heads // keys
    gate_dtype = str(params.get("gate_dtype", "fp32")).lower()
    beta_dtype = str(params.get("beta_dtype", "bf16")).lower()
    output_a = bool(params.get("output_a", True))
    use_beta_sig = bool(params.get("use_beta_sigmoid_in_kernel", True))
    semantic = (1, heads, tokens, dk)
    padded = (1, heads, chunks.padded_tokens, dk)
    meta = {
        "implementation": PREPARE_RECIPE, "lengths": lengths,
        "heads": heads, "key_heads": keys, "head_dim_k": dk, "head_dim_v": dv,
        "chunk": GDN_CHUNK32_TOKENS, "gqa_ratio": gqa, "dtype": "bf16",
        "gate_dtype": gate_dtype, "beta_dtype": beta_dtype,
        "use_qk_l2norm_in_kernel": True, "use_beta_sigmoid_in_kernel": use_beta_sig,
        "use_exp2": False, "output_a": output_a,
        "native_source": FLA_GDN_CHUNK_SOURCE, "pr_source": PR22147_SOURCE,
    }
    return OpSpec(
        op_id=f"gdn-prepare-c32:h{heads}:hk{keys}:c{chunks.total}",
        family="gdn_prepare", phase="prefill", stage=0, layer=0, requests=(),
        semantic_shape=semantic, local_shape=semantic,
        padded_shape=padded, launch_shape=padded,
        dtype="bf16", recipe=PREPARE_RECIPE, source_boundary="kernel",
        metadata=tuple(sorted(meta.items())),
    )


def build_prepare(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    meta = dict(op.metadata)
    lengths = tuple(int(v) for v in meta["lengths"])
    heads, keys = int(meta["heads"]), int(meta["key_heads"])
    dk, dv, gqa = int(meta["head_dim_k"]), int(meta["head_dim_v"]), heads // keys
    bt = GDN_CHUNK32_TOKENS
    chunks = PackedChunks.from_lengths(lengths, bt)
    gate_bytes = FP32_BYTES if meta.get("gate_dtype", "fp32") == "fp32" else BF16_BYTES
    beta_bytes = FP32_BYTES if meta.get("beta_dtype", "bf16") == "fp32" else BF16_BYTES
    output_a = bool(meta.get("output_a", True))

    def group_work(group_id: int) -> tuple[Work, Work]:
        _, _, _, rows = chunks.range(group_id // keys)
        qk_bytes = 2 * rows * dk * BF16_BYTES
        v_bytes = gqa * rows * dv * BF16_BYTES
        gb_bytes = gqa * rows * (gate_bytes + beta_bytes)
        aiv_read = float(qk_bytes + v_bytes + gb_bytes)
        a_write = gqa * rows * bt * BF16_BYTES if output_a else 0
        aiv_write = float(qk_bytes + 2 * rows * FP32_BYTES + gqa * rows * FP32_BYTES + a_write)
        aiv_vec = float(
            2 * rows * dk * L2NORM_OPS_PER_ELEM
            + gqa * rows * (dk + dv) * SIGMOID_SCALE_OPS_PER_ELEM
            + gqa * (bt * GATE_CUMSUM_OPS_PER_ROW + bt * bt * CAUSAL_DECAY_OPS_PER_PAIR
                     + VCS32_TRIANGULAR_OPS)
        )
        aiv = Work(resources={
            "hbm": aiv_read + aiv_write, "mte2": aiv_read,
            "vector": aiv_vec, "mte3": aiv_write,
        })
        aic_read = float(rows * dk * BF16_BYTES + gqa * rows * (dk + dv) * BF16_BYTES)
        wu_write = float(gqa * rows * (dk + dv) * BF16_BYTES)
        l0_bytes = float((2 * bt * dk + gqa * 2 * (bt * bt + bt * dk + bt * dv)) * BF16_BYTES)
        flops = float(2 * bt * bt * dk + gqa * (2 * bt * dk * bt + 2 * bt * dv * bt))
        aic = Work(
            matrix={"bf16": flops},
            resources={"hbm": aic_read + wu_write, "mte2": aic_read,
                       "mte1": l0_bytes, "fixpipe": wu_write},
        )
        return aiv, aic

    return _build_mixed_plan(
        geometry, tuple(range(chunks.total * keys)), group_work,
        aiv_first=True, slots=PIPELINE_WINDOW_SLOTS,
        first_label="aiv_vcs32", second_label="aic_wy32",
        extra_meta={"gqa_ratio": gqa},
    )


def make_state_op(params: Mapping[str, Any]) -> OpSpec:
    lengths, heads, keys, dk, dv = _validate_chunk32_params(params)
    chunks = PackedChunks.from_lengths(lengths, GDN_CHUNK32_TOKENS)
    use_init = bool(params.get("use_initial_state", True))
    out_final = bool(params.get("output_final_state", True))
    state_v_first = bool(params.get("state_v_first", True))
    shape = (len(lengths), heads, dv, dk) if state_v_first else (len(lengths), heads, dk, dv)
    meta = {
        "implementation": STATE_RECIPE, "lengths": lengths, "heads": heads,
        "key_heads": keys, "dk": dk, "dv": dv, "chunk": GDN_CHUNK32_TOKENS,
        "use_initial_state": use_init, "output_final_state": out_final,
        "state_v_first": state_v_first, "use_exp2": False,
        "native_source": FLA_GDN_CHUNK_SOURCE,
    }
    return OpSpec(
        op_id=f"gdn-state-c32:h{heads}:c{chunks.total}",
        family="gdn_state", phase="prefill", stage=0, layer=0, requests=(),
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="fp32", recipe=STATE_RECIPE, source_boundary="kernel",
        metadata=tuple(sorted(meta.items())),
    )


def build_state(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    meta = dict(op.metadata)
    lengths = tuple(int(v) for v in meta["lengths"])
    heads, dk, dv = int(meta["heads"]), int(meta["dk"]), int(meta["dv"])
    bt = GDN_CHUNK32_TOKENS
    use_init = bool(meta.get("use_initial_state", True))
    out_final = bool(meta.get("output_final_state", True))

    def task_work(seq_idx: int) -> tuple[Work, Work]:
        seq_len = lengths[seq_idx]
        n_chunks = (seq_len + bt - 1) // bt
        state_fp32 = float(dk * dv * FP32_BYTES)
        state_bf16 = float(dk * dv * BF16_BYTES)
        aiv_read = (state_fp32 if use_init else 0.0) + float(seq_len * (dv * BF16_BYTES + FP32_BYTES))
        aiv_write = (state_fp32 if out_final else 0.0) + n_chunks * state_bf16 + float(seq_len * dv * BF16_BYTES)
        aiv_vec = float((n_chunks * dk * dv + seq_len * dv) * STATE_DECAY_OPS_PER_ELEM)
        aiv = Work(resources={
            "hbm": aiv_read + aiv_write, "mte2": aiv_read,
            "vector": aiv_vec, "mte3": aiv_write,
        })
        aic_read = float(seq_len * (2 * dk + dv) * BF16_BYTES + n_chunks * state_bf16)
        l0_bytes = float(n_chunks * 2 * (bt * dk + dk * dv + bt * dv) * BF16_BYTES)
        flops = float(n_chunks * 2 * (2 * bt * dv * dk))
        aic = Work(matrix={"bf16": flops},
                   resources={"hbm": aic_read, "mte2": aic_read, "mte1": l0_bytes})
        return aiv, aic

    tasks = tuple(seq_idx for seq_idx in range(len(lengths)) for _ in range(heads))
    return _build_mixed_plan(
        geometry, tasks, task_work,
        aiv_first=False, slots=1,
        first_label="state_aic", second_label="state_aiv", extra_meta={},
    )


def make_output_op(params: Mapping[str, Any]) -> OpSpec:
    lengths, heads, keys, dk, dv = _validate_chunk32_params(params)
    group = min(MAX_TASK_GROUP_HEADS, heads // keys)
    shape = (1, sum(lengths), heads, dv)
    meta = {
        "implementation": OUTPUT_RECIPE, "lengths": lengths, "heads": heads,
        "key_heads": keys, "dk": dk, "dv": dv, "chunk": GDN_CHUNK32_TOKENS,
        "task_group_size": group, "use_exp2": False,
        "state_v_first": bool(params.get("state_v_first", True)),
        "native_source": FLA_GDN_CHUNK_SOURCE,
    }
    return OpSpec(
        op_id="gdn_output_c32", family="gdn_output", phase="prefill",
        stage=0, layer=0, requests=(), dtype="bf16", layout="BSND",
        recipe=OUTPUT_RECIPE, source_boundary="kernel",
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        metadata=tuple(sorted(meta.items())),
    )


def build_output(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    meta = dict(op.metadata)
    lengths = tuple(int(v) for v in meta["lengths"])
    heads, keys = int(meta["heads"]), int(meta["key_heads"])
    dk, dv, group = int(meta["dk"]), int(meta["dv"]), int(meta["task_group_size"])
    bt = GDN_CHUNK32_TOKENS
    chunks = PackedChunks.from_lengths(lengths, bt)
    groups_per_key = (heads // keys + group - 1) // group
    total_groups = chunks.total * keys * groups_per_key

    def group_work(gid: int) -> tuple[Work, Work]:
        chunk_idx = gid // (keys * groups_per_key)
        sub_idx = (gid % (keys * groups_per_key)) % groups_per_key
        active_h = min(group, heads // keys - sub_idx * group)
        _, _, _, rows = chunks.range(chunk_idx)
        aic_read = float((2 * rows * dk + active_h * (dk * dv + rows * dv)) * BF16_BYTES)
        l0_bytes = float(active_h * (2 * bt * dk + dk * dv + bt * bt + bt * dv) * BF16_BYTES)
        flops = float(active_h * (2 * bt * bt * dk + 2 * bt * dv * dk + 2 * rows * dv * bt))
        aic = Work(matrix={"bf16": flops},
                   resources={"hbm": aic_read, "mte2": aic_read, "mte1": l0_bytes})
        aiv_read = float(active_h * rows * FP32_BYTES)
        aiv_write = float(active_h * rows * dv * BF16_BYTES)
        aiv_vec = float(active_h * (
            bt * bt * CAUSAL_DECAY_OPS_PER_PAIR + rows * dv * OUTPUT_FUSE_OPS_PER_ELEM
        ))
        aiv = Work(resources={
            "hbm": aiv_read + aiv_write, "mte2": aiv_read,
            "vector": aiv_vec, "mte3": aiv_write,
        })
        return aiv, aic

    return _build_mixed_plan(
        geometry, tuple(range(total_groups)), group_work,
        aiv_first=False, slots=PIPELINE_WINDOW_SLOTS,
        first_label="out_aic", second_label="out_aiv",
        extra_meta={"task_group_size": group},
    )


@dataclass(frozen=True)
class _KernelNamespace:
    make_op: Any
    build: Any


PREPARE = _KernelNamespace(make_prepare_op, build_prepare)
STATE = _KernelNamespace(make_state_op, build_state)
OUTPUT = _KernelNamespace(make_output_op, build_output)


def recipes() -> dict[str, PlannedRecipe]:
    """Registered ``PlannedRecipe``s for the three ``Chunk32`` GDN kernels."""
    return {
        PREPARE_RECIPE: PlannedRecipe(build_prepare),
        STATE_RECIPE: PlannedRecipe(build_state),
        OUTPUT_RECIPE: PlannedRecipe(build_output),
    }
