"""Token-first arch35 GDN output with direct A-prime UB-to-L1 transfer.

[Role] Model the ChunkFwdOA5DispatchByGateType source branch, not Catlass BNSD.
[Invariants] QK/QH use 64-row tiles; AV passes the actual tail to MMAD. Q/K are loaded
  once per key head within a task group. Each value head still executes QK.
  A-prime never visits GM; FP32 intermediates remain in the owning AIV's UB.
[Timeouts & Locks] O(cores * task_group_size) live stages; lazy token chunks.
[Side-Effect Policy] Pure plans on the caller's engine, no device or file I/O.
"""
from dataclasses import dataclass, replace
import math

from ..mechanism import OperatorPlan, Work
from ..program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from ...schema.operators import OpSpec
from .gdn_prepare import PackedChunks
from .mixed_core import phase_stages, require_pair, resident, split_reads

IMPLEMENTATION = "gdn_output_arch35_token_first"
BT, K, V = 64, 128, 128


@dataclass(frozen=True)
class OutputGeometry:
    lengths: tuple[int, ...]
    heads: int
    key_heads: int
    task_group_size: int
    use_exp2: bool
    state_v_first: bool
    gate_dtype: str


def make_op(params):
    lengths = tuple(params["lengths"])
    heads, keys = int(params["heads"]), int(params["key_heads"])
    if not lengths or min(lengths) < 1 or keys < 1 or heads % keys or heads < keys:
        raise ValueError("gdn_output_requires_positive_lengths_and_integral_head_ratio")
    # Host A5ChunkTiling{Fix,Var}Len uses complete G=3 groups.
    group = 3 if heads // keys == 3 else 4
    if heads // keys > 4:
        raise ValueError("gdn_output_selected_source_requires_head_ratio_at_most_four")
    if params.get("task_group_size", group) != group:
        raise ValueError("gdn_output_task_group_disagrees_with_selected_host_tiling")
    if (params.get("dk", K), params.get("dv", V), params.get("chunk", BT)) != (K, V, BT):
        raise ValueError("gdn_output_selected_source_requires_64_128_128")
    if params.get("output_token_first", True) is not True:
        raise ValueError("gdn_output_requires_token_first_source_branch")
    if params.get("dtype", "bf16") != "bf16":
        raise ValueError("gdn_output_requires_bf16_inputs")
    gate_dtype = params.get("gate_dtype", "fp32")
    if gate_dtype not in {"bf16", "fp32"}:
        raise ValueError("gdn_output_requires_float_gate")
    config = OutputGeometry(lengths, heads, keys, group, bool(params.get("use_exp2", True)),
                            bool(params.get("state_v_first", True)), gate_dtype)
    dispatch = {"implementation": IMPLEMENTATION, "output_token_first": True,
                "use_exp2": config.use_exp2, "state_v_first": config.state_v_first,
                "gate_dtype": gate_dtype, "task_group_size": group}
    shape = (1, sum(lengths), heads, V)
    requested = heads * sum(
        (length // BT) * (2 * BT * BT * K + 2 * BT * V * K + 2 * BT * V * BT)
        + (2 * BT * BT * K + 2 * BT * V * K + 2 * (length % BT) ** 2 * V
           if length % BT else 0)
        for length in lengths)
    return OpSpec(
        op_id="gdn_output", family="gdn_output", phase="prefill", stage=0, layer=0,
        requests=(), dtype="bf16", layout="BSND", recipe=IMPLEMENTATION, source_boundary="kernel",
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        metadata=tuple(sorted({
            **config.__dict__, "dispatch": dispatch, "a_prime_transfer": "ub_to_l1",
            "h_semantic_strides": (1, K) if config.state_v_first else (V, 1),
            "source_kernel_launches": 1,
            "requested_matrix_flops": requested,
            "matrix_instruction_tile": (16, 16, 16),
        }.items())),
    )


def output_offset(token, head, channel, heads):
    """ChunkFwdOOOffset and strided MTE3 output store, in BF16 elements."""
    return (token * heads + head) * V + channel


def _work(cfg, rows, first_key):
    gate_bytes = 4 if cfg.gate_dtype == "fp32" else 2
    # Element-operation ledger: arithmetic/conversions and explicit vector
    # mask/gather/fill instructions; load/store service is accounted separately.
    gate_ops = BT * (1 + cfg.use_exp2) + BT * BT * (2 + cfg.use_exp2)
    gate_ops += rows if cfg.gate_dtype == "bf16" else 0
    gate_ops += 2 * BT if rows < BT else 0
    qkh = (2 * rows * K * 2 if first_key else 0) + K * V * 2
    stage3_ops = 2 * BT + rows * (3 * BT + 2 * BT + V)
    stage3_ops += (BT - rows) * (BT + V)
    # MMAD receives the valid extent; hardware still issues 16x16x16 BF16
    # fractals (3510 instruction performance table), not a fixed 64-row tail.
    av_rows = (rows + 15) // 16 * 16
    return (
        Work(resources={"hbm": rows * gate_bytes, "mte2": rows * gate_bytes,
                        "vector": gate_ops}),
        Work(matrix={"bf16": 2 * BT * BT * K + 2 * BT * V * K},
             resources={"hbm": qkh, "mte2": qkh,
                        "mte1": (3 * BT * K + K * V) * 2,
                        "fixpipe": (BT * BT + BT * V) * 4}),
        Work(resources={"vector": stage3_ops, "mte3": BT * BT * 2}),
        Work(matrix={"bf16": 2 * av_rows * V * av_rows},
             resources={"hbm": rows * V * 2, "mte2": rows * V * 2,
                        "mte1": (av_rows * av_rows + av_rows * V) * 2,
                        "fixpipe": rows * V * 4}),
        Work(resources={"vector": 3 * rows * V,
                        "hbm": rows * V * 2, "mte3": rows * V * 2}),
    )


def _group_heads(cfg, group):
    start = group * cfg.task_group_size
    return range(start, min(start + cfg.task_group_size, cfg.heads))


def _group_work(cfg, group, rows):
    seen = set()
    result = []
    for head in _group_heads(cfg, group):
        key = head // (cfg.heads // cfg.key_heads)
        result.append(_work(cfg, rows, key not in seen))
        seen.add(key)
    return result


def build(op, geometry):
    require_pair(geometry)
    meta = dict(op.metadata)
    cfg = OutputGeometry(**{name: meta[name] for name in OutputGeometry.__dataclass_fields__})
    chunks = PackedChunks.from_lengths(cfg.lengths)
    groups = math.ceil(cfg.heads / cfg.task_group_size)
    work, aic_reads = Work(), 0
    for length in cfg.lengths:
        full, tail = divmod(length, BT)
        for rows, count in ((BT, full), (tail, bool(tail))):
            if count:
                for group in range(groups):
                    for head in _group_work(cfg, group, rows):
                        aic_reads += count * (head[1].resources["mte2"] + head[3].resources["mte2"])
                        for stage in head:
                            work = work + stage.scaled(count)
    work = split_reads(work, aic_reads, geometry.resource_units)

    def program(context):
        def core_program(core):
            def tile(index):
                chunk, group = divmod(core + index * context.core_count, groups)
                rows = chunks.range(chunk)[3]
                heads = _group_work(cfg, group, rows)
                stages, vector_tail, cube_tail = [], {}, None
                def emit(name, value, deps=(), aiv=None, previous=(),
                         l0_after=(), compute_after=(), publish_after=(), store_after=()):
                    stages.extend(phase_stages(
                        name, value, context, core, aiv=aiv, after=deps, previous=previous,
                        l0_after=l0_after, compute_after=compute_after, publish_after=publish_after,
                        store_after=store_after))
                    return name
                for head, values in enumerate(heads):
                    aiv = head % 2
                    prior = vector_tail.get(aiv)
                    vector_tail[aiv] = emit(f"gate{head}", values[0],
                                           (prior,) if prior else (), aiv,
                                           () if prior else (f"vector_done{aiv}",))
                for head, values in enumerate(heads):
                    qk_read = values[1].resources["hbm"] - K * V * 2
                    loaded = emit(f"qk_load{head}", Work(resources={"hbm": qk_read, "mte2": qk_read}),
                                  (f"h_load{head - 1}",) if head else (),
                                  previous=("cube_done",) if not head else ())
                    h_load = emit(f"h_load{head}",
                                  Work(resources={"hbm": K * V * 2, "mte2": K * V * 2}), (loaded,))
                    qk = emit(f"qk{head}", Work(
                        matrix={"bf16": 2 * BT * BT * K},
                        resources={"mte1": 2 * BT * K * 2, "fixpipe": BT * BT * 4}),
                        (loaded,),
                        previous=("vector_done0", "vector_done1") if not head else (),
                        l0_after=(f"qh{head - 1}:l0", f"qk{head - 1}:compute") if head else (),
                        compute_after=(f"qh{head - 1}:compute", f"qk{head - 1}") if head else (),
                        publish_after=(f"qh{head - 1}:publish",) if head else ())
                    qh = emit(f"qh{head}", Work(
                        matrix={"bf16": 2 * BT * V * K},
                        resources={"mte1": (BT * K + K * V) * 2, "fixpipe": BT * V * 4}),
                        (h_load,), l0_after=(f"qk{head}:l0",) +
                        ((f"qh{head - 1}:compute",) if head else ()),
                        compute_after=(f"qk{head}:compute",) + ((f"qh{head - 1}",) if head else ()),
                        publish_after=(f"qk{head}:publish",))
                    cube_tail = f"qk_qh{head}"
                    stages.append(Stage(cube_tail, after=(qk, qh)))
                # All S1 heads on an AIV precede its S3 heads. Both AIVs
                # acknowledge each cube result before its A-prime is consumed.
                for head, values in enumerate(heads):
                    aiv = head % 2
                    vector_tail[aiv] = emit(f"a_prime{head}", values[2],
                                           (vector_tail[aiv], f"qk_qh{head}"), aiv)
                stage2 = cube_tail
                for head, values in enumerate(heads):
                    # Source queues one V head ahead of L1->L0/MMAD.
                    load_deps = (stage2,) if head < 2 else (f"av{head - 2}",)
                    if head:
                        load_deps += (f"v_load{head - 1}",)
                    read = values[3].resources["hbm"]
                    emit(f"v_load{head}", Work(resources={"hbm": read, "mte2": read}), load_deps)
                    cube_tail = emit(f"av{head}", Work(matrix=values[3].matrix, resources={
                        "mte1": values[3].resources["mte1"], "fixpipe": values[3].resources["fixpipe"]}),
                                     (cube_tail, f"v_load{head}", f"a_prime{head}"))
                for head, values in enumerate(heads):
                    aiv = head % 2
                    vector_tail[aiv] = emit(f"output{head}", values[4],
                                           (vector_tail[aiv], f"av{head}"), aiv)
                stages.append(Stage("cube_done", after=(cube_tail,)))
                stages.extend(Stage(f"vector_done{aiv}", after=(tail,))
                              for aiv, tail in vector_tail.items())
                return tuple(stages)
            count = (chunks.total * groups - 1 - core) // context.core_count + 1
            sizes = {"ub": 248 * 1024, "l1": 352 * 1024,
                     "l0a": 64 * 1024, "l0b": 64 * 1024, "l0c": 256 * 1024}
            return resident(context, core, sizes, PipelineProgram(count, 2, tile))
        return ParallelProgram(tuple(core_program(core) for core in range(
            min(context.core_count, chunks.total * groups))))
    return OperatorPlan(work, program)
