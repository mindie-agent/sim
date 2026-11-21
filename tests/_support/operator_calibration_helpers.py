"""Shared MLA/C8/indexer calibration fixtures for cross-track legacy tests.

[Role] Provide deterministic in-memory MLA prolog, C8 sparse attention, and
  quantized indexer fixture builders without importing test modules.
[Invariants] Pure declarative builders with zero ``test_*`` functions, zero
  external I/O mutations, and exact conservation helpers for ``RecipeContext``.
[Timeouts & Locks] Pure Tier-0 in-memory helpers; no device or file locks.
[Side-Effect Policy] Read-only over packaged profiles and example JSON cases.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim.backends.ascend import profiles
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.implementations import IMPLEMENTATIONS, import_module
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    RecipeBook,
    RecipeContext,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec

LAYOUT = "packed_int8_bf16_rope_fp32_scale"
SCHEDULE_ARCH35 = "arch35_c8_tile128_slots2"
_KEYPOOL_IMPL = "vllm_ascend_cann_keypool_caller"
_KEYPOOL_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
    "path": "vllm_ascend/attention/indexer_kpool.py",
    "sha256": "c7268fa24836f3ef33f9fb626ece9430c6c752a596e3bc07d230de5e9fb99e9b",
    "lines": "44-106,168-200,315-367,935-950,1112-1150",
    "adapters": {
        "cann_keypool.py": "5deb787e94c4b31f5246747778b2bbe3a9c69447b2ca9701c90b500c0da94696",
        "glm5_next_cann_keypool.py": "b02108fc0ac4e5ec77828abd2f69811511ee226ee90894500943d2175c2b4dd6",
        "prepare_nonempty_keypool_inputs.py": "1864a2affc64dee1f033aeae7ce92d9159948ee3b6bb80b6fa81c739b9745184",
    },
    "row_shard": {
        "path": "vllm_ascend/attention/indexer_kpool.py",
        "scope": "configured_caller_and_cost_balanced_TP_exchange",
    },
    "scope": "native_caller_contract; installed_binary_and_core_schedule_unbound",
}


def build_packed(op, geom, *, scratch_policy="offchip", schedule=SCHEDULE_ARCH35, compact=True, operand_modes=True):
    """Build packed C8 indexed attention through the active ``RecipeBook``."""
    book = RecipeBook.from_mapping({
        "programs": {
            "indexed_attention": {
                "kind": "indexed_attention_packed_int8",
                "scratch_policy": scratch_policy,
                "schedule": schedule,
                "compact": compact,
                "operand_modes": operand_modes,
            }
        }
    })
    return book.programs["indexed_attention"].build(op, geom)


sp = SimpleNamespace(
    Arch35QuantIndexerRecipe=lambda: RecipeBook.from_mapping({
        "lowerings": {"quant_lightning_indexer": {"implementation": "cann_quant_lightning_indexer_v2_complete"}}
    }).recipes["aclnn_quant_lightning_indexer_v2_arch35"],
)

ROOT = Path(__file__).resolve().parents[2]
GLM_CASE = (
    ROOT
    / "examples/cases/high-concurrency-20261030/glm78/"
    / "a3_glm53_w4a8_tp16_tpot88_c8.json"
)
SKUS = (("PR", "Ascend950PR_958b"), ("DT", "Ascend950DT_9582"))
DT = "Ascend950DT_9582"
HIDDEN = 4096
HEAD_WIDTH = 576
KV_RANK = 512
ROPE_WIDTH = 64
INDEX_WIDTH = 128
INDEX_HEADS = 32
INDEX_TOPK = 2048
LOCAL_TP = 8
PROLOG_QUERY_LORA_RANK = 1536

C8_PROGRAM = {
    "kind": "indexed_attention_packed_int8",
    "scratch_policy": "offchip",
    "compact": True,
    "operand_modes": True,
    "schedule": SCHEDULE_ARCH35,
}
DEVICE = {
    "model": {
        "name": "calibration_device",
        "layers": 1,
        "hidden_size": HIDDEN,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": HEAD_WIDTH,
        "intermediate_size": 1024,
        "vocab_size": 128,
    },
    "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
    "runtime": {
        "tp": 1,
        "dp": 1,
        "ep": 1,
        "pp": 1,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 256,
    },
}


def _mla_plan(**overrides):
    fields = dict(
        index=0, attention="mla", architecture="mla", ffn="dense",
        intermediate_size=1024, num_heads=8, num_kv_heads=1, head_dim=HEAD_WIDTH,
        q_lora_rank=KV_RANK, kv_lora_rank=KV_RANK, qk_nope_head_dim=KV_RANK,
        qk_rope_head_dim=ROPE_WIDTH, v_head_dim=128, mla_variant="latent_mla",
        cache_kind="window_compressed", compress_ratio=4, compressor_width=1,
        compressor_state_rows=8, compressor_kind="gated_ring", owns_long_kv=True,
        kv_owner_layer_id=0, window=4096,
    )
    fields.update(overrides)
    return LayerPlan(**fields)


def _model(plan, *, quant=None, metadata=()):
    return ModelSpec(
        name="prolog_calibration", layers=1, hidden_size=HIDDEN, num_heads=8,
        head_dim=HEAD_WIDTH, intermediate_size=1024, vocab_size=1024,
        attention="mla", ffn="dense", layer_plans=(plan,), metadata=metadata,
        quant=quant or QuantSpec("bf16"),
    )


def _kimi_quant():
    return QuantSpec(
        name="W8A8_MXFP8", weight_format="fp8_e4m3",
        activation_format="fp8_e4m3", expert_weight_format="fp8_e4m3",
        weight_block=(1, 32), expert_block=32, scale_format="fp8_e8m0",
        scale_bytes=1.0, activation_group_size=32,
        activation_scale_format="fp8_e8m0", activation_scale_pack=2,
        kv_dtype="int8", index_dtype="bf16", index_scale_dtype="none",
        category_overrides={
            "indexer_fp8": {"weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3"},
            "indexer": {"weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3"},
        },
    )


def _prolog_case():
    plan = _mla_plan(
        indexer=True, indexer_type="full", index_heads=INDEX_HEADS,
        index_dim=INDEX_WIDTH, index_topk=512, owns_index_k=True,
        owns_topk=True, cache_kind="dense", compress_ratio=0,
        q_lora_rank=PROLOG_QUERY_LORA_RANK,
    )
    return plan, _model(plan, quant=_kimi_quant())


def _declaration(ops, anchor):
    node = next(op for op in ops if op.op_id == anchor)
    return dict(dict(node.metadata)["structure_subgraphs"][-1])


def _op(op_id, family, shape):
    return OpSpec(
        op_id=op_id, family=family, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape, dtype="bf16",
        metadata=(("activation_format", "bf16"), ("weight_format", "bf16")),
    )


def _c8_case(sku):
    raw = json.loads(GLM_CASE.read_text(encoding="utf-8"))
    raw["machine"].update({"sku": sku, "num_devices": 1, "dies_per_package": 1, "ranks_per_node": 1})
    raw["runtime"].update({"tp": 1, "dp": 1, "ep": 1, "pp": 1, "max_num_seqs": 256})
    raw["model"]["layers"] = 1
    raw["runtime"]["kernels"] = {
        "programs": {"matmul": {"kind": "steady_matmul"}, "indexed_attention": dict(C8_PROGRAM)},
    }
    return raw


def _c8_op(sequences, topk, heads, prefix, paired_calls=1):
    shape = {
        "topk": topk, "compression_ratio": 1, "window": 0, "heads": heads,
        "latent": 512, "rope": 64, "cp": 1, "query_dtype": "bf16",
        "cache_dtype": "int8", "sequences": tuple(sequences),
        "prefix_cache_mode": "shared" if prefix else "private",
    }
    if paired_calls != 1:
        shape["paired_calls"] = paired_calls
    if prefix:
        shape["prefix_tokens"] = tuple([prefix] * len(sequences))
    rows = sum(query for query, _ in sequences)
    return OpSpec(
        op_id="s0.l0.attention.0", family="attention", phase="decode", stage=0,
        layer=0, requests=("r",), semantic_shape=(rows, heads, 576),
        local_shape=(rows, heads, 576), padded_shape=(rows, heads, 576),
        launch_shape=(rows, heads, 576), flops=0.0, vector_ops=0.0,
        hbm_bytes=0.0, mte2_bytes=0.0, mte3_bytes=0.0, deps=(),
        recipe="latent_attention_batch",
        metadata=(("program_kind", "indexed_attention"), ("indexed_attention_geometry", shape)),
    )


def _machine(sku, devices=1):
    device = deepcopy(profiles()[sku]["device"])
    device.pop("compute_arch", None)
    links = (
        [] if devices == 1 else
        [{"name": "ub_mesh", "capacity_gbps": 392, "latency_us": 2.5,
          "members": list(range(devices)), "semantics": "per_pair"}]
    )
    return {
        "name": sku, "sku": sku, "hardware_key": sku, "arch_family": "A5",
        "num_devices": devices, "ranks_per_node": devices, "dies_per_package": 1,
        "device": device,
        "dies": [{"rank": i, "node": 0, "package": 0} for i in range(devices)],
        "links": links,
    }


def _case(sku, devices=1, *, tp=1):
    raw = deepcopy(DEVICE)
    raw["machine"] = _machine(sku, devices)
    raw["runtime"]["tp"] = tp
    return Case.from_mapping(raw)


def _stages(program):
    """Yield every finite Stage a program graph issues."""
    if isinstance(program, KernelProgram):
        for stage in program.stages:
            yield stage
            if stage.program is not None:
                yield from _stages(stage.program)
    elif isinstance(program, PipelineProgram):
        for idx in range(program.count):
            for stage in program.tile(idx):
                yield stage
                if stage.program is not None:
                    yield from _stages(stage.program)
    elif isinstance(program, ParallelProgram):
        for branch in program.programs:
            yield from _stages(branch)
    else:
        raise AssertionError(f"program_not_finite:{type(program)!r}")


def _context(sku):
    device = _case(sku).machine.dies[0]
    rates = dict(device.rates)
    names = {role: f"{role}@r0:s0" for role in rates}
    names["cube"] = "cube@r0:s0"
    units = {role: device.units(role) for role in rates}
    units["cube"] = device.compute_units
    context_rates = {names[role]: rate for role, rate in rates.items()}
    return device, RecipeContext(
        None, {}, names, context_rates, device.compute_units,
        resource_units=units,
        compute_peaks=dict(device.compute_peaks),
        service_modes={m: (r, rt) for m, (r, rt) in dict(device.service_modes).items()},
        pool_names={pool: f"{pool}@r0:s0" for pool in dict(device.buffers)},
        pool_units={pool: device.pool_units(pool) for pool in dict(device.buffers)},
    )


def _conserved(plan, op, sku):
    """Assert every bound unit of plan work appears exactly once in issued stages."""
    _device, context = _context(sku)
    prepared = plan.work.apply(op)
    bound = replace(context, op=prepared)
    program = plan.program(bound)
    expected = bound.bind(plan.work)
    issued = {}
    for stage in _stages(program):
        for name, amount in stage.work.items():
            issued[name] = issued.get(name, 0.0) + float(amount)
    assert set(issued) <= set(expected) | {"cube"}
    for name, amount in expected.items():
        assert issued.get(name, 0.0) == pytest.approx(amount, rel=1e-9)
    return program


def _case_model(*, pool=16, index_dtype="fp8_e4m3", topk=INDEX_TOPK, compress_ratio=0):
    plan = LayerPlan(
        index=0, attention="mla", architecture="mla", ffn="dense",
        intermediate_size=1024, num_heads=8, num_kv_heads=1, head_dim=HEAD_WIDTH,
        q_lora_rank=512, kv_lora_rank=512, qk_nope_head_dim=512,
        qk_rope_head_dim=0, v_head_dim=128, mla_variant="latent_mla",
        cache_kind="dense", compress_ratio=compress_ratio, indexer=True,
        indexer_type="full", index_heads=INDEX_HEADS, index_dim=INDEX_WIDTH,
        index_topk=topk, index_kpool=pool, index_kpool_compress=True,
        index_input="hidden", owns_index_k=True, owns_topk=True,
    )
    scale_dt = "fp32" if pool == 1 and index_dtype == "fp8_e4m3" else "fp16"
    quant = QuantSpec(
        "bf16_fp8_index", weight_format="bf16", activation_format="bf16",
        native_format="bf16", expert_weight_format="bf16", kv_dtype="int8",
        index_dtype=index_dtype, index_scale_dtype=scale_dt,
    )
    reps = {} if index_dtype == "int8" else {
        "index_k": {"dtype": index_dtype, "group_size": INDEX_WIDTH if pool == 1 else 32,
                    "scale_dtype": quant.index_scale_dtype},
    }
    model = ModelSpec(
        name="indexer_calibration", layers=1, hidden_size=HIDDEN, num_heads=8,
        head_dim=HEAD_WIDTH, intermediate_size=1024, vocab_size=1024,
        attention="mla", ffn="dense", layer_plans=(plan,), quant=quant,
        metadata=(
            ("cache_representations", reps),
            ("native_keypool_caller", {
                "implementation": _KEYPOOL_IMPL, "tail_capacity": pool,
                "block_table_columns": 8195, "block_table_columns_explicit": True,
                "row_shard_tp": 1, "source": _KEYPOOL_SOURCE,
            }),
        ),
    )
    return plan, model


class _Runtime:
    def __init__(self, *, o_proj_tp, linear_proj_tp=1, metadata=()):
        self.metadata = dict(metadata)
        self.prefix_cache_mode = "private"
        self.prefix_read_mode = "auto"
        self.parallel = type(
            "Parallel", (),
            {"o_proj_tp": o_proj_tp, "linear_proj_tp": linear_proj_tp, "tp": LOCAL_TP},
        )()


class _ArchParallel:
    tp = LOCAL_TP
    cp = 1
    pcp = 1


def _arch35_op_for(ratio, views, *, topk=INDEX_TOPK, heads=INDEX_HEADS, prefix_cache_mode="private"):
    plan, model = _case_model(pool=1, index_dtype="fp8_e4m3", compress_ratio=ratio, topk=topk)
    plan = replace(plan, index_heads=heads)
    builder = Builder("s0.l0", plan, 0, views, model, "decode")
    mod = import_module(IMPLEMENTATIONS["cann_quant_lightning_indexer_v2_complete"].module)
    mod.declare(
        builder, deps=("q", "w", "cache"), views=views, plan=plan,
        quant=builder.quant, parallel=_ArchParallel(), prefix_cache_mode=prefix_cache_mode,
    )
    return builder.ops[-1]


__all__ = (
    "DT", "Geometry", "HEAD_WIDTH", "HIDDEN", "KV_RANK", "LAYOUT",
    "PROLOG_QUERY_LORA_RANK", "ParallelSpec", "SCHEDULE_ARCH35", "SKUS",
    "_Runtime", "_arch35_op_for", "_c8_case", "_c8_op", "_case_model",
    "_conserved", "_context", "_declaration", "_op", "_prolog_case", "_stages",
    "build_packed", "sp",
)
