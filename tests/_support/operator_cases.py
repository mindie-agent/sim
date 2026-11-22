"""Shared operator test-case builders replacing cross-test helper imports.

[Role] Centralize reusable raw-case, synthetic-hardware, and Engine-execution
  helpers previously imported across operator, spine, and serving test modules
  so domain tests stay orthogonal under ``TO006`` without cross-test imports.
[Invariants] Pure helper definitions; defines zero ``test_*`` functions, performs
  no host-path or environment mutation, and stays under 600 lines.
[Timeouts & Locks] In-memory builders and bounded fixture JSON reads; no locks.
[Side-Effect Policy] Read-only on repository example fixtures.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.operators.sparse_indexer.sparse_mla_metadata import IMPLEMENTATION as SPARSE_META_IMPL
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec

RECIPE_ID = "staged_gqa_decode"
MQSMLA_RECIPE = "aclnn_mixed_quant_sparse_flash_mla"
MQSMLA_FORMATS = {
    "window": {"dtype": "fp8_e4m3", "group_size": 32, "scale_dtype": "bf16"},
    "compressed_kv": {"dtype": "fp4_e2m1", "group_size": 16, "scale_dtype": "bf16"},
}
PAGED_MLA_IMPL = "ascend_paged_mla_decode_arch35"
PAGED_MLA_BUFFERS = {"l1": 516096, "l0a": 65536, "l0b": 65536, "l0c": 262144, "ub": 241920}
PAGED_MLA_MEM_IMPL = "ascend_paged_mla_decode_arch22"
PAGED_MLA_MEM_BUFFERS = {"l1": 516096, "l0a": 65536, "l0b": 65536, "l0c": 131072, "ub": 174080}

REPO_ROOT = Path(__file__).resolve().parents[2]
_PAGED_MLA_CASE = (
    REPO_ROOT
    / "examples/cases/high-concurrency-20261030/kimi-source/"
    "a5_kimi_tp8_e16_bf16_c1_p4k_source.json"
)
_MQSMLA_CASE = (
    REPO_ROOT
    / "examples/cases/high-concurrency-20261030/v41-dp8/"
    "a5_v41_fp8_tp1_dp8_ep8_tpot5ms_c1.json"
)
_INDEXED_QUANT_CASE = (
    REPO_ROOT
    / "examples/cases/high-concurrency-20261030/matrix-extension/"
    "a3_glm53flash_tp16_aclgraph_c32.json"
)


def hardware_case() -> dict:
    """Minimal synthetic 1-device case for mixed-precision and mechanism tests."""
    return {
        "machine": {
            "name": "arbitrary",
            "num_devices": 1,
            "device": {
                "memory_bytes": 1024**3,
                "compute_units": 4,
                "compute_peaks": {"bf16": 2e6, "fp32": 1e6},
                "rates": {"memory": 1e6, "vector": 1e6},
            },
        },
        "model": {
            "name": "geometry",
            "layers": 1,
            "hidden_size": 8,
            "num_heads": 1,
            "num_kv_heads": 1,
            "head_dim": 8,
            "intermediate_size": 16,
            "vocab_size": 16,
        },
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


def declared_case(*, dp=1, tp=1, pp=1, ep=1, prompt=4, output=3, requests=None) -> dict:
    """Compact declared-device case shared by MLA and recurrent helpers."""
    ranks = dp * tp * pp
    return {
        "machine": {
            "name": "declared-gpu",
            "architecture": "declared",
            "num_devices": ranks,
            "device": {
                "memory_bytes": 1024**3,
                "compute_peaks": {"bf16": 1e9, "fp32": 1e9},
                "rates": {"memory": 1e9, "vector": 1e9},
                "compute_units": 4,
            },
            "links": [
                {
                    "name": "fabric",
                    "capacity_bytes_per_ms": 1000,
                    "members": list(range(ranks)),
                    "semantics": "shared",
                }
            ],
        },
        "model": {
            "name": "geometry",
            "layers": 2,
            "hidden_size": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 8,
            "intermediate_size": 64,
            "vocab_size": 128,
        },
        "workload": {
            "requests": requests or [{"id": "r", "prompt_tokens": prompt, "output_tokens": output}]
        },
        "runtime": {"dp": dp, "tp": tp, "pp": pp, "ep": ep, "max_num_batched_tokens": 16},
    }


def mla_case(*, splits: int = 3) -> dict:
    """Compact tuple-query MLA decode case with configurable KV splits."""
    raw = declared_case(prompt=4, output=3)
    raw["model"].update(
        layers=1,
        attention="mla",
        num_kv_heads=1,
        head_dim=48,
        mla={
            "q_lora_rank": 32,
            "kv_lora_rank": 256,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 32,
            "v_head_dim": 16,
        },
    )
    raw["runtime"]["kernels"] = {
        "lowerings": {
            "mla_decode": {
                "implementation": "vllm_triton_mla_decode_tuple",
                "options": {"num_kv_splits": splits},
            }
        }
    }
    return raw


def recurrent_case():
    """Single-layer GDN recurrent case derived from ``declared_case``."""
    parsed = parse_case(declared_case())
    layer = replace(
        parsed.model.layer_plans[0],
        attention="gdn",
        linear_key_heads=1,
        linear_value_heads=2,
        linear_heads=2,
        linear_key_head_dim=3,
        linear_head_dim=2,
        conv_kernel=3,
        recurrent_state_dtype="fp32",
    )
    return replace(
        parsed,
        model=replace(parsed.model, layers=1, layer_plans=(layer,)),
        runtime=replace(parsed.runtime, pp_layer_partition=(1,)),
    )


def gqa_decode_hardware(*, units: int = 4) -> dict:
    """Synthetic hardware profile for staged GQA decode scheduling tests."""
    return {
        "machine": {
            "name": "synthetic-gqa-decode",
            "num_devices": 1,
            "device": {
                "memory_bytes": 1 << 30,
                "compute_units": units,
                "compute_peaks": {"bf16": 8e12, "fp32": 4e12},
                "rates": {"memory": 1e9, "vector": 1e9, "mte1": 1e9, "mte2": 1e9, "mte3": 1e9},
                "buffers": {
                    "l1": 1 << 20, "l0a": 1 << 18, "l0b": 1 << 18,
                    "l0c": 1 << 19, "ub": 1 << 20,
                },
                "buffer_units": {
                    "l1": units, "l0a": units, "l0b": units, "l0c": units, "ub": 2 * units,
                },
                "resource_units": {
                    "vector": 2 * units, "mte1": units, "mte2": units, "mte3": units,
                },
            },
        },
        "model": {
            "name": "geometry", "layers": 1, "hidden_size": 8,
            "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
            "intermediate_size": 16, "vocab_size": 16,
        },
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


def gqa_decode_op(requests, *, heads=4, kv_heads=2, dim=8, recipe=RECIPE_ID, **metadata_overrides):
    """Build a staged GQA decode ``OpSpec`` from ``(query, context)`` pairs."""
    metadata = {
        "requests": tuple(requests),
        "query_heads": heads,
        "kv_heads": kv_heads,
        "head_dim": dim,
        "query_dtype": "bf16",
        "kv_dtype": "bf16",
        "causal": True,
        "window": 0,
        "cp": 1,
    }
    metadata.update(metadata_overrides)
    tokens = sum(query for query, _ in requests)
    length = max((context + query for query, context in requests), default=0)
    shape = (tokens, length, heads, dim)
    return OpSpec(
        "gqa_decode",
        "attention",
        "decode",
        0,
        0,
        tuple(f"r{index}" for index in range(len(requests))),
        shape,
        shape,
        shape,
        shape,
        recipe=recipe,
        dtype="bf16",
        metadata=(("gqa_decode", metadata),),
    )


def gqa_decode_run(op, recipe=None, *, units=4, ready=0.0, external=None, raw_case=None):
    """Execute one staged GQA decode op directly on ``Engine``."""
    recipe = recipe or RecipeBook.from_mapping(
        {"programs": {"gqa_decode": {"kind": RECIPE_ID}}}
    ).programs["gqa_decode"]
    case = parse_case(raw_case or gqa_decode_hardware(units=units))
    execution = Execution(
        case,
        CostModel(case),
        recipes=RecipeBook({RECIPE_ID: recipe}),
        trace_limit=20000,
    )
    prepared, _ = execution.prepare(op)
    context = execution._context(prepared, 0, "none")
    start = execution.engine.submit("input_ready", release=ready)
    if external:
        execution.engine.submit(
            "external",
            external["work"],
            placement=external["placement"],
            owner="external",
        )
    end = recipe.lower(context).start(execution.engine, "gqa", start)
    before = {}
    if ready:
        execution.engine.advance(horizon=ready)
        before = {name: value for name, value in execution.engine.serviced.items() if value}
    execution.engine.advance(until=(end,))
    return execution.engine, end, before, context, recipe


def paged_mla_case(splits: int = 1) -> dict:
    """Load the paged MLA example case with explicit ``kv_splits``."""
    raw = json.loads(_PAGED_MLA_CASE.read_text(encoding="utf-8"))
    raw["runtime"]["kernels"]["lowerings"]["paged_mla_attention"] = {
        "implementation": PAGED_MLA_IMPL,
        "options": {"kv_splits": splits},
    }
    return raw


def paged_mla_attention(raw, batch: int = 1, context: int = 4096):
    """Select the paged MLA operator and layer graph from ``raw``."""
    graph = target_graphs(
        raw, scope="body", phase="decode", tokens=1, sequences=batch, context=context
    )[0]
    return next(o for o in graph if o.recipe == PAGED_MLA_IMPL), graph


def paged_mla_execute(raw, op):
    """Execute one paged MLA op and assert zero pre-input service and drained buffers."""
    row = operator_costs(raw, [op], input_ready_ms=7)["operators"][0]
    assert not row["service_before_input"] and row["live_tasks"] == 0
    assert all(abs(n) < 1e-7 for n in row["remaining_buffers"].values())
    return row


def paged_mla_membase_run(raw: dict, *, supply: float = 1, trace: int = 0):
    """Execute one MemBase paged-MLA dispatch record and verify service conservation."""
    cores = raw["cores"]
    units = {
        r: cores * (2 if r in ("vector", "mte2_aiv", "mte3") else 1)
        for r in ("hbm", "vector", "cube", "mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe")
    }
    units["hbm"] = 1
    rates = {r: 1e9 * supply for r in units}
    rates["cube"] = 1
    pu = {p: cores * (2 if p == "ub" else 1) for p in PAGED_MLA_MEM_BUFFERS}
    op = OpSpec(
        "mla",
        "attention",
        "decode",
        0,
        0,
        (),
        (),
        (),
        (),
        (),
        recipe=PAGED_MLA_MEM_IMPL,
        source_boundary="kernel",
        metadata=(
            (
                "paged_mla_geometry",
                {
                    "contexts": tuple(raw["contexts"]),
                    "heads": raw["heads"],
                    "page_size": 128,
                    "dispatch": raw,
                },
            ),
        ),
    )
    planner = RecipeBook.from_mapping(
        {"lowerings": {"paged_mla_attention": {"implementation": PAGED_MLA_MEM_IMPL}}}
    ).recipes[PAGED_MLA_MEM_IMPL]
    plan = planner.build(op, Geometry(cores, units, PAGED_MLA_MEM_BUFFERS, pu))
    engine = Engine(
        [Resource(r, v, units=units[r]) for r, v in rates.items()],
        pools=PAGED_MLA_MEM_BUFFERS,
        pool_units=pu,
        trace_limit=trace,
    )
    ctx = RecipeContext(
        plan.work.apply(op),
        {},
        {r: r for r in rates},
        rates,
        cores,
        resource_units=units,
        compute_peaks={"bf16": 1e12 * supply},
        pool_names={p: p for p in PAGED_MLA_MEM_BUFFERS},
        pool_capacities=PAGED_MLA_MEM_BUFFERS,
        pool_units=pu,
    )
    ready = engine.submit("input", release=7)
    done = plan.lower(ctx).start(engine, "mla", ready)
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    for role, amount in ctx.bind(plan.work).items():
        assert engine.serviced[role] == pytest.approx(amount, rel=1e-12)
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return plan, engine, done.time - 7


def mqsmla_case():
    """Load the mixed-quant sparse MLA case with explicit cache representations."""
    raw = json.loads(_MQSMLA_CASE.read_text(encoding="utf-8"))
    raw["model"].setdefault("metadata", {})["cache_representations"] = deepcopy(MQSMLA_FORMATS)
    raw["runtime"]["kernels"].setdefault("lowerings", {})["mqsmla_attention"] = MQSMLA_RECIPE
    return parse_case(raw)


def mqsmla_attention(ops, layer: int = 2):
    """Select the bound MQSMLA attention operator at ``layer``."""
    return next(op for op in ops if op.layer == layer and op.recipe == MQSMLA_RECIPE)


def indexed_quant_case(compact: bool = True) -> dict:
    """Load the packed INT8 indexed-attention example case."""
    raw = json.loads(_INDEXED_QUANT_CASE.read_text(encoding="utf-8"))
    raw["model"]["layers"] = 4
    raw["runtime"]["kernels"]["programs"]["indexed_attention"] = {
        "kind": "indexed_attention_packed_int8",
        "scratch_policy": "offchip",
        "compact": compact,
    }
    return raw


def indexed_quant_ops(raw, batch: int = 1, query: int = 1, context: int = 4096):
    """Extract indexed-attention program ops from ``raw``."""
    return [
        op
        for op in target_graphs(
            raw, scope="body", phase="decode", tokens=query, context=context, sequences=batch
        )[0]
        if dict(op.metadata).get("program_kind") == "indexed_attention"
    ]


def sparse_mla_metadata_raw_case() -> dict:
    """Build the sparse MLA metadata caller case with explicit AI_CPU lowering."""
    return {
        "model": {
            "catalog_key": "glm-5.3",
            "layers": 1,
            "layer_overrides": {"0": {"qk_rope_head_dim": 0}},
        },
        "machine": {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 8},
        "runtime": {
            "tp": 8,
            "kernels": {
                "programs": {
                    "indexed_attention": {
                        "kind": "indexed_attention_pipeline",
                        "scratch_policy": "offchip",
                        "operand_modes": False,
                        "compact": True,
                        "rope0_analytic": True,
                    }
                },
                "lowerings": {
                    "sparse_mla_metadata": {
                        "implementation": SPARSE_META_IMPL,
                        "options": {"layer_groups": [[0]]},
                    }
                },
            },
        },
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
    }


__all__ = [
    "Engine",
    "Geometry",
    "OpSpec",
    "PAGED_MLA_BUFFERS",
    "RecipeContext",
    "Resource",
    "declared_case",
    "gqa_decode_hardware",
    "gqa_decode_op",
    "gqa_decode_run",
    "hardware_case",
    "indexed_quant_case",
    "indexed_quant_ops",
    "mla_case",
    "mqsmla_attention",
    "mqsmla_case",
    "paged_mla_attention",
    "paged_mla_case",
    "paged_mla_execute",
    "paged_mla_membase_run",
    "recurrent_case",
    "sparse_mla_metadata_raw_case",
]
