"""Shared builders for linear-attention, causal-conv, and template tests.

[Role] Provide compact, single-definition case, caller, execution, and
  planning-context builders used across ``tests/operators/`` linear-attention,
  causal-conv, and template suites, plus retargeted legacy callers.
[Invariants] Zero cross-test imports, zero host-absolute path literals, zero
  mutated global state, and deterministic in-memory construction.
[Timeouts & Locks] Pure in-memory construction and bounded single-op execution;
  no sleeps, locks, or network access.
[Side-Effect Policy] Read-only against packaged model/example JSON fixtures.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from mindie_sim import target_graphs
from mindie_sim.backends import ascend
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.family_kda_recurrent import IMPLEMENTATION as KDA_REC_IMPL
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.gdn_decode_programs import CONV, GATING
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.recurrent import linear
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Software
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView

REPO_ROOT = Path(__file__).resolve().parents[2]
_KDA_CASE_DIR = "examples/cases/high-concurrency-20261030/kimi-source-paged-host"


def kda_hardware_case(
    *,
    units: int = 2,
    split_copy: bool = False,
    rates: dict[str, float] | None = None,
    compute_bf16: float | None = None,
) -> dict[str, Any]:
    """Return a minimal single-device mapping for KDA source controls."""
    base: dict[str, float] = {
        "memory": 1e12,
        "vector": 2e12,
        "mte1": 1e12,
        "fixpipe": 1e12,
        "mte3": 1e12,
    }
    resource_units: dict[str, int] = {"vector": 2 * units}
    if split_copy:
        resource_units.update(mte2_aic=units, mte2_aiv=2 * units)
        base.update(mte2_aic=1e12, mte2_aiv=2e12)
    else:
        base["mte2"] = 1e12
    if rates:
        base.update(rates)
    return {
        "machine": {
            "name": "kda-source-control",
            "num_devices": 1,
            "device": {
                "memory_bytes": 64 * 1024 ** 3,
                "compute_units": units,
                "compute_peaks": {
                    "bf16": compute_bf16 or units * 1e14,
                    "fp32": units * 2.5e13,
                },
                "rates": base,
                "resource_units": resource_units,
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
        "workload": {
            "requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}],
        },
        "runtime": {},
    }


def _base_recurrent_case():
    parsed = parse_case({
        "machine": {
            "name": "declared-gpu",
            "architecture": "declared",
            "num_devices": 1,
            "device": {
                "memory_bytes": 1024 ** 3,
                "compute_peaks": {"bf16": 1e9, "fp32": 1e9},
                "rates": {"memory": 1e9, "vector": 1e9},
                "compute_units": 4,
            },
            "links": [{
                "name": "fabric",
                "capacity_bytes_per_ms": 1000,
                "members": [0],
                "semantics": "shared",
            }],
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
            "requests": [{"id": "r", "prompt_tokens": 4, "output_tokens": 3}],
        },
        "runtime": {
            "dp": 1,
            "tp": 1,
            "pp": 1,
            "ep": 1,
            "max_num_batched_tokens": 16,
        },
    })
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


def gdn_decode_source_case(
    lengths: tuple[int, ...] = (4, 4, 4, 4),
    *,
    keys: int = 4,
    heads: int = 12,
    units: int = 64,
    queue: bool = False,
):
    """Build a GDN decode body case and op sequence."""
    parsed = _base_recurrent_case()
    layer = replace(
        parsed.model.layer_plans[0],
        linear_key_heads=keys,
        linear_value_heads=heads,
        linear_heads=heads,
        linear_key_head_dim=128,
        linear_head_dim=128,
        conv_kernel=4,
    )
    kernels: dict[str, Any] = {"lowerings": {"gdn_decode": "vllm_ascend_gdn_decode"}}
    rates = {
        "hbm": 4e12,
        "vector": 1e13,
        "mte2": 4e12,
        "mte2_aic": 4e12,
        "mte2_aiv": 4e12,
        "mte3": 4e12,
    }
    if queue:
        rates["scalar_store_issue"] = 6.4e9
        kernels["lowerings"]["gdn_decode"] = {
            "implementation": "vllm_ascend_gdn_decode",
            "options": {
                "gating_beta_store": {
                    "resource": "scalar_store_issue",
                    "queue": {
                        "outstanding": 8,
                        "granule_bytes": 128,
                        "response_ms": 0.0001,
                    },
                },
            },
        }
    device = replace(
        parsed.machine.dies[0],
        rates=tuple(rates.items()),
        compute_units=units // 2,
        resource_units=tuple((r, units) for r in rates),
    )
    parsed = replace(
        parsed,
        model=replace(parsed.model, layer_plans=(layer,)),
        runtime=replace(parsed.runtime, kernels=kernels),
        machine=replace(parsed.machine, dies=(device,)),
    )
    views = tuple(
        RequestView(str(i), q, 4096, False, "decode")
        for i, q in enumerate(lengths)
    )
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode")
    return parsed, views, ops


def execute_one(parsed, views, op):
    """Execute one op behind a delayed ready event and assert zero leaks."""
    op = replace(op, deps=("ready",))
    execution = Execution(
        parsed, CostModel(parsed), include_submission=False, trace_limit=10000,
    )
    ready = execution.engine.submit("ready", release=2)
    complete = execution.add(
        IterationWork((op,), IterationPlan(views), 1, 0, 0, None, "body", "test"),
        0,
        0,
        external_inputs={"ready": ready},
    ).completion
    execution.engine.advance(horizon=1)
    assert not any(execution.engine.serviced.values())
    execution.engine.advance(until=(complete,))
    assert not execution.engine._tasks and not execution.engine._promises
    return execution.engine, complete.time - 2


def caller(
    lengths: tuple[int, ...] = (1, 4),
    *,
    padded: int | None = None,
    dv: int = 128,
    options: dict[str, Any] | None = None,
    phase: str = "decode",
):
    """Build an ordinary KDA caller case and lowered linear op list."""
    model = load_model("kimi-k3")
    plan = next(p for p in model.layer_plans if p.attention == "kda")
    plan = replace(
        plan,
        linear_key_heads=1,
        linear_value_heads=2,
        linear_heads=2,
        linear_head_dim=dv,
        linear_key_head_dim=128,
    )
    model = replace(
        model,
        layers=1,
        hidden_size=512,
        layer_plans=(plan,),
        quant=quant_preset("bf16"),
    )
    runtime = Software.from_mapping({
        "kernels": {
            "lowerings": {
                "kda": {
                    "implementation": KDA_REC_IMPL,
                    "options": options or {},
                },
            },
        },
    })
    views = tuple(
        RequestView(str(i), q, 2048, False, phase)
        for i, q in enumerate(lengths)
    )
    ops = linear(
        prefix="kda",
        plan=plan,
        views=views,
        stage=0,
        deps="input",
        model=model,
        parallel=ParallelSpec(),
        quant=model.quant,
        mode=phase,
        padded_tokens=padded or sum(lengths),
        policy=AttentionPolicy(),
        runtime=runtime,
    ).ops
    parsed = parse_case(kda_hardware_case(units=4, split_copy=True))
    die = replace(
        parsed.machine.dies[0],
        buffers=(("ub", 256 * 1024),),
        buffer_units=(("ub", 8),),
    )
    return (
        replace(
            parsed,
            model=model,
            runtime=runtime,
            machine=replace(parsed.machine, dies=(die,)),
        ),
        ops,
    )


def packed_roles(ops):
    """Return packed KDA input projection op and its declared role slices."""
    packed = next(op for op in ops if op.recipe == "linear_qkvbfg_a")
    slices = {
        row["role"]: row["shape"]
        for row in dict(packed.metadata)["packed_projection_slices"]
    }
    return packed, slices


def case(batch: int, *, native: bool = True) -> dict[str, Any]:
    """Load the paged KDA decode example case with optional native conv selector."""
    path = REPO_ROOT / _KDA_CASE_DIR / f"a5_kimi_tp8_e16_bf16_c{batch}_p4k_source.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if native:
        raw["runtime"]["kernels"]["lowerings"]["causal_conv_update"] = CONV
    return raw


def conv_graph(raw: Any, batch: int):
    """Build target decode body ops for a KDA conv update case."""
    return target_graphs(
        raw,
        scope="body",
        phase="decode",
        tokens=1,
        context=4096,
        sequences=batch,
        apply_body_defaults=True,
    )[0]


def sku_planning_context(
    sku: str,
    op_id: str = "op0",
    *,
    requested_abi: str | None = None,
) -> tuple[Geometry, OperatorPlanningContext]:
    """Build a real Geometry and OperatorPlanningContext from a packaged SKU."""
    hw = ascend.hardware(sku, devices=1)
    die = hw.dies[0]
    geom = Geometry.from_devices(hw.dies)
    dp = datapath_for_die(die)
    ctx = OperatorPlanningContext(
        occurrence=OpOccurrence(replica=0, stage=0, rank=0, op_id=op_id),
        geometry=geom,
        datapath=dp,
        requested_abi=requested_abi,
    )
    return geom, ctx


__all__ = [
    "CONV",
    "GATING",
    "caller",
    "case",
    "conv_graph",
    "execute_one",
    "gdn_decode_source_case",
    "kda_hardware_case",
    "packed_roles",
    "sku_planning_context",
]
