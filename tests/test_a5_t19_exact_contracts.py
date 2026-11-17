"""A5-T19 exact cross-layer contracts (X1-X9) and end-to-end chains (A, D, E).

[Role]
  Execute the exact behavioral contract probes declared by A5-T19 across X1-X9
  and chains A, D, and E through public mindie_sim production entry points.

[Invariants]
  * Synthetic contract fixtures validate mechanism arithmetic, state visibility,
    materialization bytes, and event ordering; they never claim measured device
    precision or inject target durations.
  * Physical work, live capacity, HBM access, host submissions, and delivery
    events remain separately observable without subgraph-code string coupling.
  * Nominal, calibrated, and predicted service sources remain distinct and are
    normalized once without editing installed package files.

[Timeouts & Locks]
  Bounded in-memory and tmp_path execution (< 20 s total wall time); zero device
  initialization, network calls, or cross-process locks.

[Side-Effect Policy]
  All filesystem writes use pytest tmp_path with isolated HOME and MINDIE_*
  environment roots; installed package directories are never mutated.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

import mindie_sim
from mindie_sim.catalog.cache import cache_row_split
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.cache_pages import KVPageAllocation
from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.compose import assemble, compose_case
from mindie_sim.observations.resource_services import (
    RESOURCE_SERVICE_PROFILE_SCHEMA, normalize_resource_service,
)
from mindie_sim.observations.service_fitting import fit_resource_service
from mindie_sim.prediction.attention_prefill import StagedGQAPrefill, logical_causal_pairs
from mindie_sim.prediction.blocks import norm
from mindie_sim.prediction.cache_path import CacheGeometry, CacheState
from mindie_sim.prediction.capacity import (
    Allocation as PlanAllocation, AllocationPlan, Memory, MemoryPolicy,
)
from mindie_sim.prediction.communication import MessageFlow
from mindie_sim.prediction.efficiency import (
    evaluate_fixed_plan, evaluate_resource_counterfactual, prepare_fixed_plan,
    solve_sla_sizing,
)
from mindie_sim.prediction.engine import (
    Allocation as EngineAllocation, Engine, ExecutionError, Resource,
)
from mindie_sim.prediction.frontend import Frontend, SERVICE_FAMILIES
from mindie_sim.prediction.implementations import add_input_ports, audit_boundary, input_ports
from mindie_sim.prediction.matrix_mapping import grouped_owner_rectangles, grouped_owner_tiles
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call, join_profiler_rows, validate_invocation,
)
from mindie_sim.prediction.scheduling import FragmentConductor
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.prediction.transforms import activation_quantization, norm_quant
from mindie_sim.public import (
    ExtensionError, load_user_profile, simulate_graph, simulate_model,
    simulate_native_call, simulate_operator, simulate_service,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.compute_arch import NativeComputeMode, RuntimeCapability
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import (
    AllocationEffect, IterationWork, OpSpec, ProgramFragment, StateEffect, WorkItem, WorkSet,
)
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.subgraphs import PortSet
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from mindie_sim.schema.workload import IterationPlan, RequestView

MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key, sub in (("HOME", "home"), ("MINDIE_CONFIG_HOME", "config"),
                     ("MINDIE_DATA_HOME", "data"), ("MINDIE_DIAGNOSTICS_ROOT", "diag")):
        target = tmp_path / sub
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(target))


def _compact_case(*, dp: int = 1, tp: int = 1, pp: int = 1, ep: int = 1,
                  layers: int = 2, prompt: int = 4, output: int = 3,
                  requests: list[dict] | None = None) -> dict:
    ranks = dp * tp * pp
    return {
        "machine": {
            "name": "declared-compact", "architecture": "declared", "num_devices": ranks,
            "device": {
                "memory_bytes": 1024 ** 3, "compute_units": 4,
                "compute_peaks": {"bf16": 1e9, "fp32": 1e9, "int8": 2e9},
                "rates": {"memory": 1e9, "vector": 1e9},
            },
            "links": [{"name": "fabric", "capacity_bytes_per_ms": 1_000_000,
                       "members": list(range(ranks)), "semantics": "shared"}],
        },
        "model": {
            "name": "compact-geometry", "layers": layers, "hidden_size": 32,
            "num_heads": 4, "num_kv_heads": 2, "head_dim": 8,
            "intermediate_size": 64, "vocab_size": 128,
        },
        "workload": {"requests": requests or [{"id": "r0", "prompt_tokens": prompt, "output_tokens": output}]},
        "runtime": {"dp": dp, "tp": tp, "pp": pp, "ep": ep, "max_num_batched_tokens": 32},
    }


def _serving_io(prepare_ms: float = 0.5, result_ms: float = 0.5, delivery_ms: float = 1.0) -> dict:
    return {
        "host_prepare_ms": prepare_ms, "host_result_ms": result_ms,
        "h2d": {"bytes_per_s": 80_000, "bytes_per_token": 8, "latency_us": 50},
        "d2h": {"bytes_per_s": 40_000, "bytes_per_token": 4, "latency_us": 50},
        "client_delivery_ms": delivery_ms,
    }


def _fragment(name: str, work: tuple[tuple[str, float, str], ...] = (), *,
              allocations: tuple = (), state_effects: tuple = ()) -> ProgramFragment:
    items = tuple(WorkItem(res, amt, unit) for res, amt, unit in work)
    return ProgramFragment(
        name, work=WorkSet(items), allocations=allocations,
        state_effects=state_effects, ports=PortSet(), semantic_refs=None,
    )


def _sku_machine(sku: str) -> dict:
    return compose_case("qwen3.6-27b", hardware=sku, layers=1)["machine"]


def _run_transport_us(hw: Hardware, pspec: ParallelSpec, flows: tuple[MessageFlow, ...]) -> tuple[float, dict]:
    plan = Topology(hw, pspec).transport_plan(flows)
    eng = Engine(plan.resources())
    ready = {rank: eng.submit(f"r{rank}") for rank in plan.ranks}
    run = plan.start(eng, "transport", ready)
    eng.advance(until=run.completions, first=False)
    return run.report()["duration_ms"] * 1000.0, plan.lower_bound()


def test_x1_exact_assembly_counts() -> None:
    raw_2l = _compact_case(layers=2, requests=[
        {"id": "r0", "prompt_tokens": 4, "output_tokens": 4},
        {"id": "r1", "prompt_tokens": 3, "output_tokens": 4},
    ])
    raw_2l["runtime"].update(
        graph_mode="auto", graph_capture_mode="full", graph_capture_keys=[2, 7],
        submission={"graph_replay_ms": 0.25}, serving_io=_serving_io(),
    )
    svc_2l = assemble(raw_2l, scope="service", iterations=3, tokens=1, diagnostic_window=True)
    assert svc_2l["same_context"] is True
    assert (svc_2l["triggers"]["batch_prepare"], svc_2l["triggers"]["model_invocation"]) == (3, 3)
    assert (svc_2l["triggers"]["graph_replay_submissions"], svc_2l["execution"]["submission"]["graph_replays"]) == (3, 3)
    assert (svc_2l["triggers"]["request_commit_callbacks"], svc_2l["metrics"]["commit_events"]) == (6, 6)

    mdl_2l = assemble(raw_2l, scope="model", iterations=3, tokens=1, diagnostic_window=True)
    assert mdl_2l["framework_fragments"] == []
    parsed = parse_case(raw_2l)
    plan = IterationPlan((RequestView("r0", 1, 4, True, "decode"), RequestView("r1", 1, 3, True, "decode")))
    pure_prog = Frontend().program(plan, parsed.model, parsed.runtime, parsed.machine, assembly="model")
    svc_prog = Frontend().program(plan, parsed.model, parsed.runtime, parsed.machine, assembly="service")
    assert not ({op.family for op in pure_prog.work.ops} & SERVICE_FAMILIES)
    assert {"sampling", "token_commit"} <= {op.family for op in svc_prog.work.ops}

    fw = assemble(raw_2l, scope="framework", iterations=3, emitted_tokens={"r0": 1, "r1": 1})
    assert fw["model_fragment"] is None and len(fw["framework_fragments"]) >= 4
    assert _fragment("host_prepare", (("scheduler@r0", 0.5, "ms"),)).semantic_refs == ()
    assert any(frag["work"] for frag in fw["framework_fragments"])
    for frag in fw["framework_fragments"]:
        assert frag["layer"] is None and frag.get("semantic_refs", []) == []
        assert frag["origin"] == "framework" and frag["trigger"].startswith("per_")
        assert all(row["resource"] for row in frag["work"])
        ports = frag["ports"]["event_in"] + frag["ports"]["event_out"]
        assert ports and all(p["scope"] and p["direction"] in {"in", "out"} for p in ports)

    raw_4l = deepcopy(raw_2l)
    raw_4l["model"]["layers"] = 4
    svc_4l = assemble(raw_4l, scope="service", iterations=3, tokens=1, diagnostic_window=True)
    assert (svc_4l["triggers"]["batch_prepare"], svc_4l["triggers"]["model_invocation"]) == (3, 3)
    assert (svc_4l["triggers"]["graph_replay_submissions"], svc_4l["triggers"]["request_commit_callbacks"]) == (3, 6)
    assert svc_4l["execution"]["work"]["flops"] > svc_2l["execution"]["work"]["flops"]


def test_x2_exact_mixed_types_attrs_and_bucket_fidelity() -> None:
    act_val = TensorValue("act", "bf16", TensorLayout((3, 8), (8, 1), storage_shape=(4, 8)))
    state_val = TensorValue("ssm_state", "fp32", TensorLayout((3, 8), (8, 1)), state_version=7)
    idx_val = TensorValue("token_index", "int64", TensorLayout((3,), (1,)))
    q_val = TensorValue("quant_act", "int8", TensorLayout((3, 8), (8, 1)))
    s_val = TensorValue("quant_scale", "fp32", TensorLayout((3, 1), (1, 1)))
    assert act_val.layout.shape == (3, 8) and act_val.layout.storage_shape == (4, 8)
    assert (state_val.dtype, state_val.state_version, idx_val.dtype, q_val.dtype) == ("fp32", 7, "int64", "int8")
    assert s_val.dtype == "fp32" and s_val.layout.shape == (3, 1)

    a5_raw = compose_case("qwen3.6-27b", hardware="a5", layers=1, prompt_tokens=3, output_tokens=1)
    parsed = parse_case(a5_raw)
    q_spec = replace(quant_preset("w8a8"), activation_group_size=8, activation_scale_format="fp32")
    model_q = replace(parsed.model, quant=q_spec)
    norm_raw = norm(name="x2_norm", deps=(), stage=0, layer=0, requests=("r0", "r1", "r2"),
                    tokens=3, width=8, model=model_q, mode="prefill")
    norm_op = replace(norm_raw, metadata=norm_raw.metadata + (
        ("epsilon", 1e-5), ("graph_active_query_rows", 3), ("graph_physical_query_rows", 4),
    ))
    proj_op = gemm_op(op_id="x2_proj", deps=("x2_norm",), stage=0, layer=0, requests=("r0", "r1", "r2"),
                      m=3, n=8, k=8, quant=q_spec, category="dense_mlp", phase="prefill", recipe="dense_gate_up",
                      metadata=(("transpose_rhs", True), ("accumulator_dtype", "int32"), ("output_format", "bf16")))
    lowered = activation_quantization((norm_op, proj_op), "bf16")
    quant_op = next(op for op in lowered if op.family == "quantize")
    q_outputs = {row["output"]: row for row in audit_boundary(quant_op)["typed_outputs"]}
    assert q_outputs["payload"]["dtype"] == "int8" and q_outputs["payload"]["shape"] == [3, 8]
    assert q_outputs["activation_scales"]["dtype"] == "fp32" and q_outputs["activation_scales"]["shape"] in ([3], [3, 1])
    assert q_outputs["activation_scales"]["bytes"] == pytest.approx(12.0)
    assert dict(norm_op.metadata)["epsilon"] == pytest.approx(1e-5) and dict(proj_op.metadata)["transpose_rhs"] is True

    causal_pairs, dense_pairs = logical_causal_pairs(3, 0), 3 * 3
    heads, dim = 2, 8
    causal_flops, dense_flops = 2 * heads * causal_pairs * (2 * dim), 2 * heads * dense_pairs * (2 * dim)
    assert (causal_pairs, dense_pairs, causal_flops, dense_flops) == (6, 9, 384, 576)
    gqa_meta = {"requests": ((3, 0),), "query_heads": heads, "kv_heads": 2, "head_dim": dim,
                "query_dtype": "bf16", "kv_dtype": "bf16", "causal": True, "window": 0, "cp": 1}
    attn_op = OpSpec("gqa_prefill", "attention", "prefill", 0, 0, ("r0",),
                     (3, 3, heads, dim), (3, 3, heads, dim), (3, 3, heads, dim), (3, 3, heads, dim),
                     flops=causal_flops, dtype="bf16", recipe="flash_attention_prefill",
                     metadata=(("gqa_prefill", gqa_meta), ("epsilon", 1e-5)))
    staged = StagedGQAPrefill(query_tile=4, key_tile=4).report(attn_op)
    assert (staged["logical_pairs"], staged["executed_rectangle_pairs"]) == (12, 18)

    assert simulate_operator(a5_raw, (norm_op, quant_op, proj_op))["operator_sum_ms"] > 0
    fused_sub = norm_quant(lowered, "bf16", consumers=({"recipe": "dense_gate_up"},))
    fused_node = next(op for op in fused_sub if op.family == "rmsnorm_quant")
    assert simulate_graph(a5_raw, fused_sub)["duration_ms"] > 0
    assert dict(fused_node.metadata).get("epsilon", 1e-5) == pytest.approx(1e-5)

    bucket_case = _compact_case(layers=1, requests=[
        {"id": f"r{i}", "prompt_tokens": 1, "output_tokens": 2} for i in range(3)
    ])
    bucket_case["runtime"].update(
        graph_mode="auto", graph_capture_mode="full", graph_capture_keys=[4], serving_io=_serving_io(),
    )
    assert simulate_model(bucket_case, phase="decode", tokens=1, context=4, sequences=3)["body_forward_ms"] > 0
    assert simulate_service(bucket_case)["status"] == "modeled"
    svc_win = assemble(bucket_case, scope="service", iterations=1, tokens=1, diagnostic_window=True)
    assert svc_win["steps"][0]["padded_tokens"] == 4
    assert svc_win["triggers"]["request_commit_callbacks"] == 3 and len(svc_win["steps"][0]["commit_ms"]) == 3

    declared = declare_native_call(
        proj_op,
        api={"namespace": "cann_kernel", "entry": "QuantBatchMatmulV3"},
        typed={"dtype": "int8", "shapes": {"local": [3, 8, 8]}, "attrs": {"transpose_rhs": True, "epsilon": 1e-5}},
        kernels=[{"name": "QuantBatchMatmulV3", "role": "cube", "evidence": "source_declared"}],
    )
    plan = OperatorPlan(Work(matrix={"int8": proj_op.flops}, resources={"hbm": proj_op.hbm_bytes}), lambda _ctx: None)
    binding = bind_native_call(declared, plan=plan)
    validate_invocation(binding, {"dtype": "int8", "layout": binding.typed["layout"],
                                  "shapes": binding.typed["shapes"], "attrs": {"transpose_rhs": True, "epsilon": 1e-5}})
    with pytest.raises(ModelingError, match="attr_mismatch"):
        validate_invocation(binding, {"attrs": {"transpose_rhs": False, "epsilon": 1e-5}})
    assert simulate_native_call(a5_raw, (replace(declared, deps=()),))["duration_ms"] > 0
    joined = join_profiler_rows([{"name": "QuantBatchMatmulV3"}], [binding])
    assert (joined["joined"], joined["unmatched"]) == (1, 0)


def test_x3_exact_multistream_host_lookahead_and_shared_resource() -> None:
    def run_variant(*, ordered: bool = False, shared: bool = False) -> tuple[float, dict[str, float]]:
        resources = ([Resource("S", 1.0, unit="service_us", policy="fifo")] if shared else
                     [Resource("C", 1.0, unit="service_us", policy="fifo"),
                      Resource("H", 1.0, unit="service_us", policy="fifo")])
        eng = Engine(resources)
        res_a, res_b = ("S", "S") if shared else ("C", "H")
        if ordered:
            ev_b = eng.submit("framework_B", {res_b: 5.0})
            ev_a = eng.submit("model_A", {res_a: 8.0}, after=(ev_b,))
        else:
            ev_a = eng.submit("model_A", {res_a: 8.0})
            ev_b = eng.submit("framework_B", {res_b: 5.0})
        join = eng.submit("join", latency=0.0, after=(ev_a, ev_b))
        eng.advance(until=(join,), first=False)
        return float(join.time), dict(eng.serviced)

    indep_us, indep_work = run_variant()
    same_us, _ = run_variant(ordered=True)
    wait_us, _ = run_variant(ordered=True)
    shared_us, shared_work = run_variant(shared=True)
    assert indep_us == pytest.approx(8.0) and indep_work == {"C": 8.0, "H": 5.0}
    assert (same_us, wait_us, shared_us) == (pytest.approx(13.0), pytest.approx(13.0), pytest.approx(13.0))
    assert shared_work == {"S": 13.0}

    lookahead_eng = Engine([Resource("C", 1.0, unit="service_us"), Resource("HOST", 1.0, unit="service_us")])
    cur_dev = lookahead_eng.submit("device_step", {"C": 8.0})
    indep_prep = lookahead_eng.submit("next_prepare_indep", {"HOST": 2.0})
    lookahead_eng.advance(until=(cur_dev, indep_prep), first=False)
    assert (indep_prep.time, cur_dev.time) == (pytest.approx(2.0), pytest.approx(8.0))

    dep_eng = Engine([Resource("C", 1.0, unit="service_us"), Resource("HOST", 1.0, unit="service_us")])
    accept_ev = dep_eng.submit("device_accept", {"C": 8.0})
    dep_prep = dep_eng.submit("next_prepare_dep", {"HOST": 2.0}, after=(accept_ev,))
    dep_eng.advance(until=(dep_prep,), first=False)
    assert (dep_prep.started, dep_prep.time) == (pytest.approx(8.0), pytest.approx(10.0))

    raw = _compact_case(layers=1, requests=[{"id": "r0", "prompt_tokens": 4, "output_tokens": 2}])
    raw["machine"]["device"]["launch_us"] = 100.0
    raw["runtime"]["serving_io"] = _serving_io(prepare_ms=0.2, result_ms=0.2, delivery_ms=0.5)
    step0 = assemble(raw, scope="service", iterations=1, tokens=1, diagnostic_window=True)["steps"][0]
    assert step0["submitted_last_ms"] < step0["device_done_ms"]


def test_x4_exact_materialization_240_112_176_and_capture() -> None:
    raw = compose_case("qwen3.6-27b", hardware="a5", layers=1, prompt_tokens=4, output_tokens=1)
    parsed = parse_case(raw)
    q_g8 = replace(quant_preset("w8a8"), activation_group_size=8, activation_scale_format="fp32")
    model_q = replace(parsed.model, quant=q_g8)

    norm_raw = norm(name="n0", deps=("x_in",), stage=0, layer=0, requests=("r0",),
                    tokens=4, width=8, model=model_q, mode="prefill")
    norm_64 = replace(norm_raw, hbm_bytes=128.0, mte2_bytes=64.0, mte3_bytes=64.0)
    proj = gemm_op(op_id="proj", deps=("n0",), stage=0, layer=0, requests=("r0",),
                   m=4, n=8, k=8, quant=q_g8, category="dense_mlp", phase="prefill", recipe="dense_gate_up")
    ext_consumer = add_input_ports(
        vector_op(op_id="ext_norm_reader", deps=("n0",), stage=0, layer=0, requests=("r0",),
                  elements=32, ops_per_element=1, dtype="bf16", family="elementwise",
                  phase="prefill", recipe="residual_add", read_bytes_override=64, write_bytes_override=64),
        ({"op": "n0", "output": "norm", "source_op": "n0"},),
    )

    sep_ops = activation_quantization((norm_64, proj), "bf16")
    quant_op = next(op for op in sep_ops if op.family == "quantize")
    assert (norm_64.mte2_bytes, norm_64.mte3_bytes, quant_op.mte2_bytes, quant_op.mte3_bytes) == (64.0, 64.0, 64.0, 48.0)
    assert norm_64.hbm_bytes + quant_op.hbm_bytes == pytest.approx(240.0)

    nocache = CacheState(CacheGeometry(capacity_bytes=0, line_bytes=16, transfer_bytes=16))
    sep_hbm = (nocache.access("x_act", 64).hbm_read_bytes + nocache.access("norm_out", 64, write=True).hbm_write_bytes
               + nocache.access("norm_out", 64).hbm_read_bytes + nocache.access("q_out", 48, write=True).hbm_write_bytes)
    assert sep_hbm == 240

    fused_elim = norm_quant(sep_ops, "bf16", consumers=({"recipe": "dense_gate_up"},), external_output="eliminate")
    fused_112 = next(op for op in fused_elim if op.family == "rmsnorm_quant")
    assert fused_112.hbm_bytes == pytest.approx(112.0) and (fused_112.mte2_bytes, fused_112.mte3_bytes) == (64.0, 48.0)
    assert len(fused_elim) == len(sep_ops) - 1 and "n0" not in {op.op_id for op in fused_elim}
    assert audit_boundary(fused_112)["replaced_occurrences"] == ("n0", quant_op.op_id)

    sep_with_ext = activation_quantization((norm_64, proj, ext_consumer), "bf16")
    fused_mat = norm_quant(sep_with_ext, "bf16", consumers=({"recipe": "dense_gate_up"},), external_output="materialize")
    fused_176 = next(op for op in fused_mat if op.family == "rmsnorm_quant")
    assert fused_176.hbm_bytes == pytest.approx(176.0) and (fused_176.mte2_bytes, fused_176.mte3_bytes) == (64.0, 112.0)
    reconnected_ext = next(op for op in fused_mat if op.op_id == "ext_norm_reader")
    assert any(p["op"] == fused_176.op_id and p["output"] == "norm" for p in input_ports(reconnected_ext))

    illegal_in = tuple(
        replace(op, metadata=op.metadata + (("graph_captured", True), ("graph_segment", 2)))
        if op.op_id == "n0" else op for op in sep_with_ext
    )
    preserved = norm_quant(illegal_in, "bf16", consumers=({"recipe": "dense_gate_up"},), external_output="materialize")
    assert not any(op.family == "rmsnorm_quant" for op in preserved)
    assert {op.op_id for op in preserved} == {op.op_id for op in illegal_in}

    cap_case = _compact_case(layers=1)
    cap_case["machine"]["device"]["launch_us"] = 200.0
    cap_case["runtime"]["submission"] = {"graph_replay_ms": 0.15}
    k0 = vector_op(op_id="k0", deps=(), stage=0, layer=0, requests=("r0",), elements=64,
                   ops_per_element=2, dtype="bf16", family="rmsnorm", phase="prefill", recipe="rmsnorm")
    k1 = vector_op(op_id="k1", deps=("k0",), stage=0, layer=0, requests=("r0",), elements=64,
                   ops_per_element=2, dtype="bf16", family="quantize", phase="prefill", recipe="dynamic_quant")
    eager_rep = simulate_graph(cap_case, (k0, k1), replay_scope="none", include_submission=True, isolated=False)
    replay_rep = simulate_graph(cap_case, (k0, k1), replay_scope="full", include_submission=True, isolated=False)
    assert (eager_rep["execution"]["submission"]["eager_invocations"], replay_rep["execution"]["submission"]["graph_replays"]) == (2, 1)
    assert eager_rep["execution"]["work"] == replay_rep["execution"]["work"] and len(eager_rep["timeline"]) == len(replay_rep["timeline"]) == 2


def test_x5_exact_state_visibility_pool_cache_and_route() -> None:
    conductor = FragmentConductor(Engine(()), state={"kv": 7})
    cand_rej = conductor.submit(_fragment(
        "propose_reject", state_effects=(StateEffect("kv", "read", version=7), StateEffect("kv", "write", version=8)),
    ))
    rej = conductor.submit(_fragment("reject", state_effects=(StateEffect("kv", "rollback", version=8),)), after=(cand_rej.event,))
    conductor.advance(until=(rej.event,), first=False)
    v_after_reject = conductor.state.committed_version("kv")

    cand_acc = conductor.submit(_fragment(
        "propose_accept", state_effects=(StateEffect("kv", "read", version=7), StateEffect("kv", "write", version=8)),
    ))
    acc = conductor.submit(_fragment("accept", state_effects=(StateEffect("kv", "commit", version=8),)), after=(cand_acc.event,))
    read_next = conductor.submit(_fragment("next_read", state_effects=(StateEffect("kv", "read", version=8),)), after=(acc.event,))
    conductor.advance(until=(read_next.event,), first=False)
    assert (v_after_reject, conductor.state.committed_version("kv"), conductor.state.committed_version("kv")) == (7, 8, 8)

    pool_eng = Engine([Resource("C", 4096.0, unit="bytes")], pools={"kv_pool": 12288})
    pool_cond = FragmentConductor(pool_eng)
    async_copy_done = pool_eng.submit("async_copy_last_use", latency=5.0)
    committed = pool_cond.submit(
        _fragment("committed_4k", (("C", 4096.0, "bytes"),), allocations=(AllocationEffect.acquire("c_slot", "kv_pool", 4096),)),
        holds={"c_slot": pool_eng.submit("hold_committed", latency=8.0)},
    )
    candidate = pool_cond.submit(
        _fragment("candidate_4k", (("C", 4096.0, "bytes"),), allocations=(AllocationEffect.acquire("cand_slot", "kv_pool", 4096),)),
        holds={"cand_slot": async_copy_done},
    )
    scratch = pool_cond.submit(
        _fragment("scratch_4k", (("C", 4096.0, "bytes"),), allocations=(AllocationEffect.acquire("scr_slot", "kv_pool", 4096),)),
        holds={"scr_slot": pool_eng.submit("hold_scratch", latency=8.0)},
    )
    fourth = pool_cond.submit(
        _fragment("fourth_4k", (("C", 4096.0, "bytes"),),
                  allocations=(AllocationEffect.acquire("f_slot", "kv_pool", 4096), AllocationEffect.release("f_slot", "kv_pool", 4096))),
        after=(pool_eng.submit("fourth_arrival", latency=3.5),),
    )
    pool_cond.advance(horizon=4.5)
    assert committed.event.time is not None and candidate.event.time is not None and scratch.event.time is not None
    assert sum(pool_eng.allocated.values()) == pytest.approx(12288.0) and fourth.event.started is None
    pool_cond.advance(until=(fourth.event,), first=False)
    assert (candidate.release_events[0].time, fourth.event.started, fourth.event.time) == (
        pytest.approx(5.0), pytest.approx(5.0), pytest.approx(6.0)
    )

    cached = CacheState(CacheGeometry(capacity_bytes=128, line_bytes=128, transfer_bytes=32))
    assert [cached.access("kv0", 128).hbm_read_bytes, cached.access("kv0", 128).hbm_read_bytes] == [128, 0]
    uncached = CacheState(CacheGeometry(capacity_bytes=0, line_bytes=128, transfer_bytes=32))
    assert [uncached.access("kv0", 128).hbm_read_bytes, uncached.access("kv0", 128).hbm_read_bytes] == [128, 128]

    a5_case = compose_case("qwen3.6-27b", hardware="a5", layers=1, prompt_tokens=8, output_tokens=1)
    tiles_26, tiles_53 = grouped_owner_tiles((2, 6), 256, 4, 128, 4), grouped_owner_tiles((5, 3), 256, 4, 128, 4)
    rects_26, rects_53 = grouped_owner_rectangles((2, 6), 256, 4, 128, 4), grouped_owner_rectangles((5, 3), 256, 4, 128, 4)
    assert sum((2, 6)) == sum((5, 3)) == 8 and rects_26 != rects_53 and tiles_26 != tiles_53
    op_26 = gemm_op(op_id="moe_26", deps=(), stage=0, layer=0, requests=("r0",), m=8, n=256, k=256,
                    quant=quant_preset("bf16"), category="moe", phase="decode", recipe="grouped_matmul", metadata=(("group_rows", (2, 6)),))
    op_53 = gemm_op(op_id="moe_53", deps=(), stage=0, layer=0, requests=("r0",), m=8, n=256, k=256,
                    quant=quant_preset("bf16"), category="moe", phase="decode", recipe="grouped_matmul", metadata=(("group_rows", (5, 3)),))
    assert simulate_operator(a5_case, (op_26,))["operator_sum_ms"] > 0 and simulate_operator(a5_case, (op_53,))["operator_sum_ms"] > 0


def test_x6_exact_service_stability_50_95_105_and_109_flip() -> None:
    body_ms = 1000.0 / 109.0
    io_overhead_ms = 10.0 - body_ms

    def fixed_step_lower(plan, model, runtime, machine):
        tokens = sum(view.q_len for view in plan.requests)
        op = OpSpec("step_work", "vector", "body", 0, 0, tuple(v.request_id for v in plan.requests),
                    (tokens,), (tokens,), (tokens,), (tokens,),
                    flops=tokens * body_ms * 1e6,
                    vector_ops=tokens * body_ms * 1e6, dtype="bf16")
        return IterationWork((op,), plan, 1, tokens, tokens, None, "body", "x6_fixed")

    def run_rate(rate_rps: float, *, include_framework: bool, count: int = 12) -> dict:
        gap_ms = 1000.0 / rate_rps
        reqs = [{"id": f"q{i}", "prompt_tokens": 1, "output_tokens": 1, "arrival_ms": i * gap_ms} for i in range(count)]
        raw = _compact_case(layers=1, requests=reqs)
        raw["runtime"].update(max_num_seqs=1, max_num_batched_tokens=1)
        if include_framework:
            raw["runtime"]["serving_io"] = {"host_prepare_ms": io_overhead_ms * 0.5, "host_result_ms": io_overhead_ms * 0.5, "client_delivery_ms": 0.0}
        return simulate_service(raw, lower=fixed_step_lower)

    rep_50, rep_95 = run_rate(50.0, include_framework=True), run_rate(95.0, include_framework=True)
    rep_105, rep_109_opt = run_rate(105.0, include_framework=True), run_rate(105.0, include_framework=False)
    ttft_50, ttft_95 = [r["ttft_ms"] for r in rep_50["requests"]], [r["ttft_ms"] for r in rep_95["requests"]]
    ttft_105, ttft_109_opt = [r["ttft_ms"] for r in rep_105["requests"]], [r["ttft_ms"] for r in rep_109_opt["requests"]]

    assert max(ttft_50) - min(ttft_50) < 1e-6 and max(ttft_95) - min(ttft_95) < 1e-6
    assert all(ttft_105[i + 1] > ttft_105[i] + 0.4 for i in range(len(ttft_105) - 1))
    assert max(ttft_109_opt) - min(ttft_109_opt) < 1e-6 and ttft_105[-1] > ttft_109_opt[-1] + 5.0
    assert rep_95["execution"]["work"]["flops"] == pytest.approx(12 * body_ms * 1e6) and rep_95["metrics"]["makespan_ms"] > 100.0
    assert rep_95["execution"]["assembly"]["fallback_commits"] == 12
    assert rep_95["memory"]["peak_combined_bytes_per_replica_stage"][0][0] > 0 and "p50_ms" not in rep_95["metrics"]


def test_x7_exact_sku_datapath_gm_control_and_heterogeneous_placement() -> None:
    a2 = Hardware.from_mapping(_sku_machine("Ascend910B1")).dies[0]
    a3 = Hardware.from_mapping(_sku_machine("Ascend910_9392")).dies[0]
    dt = Hardware.from_mapping(_sku_machine("Ascend950DT_9581")).dies[0]
    pr = Hardware.from_mapping(_sku_machine("Ascend950PR_9579")).dies[0]

    for die in (a2, a3, dt, pr):
        assert die.compute_arch.representation_supports("fp8_e4m3", "storage")
        assert die.compute_arch.representation_supports("fp8_e4m3", "transfer")
    assert not a2.compute_arch.supports_native("matrix", "fp8_e4m3") and not a3.compute_arch.supports_native("matrix", "fp8_e4m3")
    assert dt.compute_arch.supports_native("matrix", "fp8_e4m3") and pr.compute_arch.supports_native("matrix", "fp8_e4m3")
    assert a2.compute_arch.can_cast("bf16", "fp32") and a2.compute_arch.cast_rule("bf16", "fp32") is not None
    gated_arch = replace(dt.compute_arch, native_modes=(
        NativeComputeMode("fp8_gated", "matrix", ("fp8_e4m3",), "cube", ("fp32",), implementations=("cann_fp8",)),
    ))
    with pytest.raises(ModelingError):
        gated_arch.implemented_native_mode("matrix", "fp8_e4m3", "fp32", RuntimeCapability("cann", "9.0", provides=()))

    assert a2.compute_arch.path("gm", "l0a") is not None and a3.compute_arch.path("l1", "gm") is not None
    assert a2.compute_arch.path("l0c", "ub") is None and a3.compute_arch.path("ub", "l1") is None
    assert dt.compute_arch.path("l0c", "ub") is not None and pr.compute_arch.path("ub", "l1") is not None
    assert dt.compute_arch.path("gm", "l0a") is None and pr.compute_arch.path("l1", "gm") is None

    nocache = CacheState(CacheGeometry(capacity_bytes=0, line_bytes=128, transfer_bytes=64))
    assert (nocache.access("gm_stage_buf", 4096, write=True).hbm_write_bytes, nocache.access("gm_stage_buf", 4096).hbm_read_bytes) == (4096, 4096)

    tight_ub = Engine([Resource("ub_pipe", 4096.0, unit="bytes")], pools={"ub": 4096})
    with pytest.raises(ExecutionError):
        tight_ub.submit("oversubscribed_ub", {"ub_pipe": 4096.0}, acquire=(EngineAllocation("two_slots", "ub", 8192),))

    hetero = Hardware.from_mapping({
        "name": "hetero-a5", "backend": "ascend", "sku": "Ascend950DT_9581", "num_devices": 2,
        "devices": [{"sku": "Ascend950DT_9581"}, {"sku": "Ascend950PR_9579"}],
        "links": [{"name": "shared_hccs", "members": [0, 1], "capacity_bytes_per_ms": 100_000_000,
                   "semantics": "shared", "endpoint_rates": {"0": 80_000_000, "1": 40_000_000}}],
    })
    assert (hetero.dies[0].sku, hetero.dies[1].sku) == ("Ascend950DT_9581", "Ascend950PR_9579")
    assert hetero.dies[0].memory_bytes != hetero.dies[1].memory_bytes
    dur_us, lb_01 = _run_transport_us(hetero, ParallelSpec(tp=2), (MessageFlow(0, 1, 40_000_000),))
    assert dur_us == pytest.approx(1000.0) and lb_01["service_ms"] == pytest.approx(1.0)
    assert any("shared_hccs" in name for name in lb_01["bottleneck_resources"])


def test_x8_exact_h2_kv_quant_allocation_and_frozen_access() -> None:
    bf16 = KVPageAllocation((17, 31), page_tokens=16, local_kv_heads=1, head_dim=8, kv_count=2, element_bytes=2, metadata_bytes_per_page=32)
    int8 = KVPageAllocation((17, 31), page_tokens=16, local_kv_heads=1, head_dim=8, kv_count=2, element_bytes=1, scale_bytes_per_page=8, metadata_bytes_per_page=32)
    assert bf16.logical_tokens == int8.logical_tokens == 48 and bf16.pages == int8.pages == 4
    assert bf16.physical_token_slots == int8.physical_token_slots == 64
    assert (bf16.total_allocation_bytes, int8.total_allocation_bytes) == (2176, 1184)
    assert (bf16.access_bytes, int8.access_bytes, bf16.page_tail_payload_bytes, int8.page_tail_payload_bytes) == (1664, 928, 512, 256)

    parsed_tp2 = parse_case(_compact_case(tp=2, layers=1, prompt=17, output=1))
    assert Memory(parsed_tp2).weights[0] > 0
    model_int8_kv = replace(parsed_tp2.model, metadata=(("cache_representations", {
        "mla_kv": {"dtype": "int8", "group_size": 8, "scale_dtype": "fp32"},
    }),))
    assert cache_row_split(model_int8_kv, "mla_kv", 16) == (16, 8) and cache_format(model_int8_kv, "mla_kv").row_bytes(16) == 24

    def build_h3(kv_b: int, ws_b: int, *, ws_copies: int = 1, sequential: bool = False) -> dict:
        items = [
            PlanAllocation("weights", 536_870_912, "weights"),
            PlanAllocation("runtime_persistent", 67_108_864, "runtime"),
            PlanAllocation("comm_pool", 67_108_864, "pool"),
            PlanAllocation("comm_tx", 32 * MIB, "communication", pool="comm_pool"),
            PlanAllocation("comm_rx", 32 * MIB, "communication", pool="comm_pool"),
            PlanAllocation("weights_alias", 4096, "workspace", alias_of="weights"),
            PlanAllocation("kv", kv_b, "cache"),
            PlanAllocation("workspace", ws_b, "workspace"),
        ]
        if ws_copies == 2:
            items.append(PlanAllocation("workspace_2", ws_b, "workspace", live=not sequential))
        return AllocationPlan(tuple(items), capacity_bytes=1024 ** 3).to_dict()

    r_bf16_128, r_bf16_96 = build_h3(268_566_528, 134_217_728), build_h3(268_566_528, 100_663_296)
    r_int8_128 = build_h3(134_873_088, 134_217_728)
    r_conc, r_seq = build_h3(134_873_088, 134_217_728, ws_copies=2, sequential=False), build_h3(134_873_088, 134_217_728, ws_copies=2, sequential=True)
    assert (r_bf16_128["peak_bytes"], r_bf16_128["feasible"]) == (1_073_872_896, False)
    assert (r_bf16_96["peak_bytes"], r_bf16_96["feasible"]) == (1_040_318_464, True)
    assert (r_int8_128["peak_bytes"], r_int8_128["feasible"]) == (940_179_456, True)
    assert r_bf16_128["alias_bytes_not_added"] == 4096 and r_bf16_128["pool_bytes_not_added"] == 64 * MIB
    assert r_conc["peak_bytes"] - r_seq["peak_bytes"] == 134_217_728 and r_conc["feasible"] is False and r_seq["feasible"] is True

    infeasible_svc = simulate_service(_compact_case(layers=1), memory_policy=MemoryPolicy(workspace_bytes=2 * 1024 ** 3))
    assert (infeasible_svc["status"], infeasible_svc["memory"]["status"]) == ("infeasible", "oom")


def test_x9_exact_source_separation_public_profile_and_detail_invariance(tmp_path: Path) -> None:
    nominal_rates = {"mte2_aiv": {"rate": 128_000_000.0, "work_unit": "bytes", "rate_scope": "aggregate"}}
    fitted = fit_resource_service(
        {
            "id": "h5-transfer-calibrated", "scope": "independent_resource", "resources": ["mte2_aiv"],
            "phase": "prefill", "work_unit": "bytes", "boundary": "resource_phase",
            "active_units": 1, "hardware": "Ascend950DT_9581",
            "samples": [
                {"work": 384, "duration_us": 6.0, "run_id": "r0", "split": "train", "source": "cal_0"},
                {"work": 768, "duration_us": 10.0, "run_id": "r1", "split": "train", "source": "cal_1"},
            ],
        },
        nominal_rates=nominal_rates, hardware="Ascend950DT_9581",
    )
    service_rec = fitted["service"] if "service" in fitted else fitted["record"]
    norm_svc = normalize_resource_service(service_rec, nominal_rates=nominal_rates, hardware="Ascend950DT_9581")
    norm_again = normalize_resource_service(norm_svc.to_dict(), nominal_rates=nominal_rates, hardware="Ascend950DT_9581")
    assert norm_svc.startup_us == pytest.approx(2.0) and norm_svc.effective_rate == pytest.approx(96_000_000.0)
    assert norm_svc.nominal_multiplier == pytest.approx(0.75) and norm_svc.duration_us(1536) == pytest.approx(18.0)
    assert 2.0 + 1536.0 / 128.0 == pytest.approx(14.0) and norm_again.effective_rate == pytest.approx(norm_svc.effective_rate)

    pkg_dir = Path(mindie_sim.__file__).resolve().parent
    pkg_before = sorted(p.name for p in pkg_dir.iterdir())
    profile_path = tmp_path / "user_profile_h5.json"
    profile_path.write_text(json.dumps({
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA, "hardware": "Ascend950DT_9581", "services": [norm_svc.to_dict()],
    }), encoding="utf-8")
    loaded_prof = load_user_profile(profile_path, nominal_rates=nominal_rates, hardware="Ascend950DT_9581")
    assert loaded_prof.services[0].effective_rate == pytest.approx(96_000_000.0)
    assert sorted(p.name for p in pkg_dir.iterdir()) == pkg_before
    with pytest.raises((ExtensionError, ValueError)):
        load_user_profile(profile_path, nominal_rates=nominal_rates, hardware="Ascend910B1")

    a5_case = compose_case("qwen3.6-27b", hardware="Ascend950DT_9581", layers=1, prompt_tokens=16, output_tokens=1)
    op = gemm_op(op_id="p0_gemm", deps=(), stage=0, layer=0, requests=("r0",), m=128, n=512, k=512,
                 quant=quant_preset("bf16"), category="dense_mlp", phase="prefill", recipe="dense_gate_up")
    _ = simulate_operator(a5_case, (op,))
    t0 = time.perf_counter()
    warm_rep = simulate_operator(a5_case, (op,))
    assert warm_rep["operator_sum_ms"] > 0 and (time.perf_counter() - t0) < 0.25

    g_agg = simulate_graph(a5_case, (op,), trace_limit=0, isolated=False)
    g_det = simulate_graph(a5_case, (op,), trace_limit=200, isolated=False)
    assert g_agg["duration_ms"] == pytest.approx(g_det["duration_ms"]) and g_agg["execution"]["work"] == g_det["execution"]["work"]
    assert len(g_agg.get("events", ())) == 0 and len(g_det.get("events", ())) > 0


def test_chain_a_exact_unseen_shapes_and_interventions() -> None:
    a5_case = compose_case("qwen3.6-27b", hardware="Ascend950DT_9581", layers=1, prompt_tokens=7, output_tokens=1)
    dense_unseen = gemm_op(
        op_id="unseen_dense", deps=(), stage=0, layer=0, requests=("r0",), m=5, n=19, k=13,
        quant=quant_preset("bf16"), category="dense_mlp", phase="prefill", recipe="dense_gate_up",
    )
    grouped_unseen = gemm_op(
        op_id="unseen_grouped", deps=(), stage=0, layer=0, requests=("r0",), m=7, n=37, k=17,
        quant=quant_preset("bf16"), category="moe", phase="prefill", recipe="grouped_matmul", metadata=(("group_rows", (2, 5)),),
    )
    assert dense_unseen.flops == 2 * 5 * 19 * 13 and grouped_unseen.flops == 2 * 7 * 37 * 17
    assert simulate_operator(a5_case, (dense_unseen, grouped_unseen))["operator_sum_ms"] > 0
    assert (2 * 2 * logical_causal_pairs(3, 0) * 16, 2 * 2 * 9 * 16) == (384, 576)

    plan = prepare_fixed_plan(a5_case, phase="prefill", tokens=13, context=0, sequences=1)
    base_eval = evaluate_fixed_plan(a5_case, plan=plan, phase="prefill", tokens=13, context=0, sequences=1)
    for res_name in ("cube_flops", "hbm_bw", "vector_ops"):
        cf = evaluate_resource_counterfactual(
            a5_case, resource=res_name, factor=0.5, phase="prefill", tokens=13, context=0, sequences=1, plan=plan,
        )
        assert cf["work_conservation"]["conserved"] is True
        assert cf["counterfactual"]["graph_duration_ms"] >= base_eval["graph_duration_ms"]
        if res_name in {"cube_flops", "hbm_bw"}:
            assert cf["counterfactual"]["graph_duration_ms"] > base_eval["graph_duration_ms"]

    nocache = CacheState(CacheGeometry(capacity_bytes=0, line_bytes=128, transfer_bytes=64))
    assert (nocache.access("h1_gm", 4096, write=True).hbm_write_bytes, nocache.access("h1_gm", 4096).hbm_read_bytes) == (4096, 4096)


def test_chain_d_five_interactions_and_p4_service_delivery() -> None:
    c1 = compose_case("qwen3.6-27b", hardware="a5", tp=2, layers=1, quant="w8a8", prompt_tokens=8, output_tokens=2, concurrency=1)
    c2 = compose_case("qwen3.6-27b", hardware="a5", tp=1, pcp=2, layers=1, block_size=16, prompt_tokens=16, output_tokens=2, concurrency=1)
    c3 = compose_case("qwen3.5-35b-a3b", hardware="a5", tp=1, dp=2, ep=2, layers=1, prompt_tokens=8, output_tokens=2, concurrency=2)
    c4 = compose_case("qwen3.5-35b-a3b", hardware="a5", tp=2, ep=2, layers=1, prompt_tokens=8, output_tokens=3, concurrency=1,
                      mtp_proposal_tokens=2, mtp_acceptance_length=1.8, graph_capture_mode="full")
    c4["runtime"]["graph_capture_keys"] = [1, 4]
    for cfg in (c1, c2, c3, c4):
        cfg["runtime"]["serving_io"] = _serving_io(0.05, 0.05, 0.05)

    base_pd = _compact_case(layers=1, requests=[
        {"id": "pd0", "prompt_tokens": 6, "output_tokens": 2, "prefix_tokens": 2},
        {"id": "pd1", "prompt_tokens": 6, "output_tokens": 2, "prefix_tokens": 2},
    ])
    base_pd["runtime"].update(enable_chunked_prefill=True, max_num_batched_tokens=4, max_num_seqs=1)
    c5 = deepcopy(base_pd)
    c5["pools"] = {k: {"machine": deepcopy(base_pd["machine"]), "runtime": deepcopy(base_pd["runtime"])} for k in ("prefill", "decode")}
    c5["connector"] = {"bytes_per_s": 500_000, "latency_us": 10}
    interactions = [simulate_service(cfg) for cfg in (c1, c2, c3, c4, c5)]
    assert len(interactions) == 5 and all(rep["status"] == "modeled" and rep["execution"]["live_tasks"] == 0 for rep in interactions)
    assert all(r["status"] == "served" and r["generated_tokens"] >= 2 for rep in interactions for r in rep["requests"])

    p4_case = compose_case("qwen3.6-27b", hardware="a5", tp=2, layers=1, prompt_tokens=128, output_tokens=64, concurrency=2, max_num_seqs=2, max_num_batched_tokens=256)
    p4_case["runtime"]["serving_io"] = _serving_io(0.05, 0.05, 0.1)
    p4_rep = simulate_service(p4_case)
    assert p4_rep["status"] == "modeled" and len(p4_rep["requests"]) == 2 and p4_rep["execution"]["live_tasks"] == 0
    assert all(r["status"] == "served" and r["generated_tokens"] == 64 and r["ttft_ms"] > 0 for r in p4_rep["requests"])

    dev = _sku_machine("Ascend950DT_9581")["device"]
    flows, pspec = (MessageFlow(0, 1, 1200), MessageFlow(2, 3, 1200)), ParallelSpec(tp=4)
    hw_shared = Hardware.from_mapping({"name": "h4-s", "architecture": "ascend", "num_devices": 4, "device": dev,
                                       "links": [{"name": "trunk", "members": [0, 1, 2, 3], "capacity_bytes_per_ms": 60_000, "semantics": "shared"}]})
    hw_indep = Hardware.from_mapping({"name": "h4-i", "architecture": "ascend", "num_devices": 4, "device": dev,
                                      "links": [{"name": "p01", "members": [0, 1], "capacity_bytes_per_ms": 60_000, "semantics": "shared"},
                                                {"name": "p23", "members": [2, 3], "capacity_bytes_per_ms": 60_000, "semantics": "shared"}]})
    hw_rx30 = Hardware.from_mapping({"name": "h4-r", "architecture": "ascend", "num_devices": 4, "device": dev,
                                     "links": [{"name": "p01", "members": [0, 1], "capacity_bytes_per_ms": 60_000, "semantics": "shared", "endpoint_rates": {"1": 30_000}},
                                               {"name": "p23", "members": [2, 3], "capacity_bytes_per_ms": 60_000, "semantics": "shared"}]})
    d_shared, lb_shared = _run_transport_us(hw_shared, pspec, flows)
    d_indep, _ = _run_transport_us(hw_indep, pspec, flows)
    d_rx30, _ = _run_transport_us(hw_rx30, pspec, flows)
    assert (d_shared, d_indep, d_rx30) == (pytest.approx(40.0), pytest.approx(20.0), pytest.approx(40.0))
    assert lb_shared["shared_resource_work_bytes"] == pytest.approx(2400.0)

    h2_bf16 = KVPageAllocation((17, 31), page_tokens=16, local_kv_heads=1, head_dim=8, kv_count=2, element_bytes=2, metadata_bytes_per_page=32)
    h2_int8 = KVPageAllocation((17, 31), page_tokens=16, local_kv_heads=1, head_dim=8, kv_count=2, element_bytes=1, scale_bytes_per_page=8, metadata_bytes_per_page=32)
    assert (h2_bf16.total_allocation_bytes, h2_int8.total_allocation_bytes, h2_bf16.access_bytes, h2_int8.access_bytes) == (2176, 1184, 1664, 928)


def test_chain_e_counterfactuals_sla_and_source_separation() -> None:
    base_case = compose_case("qwen3.5-35b-a3b", hardware="a5", tp=2, dp=1, ep=2, layers=1,
                             prompt_tokens=64, output_tokens=8, concurrency=2)
    fixed_plan = prepare_fixed_plan(base_case, phase="decode", tokens=1, context=64, sequences=2)
    baseline = evaluate_fixed_plan(base_case, plan=fixed_plan, phase="decode", tokens=1,
                                   context=64, sequences=2, include_capacity=True, output_budget=8)
    cf_hbm = evaluate_resource_counterfactual(
        base_case, resource="hbm_bw", factor=0.5, phase="decode", tokens=1,
        context=64, sequences=2, plan=fixed_plan, include_capacity=True, output_budget=8,
    )
    cf_cap = evaluate_resource_counterfactual(
        base_case, resource="hbm_capacity", factor=0.5, phase="decode", tokens=1,
        context=64, sequences=2, plan=fixed_plan, include_capacity=True, output_budget=8,
    )
    assert cf_hbm["baseline"]["graph_duration_ms"] == pytest.approx(baseline["graph_duration_ms"])
    assert cf_hbm["work_conservation"]["conserved"] is True
    assert cf_hbm["counterfactual"]["graph_duration_ms"] > baseline["graph_duration_ms"]
    assert cf_cap["counterfactual"]["capacity"] != cf_cap["baseline"]["capacity"]

    sla_rep = solve_sla_sizing(
        base_case, sla_ms=80.0, context=64, output_budget=8, max_sequences_per_dp=4,
        scenarios=[{"id": "baseline"}, {"id": "tight_hbm", "hbm_capacity_scale": 0.35}],
    )
    assert len(sla_rep["scenarios"]) == 2

    cands = [
        (dict(base_case, runtime=dict(base_case["runtime"], max_num_seqs=seqs, graph_capture_mode="full", graph_capture_keys=keys)), seqs)
        for seqs, keys in ((1, [1, 8]), (2, [1, 8]), (4, [4, 8]))
    ]
    evals = [evaluate_fixed_plan(c, phase="decode", tokens=1, context=64, sequences=s, include_capacity=True, output_budget=8) for c, s in cands]
    assert all(row["graph_duration_ms"] > 0 for row in evals)

    spec_case = compose_case("qwen3.5-35b-a3b", hardware="a5", tp=2, ep=2, layers=1, prompt_tokens=16,
                             output_tokens=4, concurrency=1, mtp_proposal_tokens=2, mtp_acceptance_length=1.7)
    spec_case["runtime"]["serving_io"] = _serving_io(0.05, 0.05, 0.05)
    spec_svc = simulate_service(spec_case)
    assert spec_svc["status"] == "modeled" and spec_svc["requests"][0]["generated_tokens"] == 4 and spec_svc["execution"]["live_tasks"] == 0

    nominal = {"mte2_aiv": {"rate": 128_000_000.0, "work_unit": "bytes", "rate_scope": "aggregate"}}
    cal = normalize_resource_service(
        {
            "id": "e-source-sep", "scope": "independent_resource", "resources": ["mte2_aiv"],
            "phase": "decode", "work_unit": "bytes", "boundary": "resource_phase",
            "startup_us": 2.0, "response_us": 0.0, "active_units": 1,
            "supply": {"kind": "nominal_multiplier", "multiplier": 0.75},
            "identification": {"resource_identified": True},
        },
        nominal_rates=nominal, hardware="Ascend950DT_9581",
    )
    exported = cal.to_dict()
    assert (exported["nominal_rate"], exported["effective_rate"], cal.duration_us(1536)) == (
        pytest.approx(128_000_000.0), pytest.approx(96_000_000.0), pytest.approx(18.0)
    )
