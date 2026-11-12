"""Supply denominator, invocation latency and per-layer expert placement.

[Role] Independent checks for the runtime contracts owned by the compiler,
  composition and model pipeline.  Each expected value is a closed form of
  the declared structure, not a measured time.
[Invariants]
  * A higher declared cube peak divides elapsed cube time and leaves the
    native work ledger unchanged.
  * An invocation floor is a latency event.  Two concurrent copies complete
    at the floor.  Their stage work stays the structural vector count.
  * An idle cross-node link does not replace the on-node link rate.
  * Rebinding ``operator.supply`` changes elapsed time to match a fresh
    compilation on that supply.
  * A layer plan's expert count and top-k replace the model default for that
    layer's F1/F3 signature and its uniform routing shard.
[Timeouts & Locks] In-memory arithmetic only.
[Side-Effect Policy] No filesystem writes and no device context.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math

from mindie_sim.prediction.piecewise.compiler import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
)
from mindie_sim.prediction.piecewise.composition import contention_envelope, operator_segment
from mindie_sim.prediction.piecewise.families_matrix import AIV_REPEAT_BYTES
from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.subgraphs import BoundOperator, OperatorTemplate, StaticSignature

_POLICY = ImplementationPolicy(mt=16, nt=16, cores=2)
_FLOOR = Fraction(1312, 1_000_000)


def _supply(**overrides) -> PipeSupply:
    rates = {
        "cube": 1_000_000.0, "vector": 1_000_000.0, "hbm": 1_000_000.0,
        "mte1": 1_000_000.0, "mte2": 1_000_000.0, "mte3": 1_000_000.0,
        "fixpipe": 1_000_000.0, "link": 1_000_000.0,
        "link_hccs": 1_000_000.0, "link_roce": 1_000_000.0,
    }
    rates.update(overrides.pop("rates", {}))
    return PipeSupply(
        tuple(rates.items()),
        invocations=overrides.get("invocations", (("vector", 1e-12),)),
        resource_units=overrides.get("resource_units", (("compute_units", 2), ("vector", 2))),
        flops_by_dtype=overrides.get("flops_by_dtype", (("bf16", 1_000_000.0), ("int8", 2_000_000.0))),
    )


def _bound(name: str, family: str, **groups) -> BoundOperator:
    return BoundOperator(
        OperatorTemplate(name=name, family=family, role=name),
        StaticSignature(role=name, **groups),
    )


def test_declared_cube_peak_divides_time_and_keeps_native_work() -> None:
    supply = _supply()
    compiler = OperatorPiecewiseCompiler(supply, _POLICY)
    geometry = WorkloadGeometry.homogeneous(1, 128, 128, phase="prefill")
    bf16 = compiler.compile_operator(_bound(
        "proj", "gemm", matrix=(("K", 256), ("N", 256)),
        quant=(("weight_format", "bf16"), ("act_format", "bf16")),
    )).evaluate(geometry)
    int8 = compiler.compile_operator(_bound(
        "proj", "gemm", matrix=(("K", 256), ("N", 256)),
        quant=(("weight_format", "int8"), ("act_format", "int8")),
    )).evaluate(geometry)
    assert int8.pipe_work("cube") == bf16.pipe_work("cube")
    assert int8.pipe_work("cube") > 0
    assert dict(int8.times)["cube"] * 2 == dict(bf16.times)["cube"]
    assert int8.duration_ms < bf16.duration_ms


def test_concurrent_invocation_floors_do_not_invent_vector_work() -> None:
    supply = _supply(
        rates={"vector": 1.0e12, "hbm": 1.0e12, "mte2": 1.0e12, "mte3": 1.0e12},
        invocations=(("vector", float(_FLOOR)),),
    )
    compiler = OperatorPiecewiseCompiler(supply, _POLICY)
    geometry = WorkloadGeometry.homogeneous(1, 1, 1, phase="decode")
    cost = compiler.compile_operator(_bound(
        "act", "elementwise",
        extras=(("elements", 16), ("ops_per_element", 1),
                ("read_bytes", 32), ("write_bytes", 32)),
    )).evaluate(geometry)
    body = sum(cost.stage_times, Fraction(0))
    assert body < _FLOOR
    assert cost.duration_ms == _FLOOR
    assert dict(cost.derived)["service_floor_ms"] == _FLOOR
    # The floor is not encoded as extra vector operations.
    assert cost.pipe_work("vector") < _FLOOR * supply.rate("vector")
    branches = {
        "a": [list(stage) for stage in cost.stage_work],
        "b": [list(stage) for stage in cost.stage_work],
    }
    envelope = contention_envelope(branches, supply, {"a": _FLOOR, "b": _FLOOR})
    assert envelope.makespan_ms == _FLOOR
    assert dict(envelope.branches)["a"] == _FLOOR
    assert dict(envelope.branches)["b"] == _FLOOR


def test_idle_inter_node_link_does_not_replace_the_on_node_rate() -> None:
    rates = []
    for devices in (8, 16):
        hardware = Hardware.from_mapping({
            "backend": "ascend", "sku": "Ascend950DT_9582",
            "num_devices": devices, "ranks_per_node": 8,
        })
        supply = PipeSupply.from_hardware(hardware)
        rates.append(supply)
        assert supply.rate("link") == supply.rate("link_hccs")
        assert supply.rate("link_hccs") == 392_000_000.0
    assert rates[0].rate("link") == rates[1].rate("link")
    assert rates[1].rate("link_roce") == 50_000_000.0
    assert rates[1].rate("link") != rates[1].rate("link_roce")


def test_rebound_supply_matches_a_fresh_compilation() -> None:
    fast = _supply()
    slow_rates = tuple((name, rate / 100.0) for name, rate in fast.rates)
    slow = replace(fast, rates=slow_rates, flops_by_dtype=tuple(
        (name, value / 100.0) for name, value in fast.flops_by_dtype))
    compiler = OperatorPiecewiseCompiler(fast, _POLICY)
    operator = compiler.compile_operator(_bound(
        "proj", "gemm", matrix=(("K", 256), ("N", 256)),
        quant=(("weight_format", "bf16"), ("act_format", "bf16")),
    ))
    geometry = WorkloadGeometry.homogeneous(1, 64, 64, phase="prefill")
    before = operator.evaluate(geometry)
    operator.supply = slow
    rebound = operator.evaluate(geometry)
    fresh = OperatorPiecewiseCompiler(slow, _POLICY).compile_operator(
        operator.operator).evaluate(geometry)
    assert rebound.duration_ms == fresh.duration_ms
    assert rebound.duration_ms == before.duration_ms * 100
    assert rebound.pipe_work("cube") == before.pipe_work("cube")


def _moe_model(layer1_experts: int, layer1_k: int, *,
               layer1_groups: int = 1, layer1_group_topk: int = 0) -> ModelSpec:
    common = dict(attention="gqa", ffn="moe", num_heads=4, num_kv_heads=2,
                  head_dim=16, intermediate_size=64, moe_intermediate_size=32)
    plans = [
        dict(common, index=0, num_experts=4, top_k=2, router_groups=1, router_group_topk=0),
        dict(common, index=1, num_experts=layer1_experts, top_k=layer1_k,
             router_groups=layer1_groups, router_group_topk=layer1_group_topk),
    ]
    return ModelSpec.from_mapping(dict(
        name="layer-moe", hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=64, vocab_size=128, layers=2,
        num_experts=4, top_k=2, moe_intermediate_size=32, layer_plans=plans,
    ))


def test_layer_expert_count_and_topk_replace_the_model_default() -> None:
    model = _moe_model(8, 3)
    features = FeatureConfig.from_model(
        model, tp=2, ep=2, token_tile=16, kv_tile=16, cores=2, moe_row_tile=4)
    pipeline = assemble_model(
        model, supply=_supply(), policy=_POLICY, features=features)
    by_layer = {layer.layer: layer for layer in pipeline.instantiated.layers}

    def field(layer: int, kind: str, name: str, key: str) -> int:
        instance = next(item for item in by_layer[layer].subgraphs if item.kind == kind)
        operator = next(op for op in instance.operators if op.name == name)
        return int(operator.theta.find(key))

    assert field(0, "F1", "router_topk", "E") == 4
    assert field(0, "F3", "expert_gate_up", "E") == 4
    assert field(0, "F3", "expert_gate_up", "E_local") == 2
    assert field(1, "F1", "router_topk", "E") == 8
    assert field(1, "F1", "router_topk", "k") == 3
    assert field(1, "F3", "expert_gate_up", "E") == 8
    assert field(1, "F3", "expert_gate_up", "E_local") == 4

    # 4 active tokens.  Even placement over E=4, EP=2, strided local shard:
    # K=1/2/3 -> local routed rows 2/4/6.  The model default K=2 is not used.
    geometry = WorkloadGeometry.homogeneous(4, 1, 16, phase="decode")
    for top_k, expected in ((1, 2), (2, 4), (3, 6)):
        routed = assemble_model(
            _moe_model(4, top_k), supply=_supply(), policy=_POLICY, features=features,
        ).forward(geometry)
        shard = routed.active.routing_for(1)
        assert shard is not None
        assert sum(shard) == expected
        assert len(shard) == 2


def test_layer_gate_partition_scope_follows_that_layer() -> None:
    """Grouped gate scope is the layer partition, not the model default.

    The score GEMM and the top-k scan still read every expert.  The selector
    exists only when that layer declares more than one group, and
    scope_experts is (E / groups) * group_topk.
    """
    features = FeatureConfig.from_model(
        _moe_model(8, 2), tp=2, ep=2, token_tile=16, kv_tile=16, cores=2, moe_row_tile=4)
    pipeline = assemble_model(
        _moe_model(8, 2, layer1_groups=4, layer1_group_topk=1),
        supply=_supply(), policy=_POLICY, features=features)
    by_layer = {layer.layer: layer for layer in pipeline.instantiated.layers}

    def operator(layer: int, name: str):
        instance = next(item for item in by_layer[layer].subgraphs if item.kind == "F1")
        return next(op for op in instance.operators if op.name == name)

    def names(layer: int) -> set[str]:
        instance = next(item for item in by_layer[layer].subgraphs if item.kind == "F1")
        return {op.name for op in instance.operators}

    assert "router_group_select" not in names(0)
    assert operator(0, "router_topk").theta.find("scope_experts") == 4
    assert operator(0, "router_topk").theta.find("groups") == 1
    select = operator(1, "router_group_select")
    assert select.theta.find("E") == 8
    assert select.theta.find("groups") == 4
    assert select.theta.find("group_topk") == 1
    assert select.theta.find("scope_experts") == 2
    assert select.theta.find("scope") == "group_partition"
    assert operator(1, "router_topk").theta.find("E") == 8
    assert operator(1, "router_topk").theta.find("scope_experts") == 2
    assert operator(1, "router_gemm").theta.find("scope_experts") == 2
    # A wider group top-k admits more experts and does not change the scan of E.
    wider = assemble_model(
        _moe_model(8, 2, layer1_groups=4, layer1_group_topk=2),
        supply=_supply(), policy=_POLICY, features=features)
    wider_f1 = next(item for item in wider.instantiated.layers[1].subgraphs if item.kind == "F1")
    wider_top = next(op for op in wider_f1.operators if op.name == "router_topk")
    wider_select = next(op for op in wider_f1.operators if op.name == "router_group_select")
    assert wider_top.theta.find("scope_experts") == 4
    assert wider_top.theta.find("E") == select.theta.find("E")
    geometry = WorkloadGeometry.homogeneous(4, 1, 16, phase="decode")
    compiled = OperatorPiecewiseCompiler(_supply(), _POLICY)
    narrow_cost = compiled.compile_operator(select).evaluate(geometry)
    wide_cost = compiled.compile_operator(wider_select).evaluate(geometry)
    # The selector reads every expert score: tokens * E * 2.  The vector pipe
    # also charges one AIV repeat per core per token wave, and that repeat is
    # the larger term at this shape.  group_topk changes scope, not this work.
    structural = 4 * 8 * 2
    waves = math.ceil(4 / _POLICY.mt)
    cores = min(_POLICY.cores, 32)
    repeat = waves * cores * math.ceil(AIV_REPEAT_BYTES / 2)
    assert narrow_cost.pipe_work("vector") == wide_cost.pipe_work("vector") == max(structural, repeat)


def test_later_kernel_floor_waits_for_the_previous_kernel() -> None:
    supply = _supply(invocations=(("vector", float(_FLOOR)),))
    compiler = OperatorPiecewiseCompiler(supply, _POLICY)
    geometry = WorkloadGeometry.homogeneous(1, 1, 1, phase="decode")
    long = compiler.compile_operator(_bound(
        "long_act", "elementwise",
        extras=(("elements", 2_000_000), ("ops_per_element", 1),
                ("read_bytes", 4), ("write_bytes", 4)),
    )).evaluate(geometry)
    short = compiler.compile_operator(_bound(
        "short_act", "elementwise",
        extras=(("elements", 16), ("ops_per_element", 1),
                ("read_bytes", 32), ("write_bytes", 32)),
    )).evaluate(geometry)
    assert sum(long.stage_times, Fraction(0)) > _FLOOR
    assert short.duration_ms == _FLOOR
    long_stages = [list(stage) for stage in long.stage_work]
    short_stages = [list(stage) for stage in short.stage_work]
    first = contention_envelope(
        {"a": long_stages, "b": long_stages}, supply,
        {"a": long.duration_ms, "b": long.duration_ms})
    second = contention_envelope(
        {"a": short_stages, "b": short_stages}, supply,
        {"a": short.duration_ms, "b": short.duration_ms})
    flat = contention_envelope(
        {"a": long_stages + short_stages, "b": long_stages + short_stages}, supply,
        {"a": long.duration_ms + short.duration_ms,
         "b": long.duration_ms + short.duration_ms})
    chain = contention_envelope({}, supply, chains={
        "a": (operator_segment(long, supply), operator_segment(short, supply)),
        "b": (operator_segment(long, supply), operator_segment(short, supply)),
    })
    assert chain.makespan_ms == first.makespan_ms + second.makespan_ms
    assert chain.makespan_ms > flat.makespan_ms
    assert dict(chain.resource_work)["vector"] == 2 * (
        long.pipe_work("vector") + short.pipe_work("vector"))


def test_rebind_previous_op_does_not_change_the_next_op_work() -> None:
    fast = _supply()
    slow = replace(fast, rates=tuple((name, rate / 100.0) for name, rate in fast.rates),
                   flops_by_dtype=tuple((name, value / 100.0) for name, value in fast.flops_by_dtype))
    compiler = OperatorPiecewiseCompiler(fast, _POLICY)
    geometry = WorkloadGeometry.homogeneous(1, 64, 64, phase="prefill")
    signature = dict(matrix=(("K", 256), ("N", 256)),
                     quant=(("weight_format", "bf16"), ("act_format", "bf16")))
    first = compiler.compile_operator(_bound("first", "gemm", **signature))
    second = compiler.compile_operator(_bound("second", "gemm", **signature))
    before_first = first.evaluate(geometry)
    before_second = second.evaluate(geometry)
    first.supply = slow
    after_first = first.evaluate(geometry)
    after_second = second.evaluate(geometry)
    assert after_first.pipe_work("cube") == before_first.pipe_work("cube")
    assert after_first.duration_ms == before_first.duration_ms * 100
    assert after_second.pipe_work("cube") == before_second.pipe_work("cube")
    assert after_second.duration_ms == before_second.duration_ms
    chain = contention_envelope({}, fast, chains={"main": (
        operator_segment(after_first, fast),
        operator_segment(after_second, fast),
    )})
    # One stream, no second contender: the causal completion is the sum of the
    # two isolated durations.  The first operator's slower supply is a cap on
    # its own stages; the second operator's native cube work is unchanged.
    assert chain.isolated_max_ms == after_first.duration_ms + after_second.duration_ms
    assert abs(chain.makespan_ms - chain.isolated_max_ms) <= chain.isolated_max_ms / 10**9
    assert dict(chain.resource_work)["cube"] == (
        after_first.pipe_work("cube") + after_second.pipe_work("cube"))


def test_d2d_host_named_link_keeps_real_intra_route_and_cap() -> None:
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950DT_9582",
        "num_devices": 2, "ranks_per_node": 2,
        "links": [{"name": "ub-host0", "channel": "d2d_intra_node",
                   "capacity_bytes_per_ms": 392_000_000.0,
                   "members": [0, 1]}],
    })
    ordinary = PipeSupply.from_hardware(hardware)
    assert ordinary.rate("link_hccs") == 392_000_000.0
    slow = replace(hardware, links=(replace(
        hardware.links[0], capacity_bytes_per_ms=3_920_000.0),))
    assert PipeSupply.from_hardware(slow).rate("link_hccs") == 3_920_000.0
    capped = replace(ordinary, structure_caps=(("link_hccs", 3_920_000.0),))
    assert capped.effective("link_hccs") == 3_920_000.0


def _aggregate_cost(link_inv=None, cap=None):
    extras = {"invocations": (("link_rate_intra_bytes_per_ms", 10_000_000.0),)}
    if link_inv is not None:
        extras["invocations"] = link_inv
    base = _supply(rates={"link_hccs": 1_000_000.0, "link": 1_000_000.0},
                   **extras)
    if cap is not None:
        base = replace(base, structure_caps=(("link_hccs", cap),))
    from mindie_sim.schema.subgraphs import BoundOperator, OperatorTemplate, StaticSignature
    op = BoundOperator(
        OperatorTemplate(name="tp_all_gather", family="collective", role="tp"),
        StaticSignature(role="tp", quant=(("wire_bytes", 1024.0),),
                        parallel=(("members", 2),),
                        tiles=(("cores", 1),),
                        extras=(("ranks_per_node", 2), ("mtu_bytes", 1))))
    return (OperatorPiecewiseCompiler(base, _POLICY).compile_operator(op)
            .evaluate(WorkloadGeometry.homogeneous(1, 128, 129)))


def test_selected_route_never_overwrites_lower_cap_or_source_link() -> None:
    capped = _aggregate_cost(cap=1_000_000.0)
    ordinary = _aggregate_cost()
    # The selected invocation route is faster than the declared source/cap;
    # it must not scale the real lower route upward.
    assert capped.pipe_time("link_hccs") >= ordinary.pipe_time("link_hccs")
    assert dict(capped.derived).get("service_floor_ms", Fraction(0)) == 0
