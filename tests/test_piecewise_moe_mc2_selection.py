"""Source-selected MC2 transport and typed FP4 grouped composite invariants."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mc2_phase import build as mc2_build
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.moe_comm_selection import (
    ALLTOALL, MC2, MoeCommConfig, source_mc2_tokens_capacity,
)
from mindie_sim.prediction.operators.mc2_fullmesh import dispatch_layout
from mindie_sim.prediction.piecewise.families_moe_mc2 import Mc2LaunchView
from mindie_sim.prediction.piecewise.compiler import (
    ImplementationPolicy, PipeSupply,
)
from mindie_sim.prediction.piecewise.geometry import (
    RequestGeometry, WorkloadGeometry,
)
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.operators import OpSpec

SUPPLY = PipeSupply((
    ("cube", 1_000_000.0), ("vector", 1_000_000.0), ("hbm", 1_000_000.0),
    ("mte1", 2_000_000.0), ("mte2", 2_000_000.0), ("mte3", 2_000_000.0),
    ("fixpipe", 2_000_000.0), ("link", 1_000_000.0),
))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4,
                              split_target=1000, max_splits=4)


def _model(*, quant=None, experts=8, top_k=2):
    plan = dict(index=0, attention="gqa", ffn="moe", num_heads=4,
                num_kv_heads=2, head_dim=16, intermediate_size=128,
                num_experts=experts, top_k=top_k, moe_intermediate_size=32,
                num_shared_experts=0)
    raw = dict(name="toy", hidden_size=64, num_heads=4, num_kv_heads=2,
               head_dim=16, intermediate_size=128, vocab_size=128, layers=1,
               layer_plans=[plan], num_experts=experts, top_k=top_k,
               moe_intermediate_size=32, num_shared_experts=0)
    if quant is not None:
        raw["quant"] = quant
    return ModelSpec.from_mapping(raw)


def _features(model, *, additional=None, quant=None):
    values = dict(token_tile=16, kv_tile=16, cores=2, moe_row_tile=4,
                  split_target=1000, max_splits=4, tp=1, dp=4, ep=4)
    features = FeatureConfig.from_model(model, **values)
    meta = dict(features.metadata)
    meta.update({
        "additional_config": dict(additional or {}),
        "arch_family": "A5",
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 64,
    })
    return replace(features, metadata=tuple(sorted(meta.items())),
                   graph_buckets=(1, 2, 4, 8, 16, 32, 64),
                   moe_recipe="alltoall")


def _pipeline(model, features):
    return assemble_model(model, supply=SUPPLY, policy=POLICY,
                          features=features)


def _rows(prediction, kind):
    return tuple(row.cost for layer in prediction.layers for row in layer.rows
                 if row.cost.kind == kind)


def _decode2():
    return WorkloadGeometry((
        RequestGeometry(1, 0, phase="decode"),
        RequestGeometry(1, 0, phase="decode"),
    ), phase="decode")


def _prefill1024():
    return WorkloadGeometry((RequestGeometry(1024, 0, phase="prefill"),),
                            phase="prefill", cohort_batch=1, local_batch=1)


def test_source_capacity_and_world_size_selector_use_real_conditions():
    assert source_mc2_tokens_capacity(
        max_num_reqs=64, uniform_decode_query_len=1,
        capture_sizes=(1, 2, 4, 8, 16, 32, 64), tp=1,
        enable_prefill_mc2=False, max_num_batched_tokens=8192,
        enable_fused_mc2=True, mega_moe_supported=False) == 64
    config = MoeCommConfig(
        mode="dynamic", policy="capacity_and_world_size", capacity=64,
        world_size=16, tp=1, top_k=8, experts_per_device=16,
        enable_fused_mc2=True, capture_sizes=(1, 2, 4, 8, 16, 32, 64))
    assert config.select(2) == MC2
    assert config.select(64) == MC2
    assert config.select(65) == ALLTOALL
    assert config.select(1024) == ALLTOALL


def test_piecewise_f2_selects_mc2_by_local_tokens_not_phase_name():
    model = _model(experts=8, top_k=2)
    pipe = _pipeline(model, _features(
        model, additional={"enable_fused_mc2": 1}))
    decode = _rows(pipe.forward(_decode2()), "F2")
    by_name = {cost.name: cost for cost in decode}
    assert by_name["mc2_dispatch"].duration_ms > 0
    assert by_name["mc2_combine"].duration_ms > 0
    assert by_name["dispatch"].duration_ms == 0
    assert by_name["combine"].duration_ms == 0
    assert by_name["quantize"].duration_ms == 0
    prefill = _rows(pipe.forward(_prefill1024()), "F2")
    names = {cost.name: cost for cost in prefill}
    assert names["dispatch"].duration_ms > 0
    assert names["combine"].duration_ms > 0
    assert names["mc2_dispatch"].duration_ms == 0
    assert names["mc2_combine"].duration_ms == 0


def test_mc2_rows_own_framed_source_work_and_keep_generic_nodes_separate():
    model = _model(experts=8, top_k=2)
    pipe = _pipeline(model, _features(
        model, additional={"enable_fused_mc2": 1}))
    rows = {cost.name: cost for cost in _rows(pipe.forward(_decode2()), "F2")}
    dispatch = rows["mc2_dispatch"]
    combine = rows["mc2_combine"]
    layout = dispatch_layout(64, input_dtype="bf16", payload_dtype="bf16")
    assert dispatch.get("mc2_packet_bytes") == layout.packet_bytes
    assert dispatch.get("mc2_phase_count") == 14
    assert combine.get("mc2_phase_count") == 7
    assert dispatch.get("mc2_link_bytes") > 0
    assert dict(dispatch.work)["mte2"] > 0
    assert dict(dispatch.work)["hbm"] > 0
    assert dict(combine.work)["vector"] > 0
    # The retained generic node is a distinct family/owner and is not renamed
    # to the native source kernel.
    assert rows["dispatch"].duration_ms == 0
    assert "MoeDistribute" not in rows["dispatch"].formula


def test_fp4_grouped_composite_owns_typed_payload_scale_and_down_bf16():
    quant = dict(
        name="w4a4", weight_format="mxfp4", activation_format="mxfp4",
        expert_weight_format="mxfp4", weight_block=(1, 32), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1.0, activation_group_size=32,
        activation_scale_format="fp8_e8m0")
    model = _model(quant=quant, experts=8, top_k=2)
    pipe = _pipeline(model, _features(model))
    f3 = pipe.instantiated.find("F3", layer=0)
    assert [op.name for op in f3.operators] == ["expert_gate_up", "expert_down"]
    prediction = pipe.forward(_prefill1024())
    rows = {cost.name: cost for cost in _rows(prediction, "F3")}
    gate = rows["expert_gate_up"]
    down = rows["expert_down"]
    gate_materialization = dict(gate.materialization)
    assert set(gate_materialization) == {
        "expert_gate_up_output_payload", "expert_gate_up_output_scale"}
    assert gate_materialization["expert_gate_up_output_scale"] > 0
    assert list(dict(down.materialization)) == ["expert_down_output"]
    assert dict(gate.work)["vector"] > 0
    assert dict(down.work)["mte2"] > 0


def test_fp4_mc2_dispatch_uses_packed_payload_and_e8m0_scale_rows():
    quant = dict(
        name="w4a4", weight_format="mxfp4", activation_format="mxfp4",
        expert_weight_format="mxfp4", weight_block=(1, 32), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1.0, activation_group_size=32,
        activation_scale_format="fp8_e8m0")
    model = _model(quant=quant, experts=8, top_k=2)
    pipe = _pipeline(model, _features(
        model, additional={"enable_fused_mc2": 1}))
    rows = {cost.name: cost for cost in _rows(pipe.forward(_decode2()), "F2")}
    dispatch = rows["mc2_dispatch"]
    layout = dispatch_layout(64, input_dtype="bf16",
                             payload_dtype="fp4_e2m1", quant_mode="mx")
    assert dispatch.get("mc2_packet_bytes") == layout.packet_bytes
    assert dispatch.get("mc2_scale_bytes") == layout.scale_bytes == 2
    assert dispatch.get("mc2_phase_count") == 14
    assert dict(dispatch.materialization)["peer_traffic"] > 0
    assert dict(dispatch.work)["mte2"] > 0


def test_source_native_fusion_role_selects_grouped_composite_for_bf16_header():
    model = _model(experts=8, top_k=2)
    features = _features(model)
    meta = dict(features.metadata)
    meta["native_fusion_roles"] = ("grouped_gate_up_activation_quant",
                                   "hc_pre", "hc_post")
    features = replace(features, metadata=tuple(sorted(meta.items())))
    pipe = _pipeline(model, features)
    f3 = pipe.instantiated.find("F3", layer=0)
    assert [op.name for op in f3.operators] == ["expert_gate_up", "expert_down"]
    rows = {cost.name: cost for cost in _rows(pipe.forward(_prefill1024()), "F3")}
    assert set(rows) == {"expert_gate_up", "expert_down"}
    assert list(dict(rows["expert_gate_up"].materialization)) == [
        "expert_gate_up_output_payload"]
    assert rows["expert_down"].duration_ms > 0
    # The role is the selector: an hc_pre-only declaration keeps the prior
    # separate activation owner instead of silently fusing the family.
    only_meta = dict(features.metadata)
    only_meta["native_fusion_roles"] = ("hc_pre",)
    only = _pipeline(model, replace(
        features, metadata=tuple(sorted(only_meta.items()))))
    only_f3 = only.instantiated.find("F3", layer=0)
    assert [op.name for op in only_f3.operators] == [
        "expert_gate_up", "expert_activation", "expert_down"]


def test_fp8_w8_role_keeps_old_separate_grouped_route():
    quant = dict(
        name="w8a8", weight_format="int8", activation_format="fp8_e4m3",
        expert_weight_format="int8", weight_block=(128, 128),
        expert_block=0, scale_format="fp32", scale_bytes=4.0,
        activation_group_size=0, activation_scale_format="fp32")
    model = _model(quant=quant, experts=8, top_k=2)
    features = _features(model)
    meta = dict(features.metadata)
    meta["native_fusion_roles"] = ("grouped_gate_up_activation_quant",)
    features = replace(features, metadata=tuple(sorted(meta.items())))
    pipe = _pipeline(model, features)
    f3 = pipe.instantiated.find("F3", layer=0)
    assert [op.name for op in f3.operators] == [
        "expert_gate_up_QuantV2", "expert_gate_up", "expert_activation",
        "expert_down_QuantV2", "expert_down"]


def test_a5_selector_is_not_the_fused_flag_and_unknown_deployments_stay_static():
    model = _model(experts=8, top_k=2)
    for flag in (0, 1):
        features = _features(model, additional={"enable_fused_mc2": flag})
        config = MoeCommConfig.from_features(
            features, ep=4, top_k=2, experts=8)
        assert config is not None
        assert config.select(2) == MC2
        assert config.select(1024) == ALLTOALL
    unknown = _features(model)
    meta = dict(unknown.metadata)
    meta.pop("additional_config", None)
    meta.pop("arch_family", None)
    unknown = replace(unknown, metadata=tuple(sorted(meta.items())))
    assert MoeCommConfig.from_features(
        unknown, ep=4, top_k=2, experts=8) is None
    # A loaded A5 R64 deployment is provenance even when the fused field is
    # absent; ordinary MC2 still follows the capacity crossover.
    provenance_meta = dict(meta)
    provenance_meta["arch_family"] = "A5"
    provenance_meta["source_lowerings"] = ("native_fusion",)
    provenance = replace(unknown, metadata=tuple(sorted(provenance_meta.items())))
    loaded = MoeCommConfig.from_features(
        provenance, ep=4, top_k=2, experts=8)
    assert loaded is not None and loaded.select(2) == MC2


def test_allgather_mode_keeps_a_positive_separate_owner_and_mega_stays_unknown():
    model = _model(experts=8, top_k=8)
    features = _features(
        model, additional={"enable_fused_mc2": 0}).project(tp=1, dp=2, ep=2)
    pipe = _pipeline(model, features)
    rows = {cost.name: cost for cost in _rows(pipe.forward(_prefill1024()), "F2")}
    assert rows["moe_allgather"].duration_ms > 0
    assert rows["dispatch"].duration_ms == 0
    assert rows["mc2_dispatch"].duration_ms == 0
    mega_features = _features(
        model, additional={"enable_fused_mc2": 1, "enable_mega_moe": 1})
    assert MoeCommConfig.from_features(
        mega_features, ep=2, top_k=8, experts=8) is None


def test_mc2_uses_true_intra_inter_link_classes_from_declared_topology():
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    rates = dict(SUPPLY.rates)
    rates.update(link_hccs=392_000_000.0, link_roce=2_000_000.0)
    two_node = replace(
        SUPPLY, rates=tuple(sorted(rates.items())),
        invocations=(("ranks_per_node", 2),))
    pipe = assemble_model(model, supply=two_node, policy=POLICY, features=features)
    rows = {cost.name: cost for cost in _rows(pipe.forward(_decode2()), "F2")}
    work = dict(rows["mc2_dispatch"].work)
    assert work["link_hccs"] > 0 and work["link_roce"] > 0
    # No witnessed per-rank source route is present: the declared statistical
    # policy still owns expected payload bytes on both real link classes. It is
    # not integer frames and not zero transferred work.
    combine_work = dict(rows["mc2_combine"].work)
    assert combine_work["link_hccs"] > 0 and combine_work["link_roce"] > 0
    assert rows["mc2_dispatch"].duration_ms > 0
    assert rows["mc2_combine"].duration_ms > 0
    one_node = replace(
        SUPPLY, rates=tuple(sorted(rates.items())),
        invocations=(("ranks_per_node", 8),))
    single = assemble_model(model, supply=one_node, policy=POLICY,
                            features=features)
    single_work = dict({
        cost.name: cost for cost in _rows(single.forward(_decode2()), "F2")
    }["mc2_dispatch"].work)
    assert single_work.get("link_hccs", 0) > 0
    assert single_work.get("link_roce", 0) == 0


def test_plain_u8_storage_is_not_used_as_fp4_selection():
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    meta = dict(features.metadata)
    meta["checkpoint_storage_dtypes"] = {"routed_expert": "U8"}
    features = replace(features, metadata=tuple(sorted(meta.items())))
    pipe = _pipeline(model, features)
    rows = {cost.name: cost for cost in _rows(pipe.forward(_decode2()), "F2")}
    # No signed routed method was declared, so the row remains BF16 and does
    # not acquire an invented FP4/E8M0 scale from the bare U8 packing dtype.
    assert rows["mc2_dispatch"].get("mc2_scale_bytes") == 0


def test_mc2_uses_declared_local_received_ledger_and_exposes_unpriced_scalars():
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    meta = dict(features.metadata)
    meta["moe_rank_index"] = 0
    meta["moe_received_rows"] = {0: 3}
    features = replace(features, metadata=tuple(sorted(meta.items())))
    pipe = _pipeline(model, features)
    rows = {cost.name: cost for cost in _rows(pipe.forward(_decode2()), "F2")}
    dispatch = rows["mc2_dispatch"]
    assert dispatch.get("mc2_received_assignments") == 3
    assert dispatch.get("mc2_received_declared") == 1
    assert dispatch.get("mc2_unpriced_primitive_total") > 0


def test_mc2_group_ready_skew_is_not_applied_without_actual_handles():
    model = _model(experts=8, top_k=2)
    base_features = _features(model, additional={"enable_fused_mc2": 1})
    base = {cost.name: cost for cost in _rows(
        _pipeline(model, base_features).forward(_decode2()), "F2")}
    meta = dict(base_features.metadata)
    meta["moe_group_ready"] = {
        "schema": "mindie-sim/moe-group-ready/1", "group_size": 4,
        "skew_ms": 0.01, "basis": "source_declared"}
    features = replace(base_features, metadata=tuple(sorted(meta.items())))
    hooked = {cost.name: cost for cost in _rows(
        _pipeline(model, features).forward(_decode2()), "F2")}
    # A scalar skew is not a group handle: it must not block the early rank's
    # local routing/quant/count or repeat as a per-layer fixed wait.
    assert hooked["mc2_dispatch"].get("mc2_group_ready_declared") == 0
    assert (hooked["mc2_dispatch"].duration_ms ==
            base["mc2_dispatch"].duration_ms)


def test_mc2_schedules_per_peer_send_receive_handles_from_source_weights():
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    meta = dict(features.metadata)
    meta["moe_peer_weights"] = (0.2, 0.3, 0.5)
    features = replace(features, metadata=tuple(sorted(meta.items())))
    pipe = _pipeline(model, features)
    rows = {cost.name: cost for cost in _rows(pipe.forward(_decode2()), "F2")}
    native = rows["mc2_dispatch"].native_program
    program = native.program
    assert program.peer_count == 4
    assert len(program.peer_labels) == 3
    assert program.peer_weights == (Fraction(1, 5), Fraction(3, 10), Fraction(1, 2))
    assert rows["mc2_dispatch"].get("mc2_peer_count") == 3
    assert rows["mc2_dispatch"].get("mc2_peer_weight_declared") == 1
    resources = tuple(
        Resource(name, rate, "work", "fair", 2)
        for name, rate in SUPPLY.rates)
    engine = Engine(resources)
    ready = engine.submit("ready")
    event, handles = native.start_with_handles(engine, "native", ready)
    engine.advance(until=(event,))
    send_times = {
        item["send"].time
        for phase in handles["peer_events"].values()
        for item in phase.values()}
    assert len(send_times) > 1


def test_mc2_peer_classes_split_local_hccs_and_remote_roce():
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    meta = dict(features.metadata)
    meta["moe_rank_index"] = 0
    meta["moe_peer_weights"] = (0.1, 0.8, 0.1)
    features = replace(features, metadata=tuple(sorted(meta.items())))
    supply = replace(SUPPLY, invocations=(("ranks_per_node", 2),))
    pipe = assemble_model(model, supply=supply, policy=POLICY,
                          features=features)
    rows = {cost.name: cost for cost in _rows(pipe.forward(_decode2()), "F2")}
    native = rows["mc2_dispatch"].native_program
    program = native.program
    assert program.peer_labels == (1, 2, 3)
    assert [program._peer_class(i, peer)
            for i, peer in enumerate(program.peer_labels)] == [
                "hccs", "roce", "roce"]
    assert rows["mc2_dispatch"].get("mc2_peer_class_basis") == 1
    resources = tuple(
        Resource(name, rate, "work", "fair", 2)
        for name, rate in supply.rates)
    engine = Engine(resources)
    ready = engine.submit("ready")
    event, handles = native.start_with_handles(engine, "native", ready)
    engine.advance(until=(event,))
    for phase, peers in handles["peer_events"].items():
        for peer, item in peers.items():
            expected = "hccs" if peer == 1 else "roce"
            assert item["class"] == expected
    expected = {}
    for links in program.link_by_phase.values():
        for pipe, amount in links.items():
            key = program.names.get(pipe, pipe)
            expected[key] = expected.get(key, 0) + float(amount)
    for pipe, amount in expected.items():
        assert engine.serviced[pipe] == pytest.approx(amount)


def test_mc2_native_program_accepts_phase_scoped_real_peer_ready_events():
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    meta = dict(features.metadata)
    meta["moe_rank_index"] = 0
    features = replace(features, metadata=tuple(sorted(meta.items())))
    rows = {cost.name: cost for cost in _rows(
        _pipeline(model, features).forward(_decode2()), "F2")}
    native = rows["mc2_dispatch"].native_program
    resources = tuple(
        Resource(name, rate, "work", "fair", 2)
        for name, rate in SUPPLY.rates)
    engine = Engine(resources)
    ready = engine.submit("ready")
    peer = native.program.peer_labels[0]
    phase = next(name for name, links in native.program.link_by_phase.items() if links)
    incoming = engine.submit("peer_ready_arrival")
    local = engine.future("local_send_arrival")
    peer_ready = {phase: {peer: incoming}}
    placeholders = {(0, peer, phase): local, (peer, 0, phase): incoming}
    event, handles = native.start_with_handles(
        engine, "native", ready, peer_ready=peer_ready,
        send_placeholders=placeholders)
    engine.advance(until=(event, local))
    trace = handles["peer_events"]
    assert trace
    assert local.done
    assert all(item["receive"].time >= incoming.time
               for item in trace[phase].values()
               if item.get("receive") is not None)



def test_mc2_peer_ready_is_phase_specific_not_shared_payload_wait():
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    meta = dict(features.metadata)
    meta["moe_rank_index"] = 0
    features = replace(features, metadata=tuple(sorted(meta.items())))
    rows = {cost.name: cost for cost in _rows(
        _pipeline(model, features).forward(_decode2()), "F2")}
    native = rows["mc2_dispatch"].native_program
    resources = tuple(
        Resource(name, rate, "work", "fair", 2)
        for name, rate in SUPPLY.rates)
    engine = Engine(resources)
    ready = engine.submit("ready")
    late = engine.submit("late", release=10.0)
    peer = native.program.peer_labels[0]
    phase = "count_status_exchange"
    placeholders = {
        (0, peer, phase): engine.future("local_count_arrival"),
        (peer, 0, phase): late,
    }
    event, handles = native.start_with_handles(
        engine, "native", ready, peer_ready={phase: {peer: late}},
        send_placeholders=placeholders)
    engine.advance(until=(event,))
    assert (handles["peer_events"][phase][peer]["receive"].time >= 10.0)


def test_mc2_peer_latency_uses_declared_class_and_launch_view():
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    meta = dict(features.metadata)
    meta["moe_rank_index"] = 0
    features = replace(features, metadata=tuple(sorted(meta.items())))
    supply = replace(SUPPLY, invocations=(
        ("ranks_per_node", 2),
        ("link_latency_intra_ms", 0.0018),
        ("link_latency_inter_ms", 0.005),
    ))
    pipe = assemble_model(model, supply=supply, policy=POLICY,
                          features=features)
    rows = {cost.name: cost for cost in _rows(pipe.forward(_decode2()), "F2")}
    native = rows["mc2_dispatch"].native_program
    resources = tuple(
        Resource(name, rate, "work", "fair", 2)
        for name, rate in supply.rates)
    engine = Engine(resources)
    ready = engine.submit("ready")
    event, handles = native.start_with_handles(engine, "native", ready)
    engine.advance(until=(event,))
    assert handles["ports"][1]["class"] == "hccs"
    assert handles["ports"][2]["class"] == "roce"
    assert handles["ports"][1]["latency_ms"] == Fraction(9, 5000)
    assert handles["ports"][2]["latency_ms"] == Fraction(1, 200)
    assert handles["launch"].rank == 0
    # A launch-local view for a different rank does not mutate the cached
    # program's source rank or peer labels.
    other = Mc2LaunchView(
        rank=1, peer_labels=(0, 2, 3), node_ranks=2,
        intra_latency_ms=Fraction(9, 5000),
        inter_latency_ms=Fraction(1, 200))
    second = Engine(resources)
    second_ready = second.submit("ready")
    second_event, second_handles = native.start_with_handles(
        second, "native", second_ready, launch=other)
    second.advance(until=(second_event,))
    assert second_handles["launch"].rank == 1
    assert native.program.view.rank == 0


def test_mc2_source_send_key_releases_only_matching_peer_phase():
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    meta = dict(features.metadata)
    meta["moe_rank_index"] = 0
    features = replace(features, metadata=tuple(sorted(meta.items())))
    supply = replace(SUPPLY, invocations=(
        ("ranks_per_node", 2),
        ("link_latency_intra_ms", 0.0018),
        ("link_latency_inter_ms", 0.005),
    ))
    pipe = assemble_model(model, supply=supply, policy=POLICY,
                          features=features)
    rows = {cost.name: cost for cost in _rows(pipe.forward(_decode2()), "F2")}
    native = rows["mc2_dispatch"].native_program
    resources = tuple(
        Resource(name, rate, "work", "fair", 2)
        for name, rate in supply.rates)
    engine = Engine(resources)
    ready = engine.submit("ready")
    late = engine.submit("late", release=10.0)
    source_send = {
        (2, 0, "count_status_exchange"): late,
        (1, 0, "count_status_exchange"): engine.submit("early_1_0"),
    }
    event, handles = native.start_with_handles(
        engine, "native", ready, source_send=source_send)
    engine.advance(until=(event,))
    assert handles["peer_events"]["count_status_exchange"][2]["receive"].time >= 10.0
    assert handles["peer_events"]["count_status_exchange"][1]["receive"].time < 10.0


def test_mc2_preallocated_send_placeholder_binds_only_matching_receiver():
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    meta = dict(features.metadata)
    meta["moe_rank_index"] = 0
    meta["moe_peer_weights"] = (0.1, 0.8, 0.1)
    features = replace(features, metadata=tuple(sorted(meta.items())))
    supply = replace(SUPPLY, invocations=(("ranks_per_node", 2),))
    pipe = assemble_model(model, supply=supply, policy=POLICY,
                          features=features)
    rows = {cost.name: cost for cost in _rows(pipe.forward(_decode2()), "F2")}
    native = rows["mc2_dispatch"].native_program
    resources = tuple(
        Resource(name, rate, "work", "fair", 2)
        for name, rate in supply.rates)
    engine = Engine(resources)
    ready = engine.submit("ready")
    phase = "count_status_exchange"
    placeholder = engine.future("send_0_2_count")
    late = engine.submit("arrival_2_0", release=10.0)
    placeholders = {(0, peer, phase): engine.future(
        f"send_0_{peer}_count") for peer in native.program.peer_labels}
    placeholders[(0, 2, phase)] = placeholder
    placeholders[(2, 0, phase)] = late
    placeholders[(1, 0, phase)] = engine.submit("arrival_1_0")
    placeholders[(3, 0, phase)] = engine.submit("arrival_3_0")
    event, handles = native.start_with_handles(
        engine, "native", ready, send_placeholders=placeholders)
    engine.advance(until=(event,))
    assert placeholder.done
    assert (handles["peer_events"][phase][2]["receive"].time >= late.time)
    assert (handles["peer_events"][phase][1]["receive"].time < late.time)


def test_mc2_launch_view_remaps_canonical_resource_names():
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    rows = {cost.name: cost for cost in _rows(
        _pipeline(model, features).forward(_decode2()), "F2")}
    native = rows["mc2_dispatch"].native_program
    view = Mc2LaunchView(
        rank=0, peer_labels=(1, 2, 3), node_ranks=2,
        resource_names=(("vector", "vector@r9"), ("link_hccs", "link@r9")))
    context = native.program._context(view)
    assert context.resource_names["vector"] == "vector@r9"
    assert context.resource_names["link_hccs"] == "link@r9"


def test_mc2_maskcnt_c2_c32_and_k3_typed_counts_not_double_charged():
    roles = dict(hbm=1, vector=64, mte2_aiv=64, mte3=64)
    geometry = Geometry(32, roles)

    def count_work(tokens, topk, experts, ep):
        raw = dict(
            kind="dispatch", width=5120, dtype="bf16", payload_dtype="bf16",
            dispatch_quant="none", source_tokens=tokens, top_k=topk,
            source_assignments=tokens * topk,
            received_assignments=tokens * topk, experts=experts, ep=ep,
            phase="count_compute")
        op = OpSpec("count", "mc2_protocol", "decode", 0, 0, (), (), (), (), (),
                    metadata=(("mc2_phase", raw),))
        plan = mc2_build(op, geometry)
        owners = plan.metadata["primitive_services"]["owners"]
        typed = {}
        for items in owners.values():
            for item in items.values():
                for role, value in item["counts"].items():
                    typed[role] = typed.get(role, 0) + value
        return plan.work.resources["vector"], typed

    c2_vec, c2 = count_work(1, 6, 384, 16)
    c32_vec, c32 = count_work(2, 6, 384, 16)
    k3_vec, k3 = count_work(2, 16, 896, 32)
    assert c2_vec == c32_vec == 27648  # generic flags only
    assert c2["pipe_v_lanes"] == 13824
    assert c32["pipe_v_lanes"] == 27648
    assert k3["pipe_v_lanes"] == 172032
    assert c2["reduce_sum_issue"] == 1920
    assert c32["reduce_sum_issue"] == 4224
    assert k3["reduce_sum_issue"] == 27776
    assert c2["ub_read_bytes"] == 9216
    assert c32["ub_read_bytes"] == 18432
    assert c2["pipe_v_issue"] == 1536
    assert k3["pipe_v_issue"] == 3584


def test_mc2_receive_flags_waits_real_arrival_without_priced_pipe_s():
    from copy import copy
    from mindie_sim.prediction.engine import Engine, Resource
    model = _model(experts=8, top_k=2)
    features = _features(model, additional={"enable_fused_mc2": 1})
    meta = dict(features.metadata)
    meta["moe_rank_index"] = 0
    features = replace(features, metadata=tuple(sorted(meta.items())))
    one = WorkloadGeometry((RequestGeometry(1, 0, phase="decode"),),
                           phase="decode")
    rows = {cost.name: cost for cost in _rows(
        _pipeline(model, features).forward(one), "F2")}
    native = rows["mc2_combine"].native_program
    inner = copy(native.program)
    inner.rates = {name: rate for name, rate in inner.rates.items()
                   if name != "pipe_s_issue"}
    native = type(native)(inner, native.work, native.duration)
    resources = tuple(Resource(name, rate, "work", "fair", 2)
                      for name, rate in inner.rates.items())
    engine = Engine(resources, trace_limit=100)
    ready = engine.submit("ready")
    late = engine.submit("late_arrival", release=10.0)
    event, handles = native.start_with_handles(
        engine, "native", ready,
        send_placeholders={(1, 0, "payload_exchange"): late})
    engine.advance(until=(event,))
    assert handles["receive_flags_owners"]
    assert handles["receive_flags_incoming"]
    assert handles["phase_complete"]["receive_flags"].time >= late.time
    assert not handles["receive_flags_retry"]
    assert not [row for row in engine.trace
                if "receive_flags:failed" in row["task"]]
    reads = [row for row in engine.trace
             if row["task"].endswith("receive_flags:owner0:read_flags")]
    assert reads
    assert min(row["start_ms"] for row in reads) >= late.time
    assert handles["phase_complete"]["receive_flags"].time >= late.time


