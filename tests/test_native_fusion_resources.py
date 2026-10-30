"""Selected native resources reuse the real builders and causal engine."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs
from mindie_sim.prediction.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.native_fusion_traffic import apply_traffic_profile
from mindie_sim.prediction.operator_analysis import target_graphs, target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure_physical import node_physical


ROOT = Path(__file__).resolve().parents[1]
HC_PROFILE = "hc_pre_m_k_split_175ef4d4"
MLA_PROFILE = "mla_prolog_arch35_split_m_n_ae923905"
GMM_PROFILE = "grouped_swiglu_quant_int4_post_ae923905"


def _case(name, roles, profile="preserve_workspaces", peaks=None, **source_lowerings):
    raw = json.loads((ROOT / "examples/cases" / (name + ".json")).read_text())
    if peaks:
        raw["machine"].setdefault("device", {}).setdefault(
            "compute_peaks", {}).update(peaks)
    case = parse_case(raw)
    kernels = {**case.runtime.kernels, "lowerings": {
        **case.runtime.kernels.get("lowerings", {}), **source_lowerings,
        "native_fusion": {"implementation": "declared_native_fusion",
                          "options": {"roles": roles, "traffic_profile": profile}}}}
    return replace(case, runtime=replace(case.runtime, kernels=kernels))


def _group(ops, role):
    return next((declaration, members) for declaration, members in _groups(ops)
                if declaration["role"] == role)


def _phases(op):
    return {phase["recipe"]: phase for phase in dict(op.metadata)["fusion_phases"]}


def _external(ops):
    ids = {op.op_id for op in ops}
    return {dep: 0 for op in ops for dep in op.deps if dep not in ids}


def test_receive_only_kimi_rank_preserves_expert_fusion_without_local_situ():
    raw = json.loads((ROOT / 'examples/cases/kimi-k3-a5-sp-right-pad-conditional.json').read_text())
    raw['model']['layers'] = 2
    graphs = target_graphs(raw, phase='prefill', tokens=64, scope='body',
                          sequences=[1, 0, 0, 0])
    for replica, ops in graphs.items():
        roles = {dict(op.metadata).get('fusion_role') for op in ops}
        assert 'grouped_gate_up_situ_quant' in roles
        assert ('situ_quant' in roles) == (replica == 0)
        assert any(op.family == 'alltoall' and op.link_bytes > 0 for op in ops)
    result = graph_costs(raw, graphs, external_inputs={'analysis_input': 0}, isolated=False)
    assert result['duration_ms'] > 0
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    # The idle source has real expert compute and return traffic; exempting a
    # nonexistent local activation does not disable its physical participant.
    assert result['execution']['serviced']['matrix@r1:s0'] > 0


def test_hc_fusion_runs_one_launch_with_source_gram_and_external_outputs():
    from mindie_sim.prediction.family_hc import pre_phase_work
    from mindie_sim.prediction.mechanism import Geometry

    # Explicit test supply for the source's HF32 projection phase; not an A3
    # hardware-rate claim.
    case = _case("v4f-a3-w8a8-tp8", ["hc_pre"], HC_PROFILE,
                 peaks={"hf32": 2.0e14})
    semantic = target_operators(case, phase="prefill", tokens=8, resource_fusion=False)
    declaration, members = _group(semantic, "hc_pre")
    physical = lower_native_fusions(members, case.runtime)
    assert len(members) == 6 and len(physical) == 1
    fused = physical[0]
    phases = _phases(fused)
    old = {op.recipe: op for op in members}
    tokens, mixed, hc = (declaration["tensor_view"][key] for key in ("tokens", "mixed", "hc"))
    # The semantic declaration remains inspectable. Source preparation replaces
    # its standalone RMS work with a shared-input Gram computation.
    writes = tokens * mixed * 4
    reads = writes + tokens * hc * 4
    assert fused.hbm_bytes == sum(op.hbm_bytes for op in members) - reads - writes
    assert fused.flops == sum(op.flops for op in members)
    assert fused.vector_ops == sum(op.vector_ops for op in members)
    for recipe in ("hc_values_fp32", "hc_projection", "hc_inverse_rms"):
        assert phases[recipe]["hbm_bytes"] == old[recipe].hbm_bytes
    assert phases["hc_sinkhorn"]["mte3_bytes"] == old["hc_sinkhorn"].mte3_bytes
    assert phases["hc_pre_mix"]["mte3_bytes"] == old["hc_pre_mix"].mte3_bytes
    reference = graph_costs(case, members, external_inputs=_external(members),
                            include_submission=True, isolated=False)
    selected = graph_costs(case, physical, external_inputs=_external(members),
                           include_submission=True, isolated=False)
    before, after = reference["execution"]["serviced"], selected["execution"]["serviced"]
    timeline = selected['timeline'][0]
    assert timeline['fusion_role'] == 'hc_pre'
    assert timeline['fusion_group_key'] == dict(fused.metadata)['fusion_group_key']
    assert timeline['physical_source_program'] == HC_PROFILE
    assert timeline['native_implementation'] == declaration['implementation']
    assert before["host@r0:s0"] == pytest.approx(6 * after["host@r0:s0"])
    device = case.machine.dies[0]
    source = pre_phase_work(
        declaration["tensor_view"],
        Geometry(device.compute_units, {"vector": device.units("vector")}))
    assert after["memory@r0:s0"] == pytest.approx(
        sum(work.resources.get("hbm", 0) for work in source.values()))
    assert after["vector@r0:s0"] == pytest.approx(
        sum(work.resources.get("vector", 0) for work in source.values()))
    assert after["matrix@r0:s0"] == pytest.approx(
        sum(1000 * amount / device.peak(dtype)
            for work in source.values() for dtype, amount in work.matrix.items()))
    assert after["matrix@r0:s0"] > before["matrix@r0:s0"]
    assert selected["execution"]["live_tasks"] == selected["execution"]["live_programs"] == 0


def test_default_fusion_retains_all_work_and_only_explicit_selection_changes_launches():
    case = _case("v4f-a3-w8a8-tp8", ["hc_pre"])
    semantic = target_operators(case, phase="prefill", tokens=8, resource_fusion=False)
    _, members = _group(semantic, "hc_pre")
    kernels = {**case.runtime.kernels, "lowerings": {
        key: value for key, value in case.runtime.kernels["lowerings"].items()
        if key != "native_fusion"}}
    baseline = replace(case.runtime, kernels=kernels)
    assert lower_native_fusions(members, baseline) == tuple(members)
    fused = lower_native_fusions(members, case.runtime)[0]
    for field in ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops"):
        assert getattr(fused, field) == sum(getattr(op, field) for op in members)
    boundary = dict(fused.metadata)["fusion_boundary"]
    assert boundary["traffic"]["reused_edges"] == []
    assert boundary["evidence"]["capture_binary_identity"] == "not_established"


def test_extended_resource_demand_adds_to_existing_phase_work():
    case = _case("v4f-a3-w8a8-tp8", ["hc_pre"])
    semantic = target_operators(case, phase="prefill", tokens=8, resource_fusion=False)
    _, members = _group(semantic, "hc_pre")
    members[0] = replace(members[0], metadata=tuple({
        **dict(members[0].metadata), "resource_work": {"vector": 1000}}.items()))
    physical = lower_native_fusions(members, case.runtime)
    reference = graph_costs(case, members, external_inputs=_external(members), isolated=False)
    selected = graph_costs(case, physical, external_inputs=_external(members), isolated=False)
    assert selected["execution"]["serviced"]["vector@r0:s0"] == (
        reference["execution"]["serviced"]["vector@r0:s0"])
    assert physical[0].vector_ops == sum(op.vector_ops for op in members) + 1000


def test_mla_and_indexer_alias_live_outputs_without_deleting_query_workspaces():
    case = _case("glm53-a3-w4a8", ["mla_prolog", "lightning_indexer"],
                 {"mla_prolog": MLA_PROFILE},
                 mla_prolog="vllm_ascend_mla_prolog_v3")
    semantic = target_operators(case, phase="prefill", tokens=8, resource_fusion=False)
    physical = lower_native_fusions(semantic, case.runtime)
    declaration, members = _group(semantic, "mla_prolog")
    fused = next(op for op in physical if dict(op.metadata).get("fusion_group_key") == declaration["key"])
    before, after = {op.recipe: op for op in members}, _phases(fused)
    for recipe in ("q_norm", "dynamic_quantize", "rotary", "kv_store", "q_b"):
        assert after[recipe]["mte3_bytes"] == before[recipe].mte3_bytes
    assert after["kv_norm"]["mte3_bytes"] == 0
    assert after["kv_store"]["mte2_bytes"] == 0
    query = next(op for op in physical if op.op_id == "s0.l0.index_q")
    attention = next(op for op in physical if op.layer == 0 and op.family == "attention")
    assert fused.op_id in query.deps and fused.op_id in attention.deps
    indexer = next(op for op in physical if dict(op.metadata).get("fusion_role") == "lightning_indexer")
    _, index_members = _group(semantic, "lightning_indexer")
    assert indexer.hbm_bytes == sum(op.hbm_bytes for op in index_members)
    positions = {op.op_id: index for index, op in enumerate(physical)}
    assert all(positions[dep] < positions[op.op_id] for op in physical for dep in op.deps
               if dep in positions)


def test_external_consumer_preserves_the_producer_write_even_when_edge_is_local():
    # Explicit test supply for the source's HF32 projection phase; not an A3
    # hardware-rate claim.
    case = _case("v4f-a3-w8a8-tp8", ["hc_pre"], HC_PROFILE,
                 peaks={"hf32": 2.0e14})
    semantic = target_operators(case, phase="prefill", tokens=8, resource_fusion=False)
    declaration, members = _group(semantic, "hc_pre")
    affine = next(op for op in members if op.recipe == "hc_normalize_scale_bias_gate")
    adjusted, ledger = apply_traffic_profile(
        members, declaration, HC_PROFILE, external_outputs=(affine.op_id,))
    actual = next(op for op in adjusted if op.op_id == affine.op_id)
    assert actual.mte3_bytes == affine.mte3_bytes
    assert ledger["eliminated_writes"] == 0
    assert ledger["eliminated_reads"] > 0


@pytest.mark.parametrize("name,profile", [
    ("glm53-a3-w4a8", GMM_PROFILE),
    ("v4f-a3-w8a8-tp8", "grouped_swiglu_quant_int8_split_workspace_ae923905"),
])
def test_grouped_postprocess_eliminates_only_swiglu_quant_intermediate(name, profile):
    case = _case(name, ["grouped_gate_up_activation_quant"], profile)
    semantic = target_operators(case, phase="prefill", tokens=8, resource_fusion=False)
    _, members = _group(semantic, "grouped_gate_up_activation_quant")
    fused = lower_native_fusions(members, case.runtime)[0]
    phases = dict(fused.metadata)["fusion_phases"]
    original_matrix = next(op for op in members if op.family == "grouped_gemm")
    matrix = next(op for op in phases if op["family"] == "grouped_gemm")
    activation = next(op for op in members if op.family == "swiglu")
    assert matrix["mte3_bytes"] == original_matrix.mte3_bytes
    assert fused.hbm_bytes == sum(op.hbm_bytes for op in members) - 2 * activation.mte3_bytes
    assert not any(op["recipe"] == "expert_down" for op in phases)
    assert fused.vector_ops == sum(op.vector_ops for op in members)


def test_resource_cohorts_ignore_layer_identity_but_retain_selected_traffic_recipe():
    # Explicit test supply for the source's HF32 projection phase; not an A3
    # hardware-rate claim.
    case = _case("v4f-a3-w8a8-tp8", ["hc_pre"], HC_PROFILE,
                 peaks={"hf32": 2.0e14})
    semantic = target_operators(case, phase="prefill", tokens=8, resource_fusion=False)
    physical = lower_native_fusions(semantic, case.runtime)
    fused = [op for op in physical if op.family == "native_fusion"]
    signatures = {node_physical(op)["physical_signature"] for op in fused}
    assert len({op.layer for op in fused}) > 1 and len(signatures) == 1
    for op in fused:
        phases = dict(op.metadata)["fusion_phases"]
        assert [phase["op_id"] for phase in phases] == [str(i) for i in range(len(phases))]
    preserved = _case("v4f-a3-w8a8-tp8", ["hc_pre"])
    other = lower_native_fusions(semantic, preserved.runtime)
    other = next(op for op in other if op.family == "native_fusion")
    assert node_physical(other)["physical_signature"] not in signatures


def test_unknown_chunk_stage_cannot_be_priced_as_the_complete_native_algorithm():
    case = _case("kimi-k3-a5", ["kda_chunk"])
    semantic = target_operators(case, phase="prefill", tokens=8, resource_fusion=False)
    with pytest.raises(ValueError, match="native_fusion_requires_complete_single_kernel"):
        lower_native_fusions(semantic, case.runtime)
    no_match = _case("v4f-a3-w8a8-tp8", ["mla_prolog"])
    with pytest.raises(ValueError, match="native_fusion_selected_roles_unmatched:mla_prolog"):
        target_operators(no_match, phase="prefill", tokens=8)


def test_local_reuse_cannot_silently_disagree_with_shape_generated_memory_program():
    # Explicit test supply for the source's HF32 projection phase; not an A3
    # hardware-rate claim.
    case = _case("v4f-a3-w8a8-tp8", ["hc_pre"], HC_PROFILE,
                 peaks={"hf32": 2.0e14})
    kernels = {**case.runtime.kernels, "recipes": {
        "hc_normalize_scale_bias_gate": {
            "kind": "tiled_pointwise", "input_bytes_per_element": 4,
            "outputs": [{"bytes_per_element": 4}]}}}
    case = replace(case, runtime=replace(case.runtime, kernels=kernels))
    with pytest.raises(ValueError, match="local_reuse_requires_work_preserving_phase_recipe"):
        target_operators(case, phase="prefill", tokens=8)


def test_resource_projection_preserves_declared_capture_boundaries():
    case = _case("v4f-a3-w8a8-tp8", ["hc_pre"])
    case = replace(case, runtime=replace(
        case.runtime, graph_mode="auto", graph_capture_mode="piecewise",
        piecewise_capture_keys=(8,), mixed_mode="mixed"))
    semantic = target_graphs(case, phase="prefill", tokens=8, resource_fusion=False)
    physical = target_graphs(case, phase="prefill", tokens=8)
    assert physical == {rank: lower_native_fusions(ops, case.runtime)
                        for rank, ops in semantic.items()}
    _, members = _group(next(iter(semantic.values())), "hc_pre")
    assert all(dict(op.metadata)["graph_captured"] for op in members)
    fused = lower_native_fusions(members, case.runtime)[0]
    assert dict(fused.metadata)["graph_segment"] == dict(members[0].metadata)["graph_segment"]
    # An explicit split through the middle of the group is a real dispatch
    # boundary; folding it cannot silently extend the graph's capture coverage.
    split = replace(case, runtime=replace(
        case.runtime, splitting_ops=("hc_inverse_rms",)))
    with pytest.raises(ValueError, match="native_fusion_crosses_graph_capture_boundary"):
        target_graphs(split, phase="prefill", tokens=8)
