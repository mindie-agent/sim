"""HC source work, memory conservation and finite dependency execution."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.ops import vector_op
from mindie_sim.prediction.family_hc import (
    POST_PROFILE, PRE_PROFILE, post_buffer_bytes, post_chunk_work, post_work,
    pre_geometry, pre_phase_work, selected_plan,
)
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.structure import export_structure


ROOT = Path(__file__).resolve().parents[1]


def _declared(name, role, profile, tokens, *, peaks=None):
    raw = json.loads((ROOT / "examples/cases" / (name + ".json")).read_text())
    if peaks:
        raw["machine"].setdefault("device", {}).setdefault(
            "compute_peaks", {}).update(peaks)
    raw.setdefault("runtime", {}).setdefault("kernels", {}).setdefault("lowerings", {})[
        "native_fusion"] = {
            "implementation": "declared_native_fusion",
            "options": {"roles": [role], "traffic_profile": profile},
        }
    case = parse_case(raw)
    semantic = target_operators(case, phase="prefill", tokens=tokens, resource_fusion=False)
    declaration, members = next((d, m) for d, m in _groups(semantic) if d["role"] == role)
    op = lower_native_fusions(members, case.runtime)[0]
    device = case.machine.dies[0]
    geometry = Geometry(
        device.compute_units,
        {role: device.units(role) for role in (*dict(device.rates), "cube", "host")},
        dict(device.buffers), dict(device.buffer_units))
    return case, declaration["tensor_view"], op, geometry


@pytest.mark.parametrize("name,tokens,hidden,m_parts,k_parts", [
    ("v4f-a3-w8a8-tp8", 4096, 4096, 16, 1),
    ("v4pro-a3-w4a8-tp16", 1024, 7168, 4, 5),
])
def test_hc_pre_shared_l1_gram_and_split_k_work(name, tokens, hidden, m_parts, k_parts):
    # These dimensions come from independent catalog cases, not profiler rows.
    _, view, op, geometry = _declared(name, "hc_pre", PRE_PROFILE, tokens)
    assert view["hidden"] == hidden
    physical = pre_geometry(view, geometry)
    assert physical.m_partitions == m_parts
    assert physical.k_partitions == k_parts
    assert physical.cast_ring_bytes == geometry.compute_units * 2 * 256 * 1024 * 4
    phases = pre_phase_work(view, geometry)
    cube = phases["projection_and_gram"]
    width = 4 * hidden
    # One FP32-X load supplies both the projection and 16-row Gram products.
    weights = physical.m_tile_count * width * 24 * 4
    assert cube.resources["mte2_aic"] - weights == tokens * width * 4
    assert cube.matrix == {
        "hf32": 2 * tokens * width * 24,
        "fp32": 2 * tokens * width * 16,
    }
    partials = tokens * k_parts * (24 + 16) * 4
    # Fixp/CopyOut writes the projection and Gram partials, not AIV MTE3.
    assert cube.resources["fixpipe"] == partials
    assert "mte3" not in cube.resources
    assert phases["coefficients"].resources["mte2_aiv"] >= partials
    assert phases["cast"].resources["mte3"] == tokens * width * 4
    plan = selected_plan(op, geometry)
    assert plan.work.resources["hbm"] == sum(
        work.resources.get("hbm", 0) for work in phases.values())


def test_hc_pre_single_program_execution_uses_source_work_and_preserves_ready_boundary():
    # Test-only explicit HF32 supply: A3 descriptors must declare their own;
    # this value is not an A3 rate claim.
    case, view, op, geometry = _declared(
        "v4f-a3-w8a8-tp8", "hc_pre", PRE_PROFILE, 64,
        peaks={"hf32": 2.0e14})
    expected = selected_plan(op, geometry).work
    report = operator_costs(case, (op,), input_ready_ms=17)
    row = report["operators"][0]
    device = case.machine.dies[0]
    assert row["service_before_input"] == {}
    assert row["serviced"]["memory@r0:s0"] == pytest.approx(expected.resources["hbm"])
    assert row["serviced"]["vector@r0:s0"] == pytest.approx(expected.resources["vector"])
    assert row["serviced"]["matrix@r0:s0"] == pytest.approx(
        sum(1000 * amount / device.peak(dtype)
            for dtype, amount in expected.matrix.items()))
    assert row["live_tasks"] == 0
    assert row["operator"]["hbm_bytes"] == expected.resources["hbm"]
    assert row["operator"]["matrix_flops"] == expected.matrix


def test_hc_manifest_exports_prepared_source_work_without_changing_semantic_members():
    case, view, op, geometry = _declared("v4f-a3-w8a8-tp8", "hc_pre", PRE_PROFILE, 32)
    manifest = export_structure(case, phase="prefill", tokens=32, scope="body")
    selected = next(group for group in manifest["subgraphs"] if group["role"] == "hc_pre")
    expected = selected_plan(op, geometry).work
    work = selected["physical_work"]
    assert work["hbm_bytes"] == expected.resources["hbm"]
    assert work["matrix_flops"] == expected.matrix
    assert work["vector_ops"] == expected.resources["vector"]
    assert work["launches"] == 1
    members = [node for node in manifest["nodes"] if node["node_id"] in selected["node_ids"]]
    assert len(members) == 6
    assert work["matrix_flops"]["fp32"] > sum(
        node["physical_work"]["matrix_flops"].get("fp32", 0) for node in members)
    assert all(node["event_matchable"] is False for node in members)
    resource = next(node for node in manifest["resource_operators"]
                    if node["node_id"] in selected["resource_boundary"]["operator_ids"])
    assert resource["physical_work"] == work
    # Prepared export retains the physical split services in operator metadata.
    split = resource["operator"]["metadata"]["resource_work"]
    assert split["mte2_aiv"] == expected.resources["mte2_aiv"]
    assert split["mte2_aic"] == expected.resources["mte2_aic"]
    assert split["fixpipe"] == expected.resources["fixpipe"]


def test_hc_source_rejects_an_external_intermediate_absent_from_its_native_outputs():
    case, _, _, geometry = _declared("v4f-a3-w8a8-tp8", "hc_pre", PRE_PROFILE, 8)
    semantic = target_operators(case, phase="prefill", tokens=8, resource_fusion=False)
    _, members = next((d, m) for d, m in _groups(semantic) if d["role"] == "hc_pre")
    affine = next(op for op in members if op.recipe == "hc_normalize_scale_bias_gate")
    consumer = vector_op(
        op_id="external_affine_consumer", deps=(affine.op_id,), stage=0, layer=0,
        requests=(), elements=8 * 24, dtype="fp32", family="elementwise",
        phase="prefill", recipe="consume_affine", ops_per_element=1)
    physical = lower_native_fusions((*members, consumer), case.runtime)
    with pytest.raises(ModelingError, match="cannot_export_internal_intermediate"):
        selected_plan(physical[0], geometry)


def test_hc_post_tail_chunks_read_coefficients_once_and_include_fp32_conversions():
    view = {"tokens": 43, "hidden": 2112, "hc": 4, "dtype": "bf16",
            "coefficient_dtype": "fp32"}
    first = dict(post_chunk_work(view, 2048, first=True))
    tail = dict(post_chunk_work(view, 64, first=False))
    assert tail["coefficients"].resources == {}
    chunked_read = sum(work.resources.get("mte2", 0)
                       for work in (*first.values(), *tail.values()))
    expected_read = 2112 * (1 + 4) * 2 + (4 + 16) * 4
    assert chunked_read == expected_read
    total = post_work(view)
    assert total.resources["mte2"] == 43 * expected_read
    assert total.resources["mte3"] == 43 * 2112 * 4 * 2
    mathematical = 43 * 2112 * 4 * (2 * 4 + 1)
    conversions = 43 * 2112 * (1 + 2 * 4)
    assert total.resources["vector"] == mathematical + conversions
    # Source reserves its configured 2048-column queues even for this width.
    assert post_buffer_bytes(view) == 164640


def test_hc_post_source_pipeline_conserves_work_and_costs_dependent_phases():
    case, view, op, geometry = _declared("v4f-a3-w8a8-tp8", "hc_post", POST_PROFILE, 43)
    expected = post_work(view, read_role="mte2_aiv")
    row = operator_costs(case, (op,), input_ready_ms=7)["operators"][0]
    assert row["service_before_input"] == {}
    assert row["serviced"]["memory@r0:s0"] == pytest.approx(expected.resources["hbm"])
    assert row["serviced"]["vector@r0:s0"] == pytest.approx(expected.resources["vector"])
    assert not any(row["remaining_buffers"].values())
    assert row["live_tasks"] == 0
    device = case.machine.dies[0]
    memory_only = expected.resources["hbm"] / dict(device.rates)["hbm"] * 1000
    assert row["duration_ms"] > memory_only
    # A second call uses the same finite plan; no state survives the first one.
    again = operator_costs(case, (replace(op, op_id="another_occurrence"),))["operators"][0]
    assert again["duration_ms"] == pytest.approx(row["duration_ms"])
    assert again["serviced"] == pytest.approx(row["serviced"])
