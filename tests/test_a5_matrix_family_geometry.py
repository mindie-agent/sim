"""A5 matrix-family source plans conserve declared and executed service.

The real Kimi K3 A5 case supplies an ordinary shared-expert FP8 x FP4 single
WeightNz matrix (GroupedMatmul's shared projection is not a routed bank).  The
same source program is then exercised at unsampled M/N/K and against an
independent tile-count oracle.  Small-M gate/up SituQuant and the typed U8
widen inventory are checked on their own source geometry without weights or a
device.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs, target_operators
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction import family_gate_up_situ_mxfp4 as gate_situ
from mindie_sim.prediction import family_grouped_gemm
from mindie_sim.prediction import family_mx_matrix
from mindie_sim.prediction import shared_w4a8
from mindie_sim.prediction.gmmsq_quant_typed import work as gmmsq_work
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.mxfp4_geometry import Config, align, owner_tiles
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.common import ModelingError

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/cases/kimi-k3-a5-sp-right-pad-conditional.json"
PREPARED = (ROOT / "examples/cases/high-concurrency-20261030/v41-mx32-conditional/"
            "a5_v41_fp8_tp1_dp8_ep8_c128_p2k-prepared-mx32-source.json")
CORES = 32
AIVS = 64


def _case():
    return parse_case(json.loads(CASE.read_text()))


@pytest.fixture(scope="module")
def case():
    return _case()


@pytest.fixture(scope="module")
def shared_ops(case):
    ops = target_graphs(case, scope="body", phase="decode", tokens=1,
                        context=2048, sequences=1, resource_fusion=False)[0]
    selected = tuple(
        op for op in ops
        if dict(op.metadata).get("ffn_category") == "shared_expert"
        and op.family in {"gate_up_projection", "down_projection"})
    assert len(selected) >= 2
    gate, down = selected[0], selected[1]
    assert gate.family == "gate_up_projection"
    assert down.family == "down_projection"
    return gate, down


@pytest.fixture(scope="module")
def grouped_base(case):
    ops = target_graphs(case, scope="body", phase="decode", tokens=1,
                        context=2048, sequences=1, resource_fusion=False)[0]
    return next(op for op in ops if op.family == "grouped_gemm")


def _assert_declared_equals_served(result):
    assert not result["service_before_input"]
    assert result["live_tasks"] == 0
    for resource, delta in result["service_delta"].items():
        served = result["serviced"].get(resource, 0.0)
        assert abs(delta) <= max(1e-6, abs(served) * 1e-9), (
            resource, delta, served)


def _retyped(op, m, n, k):
    """Replace one ordinary matrix shape and its dependent scale declarations."""
    metadata = dict(op.metadata)
    k_groups = (k + 63) // 64
    metadata["activation_scale_shape"] = (m, k_groups, 2)
    metadata["weight_scale_shape"] = (n, k_groups, 2)
    shape = (m, n, k)
    return replace(op, local_shape=shape, semantic_shape=shape, launch_shape=shape,
                   metadata=tuple(metadata.items()))


def test_real_kimi_shared_mxfp4_plan_equals_executed_service(case, shared_ops):
    geometry = Geometry.from_devices(case.machine.dies)
    assert geometry.resource_units["vector"] == AIVS
    for op in shared_ops:
        assert shared_w4a8.matches_source_interface(op)
        plan = shared_w4a8.build(op, geometry)
        result = operator_costs(case, (op,), input_ready_ms=7)["operators"][0]
        _assert_declared_equals_served(result)
        detail = result["operator"]["metadata"]["program_execution"]["shared_w4a8"]
        m, n, k = op.local_shape
        assert detail["logical_flops"] == 2 * m * n * k
        assert detail["issued_cube_flops"] == 2 * align(m, 16) * n * k
        assert detail["init_mx_bytes"] == shared_w4a8.INIT_PERMUTATION_BYTES * AIVS
        assert result["serviced"]["fixpipe@r0:s0"] == pytest.approx(2 * m * n)
        # The source AIC output is Fixpipe, not an AIV MTE3 publication.
        assert plan.work.matrix == pytest.approx({"fp8_e4m3": detail["issued_cube_flops"]})


@pytest.mark.parametrize("m,n,k", [
    (1, 64, 64),
    (3, 320, 448),
    (17, 1088, 1088),
    (257, 320, 512),
])
def test_shared_mxfp4_unsampled_geometry_conserves_each_owner(case, shared_ops, m, n, k):
    gate, _down = shared_ops
    op = _retyped(gate, m, n, k)
    result = operator_costs(case, (op,), input_ready_ms=7)["operators"][0]
    _assert_declared_equals_served(result)
    detail = result["operator"]["metadata"]["program_execution"]["shared_w4a8"]
    tiles = tuple(
        (tile[3], tile[4]) for owner in range(CORES)
        for tile in owner_tiles(Config(k, n, (m,)), owner, CORES))
    issued = 2 * sum(align(rows, 16) * columns * k for rows, columns in tiles)
    assert detail["logical_flops"] == 2 * m * n * k
    assert detail["issued_cube_flops"] == issued
    assert detail["mmad_parameter_m"] == tuple(sorted({rows for rows, _ in tiles}))
    assert detail["fixpipe_m"] == tuple(sorted({rows for rows, _ in tiles}))
    assert detail["l0a_issue_fractal_m"] == tuple(sorted(
        {align(rows, 16) for rows, _ in tiles}))
    assert result["serviced"]["fixpipe@r0:s0"] == pytest.approx(2 * m * n)
    assert detail["init_mx_bytes"] == shared_w4a8.INIT_PERMUTATION_BYTES * AIVS


def test_shared_mxfp4_source_interface_rejects_near_miss_declarations(shared_ops):
    gate, _down = shared_ops
    assert shared_w4a8.matches_source_interface(gate)
    mutations = (
        {"weight_format": "fp8_e4m3"},
        {"activation_format": "bf16"},
        {"output_format": "fp32"},
        {"matrix_groups": 2},
        {"matrix_group_rows": (8,)},
        {"bias": {"present": True}},
        {"activation_scale_shape": (8, 1, 2)},
        {"weight_layout": "FRACTAL_Z"},
        {"ffn_category": "routed_expert"},
        {"quant_category": "routed_expert"},
    )
    for change in mutations:
        metadata = {**dict(gate.metadata), **change}
        candidate = replace(gate, metadata=tuple(metadata.items()))
        assert not shared_w4a8.matches_source_interface(candidate), change
        with pytest.raises(ModelingError):
            shared_w4a8.config_from_op(candidate)


def _gate_op(base, rows, width, k):
    metadata = {
        key: value for key, value in dict(base.metadata).items()
        if key not in {"matrix_group_rows", "matrix_weight_bank_groups",
                       "matrix_groups", "grouped_dequant"}
    }
    metadata["mxfp4_gate_config"] = {
        "group_rows": tuple(rows), "output_width": width, "k": k}
    shape = (sum(rows), 2 * width, k)
    return replace(base, recipe=gate_situ.RECIPE, flops=0, matrix_flops=(),
                   vector_ops=0, hbm_bytes=0, mte2_bytes=0, mte3_bytes=0,
                   local_shape=shape, semantic_shape=shape, launch_shape=shape,
                   deps=(), metadata=tuple(metadata.items()))


@pytest.mark.parametrize("rows,width,k", [
    ((1,), 128, 64),
    ((2, 0, 3), 320, 448),
    ((5, 7), 192, 512),
    ((33,), 256, 1024),
])
def test_gate_up_situ_source_tiles_conserve_issued_and_published_work(
        case, grouped_base, rows, width, k):
    geometry = Geometry.from_devices(case.machine.dies)
    op = _gate_op(grouped_base, rows, width, k)
    book = RecipeBook({gate_situ.RECIPE: PlannedRecipe(gate_situ.build)})
    plan = gate_situ.build(op, geometry)
    result = operator_costs(case, (op,), recipes=book,
                            input_ready_ms=7)["operators"][0]
    _assert_declared_equals_served(result)
    detail = plan.metadata
    assert detail["source_output_tiles"] == sum(
        (bank + 127) // 128 for bank in rows) * ((width + 127) // 128)
    assert detail["physical_cube_flops"] == detail["physical_cube_flops_derived"]
    # Every output tile publishes two API-padded halves to Fixpipe.
    expected_fixpipe = sum(
        2 * (((bank_rows + 1) // 2) * 2) * min(128, width - column) * 4
        for bank_rows, column in _gate_extents(rows, width))
    assert detail["source_api_fixpipe_bytes"] == expected_fixpipe
    assert result["serviced"]["fixpipe@r0:s0"] == pytest.approx(expected_fixpipe)
    # Every source row bank contributes exactly its integer rows to logical FLOPs.
    assert detail["logical_valid_flops"] == 2 * 2 * sum(rows) * width * k


def _gate_extents(rows, width):
    for bank in rows:
        if not bank:
            continue
        for start in range(0, bank, 128):
            bank_rows = min(128, bank - start)
            for column in range(0, width, 128):
                yield bank_rows, column


def test_gate_up_situ_zero_bank_skips_tiles_and_does_not_invent_work(case):
    geometry = Geometry.from_devices(case.machine.dies)
    base = next(op for op in target_graphs(
        case, scope="body", phase="decode", tokens=1, context=2048,
        sequences=1, resource_fusion=False)[0] if op.family == "grouped_gemm")
    plan = gate_situ.build(_gate_op(base, (0, 0), 128, 64), geometry)
    assert plan.work.matrix == {}
    assert plan.work.resources == {}
    assert plan.metadata["source_output_tiles"] == 0
    assert plan.metadata["active_aic_owners"] == 0
    assert plan.metadata["producer_packed_weight_bytes"] == 0


@pytest.mark.parametrize("groups", [0, 1, 63, 64, 65, 257])
def test_gmmsq_u8_inventory_charges_declared_packing_once(groups):
    typed, counts, inventory = gmmsq_work(groups)
    vf_groups = (groups + 63) // 64
    assert counts.vf_groups == vf_groups
    assert inventory.counts["LoadAlignU8"].calls == groups
    assert inventory.counts["StoreAlignU8"].calls == 2 * groups
    assert inventory.counts["CreateMask"].calls == vf_groups
    assert inventory.counts["DuplicateU8"].calls == 3 * vf_groups
    assert counts.packed_bytes == 256 * groups
    assert counts.expanded_bytes == 512 * groups
    assert typed.resources.get("reg_issue_slots", 0) == 4 * vf_groups + 12 * groups
    assert typed.resources.get("reg_data_writes", 0) == 3 * vf_groups + 8 * groups
    assert typed.resources.get("reg_ub_accesses", 0) == 3 * groups
    assert typed.resources.get("reg_ub_stores", 0) == 2 * groups
    # U8 widen has no FP16/FP32 vector-lane resource, so it cannot borrow
    # another precision's lane capacity.
    assert not {"vector", "reg_alu_lanes", "reg_shuffle_lanes"} & typed.resources.keys()


def test_prepared_mx32_real_case_at_unsampled_batch_keeps_declared_storage():
    """One real paired-case slice outside the retained B1/B16 cohort."""
    raw = json.loads(PREPARED.read_text())
    raw["runtime"]["kernels"]["lowerings"]["mx_matrix"] = {
        "implementation": "ascend_prepared_mx32_matrix",
        "options": {
            "projection_recipes": ["q_b", "grouped_output_a", "grouped_output_b"],
            "scale_storage": "replace_checkpoint_scales",
        },
    }
    ops = target_graphs(raw, scope="body", phase="decode", tokens=1,
                        context=2048, sequences=[4] + [0] * 7)[0]
    op = next(op for op in ops
              if dict(op.metadata).get("semantic_recipe") == "grouped_output_a")
    metadata = dict(op.metadata)
    m, n, k = op.local_shape
    groups = metadata["matrix_groups"]
    assert groups == 8 and m % groups == 0
    payload = groups * n * k
    scales = groups * n * (k // 64) * 2
    assert metadata["packed_weight_bytes"] == payload + scales
    assert metadata["prepared_weight_scale_bytes"] == scales
    assert metadata["weight_representation"] == "fp8_block1x32_fp8_e8m0"
    assert metadata["fixpipe_bytes"] == 2 * m * n
    assert op.mte3_bytes == 0
    result = operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]
    assert result["serviced"]["fixpipe@r0:s0"] == pytest.approx(2 * m * n)
    assert result["serviced"]["memory@r0:s0"] == pytest.approx(op.hbm_bytes)
    assert not any(result["serviced"].get(name + "@r0:s0", 0)
                   for name in ("mte3", "vector"))
    assert not result["service_before_input"]
    assert result["live_tasks"] == 0
    assert not any(result["remaining_buffers"].values())
    # The MX scale storage is a weight-view replacement, not a second weight
    # allocation or an invented GM transpose copy.
    assert metadata["prepared_scale_storage"] == "replace_checkpoint_scales"
    assert metadata["matrix_output_contract"] == "aic_fixpipe_bf16_to_gm"
    assert family_mx_matrix.PROFILE == op.recipe


def _explicit_grouped_op(rows, n, k, realization):
    op = gemm_op(
        op_id="explicit_grouped", deps=(), stage=0, layer=0, requests=(),
        m=rows, n=n, k=k, quant=QuantSpec("bf16"), category="routed_expert",
        phase="prefill", family="grouped_gemm", recipe="grouped_gate_up")
    metadata = {**dict(op.metadata),
                "routing_realization": realization,
                "routing_basis": realization["basis"]}
    return replace(op, local_shape=(rows, n, k), metadata=tuple(metadata.items()),
                   flops=2 * rows * n * k)


def test_explicit_token_basis_consumes_destination_bank_and_global_slice():
    realization = {
        "basis": "explicit_token_routing_integer",
        "expert_placement": ((0, 0), (0, 1), (1, 0), (1, 1)),
        "local_expert_rows": (2, 2),
        "representative_expert_ids": (0, 1),
        "local_weight_expert_ids": (0, 1),
        "global_expert_rows": (2, 2, 2, 2),
        "representative_rank": 0,
        "valid_rows": 4, "received_rows": 4, "capacity_rows": 8,
    }
    lowered = family_grouped_gemm.lower_op(
        _explicit_grouped_op(4, 8, 4, realization))
    metadata = dict(lowered.metadata)
    assert metadata["routing_rows_basis"] == "explicit_destination_bank"
    assert metadata["routing_valid_rows"] == 4
    assert metadata["routing_received_rows"] == 4
    assert metadata["routing_capacity_rows"] == 8
    assert metadata["representative_expert_ids"] == (0, 1)
    assert metadata["global_expert_slice"] == (2, 2)
    assert metadata["global_expert_rows"] == (2, 2, 2, 2)


def test_explicit_token_basis_rejects_non_contiguous_local_bank():
    realization = {
        "basis": "explicit_token_routing_integer",
        "expert_placement": ((0, 0), (0, 1), (1, 0), (1, 2)),
        "local_expert_rows": (2, 2),
        "representative_expert_ids": (2, 3),
        "local_weight_expert_ids": (0, 1),
        "global_expert_rows": (2, 2, 2, 2),
        "representative_rank": 1,
    }
    with pytest.raises(ModelingError, match="explicit_local_bank_slot_out_of_range"):
        family_grouped_gemm.lower_op(_explicit_grouped_op(4, 8, 4, realization))


def test_explicit_token_basis_accepts_contiguous_physical_bank_on_rank_one():
    realization = {
        "basis": "explicit_token_routing_integer",
        "expert_placement": "contiguous_ep",
        "local_expert_rows": (2, 2),
        "representative_expert_ids": (2, 3),
        "local_weight_expert_ids": (0, 1),
        "global_expert_rows": (2, 2, 2, 2),
        "representative_rank": 1,
        "represented_ep_ranks": (1,),
    }
    lowered = family_grouped_gemm.lower_op(
        _explicit_grouped_op(4, 8, 4, realization))
    metadata = dict(lowered.metadata)
    assert metadata["representative_expert_ids"] == (2, 3)
    assert metadata["global_expert_slice"] == (2, 2)
    assert metadata["routing_representative_rank"] == 1


def test_explicit_token_route_enters_real_grouped_source_lowering():
    from _support.moe_hc_mc2_fixtures import explicit_route_case as _case
    raw = _case(route=((6, 10, 8, 8), True))
    raw["runtime"]["kernels"] = {"lowerings": {
        "grouped_gemm": {"implementation": "ascendc_grouped_nd_bf16"}}}
    ops = target_operators(raw, phase="prefill", tokens=6, scope="body")
    grouped = [op for op in ops if op.family == "grouped_gemm"]
    assert len(grouped) == 2
    for op in grouped:
        metadata = dict(op.metadata)
        assert metadata["routing_basis"] == "explicit_token_routing_integer"
        assert metadata["routing_rows_basis"] == "explicit_destination_bank"
        assert tuple(metadata["matrix_group_rows"]) == (8, 8)
        assert metadata["global_expert_slice"] == (8, 8)
    result = operator_costs(raw, (grouped[0],), input_ready_ms=0)["operators"][0]
    assert result["serviced"]["matrix@r0:s0"] > 0
    assert result["live_tasks"] == 0


def test_grouped_valid_received_capacity_keep_interface_rows_separate_from_emission():
    realization = {
        "basis": "explicit_token_routing_integer",
        "expert_placement": "contiguous_ep",
        "local_expert_rows": (16, 0),
        "representative_expert_ids": (0, 1),
        "local_weight_expert_ids": (0, 1),
        "global_expert_rows": (16, 0),
        "representative_rank": 0,
        "represented_ep_ranks": (0,),
        "valid_rows": 16, "received_rows": 32, "capacity_rows": 64,
        "capacity_slack": 32, "padding_rows": 16,
    }
    lowered = family_grouped_gemm.lower_op(
        _explicit_grouped_op(16, 8, 4, realization))
    metadata = dict(lowered.metadata)
    assert lowered.local_shape[0] == 16
    assert metadata["local_expert_rows"] == (16, 0)
    assert metadata["routing_valid_rows"] == 16
    assert metadata["routing_received_rows"] == 32
    assert metadata["routing_capacity_rows"] == 64
    assert metadata["routing_capacity_slack"] == 32
    assert metadata["routing_padding_rows"] == 16
    # Capacity is provenance only; emitted matrix rows come from valid local
    # regions, not from the 64-row interface capacity.
    assert lowered.local_shape == (16, 8, 4)


def test_native_grouped_samples_keep_interface_capacity_and_issued_regions():
    samples = {
        "grouped_v4_16_32_64": {
            "geometry": {"valid_rows": 16, "received_rows": 32,
                         "interface_rows": 32, "capacity_rows": 64,
                         "padding_rows": 16, "capacity_slack": 32},
            "regions": [{"rows": 16, "issued_macs": 65536},
                        {"rows": 0, "issued_macs": 0}],
            "work": {"cube": {"issued_macs": 65536}},
        },
        "gmm_swiglu_one_one": {
            "geometry": {"valid_rows": 2, "received_rows": 2,
                         "interface_rows": 2, "capacity_rows": 2,
                         "padding_rows": 0, "capacity_slack": 0},
            "regions": [{"rows": 1, "issued_macs": 65536},
                        {"rows": 1, "issued_macs": 65536}],
            "work": {"cube": {"issued_macs": 131072}},
        },
        "gmm_swiglu_two_zero": {
            "geometry": {"valid_rows": 2, "received_rows": 2,
                         "interface_rows": 2, "capacity_rows": 2,
                         "padding_rows": 0, "capacity_slack": 0},
            "regions": [{"rows": 2, "issued_macs": 65536},
                        {"rows": 0, "issued_macs": 0}],
            "work": {"cube": {"issued_macs": 65536}},
        },
    }
    for name, package in samples.items():
        contract = family_grouped_gemm.native_grouped_row_contract(package)
        assert contract["interface_rows"] == 32 if name == "grouped_v4_16_32_64" else 2
        assert contract["capacity_rows"] == 64 if name == "grouped_v4_16_32_64" else 2
        assert contract["valid_rows"] == 16 if name == "grouped_v4_16_32_64" else 2
        assert sum(contract["group_rows"]) == contract["valid_rows"]
        assert contract["nnz_rows"] == contract["valid_rows"]
        assert contract["active_regions"] == tuple(
            i for i, rows in enumerate(contract["group_rows"]) if rows)
        assert contract["issued_macs"] == package["work"]["cube"]["issued_macs"]
    # The native package's own quantized program is replayed by
    # replay_native_leaf.py.  This family test intentionally does not replace
    # it with a BF16 source plan.
