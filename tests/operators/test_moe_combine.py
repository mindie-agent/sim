"""Independent arithmetic, dispatch, ownership and work oracles for MoE combine.

The selected source is the current-checkout Triton ``moe_fused_mul_sum_kernel``
with the no-expert-map branch and the explicit SM89 heuristic.  These tests
establish source identity, threshold dispatch, masked tail work, repeated
weight traffic, per-SM CTA ownership, bounded live state and an independent
NumPy weighted-sum oracle.  A green contract run is structural only; it is not
accuracy acceptance.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.operators import moe_combine as mc
from mindie_sim.prediction.program import (
    ParallelProgram, PipelineProgram, RecipeBook, RecipeContext,
)
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.common import ModelingError

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/fixtures/operator_cases/moe_combine.json"
MANIFEST = ROOT / "tests/fixtures/operator_sources/moe_combine.json"

ROLES = ("hbm", "mte2", "mte3", "vector")


def hardware_case(*, units=32, split_copy=False, vector_per_unit=1e9,
                  memory_per_unit=1e12, copy_per_unit=1e9, vector_units=None):
    """Synthetic hardware where every declared unit has a fixed per-unit rate.

    Case rates are per second; the engine divides them by 1000.  With
    ``vector_units == units`` a CTA pinned to one owner receives exactly
    ``vector_per_unit`` ops/s regardless of the device width.
    """
    vector_units = units if vector_units is None else vector_units
    rates = {
        "memory": units * memory_per_unit,
        "vector": units * vector_per_unit,
    }
    if split_copy:
        rates.update(read=units * copy_per_unit, write=units * copy_per_unit)
    return {
        "machine": {
            "name": "independent-synthetic-control", "num_devices": 1,
            "device": {
                "memory_bytes": 1 << 30, "compute_units": units,
                "compute_peaks": {"bf16": 1e14, "fp16": 1e14, "fp32": 2.5e13},
                "rates": rates, "resource_units": {"vector": vector_units},
            },
        },
        "model": {
            "name": "unlabeled-geometry", "layers": 1, "hidden_size": 8,
            "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
            "intermediate_size": 16, "vocab_size": 16,
        },
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


def params(tokens=3, hidden=64, topk=2, *, dtype="bf16", weight_dtype="fp32",
           **overrides):
    value = {
        "tokens": tokens, "hidden": hidden, "topk": topk,
        "dtype": dtype, "weight_dtype": weight_dtype,
        "compute_capability": mc.COMPUTE_CAPABILITY,
        "implementation": mc.IMPLEMENTATION,
        "apply_router_weight_on_input": False,
        "has_expert_map": False,
    }
    value.update(overrides)
    return value


def campaign_cases():
    return json.loads(CASES.read_text())["cases"]


def context_for(op, units=32, *, vector_units=None, copy_roles=True):
    vector_units = units if vector_units is None else vector_units
    roles = ROLES if copy_roles else ("hbm", "vector")
    return RecipeContext(
        op=op, work={}, resource_names={role: role for role in roles},
        rates={role: 1.0 for role in roles},
        core_count=units, resource_units={"vector": vector_units},
    )


def iter_cta_tiles(program):
    """Yield ``(global_cta_index, stages)`` for a per-owner CTA program.

    This is used only by small oracles.  The huge-token test inspects stream
    counts without materializing any tile.
    """
    owners = len(program.programs)
    for core, stream in enumerate(program.programs):
        for local in range(stream.count):
            yield core + local * owners, stream.tile(local)


def reference_work(tokens, hidden, topk, input_bytes, weight_bytes,
                   block_m, block_k):
    """Independently enumerate every source CTA and its valid lanes."""
    m_tiles = (tokens + block_m - 1) // block_m
    k_tiles = (hidden + block_k - 1) // block_k
    a_upconvert = 1 if input_bytes < 4 else 0
    w_upconvert = 1 if weight_bytes < 4 else 0
    out_downconvert = 1 if input_bytes < 4 else 0
    per_program = (
        block_m * block_k
        + block_m * block_k * topk * (2 + a_upconvert)
        + block_m * topk * w_upconvert
        + block_m * block_k * out_downconvert
    )
    hbm = mte2 = mte3 = 0
    for pid_m in range(m_tiles):
        valid_m = min(block_m, tokens - pid_m * block_m)
        for pid_k in range(k_tiles):
            valid_k = min(block_k, hidden - pid_k * block_k)
            a_bytes = topk * valid_m * valid_k * input_bytes
            w_bytes = topk * valid_m * weight_bytes
            stored = valid_m * valid_k * input_bytes
            hbm += a_bytes + w_bytes + stored
            mte2 += a_bytes + w_bytes
            mte3 += stored
    return {
        "m_tiles": m_tiles, "k_tiles": k_tiles,
        "programs": m_tiles * k_tiles,
        "per_program": per_program,
        "hbm": hbm, "mte2": mte2, "mte3": mte3,
        "vector": m_tiles * k_tiles * per_program,
    }


def program_totals(op, units=32, *, vector_units=None, copy_roles=True):
    vector_units = units if vector_units is None else vector_units
    plan = mc.build(op, Geometry(units, {"vector": vector_units}, {}))
    program = plan.program(
        context_for(op, units, vector_units=vector_units,
                    copy_roles=copy_roles)
    )
    assert isinstance(program, ParallelProgram)
    totals: dict[str, float] = {}
    count = 0
    for _index, stages in iter_cta_tiles(program):
        count += 1
        assert tuple(stage.name for stage in stages) == ("load", "fma", "store")
        assert stages[1].after == ("load",)
        assert stages[2].after == ("fma",)
        for stage in stages:
            for role, amount in stage.work.items():
                totals[role] = totals.get(role, 0) + amount
    assert count == sum(stream.count for stream in program.programs)
    return plan, program, totals


def to_bf16(values):
    """Round FP32 to BF16 with round-to-nearest-even, without torch."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def weighted_sum_oracle(inputs, weights):
    """Explicit source-order FP32 accumulation and output downconvert."""
    a = np.asarray(inputs, dtype=np.float32)
    b = np.asarray(weights, dtype=np.float32)
    if a.ndim != 3 or b.shape != a.shape[:2]:
        raise ValueError("oracle_requires_inputs[m,topk,k]_and_weights[m,topk]")
    acc = np.zeros((a.shape[0], a.shape[2]), dtype=np.float32)
    for n in range(a.shape[1]):
        acc = acc + a[:, n, :] * b[:, n, None]
    return to_bf16(acc)


def assert_service_equal(actual, expected, *, rel=1e-12):
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=rel, abs=1e-12)


def test_source_identity_matches_the_campaign_manifest():
    manifest = json.loads(MANIFEST.read_text())
    task = manifest
    sources = {item["path"]: item for item in task["sources"]}
    for path, digest, size in (
        (mc.SOURCE_PATH, mc.SOURCE_SHA256, mc.SOURCE_BYTES),
        (mc.CONTRACT_PATH, mc.CONTRACT_SHA256, mc.CONTRACT_BYTES),
    ):
        entry = sources[path]
        assert entry["sha256"] == digest
        assert entry["bytes"] == size
        source = Path(path)
        if source.exists():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert dict(mc.make_op(params()).metadata).get("source_origin") == (
        "current_checkout_not_historical_capture_image"
    )


def test_campaign_cases_preserve_splits_and_expected_dispatch():
    cases = campaign_cases()
    assert [case["split"] for case in cases].count("development") == 3
    assert [case["split"] for case in cases].count("heldout_geometry") == 2
    for case in cases:
        op = mc.make_op(dict(case["params"]))
        selected = dict(op.metadata)["dispatch"]
        assert {key: selected[key] for key in case["expected_dispatch"]} == (
            case["expected_dispatch"]
        )
        assert selected["kernel"] == mc.KERNEL
        assert selected["wrapper"] == "moe_fused_mul_sum"
        assert selected["source_kernel_launches"] == 1
        assert selected["accumulation_dtype"] == "fp32"
        assert selected["compute_capability"] == mc.COMPUTE_CAPABILITY
        assert op.recipe == mc.RECIPE
        assert op.link_bytes == 0


@pytest.mark.parametrize("case", campaign_cases(), ids=lambda case: case["id"])
@pytest.mark.parametrize("split_copy", [False, True])
def test_campaign_cases_execute_the_declared_work(case, split_copy):
    op = mc.make_op(dict(case["params"]))
    geometry = Geometry(32, {"vector": 64}, {})
    plan = mc.build(op, geometry)
    report = verify_operator(
        hardware_case(split_copy=split_copy), op, mc.build,
    )
    assert report["status"] == "passed"
    assert report["finite_completion"] and report["buffers_released"]
    assert report["work_mismatches"] == {}
    row = report["result"]
    declared = row["declared_service"]
    assert_service_equal(row["serviced"], declared)
    assert declared["memory@r0:s0"] == plan.work.resources["hbm"]
    assert declared["vector@r0:s0"] == plan.work.resources["vector"]
    if split_copy:
        assert declared["mte2@r0:s0"] == plan.work.resources["mte2"]
        assert declared["mte3@r0:s0"] == plan.work.resources["mte3"]
    assert not any("link@" in key for key in declared)
    assert row["operator"]["hbm_bytes"] == plan.work.resources["hbm"]
    assert row["operator"]["vector_ops"] == plan.work.resources["vector"]
    assert row["operator"]["flops"] == 0
    assert row["operator"].get("matrix_flops", {}) == {}
    assert row["peak_buffers"] == {}


@pytest.mark.parametrize("case", campaign_cases(), ids=lambda case: case["id"])
def test_campaign_work_matches_independent_cta_enumeration(case):
    op = mc.make_op(dict(case["params"]))
    selected = dict(op.metadata)["dispatch"]
    input_bytes = 4 if case["params"]["dtype"] == "fp32" else 2
    weight_bytes = 4 if case["params"]["weight_dtype"] == "fp32" else 2
    expected = reference_work(
        case["params"]["tokens"], case["params"]["hidden"],
        case["params"]["topk"], input_bytes, weight_bytes,
        selected["BLOCK_M"], selected["BLOCK_K"],
    )
    plan, program, totals = program_totals(op)
    assert plan.work.resources == {
        "hbm": expected["hbm"], "mte2": expected["mte2"],
        "mte3": expected["mte3"], "vector": expected["vector"],
    }
    assert sum(stream.count for stream in program.programs) == expected["programs"]
    assert totals == plan.work.resources


@pytest.mark.parametrize(
    "tokens,hidden,expected",
    [
        (1, 128, (1, 256, 4, 4)),    # BLOCK_M <= 4; BLOCK_K floor 256
        (4, 256, (1, 256, 4, 4)),
        (5, 257, (2, 512, 4, 4)),    # first BLOCK_M step; next pow2 <= 512
        (32, 512, (2, 512, 4, 4)),
        (33, 1024, (4, 1024, 16, 2)),  # BLOCK_M step and large-tile stages
        (128, 1024, (4, 1024, 16, 2)),
        (129, 1024, (16, 1024, 16, 2)),  # 16-row tile threshold
        (1024, 1024, (16, 1024, 16, 2)),
        (1025, 1024, (8, 1024, 16, 2)),  # final BLOCK_M branch
        (7, 512, (2, 512, 4, 4)),
        (7, 513, (2, 1024, 8, 4)),
        (7, 1025, (2, 1024, 8, 4)),   # BLOCK_K ceiling, not next pow2 2048
        (7, 2049, (2, 1024, 8, 4)),
    ],
)
def test_tile_dispatch_thresholds_follow_the_sm89_heuristic(tokens, hidden, expected):
    op = mc.make_op(params(tokens=tokens, hidden=hidden))
    selected = dict(op.metadata)["dispatch"]
    assert (
        selected["BLOCK_M"], selected["BLOCK_K"],
        selected["num_warps"], selected["num_stages"],
    ) == expected
    assert selected["is_fp32"] is False
    assert selected["grid"] == (
        math.ceil(hidden / expected[1]), math.ceil(tokens / expected[0]),
    )


@pytest.mark.parametrize(
    "tokens,hidden,expected",
    [
        (1, 128, (1, 256, 8, 4)),
        (5, 100, (2, 256, 8, 4)),
        (200, 64, (4, 256, 16, 4)),
    ],
)
def test_fp32_io_uses_the_source_fp32_tile_branch(tokens, hidden, expected):
    op = mc.make_op(params(tokens=tokens, hidden=hidden, dtype="fp32",
                           weight_dtype="fp32"))
    selected = dict(op.metadata)["dispatch"]
    assert (
        selected["BLOCK_M"], selected["BLOCK_K"],
        selected["num_warps"], selected["num_stages"],
    ) == expected
    assert selected["is_fp32"] is True


@pytest.mark.parametrize(
    "tokens,hidden,topk,dtype,weight_dtype",
    [
        (1, 1, 1, "bf16", "fp32"),
        (5, 130, 3, "bf16", "fp32"),
        (33, 1024, 2, "bf16", "fp32"),
        (17, 513, 4, "bf16", "bf16"),
        (9, 300, 5, "fp16", "fp16"),
        (7, 64, 2, "fp32", "fp32"),
        (7, 257, 3, "fp32", "bf16"),
    ],
)
def test_work_and_program_match_an_independent_tile_enumeration(
    tokens, hidden, topk, dtype, weight_dtype,
):
    op = mc.make_op(params(tokens=tokens, hidden=hidden, topk=topk,
                           dtype=dtype, weight_dtype=weight_dtype))
    selected = dict(op.metadata)["dispatch"]
    input_bytes = 4 if dtype == "fp32" else 2
    weight_bytes = 4 if weight_dtype == "fp32" else 2
    expected = reference_work(
        tokens, hidden, topk, input_bytes, weight_bytes,
        selected["BLOCK_M"], selected["BLOCK_K"],
    )
    plan, program, totals = program_totals(op)
    assert plan.work.resources == {
        "hbm": expected["hbm"], "mte2": expected["mte2"],
        "mte3": expected["mte3"], "vector": expected["vector"],
    }
    assert sum(stream.count for stream in program.programs) == expected["programs"]
    expected_owners = min(expected["programs"], 32)
    assert len(program.programs) == expected_owners
    assert all(isinstance(stream, PipelineProgram)
               for stream in program.programs)
    # Every source CTA appears exactly once across the owner streams.
    indices = [index for index, _ in iter_cta_tiles(program)]
    assert indices == list(range(expected["programs"]))
    assert totals == plan.work.resources


def test_masked_arithmetic_is_full_tile_and_tail_store_is_valid_only():
    op = mc.make_op(params(tokens=5, hidden=130, topk=3))
    plan, program, _ = program_totals(op)
    g = mc._TileGeometry(
        tokens=5, hidden=130, topk=3, block_m=2, block_k=256,
        num_warps=4, num_stages=4, input_bytes=2, weight_bytes=4,
        output_bytes=2, is_fp32=False,
    )
    assert g.m_tiles == 3 and g.k_tiles == 1
    assert g.programs == sum(stream.count for stream in program.programs) == 3
    assert g.vector_ops_per_program == 512 * (1 + 3 * (2 + 1) + 1) == 5632
    assert plan.work.resources["vector"] == 3 * 5632
    valid_arithmetic = 5 * 3 * 130 * 2
    assert plan.work.resources["vector"] > valid_arithmetic

    loads, stores = [], []
    for index, stages in iter_cta_tiles(program):
        pid_k, pid_m, valid_m, valid_k = g.coordinates(index)
        assert pid_k == 0 and pid_m == index
        assert valid_k == 130
        assert valid_m == (2 if index < 2 else 1)
        loads.append(stages[0].work["hbm"])
        stores.append(stages[2].work["hbm"])
    assert loads == [3 * 2 * 130 * 2 + 3 * 2 * 4,
                     3 * 2 * 130 * 2 + 3 * 2 * 4,
                     3 * 1 * 130 * 2 + 3 * 1 * 4]
    assert stores == [2 * 130 * 2, 2 * 130 * 2, 1 * 130 * 2]
    assert sum(stores) == plan.work.resources["mte3"] == 5 * 130 * 2


def test_topk_weights_are_read_again_by_each_k_tile():
    op = mc.make_op(params(tokens=5, hidden=1100, topk=2))
    plan, program, _ = program_totals(op)
    g = mc._TileGeometry(
        tokens=5, hidden=1100, topk=2, block_m=2, block_k=1024,
        num_warps=8, num_stages=4, input_bytes=2, weight_bytes=4,
        output_bytes=2, is_fp32=False,
    )
    assert g.k_tiles == 2 and g.m_tiles == 3 and g.programs == 6
    assert g.weight_reads_per_k_tile == 5 * 2 == 10
    assert g.weight_read_bytes == 2 * 5 * 2 * 4 == 80
    expected_hbm = 5 * 2 * 1100 * 2 + 80 + 5 * 1100 * 2
    assert expected_hbm == 33080
    assert plan.work.resources["hbm"] == expected_hbm
    # Same pid_m, different pid_k: identical weight bytes, different A tail.
    # Triton's first grid dimension varies fastest.
    by_index = {index: stages for index, stages in iter_cta_tiles(program)}
    first = by_index[0][0].work["hbm"]
    second = by_index[1][0].work["hbm"]
    assert first == 2 * 2 * 1024 * 2 + 2 * 2 * 4
    assert second == 2 * 2 * 76 * 2 + 2 * 2 * 4
    assert first != second
    # The un-repeated lower bound would be only one K column of weights.
    assert plan.work.resources["mte2"] == 5 * 2 * 1100 * 2 + 80


@pytest.mark.parametrize("vector_multiple", [1, 2])
def test_singleton_grid_is_owned_by_one_sm_at_every_device_width(vector_multiple):
    """Counterexample: idle SMs must not serve a grid(1,1) CTA.

    At a fixed 1e9 vector ops/s per SM the source CTA executes 2048 lane
    operations.  It takes 0.002048 ms on one SM and must take the same on a
    32-SM device.  The previous global unpinned pipeline returned ~6.48e-5 ms
    (32x) because all SMs shared the single CTA's vector work.
    """
    op = mc.make_op(params(tokens=1, hidden=128, topk=2))
    assert dict(op.metadata)["programs"] == 1
    assert op.vector_ops == 2048

    one = verify_operator(
        hardware_case(units=1, vector_per_unit=1e9,
                      memory_per_unit=1e18,
                      vector_units=vector_multiple), op, mc.build,
    )
    many = verify_operator(
        hardware_case(units=32, vector_per_unit=1e9,
                      memory_per_unit=1e18,
                      vector_units=32 * vector_multiple), op, mc.build,
    )
    assert one["status"] == many["status"] == "passed"
    for report in (one, many):
        assert report["result"]["duration_ms"] == pytest.approx(
            0.002048, rel=1e-9,
        )
        assert_service_equal(report["result"]["serviced"],
                             report["result"]["declared_service"])
    # Explicitly reject the old 32x idle-SM result.
    assert many["result"]["duration_ms"] != pytest.approx(0.000064, rel=0.01)


def test_64_ctas_use_32_owned_sms_without_inventing_owners():
    """A full 32-CTA wave scales from 1 to 32 SMs by the owner count."""
    op = mc.make_op(params(tokens=1024, hidden=1024, topk=1))
    selected = dict(op.metadata)["dispatch"]
    assert selected["BLOCK_M"] == 16 and selected["BLOCK_K"] == 1024
    programs = dict(op.metadata)["programs"]
    assert programs == 64
    per_cta_vector = dict(op.metadata)["vector_ops_per_program"]
    assert per_cta_vector == 81920

    slow = verify_operator(
        hardware_case(units=1, vector_per_unit=1e9,
                      memory_per_unit=1e18), op, mc.build,
    )["result"]
    fast = verify_operator(
        hardware_case(units=32, vector_per_unit=1e9,
                      memory_per_unit=1e18), op, mc.build,
    )["result"]
    # 64 sequential CTAs on 1 SM versus 2 CTAs on each of 32 SMs.
    per_cta_ms = per_cta_vector / (1e9 / 1000)
    assert slow["duration_ms"] == pytest.approx(64 * per_cta_ms, rel=1e-9)
    assert fast["duration_ms"] == pytest.approx(2 * per_cta_ms, rel=1e-9)
    assert slow["duration_ms"] / fast["duration_ms"] == pytest.approx(32, rel=1e-9)
    assert_service_equal(slow["serviced"], slow["declared_service"])
    assert_service_equal(fast["serviced"], fast["declared_service"])


def test_copy_stages_are_pinned_to_one_owner_per_cta():
    """A single CTA cannot borrow the other 31 split copy engines."""
    op = mc.make_op(params(tokens=1, hidden=128, topk=2))
    plan, program, totals = program_totals(op)
    assert totals == plan.work.resources
    (load, fma, store), = (stages for _, stages in iter_cta_tiles(program))
    assert load.placement == {"mte2": (0,)}
    assert fma.placement == {"vector": (0,)}
    assert store.placement == {"mte3": (0,)}
    # Global memory remains shared: no HBM placement is declared.
    for stage in (load, fma, store):
        assert "hbm" not in stage.placement

    slow = verify_operator(
        hardware_case(units=1, split_copy=True, vector_per_unit=1e15,
                      memory_per_unit=1e18, copy_per_unit=1e9),
        op, mc.build,
    )["result"]
    fast = verify_operator(
        hardware_case(units=32, split_copy=True, vector_per_unit=1e15,
                      memory_per_unit=1e18, copy_per_unit=1e9),
        op, mc.build,
    )["result"]
    load_bytes = 2 * 1 * 128 * 2 + 2 * 1 * 4
    store_bytes = 1 * 128 * 2
    copy_unit_ms = 1e9 / 1000  # bytes/ms for one owned copy unit
    vector_ms = op.vector_ops / (1e15 / 1000)
    expected = (load_bytes / copy_unit_ms + vector_ms
                + store_bytes / copy_unit_ms)
    assert slow["duration_ms"] == pytest.approx(expected, rel=1e-9)
    assert fast["duration_ms"] == pytest.approx(expected, rel=1e-9)
    assert slow["duration_ms"] != pytest.approx(expected / 32, rel=0.01)


def test_every_grid_cta_owns_one_unit_and_covers_the_exact_tail():
    op = mc.make_op(params(tokens=1025, hidden=1024, topk=1))
    plan, program, totals = program_totals(op, copy_roles=False)
    g = mc._TileGeometry(
        tokens=1025, hidden=1024, topk=1, block_m=8, block_k=1024,
        num_warps=16, num_stages=2, input_bytes=2, weight_bytes=4,
        output_bytes=2, is_fp32=False,
    )
    assert (g.m_tiles, g.k_tiles, g.grid) == (129, 1, (1, 129))
    assert len(program.programs) == 32
    counts = [stream.count for stream in program.programs]
    assert counts[0] == 5 and counts[1:] == [4] * 31
    assert sum(counts) == 129
    seen = []
    for core, stream in enumerate(program.programs):
        for local in range(stream.count):
            index = core + local * 32
            seen.append(index)
            stages = stream.tile(local)
            load, fma, store = stages
            # Each CTA is pinned to its owner's single vector unit.
            assert fma.placement == {"vector": (core,)}
            # Unified memory: no copy-engine placement exists.
            assert load.placement == {} and store.placement == {}
            expected_load, expected_compute, expected_store = g.tile_work(index)
            assert load.work["hbm"] == expected_load.resources["hbm"]
            assert fma.work["vector"] == expected_compute.resources["vector"]
            assert store.work["hbm"] == expected_store.resources["hbm"]
            _pid_k, pid_m, valid_m, valid_k = g.coordinates(index)
            assert valid_k == 1024
            assert valid_m == (1 if index == 128 else 8)
    assert sorted(seen) == list(range(129))
    assert totals == {
        role: plan.work.resources[role] for role in ("hbm", "vector")
    }
    assert plan.work.resources["mte3"] == 1025 * 1024 * 2


def test_no_latency_floor_or_hidden_workspace_is_added():
    op = mc.make_op(params(tokens=1, hidden=1, topk=1))
    plan = mc.build(op, Geometry(32, {"vector": 32}, {}))
    # A read 2 B + one FP32 weight 4 B + 256-lane tile (1 + 3 + 1) + output 2 B.
    assert plan.work.resources == {
        "hbm": 2 + 4 + 2, "mte2": 2 + 4, "mte3": 2, "vector": 5 * 256,
    }
    report = verify_operator(
        hardware_case(units=32, vector_per_unit=1e9,
                      memory_per_unit=1e12), op, mc.build,
    )
    assert report["status"] == "passed"
    row = report["result"]
    memory_rate = 32 * 1e12 / 1000  # bytes/ms, shared by all SMs
    expected_ms = 6 / memory_rate + 1280 / (1e9 / 1000) + 2 / memory_rate
    assert row["duration_ms"] == pytest.approx(expected_ms, rel=1e-9)
    assert row["duration_ms"] < 4e-3
    assert row["remaining_buffers"] == {} and row["live_tasks"] == 0
    assert row["peak_buffers"] == {}
    metadata = dict(op.metadata)
    assert metadata["kernel_latency_floor_ms"] == 0.0
    assert metadata["workspace_bytes"] == 0
    assert metadata["allocation_pools"] == ()


def test_operator_cost_is_independent_of_graph_identity():
    op = mc.make_op(params(tokens=17, hidden=130, topk=3))
    recipes = RecipeBook(recipes={mc.RECIPE: PlannedRecipe(mc.build)})
    first = operator_costs(hardware_case(), (op,), recipes=recipes)["operators"][0]
    renamed = replace(op, op_id="different", layer=99, requests=("unseen",))
    second = operator_costs(
        hardware_case(), (renamed,), recipes=recipes,
    )["operators"][0]
    assert first["duration_ms"] == second["duration_ms"]
    assert first["serviced"] == second["serviced"]


def test_program_is_lazy_for_a_huge_token_count():
    tokens, hidden, topk = 10**9, 1024, 2
    op = mc.make_op(params(tokens=tokens, hidden=hidden, topk=topk))
    plan = mc.build(op, Geometry(32, {"vector": 64}, {}))
    program = plan.program(context_for(op))
    # Tokens > 1024 select the source's eight-row BLOCK_M branch.
    g = mc._TileGeometry(
        tokens=tokens, hidden=hidden, topk=topk, block_m=8, block_k=1024,
        num_warps=16, num_stages=2, input_bytes=2, weight_bytes=4,
        output_bytes=2, is_fp32=False,
    )
    assert g.programs == 125_000_000
    assert len(program.programs) == 32
    assert sum(stream.count for stream in program.programs) == g.programs
    assert all(stream.count == 125_000_000 // 32
               for stream in program.programs)
    assert plan.work.resources["vector"] == g.programs * g.vector_ops_per_program
    assert math.isfinite(plan.work.resources["hbm"])


def test_execution_keeps_live_cta_state_bounded_and_conserves_work():
    """Actual execution: at most one resident tile per owner stream."""
    op = mc.make_op(params(tokens=8192, hidden=1024, topk=1))
    geometry = Geometry(32, {"vector": 32}, {})
    plan = mc.build(op, geometry)
    program = plan.program(context_for(op, copy_roles=False))
    total_programs = sum(stream.count for stream in program.programs)
    assert total_programs == 1024

    engine = Engine([
        Resource("memory", 32e9, "bytes", units=32),
        Resource("vector", 32e6, "ops", units=32),
    ])
    context = RecipeContext(
        op=op, work={}, resource_names={"hbm": "memory", "vector": "vector"},
        rates={"memory": 32e9, "vector": 32e6},
        core_count=32, resource_units={"vector": 32},
    )
    ready = engine.submit("ready")
    complete = plan.lower(context).start(engine, "owned", ready)
    engine.advance(until=(complete,))
    summary = engine.summary()
    assert summary["live_tasks"] == 0 and summary["live_programs"] == 0
    # 1024 CTAs must not be submitted up front: only the 32 owner windows.
    assert summary["peak_live_tasks"] < 400
    assert engine.serviced["vector"] == plan.work.resources["vector"]
    assert engine.serviced["memory"] == plan.work.resources["hbm"]
    assert engine.completed_count > total_programs


def test_execution_trace_pins_each_cta_to_its_owner():
    """The running engine sees one owner placement per materialized fma."""
    op = mc.make_op(params(tokens=5, hidden=130, topk=3))
    plan = mc.build(op, Geometry(4, {"vector": 4}, {}))
    context = RecipeContext(
        op=op, work={}, resource_names={"hbm": "memory", "vector": "vector"},
        rates={"memory": 4e9, "vector": 4e6},
        core_count=4, resource_units={"vector": 4},
    )
    engine = Engine([
        Resource("memory", 4e9, "bytes", units=4),
        Resource("vector", 4e6, "ops", units=4),
    ], trace_limit=64)
    ready = engine.submit("ready")
    complete = plan.lower(context).start(engine, "op", ready)
    engine.advance(until=(complete,))
    fma = [entry for entry in engine.trace if entry["task"].endswith(":fma")]
    assert len(fma) == 3
    assert sorted(entry["placement"]["vector"] for entry in fma) == [
        (0,), (1,), (2,),
    ]
    assert engine.serviced["vector"] == pytest.approx(
        plan.work.resources["vector"], rel=1e-12,
    )
    assert engine.summary()["live_tasks"] == 0


def test_incompatible_physical_unit_partition_is_rejected():
    op = mc.make_op(params(tokens=1025, hidden=1024, topk=1))
    plan = mc.build(op, Geometry(32, {"vector": 48}, {}))
    program = plan.program(context_for(op, vector_units=48))
    with pytest.raises(ModelingError, match="ownership_geometry_incompatible"):
        program.programs[0].tile(0)


def test_build_rejects_stale_dispatch_metadata():
    op = mc.make_op(params(tokens=5, hidden=130, topk=3))
    metadata = dict(op.metadata)
    metadata["block_m"] = 99
    stale = replace(op, metadata=tuple(sorted(metadata.items())))
    with pytest.raises(ModelingError, match="metadata_mismatch:block_m"):
        mc.build(stale, Geometry(32, {"vector": 64}, {}))
    metadata = dict(op.metadata)
    dispatch = dict(metadata["dispatch"])
    dispatch["BLOCK_K"] = 1
    metadata["dispatch"] = dispatch
    stale = replace(op, metadata=tuple(sorted(metadata.items())))
    with pytest.raises(ModelingError, match="dispatch_mismatch:BLOCK_K"):
        mc.build(stale, Geometry(32, {"vector": 64}, {}))
    metadata = dict(op.metadata)
    dispatch = dict(metadata["dispatch"])
    dispatch["wrapper"] = "topk_weight_and_reduce"
    metadata["dispatch"] = dispatch
    stale = replace(op, metadata=tuple(sorted(metadata.items())))
    with pytest.raises(ModelingError, match="dispatch_mismatch:wrapper"):
        mc.build(stale, Geometry(32, {"vector": 64}, {}))


def test_numpy_weighted_sum_oracle_keeps_fp32_accumulation():
    rng = np.random.default_rng(20261028)
    inputs = to_bf16(rng.standard_normal((5, 4, 130)).astype(np.float32))
    weights = rng.standard_normal((5, 4)).astype(np.float32)
    actual = weighted_sum_oracle(inputs, weights)
    exact = np.einsum(
        "mtk,mt->mk", inputs.astype(np.float64), weights.astype(np.float64),
    )
    assert np.allclose(actual.astype(np.float64), exact,
                       rtol=2**-7, atol=2**-8)
    relative = np.abs(actual.astype(np.float64) - exact) / np.abs(exact)
    assert relative[np.abs(exact) > 0.1].max() < 2**-7
    assert actual.dtype == np.float32

    # A BF16 accumulator loses low-order contributions that the source keeps
    # until its final downconvert.
    a = to_bf16(np.array([[[1.0]] + [[0.001]] * 7], dtype=np.float32))
    b = np.ones((1, 8), dtype=np.float32)
    fp32_result = weighted_sum_oracle(a, b)
    bf16_acc = np.zeros((1, 1), dtype=np.float32)
    for n in range(8):
        bf16_acc = to_bf16(bf16_acc + a[:, n, :] * b[:, n, None])
    assert fp32_result[0, 0] != bf16_acc[0, 0]
    assert float(fp32_result[0, 0]) == pytest.approx(1.0078125)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"has_expert_map": True}, "expert_map_branch_not_selected"),
        ({"apply_router_weight_on_input": True},
         "apply_router_weight_on_input_not_supported"),
        ({"implementation": "moe_sum"}, "unsupported_implementation"),
        ({"compute_capability": 90}, "compute_capability_89"),
        ({"compute_capability": 80}, "compute_capability_89"),
        ({"compute_capability": 89.0}, "compute_capability_89"),
        ({"dtype": "fp8"}, "unsupported_dtype:dtype"),
        ({"weight_dtype": "int8"}, "unsupported_dtype:weight_dtype"),
        ({"tokens": 0}, "positive_integer:tokens"),
        ({"tokens": True}, "positive_integer:tokens"),
        ({"hidden": -1}, "positive_integer:hidden"),
        ({"topk": 0}, "positive_integer:topk"),
        ({"has_expert_map": 0}, "requires_explicit_boolean"),
        ({"apply_router_weight_on_input": 0}, "requires_explicit_boolean"),
        ({"topk_ids": np.zeros((1, 1), dtype=np.int64)},
         "unknown_parameters:topk_ids"),
    ],
)
def test_unselected_or_invalid_source_paths_are_rejected(overrides, match):
    with pytest.raises(ModelingError, match=match):
        mc.make_op(params(**overrides))


def test_missing_and_non_mapping_parameters_are_rejected():
    with pytest.raises(ModelingError, match="params_must_be_a_mapping"):
        mc.make_op([])
    missing = params()
    del missing["weight_dtype"]
    with pytest.raises(ModelingError, match="missing_parameters:weight_dtype"):
        mc.make_op(missing)


def test_tile_address_oracle_covers_inputs_once_and_repeats_weights_per_tile():
    tokens, hidden, topk, block_m, block_k = 5, 130, 3, 2, 256
    m_tiles = math.ceil(tokens / block_m)
    k_tiles = math.ceil(hidden / block_k)
    input_addresses, output_addresses = [], []
    weight_addresses = []
    for pid_m in range(m_tiles):
        valid_m = min(block_m, tokens - pid_m * block_m)
        for pid_k in range(k_tiles):
            valid_k = min(block_k, hidden - pid_k * block_k)
            for lane_m in range(valid_m):
                row = pid_m * block_m + lane_m
                for lane_k in range(valid_k):
                    column = pid_k * block_k + lane_k
                    input_addresses.append((row, column))
                    output_addresses.append((row, column))
                for expert in range(topk):
                    weight_addresses.append((row, expert))
    assert len(input_addresses) == len(set(input_addresses)) == tokens * hidden
    assert len(output_addresses) == len(set(output_addresses)) == tokens * hidden
    # One K tile here; every weight pair is read exactly once.
    assert len(weight_addresses) == len(set(weight_addresses)) == tokens * topk

    # With two K tiles, the same (token, expert) weight address is read twice.
    hidden = 300
    k_tiles = math.ceil(hidden / block_k)
    repeated = []
    for pid_m in range(m_tiles):
        valid_m = min(block_m, tokens - pid_m * block_m)
        for _pid_k in range(k_tiles):
            for lane_m in range(valid_m):
                row = pid_m * block_m + lane_m
                for expert in range(topk):
                    repeated.append((row, expert))
    assert len(repeated) == 2 * tokens * topk
    assert len(set(repeated)) == tokens * topk
    assert len(repeated) == 2 * len(set(repeated))
