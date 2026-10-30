"""Standalone historical GDN scan_prefill composite and semantic binding."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim.observations.gdn_binding import bind_observation
from mindie_sim.observations.historical import call_observations
from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.call_builders import build_call
from mindie_sim.prediction.call_gdn import make_call
from mindie_sim.prediction.graph_analysis import graph_costs


SMALL = {"batch": 2, "tokens": 65, "k_heads": 2, "v_heads": 6,
         "k_dim": 128, "v_dim": 128, "chunk": 64}
CASE = Path("examples/calls/gdn-historical-device.json")
EXAMPLE = Path("examples/calls/gdn-scan-prefill.json")


def _case():
    return json.loads(CASE.read_text())


def _call(params=None):
    return make_call("scan_prefill", {**SMALL, **(params or {})})


def _observed():
    protocol = json.loads(Path("tests/fixtures/observations/gdn-call-protocol.json").read_text())
    return next(row for row in call_observations(protocol)
                if row["identity"]["conditions"]["role"] == "scan_prefill")


def _by_recipe(call):
    return {op.recipe: op for op in call.ops}


def test_composite_includes_native_prepare_state_output_and_export_materializations():
    call = _call()
    by = _by_recipe(call)
    for recipe in ("gdn_prepare_arch35_mix_aic_aiv_1_2", "chunk_fwd_h",
                   "gdn_output_arch35_token_first"):
        assert by[recipe].source_boundary == "kernel"
    assert not any(name in by for name in ("qkv_concatenate", "gather_recurrent_state",
                                           "scatter_recurrent_state", "core_output_copy"))
    assert by["gdn_prepare_arch35_mix_aic_aiv_1_2"].semantic_shape == (1, 6, 130, 128)
    assert by["gdn_output_arch35_token_first"].semantic_shape == (1, 130, 6, 128)
    assert by["q_hat_export"].semantic_shape == (1, 130, 2, 128)
    assert by["k_hat_export"].semantic_shape == (1, 130, 2, 128)
    assert by["g_cumsum_export"].semantic_shape == (1, 130, 6)
    assert by["chunk_fwd_h"].deps == (by["gdn_prepare_arch35_mix_aic_aiv_1_2"].op_id,)
    copies = {name: by[name] for name in ("q_head_first", "k_head_first", "v_head_first",
                                          "g_head_first", "beta_fp32", "beta_head_first")}
    for name in ("q_head_first", "k_head_first", "v_head_first", "g_head_first"):
        metadata = dict(copies[name].metadata)
        assert metadata["source_stream"] == "current"
    assert dict(copies["q_head_first"].metadata)["source_input"] == "q"
    assert dict(copies["k_head_first"].metadata)["source_input"] == "k"
    assert dict(copies["v_head_first"].metadata)["source_input"] == "v"
    assert dict(copies["g_head_first"].metadata)["source_input"] == "g"
    assert dict(copies["beta_fp32"].metadata)["source_input"] == "beta"
    assert copies["beta_fp32"].dtype == copies["beta_head_first"].dtype == "fp32"
    assert copies["g_head_first"].dtype == "fp32"
    assert call.ledger["state"]["initial"]["dtype"] == "fp32"
    assert call.ledger["state"]["final"]["dtype"] == "fp32"
    assert call.ledger["exports"]["q_hat"]["shape"] == [1, 130, 2, 128]
    assert call.ledger["exports"]["g_cumsum"]["dtype"] == "fp32"


def test_full_and_tail_chunks_and_qk_v_head_grouping_keep_actual_extents():
    full = _call({"batch": 1, "tokens": 64, "k_heads": 2, "v_heads": 4})
    tail = _call({"batch": 1, "tokens": 65, "k_heads": 2, "v_heads": 4})
    full_by, tail_by = _by_recipe(full), _by_recipe(tail)
    assert full_by["gdn_prepare_arch35_mix_aic_aiv_1_2"].padded_shape == (1, 4, 64, 128)
    assert tail_by["gdn_prepare_arch35_mix_aic_aiv_1_2"].semantic_shape == (1, 4, 65, 128)
    assert tail_by["gdn_prepare_arch35_mix_aic_aiv_1_2"].padded_shape == (1, 4, 128, 128)
    assert tail_by["q_head_first"].semantic_shape == (1, 2, 65, 128)
    assert tail_by["k_head_first"].semantic_shape == (1, 2, 65, 128)
    assert tail_by["v_head_first"].semantic_shape == (1, 4, 65, 128)
    assert tail_by["q_hat_export"].semantic_shape == (1, 65, 2, 128)
    assert tail_by["gdn_output_arch35_token_first"].semantic_shape == (1, 65, 4, 128)
    with pytest.raises(Exception, match="head_grouping"):
        _call({"k_heads": 3, "v_heads": 4})


def test_public_build_call_reaches_native_work_without_serving_state_movement():
    declaration = json.loads(EXAMPLE.read_text())
    declaration["arguments"].update(SMALL)
    call = build_call(declaration)
    assert declaration["arguments"]["role"] == "scan_prefill"
    case = _case()
    report = call_costs(case, call.ops, recipes=call.recipes)
    assert report["boundary"] == "device_graph"
    assert report["single_call"]["families"] == {
        "tensor_copy": 15, "gdn_prepare": 1, "gdn_state": 1, "gdn_output": 1}
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert report["parameters"]["calibration"] == "none"
    graph = graph_costs(case, call.ops, recipes=call.recipes)
    assert graph["operator_occurrences"] == len(call.ops) == 18
    rows = graph["isolated"]["operators"]
    assert all(row["declared_service"] == pytest.approx(row["serviced"]) for row in rows)
    assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0


def test_binding_requires_producer_and_all_relevant_call_arguments():
    observed = _observed()
    bound = bind_observation(observed)
    assert bound["status"] == "bound" and bound["mismatches"] == []
    assert bound["call"].ledger["boundary"] == "device_work_of_complete_call"
    changed = deepcopy(observed)
    changed["samples"] = [{"trial": 0, "repeat": 0, "duration_ms": 1e12}]
    changed["statistics"] = {"sample_median_ms": 1e12}
    changed["canary"] = {"ok": True}
    changed["model_geometry_label"] = "different-label"
    same = bind_observation(changed)
    assert same["status"] == "bound"
    assert same["call"].ops == bound["call"].ops
    assert same["call"].ledger == bound["call"].ledger
    for mutate in ("producer", "contract", "condition", "implementation"):
        candidate = deepcopy(observed)
        if mutate == "producer":
            candidate["identity"]["producer_sha256"] = "modified"
        elif mutate == "contract":
            candidate["identity"]["call_contract"]["chunk_size"] = 32
        elif mutate == "condition":
            candidate["identity"]["conditions"]["state_bytes"] += 4
        else:
            candidate["identity"]["implementation"] = "other.wrapper"
        assert bind_observation(candidate)["status"] == "semantic_mismatch"
    # Explicit construction never reads a capture.
    explicit = _call()
    assert explicit.ops and explicit.ledger["source"]["members"]


def _retained_conditions():
    for heads in (48, 32):
        for batch in (1, 4):
            for tokens in (4096, 8192, 16384, 32768):
                yield {"batch": batch, "tokens": tokens, "k_heads": 16,
                       "v_heads": heads, "k_dim": 128, "v_dim": 128,
                       "chunk": 64, "chunks": (batch * tokens + 63) // 64,
                       "state_bytes": batch * heads * 128 * 128 * 4}


def test_all_sixteen_retained_geometries_bind_and_reach_prepare_h_o():
    base = _observed()
    for condition in _retained_conditions():
        observed = deepcopy(base)
        observed["identity"]["conditions"] = {
            "role": "scan_prefill", "phase": "prefill",
            "impl": "fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd",
            **condition}
        bound = bind_observation(observed)
        assert bound["status"] == "bound", bound["mismatches"]
        by = {op.recipe for op in bound["call"].ops}
        assert {"gdn_prepare_arch35_mix_aic_aiv_1_2", "chunk_fwd_h",
                "gdn_output_arch35_token_first", "q_hat_export"} <= by
