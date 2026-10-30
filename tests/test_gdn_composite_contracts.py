"""Independent native-wrapper allocation, view and mixed-request controls."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.call_gdn import make_call
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.schema.workload import RequestView
from test_gdn_integration import native_case


def _call(**changes):
    return make_call("scan_prefill", {
        "batch": 1, "tokens": 1, "k_heads": 1, "v_heads": 1,
        "k_dim": 128, "v_dim": 128, "chunk": 64, **changes,
    })


def _case():
    return json.loads(Path("examples/calls/gdn-historical-device.json").read_text())


def _exports(call):
    return {
        dict(op.metadata)["target_output"].removeprefix("caller:"): op
        for op in call.ops
        if dict(op.metadata).get("materialization_reason") == "caller_owned_output"
    }


def test_singleton_transposes_are_free_but_separate_caller_buffers_are_not():
    call = _call()
    copies = [op for op in call.ops if op.family == "tensor_copy"]
    beta = next(op for op in copies if op.recipe == "beta_fp32")
    exports = _exports(call)
    # Independent allocation arithmetic from ctypes output shapes: two BF16
    # 128-vectors, two FP32 scalars, one BF16 [1,64] A row and one FP32 gate.
    payloads = {"q_hat": 256, "k_hat": 256, "q_rstd": 4, "k_rstd": 4,
                "A": 128, "g_cumsum": 4}
    assert set(exports) == payloads.keys()
    assert len(copies) == 1 + len(payloads)
    assert (beta.mte2_bytes, beta.mte3_bytes) == (2, 4)
    for name, amount in payloads.items():
        copied = exports[name]
        assert copied.mte2_bytes == copied.mte3_bytes == amount
        assert dict(copied.metadata)["source_output"] == name
        assert copied.deps == ("gdn_scan_prefill.gdn_prepare",)
    assert sum(op.hbm_bytes for op in copies) == 1310
    assert not any(dep.startswith("external:") or not dep
                   for op in call.ops for dep in op.deps)


def test_caller_exports_keep_real_rows_and_native_rstd_layout_with_chunk_tails():
    call = _call(batch=2, tokens=65, k_heads=2, v_heads=6)
    exports = _exports(call)
    total, keys, heads = 130, 2, 6
    shapes = {"q_hat": (1, total, keys, 128), "k_hat": (1, total, keys, 128),
              "q_rstd": (1, keys, total), "k_rstd": (1, keys, total),
              "A": (1, heads, total, 64), "g_cumsum": (1, total, heads)}
    amounts = {"q_hat": total * keys * 128 * 2, "k_hat": total * keys * 128 * 2,
               "q_rstd": total * keys * 4, "k_rstd": total * keys * 4,
               "A": total * heads * 64 * 2, "g_cumsum": total * heads * 4}
    for name, copied in exports.items():
        assert copied.local_shape == shapes[name]
        target = dict(copied.metadata)["target_layout"]
        assert target["shape"] == target["storage_shape"] == shapes[name]
        assert copied.mte2_bytes == copied.mte3_bytes == amounts[name]
    by = {op.recipe: op for op in call.ops}
    assert by["q_hat_export"].op_id in exports["q_hat"].deps
    assert by["k_hat_export"].op_id in exports["k_hat"].deps
    assert by["g_cumsum_export"].op_id in exports["g_cumsum"].deps
    assert exports["q_rstd"].deps == exports["k_rstd"].deps == exports["A"].deps
    assert not ({"beta_eff", "h", "o", "final_state"} & exports.keys())


def test_unrequested_beta_eff_has_no_native_store_and_external_inputs_wait():
    call = _call()
    prepare = next(op for op in call.ops if op.family == "gdn_prepare")
    assert dict(prepare.metadata)["output_beta"] is False
    explicit_beta = replace(
        prepare, op_id="explicit_beta",
        metadata=tuple({**dict(prepare.metadata), "output_beta": True}.items()))
    costs = operator_costs(_case(), (prepare, explicit_beta), recipes=call.recipes)["operators"]
    by = {row["operator"]["metadata"]["output_beta"]: row["operator"] for row in costs}
    assert by[True]["mte3_bytes"] - by[False]["mte3_bytes"] == 4
    assert by[True]["hbm_bytes"] - by[False]["hbm_bytes"] == 4
    now = call_costs(_case(), call.ops, recipes=call.recipes)
    delayed = call_costs(_case(), call.ops, recipes=call.recipes,
                         input_ready_ms=7, trace_limit=20000)
    assert delayed["complete_ms"] == pytest.approx(7 + now["duration_ms"])
    assert delayed["duration_ms"] == pytest.approx(now["duration_ms"])
    serviced = [event for event in delayed["events"] if event["demand"]]
    assert serviced and min(event["start_ms"] for event in serviced) >= 7
    assert delayed["execution"]["live_tasks"] == delayed["execution"]["live_programs"] == 0


def test_mixed_serving_native_scope_keeps_prefill_requests_and_exports():
    case = native_case()
    views = (RequestView("d", 1, 33, True, "decode"),
             RequestView("p0", 17, 0, False, "prefill"),
             RequestView("p1", 65, 23, False, "prefill"))
    ops, _ = Frontend().body(case.model, case.runtime, views, mode="mixed")
    native = [op for op in ops if op.family in {"gdn_prepare", "gdn_state", "gdn_output"}]
    assert len(native) == 3
    assert all(op.requests == ("p0", "p1") for op in native)
    assert all(dict(op.metadata)["lengths"] == (17, 65) for op in native)
    copied = next(op for op in ops if op.recipe == "q_hat_caller_copy")
    assert copied.mte3_bytes == 82 * 128 * 2
    # A size-one key-head axis preserves the native buffer's address map;
    # the caller's separate output still receives a real copy.
    assert copied.deps == (native[0].op_id,)
    assert any(op.recipe == "qkv_concatenate" for op in ops)
    assert any(op.recipe == "gather_recurrent_state" for op in ops)
    assert any(op.recipe == "scatter_recurrent_state" for op in ops)


def test_explicit_call_does_not_silently_replace_other_source_branches():
    base = _call()
    selected = _call(scale=128 ** -0.5, chunk_size=64, phase="prefill",
                     a_log=None, dt_bias=None)
    assert base.ops == selected.ops
    for changed in ({"scale": 1.0}, {"chunk_size": 32}, {"phase": "decode"},
                    {"a_log": "device_tensor"}, {"dt_bias": "device_tensor"}):
        with pytest.raises(ValueError, match="argument_not_selected"):
            _call(**changed)
