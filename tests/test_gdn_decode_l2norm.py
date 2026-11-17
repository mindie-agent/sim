"""Canonical decode GDN exposes source q/k L2-normalization work.

[Role]
  Drive the ordinary decode frontend and the native prepare plan; assert that
  q/k normalization work is present exactly when the decode path selects it.

[Invariants]
  * One q and one k decode L2-normalization op exist for decode only, each
    with five vector operations per normalized element.
  * The native prepare plan includes the register-level normalization work
    only when ``use_qk_l2norm_in_kernel`` is true.
  * Chunked prefill and non-GDN attention never emit the decode normalization
    ops.

[Side-Effect Policy]
  In-memory plans and private Engine instances only.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.gdn_decode_programs import GATING
from mindie_sim.prediction.gdn_recurrent import build as gdn_build
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators import gdn_prepare
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView


def _case(attention="gdn"):
    parsed = parse_case({
        "machine": {
            "name": "declared-gpu", "architecture": "declared", "num_devices": 1,
            "device": {
                "memory_bytes": 1024 ** 3,
                "compute_peaks": {"bf16": 1e9, "fp32": 1e9},
                "rates": {"memory": 1e9, "vector": 1e9},
                "compute_units": 4,
            },
            "links": [{"name": "fabric", "capacity_bytes_per_ms": 1000,
                       "members": [0], "semantics": "shared"}],
        },
        "model": {
            "name": "geometry", "layers": 1, "hidden_size": 32,
            "num_heads": 4, "num_kv_heads": 2, "head_dim": 8,
            "intermediate_size": 64, "vocab_size": 128,
        },
        "workload": {"requests": [
            {"id": "r", "prompt_tokens": 4, "output_tokens": 3}]},
        "runtime": {"dp": 1, "tp": 1, "pp": 1, "ep": 1,
                    "max_num_batched_tokens": 16},
    })
    layer = replace(
        parsed.model.layer_plans[0], attention=attention,
        linear_key_heads=4, linear_value_heads=12, linear_heads=12,
        linear_key_head_dim=128, linear_head_dim=128, conv_kernel=4,
        recurrent_state_dtype="fp32")
    kernels = {"lowerings": {"gdn_decode": "vllm_ascend_gdn_decode"}}
    rates = {"hbm": 4e12, "vector": 1e13, "mte2": 4e12,
             "mte2_aic": 4e12, "mte2_aiv": 4e12, "mte3": 4e12}
    device = replace(parsed.machine.dies[0], rates=tuple(rates.items()),
                     compute_units=32,
                     resource_units=tuple((r, 64) for r in rates))
    return replace(parsed, model=replace(parsed.model, layer_plans=(layer,)),
                   runtime=replace(parsed.runtime, pp_layer_partition=(1,),
                                   kernels=kernels),
                   machine=replace(parsed.machine, dies=(device,)))


def _decode_ops(parsed, *, mode="decode", lengths=(4,)):
    views = tuple(RequestView(str(i), q, 4096, False, mode)
                  for i, q in enumerate(lengths))
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode=mode)
    return ops


def test_decode_emits_one_q_and_one_k_l2norm():
    parsed = _case()
    ops = _decode_ops(parsed)
    names = [op.op_id for op in ops if "decode_" in op.op_id and "l2norm" in op.op_id]
    assert len(names) == 2
    assert any("decode_q_l2norm" in name for name in names)
    assert any("decode_k_l2norm" in name for name in names)


def test_chunked_prefill_does_not_emit_decode_l2norm():
    parsed = _case()
    ops = _decode_ops(parsed, mode="prefill")
    assert not any("decode_q_l2norm" in op.op_id or "decode_k_l2norm" in op.op_id
                   for op in ops)


def test_non_gdn_attention_has_no_decode_l2norm():
    parsed = _case(attention="gqa")
    ops = _decode_ops(parsed)
    assert not any("decode_q_l2norm" in op.op_id or "decode_k_l2norm" in op.op_id
                   for op in ops)


def _prepare_plan(*, use_qk_l2norm):
    op = gdn_prepare.make_op({
        "lengths": (4,) * 4, "heads": 12, "key_heads": 4,
        "dk": 128, "dv": 128, "chunk": 64, "dtype": "bf16",
        "gate_dtype": "fp32", "use_exp2": True,
        "use_qk_l2norm_in_kernel": use_qk_l2norm,
        "use_gate_in_kernel": False, "use_beta_sigmoid_in_kernel": False,
        "output_a": True, "output_beta": False,
    })
    geometry = Geometry(4, {"vector": 8, "hbm": 8, "mte2": 8, "mte3": 8,
                            "cube": 8, "fixpipe": 8, "mte1": 8},
                        {}, {})
    return gdn_prepare.build(op, geometry)


def test_native_prepare_normalization_toggles_register_work():
    normalized = _prepare_plan(use_qk_l2norm=True)
    plain = _prepare_plan(use_qk_l2norm=False)
    normalized_work = dict(normalized.work.resources)
    plain_work = dict(plain.work.resources)
    assert normalized_work.get("vector", 0) > plain_work.get("vector", 0)
    assert sum(normalized_work.values()) > sum(plain_work.values())
    assert any(normalized_work.get(role, 0) > plain_work.get(role, 0)
               for role in normalized_work)


def test_decode_gating_selected_for_gdn_layer():
    parsed = _case()
    ops = _decode_ops(parsed)
    assert sum(op.recipe == GATING for op in ops) == 1
    recurrent = [op for op in ops if op.family == "recurrent_vector"]
    assert len(recurrent) == 1
    plan = gdn_build(recurrent[0], Geometry.from_devices(parsed.machine.dies))
    assert dict(plan.work.resources)


def test_decode_l2norm_ops_execute_on_shared_engine():
    parsed = _case()
    ops = _decode_ops(parsed)
    target = next(op for op in ops if "decode_q_l2norm" in op.op_id)
    target = replace(target, deps=("ready",))
    execution = Execution(parsed, CostModel(parsed), include_submission=False,
                          trace_limit=1000)
    ready = execution.engine.submit("ready", release=2)
    complete = execution.add(
        IterationWork((target,), IterationPlan(()), 1, 0, 0, None, "body", "test"),
        0, 0, external_inputs={"ready": ready}).completion
    execution.engine.advance(until=(complete,))
    assert not execution.engine._tasks and not execution.engine._promises
    assert execution.engine.serviced["vector@r0:s0"] == pytest.approx(
        4 * 4 * 128 * 5)
