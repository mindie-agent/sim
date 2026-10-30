"""Public serving handoff controls with explicit synthetic compute/link rates."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from mindie_sim import predict_pd
from mindie_sim.schema.operators import IterationWork, OpSpec

ROOT = Path(__file__).resolve().parents[2]


def lower(plan, model, runtime, machine):
    tokens = sum(view.q_len for view in plan.requests)
    shape = (tokens,)
    op = OpSpec("work", "gemm", "body", 0, 0,
                tuple(view.request_id for view in plan.requests),
                shape, shape, shape, shape, flops=tokens * 1e6, dtype="bf16")
    return IterationWork((op,), plan, 1, tokens, tokens, None, "body", "synthetic_control")


pool = {
    "machine": {"num_devices": 1, "device": {
        "memory_bytes": 1024 ** 3, "compute_peaks": {"bf16": 1e9},
        "rates": {"memory": 1e9, "vector": 1e9}}},
    "runtime": {"max_num_batched_tokens": 16},
}
raw = {
    "model": {"layers": 2, "hidden_size": 32, "num_heads": 4, "num_kv_heads": 2,
              "head_dim": 8, "intermediate_size": 64, "vocab_size": 128},
    "workload": {"requests": [{"id": "r", "prompt_tokens": 3, "output_tokens": 3}]},
    "pools": {name: deepcopy(pool) for name in ("prefill", "decode")},
    "connector": {"link": {"name": "handoff", "capacity_bytes_per_ms": 128}},
}
rows = []
for mode in ("bandwidth", "framing", "switch_propagation", "switch_capacity", "hbm"):
    for factor in (.5, 1, 2):
        value = deepcopy(raw)
        link = value["connector"]["link"]
        if mode == "bandwidth":
            link["capacity_bytes_per_ms"] *= factor
        elif mode == "framing":
            link.update(transaction_bytes=256, protocol_overhead_bytes=int(64 * factor))
        elif mode == "switch_propagation":
            link.update(switch_hops=2, switch_hop_latency_us=500 * factor)
        elif mode == "switch_capacity":
            link["switch_capacity_bytes_per_ms"] = 32 * factor
        else:
            for selected in value["pools"].values():
                selected["machine"]["device"]["rates"]["memory"] = 32000 * factor
        result = predict_pd(value, lower=lower)
        execution = result["execution"]
        rows.append({"scan": mode, "factor": factor,
                     "finish_ms": result["requests"][0]["finish_ms"],
                     "handoff_bytes": execution["handoff_bytes"],
                     "wire_bytes": execution["connector"]["wire_bytes"],
                     "transactions": execution["connector"]["transactions"],
                     "serviced": execution["serviced"], "live_tasks": execution["live_tasks"]})
        assert execution["handoff_bytes"] == 384
        assert execution["serviced"]["prefill:memory@r0:s0"] == 384
        assert execution["serviced"]["decode:memory@r0:s0"] == 384
        assert execution["live_tasks"] == 0
expected = {
    "bandwidth": [11, 8, 6.5], "framing": [9.5, 10, 11],
    "switch_propagation": [8.5, 9, 10], "switch_capacity": [29, 17, 11],
    "hbm": [29, 17, 11],
}
for scan, values in expected.items():
    assert [row["finish_ms"] for row in rows if row["scan"] == scan] == values
files = ["mindie_sim/prediction/pd_connector.py", "mindie_sim/prediction/disaggregated.py"]
print(json.dumps({
    "boundary": "public predict_pd serving with synthetic three-token prefill and two decode steps",
    "evidence_kind": "structural_and_capacity_controls_not_hardware_calibration",
    "connector_endpoints": {"0": "prefill_pool", "1": "decode_pool"},
    "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files},
    "scans": rows,
}, indent=2))
