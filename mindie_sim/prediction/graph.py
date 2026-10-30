"""Graph capture as a software dispatch policy.

[Role] Select declared capture sizes and mark contiguous captured regions.
[Invariants] Capture changes padding and submissions, not model semantics.
  Measured timing rows never determine whether a capture exists.
[Timeouts & Locks] Linear traversal of one active operator graph.
[Side-Effect Policy] Returns new records, no compiler or device interaction.
"""
from __future__ import annotations

from dataclasses import replace


class GraphPolicy:
    def select(self, views, runtime):
        tokens = sum(view.q_len for view in views)
        mode = runtime.graph_mode.lower()
        if mode in {"none", "off", "eager", "disabled"} or runtime.deepep_high_throughput:
            return None, "none"
        decode = all(view.phase == "decode" for view in views)
        uniform = len({view.q_len for view in views}) == 1
        full = runtime.graph_capture_mode == "full" and decode and uniform and not runtime.invalid_full_modes
        sizes = (runtime.graph_capture_keys or runtime.graph_buckets) if full else runtime.piecewise_capture_keys
        scope = "full" if full else "piecewise"
        if not full and not (decode or runtime.mixed_mode not in {"none", "off"}):
            return None, "none"
        bucket = next((size for size in sorted(sizes) if size >= tokens), None)
        return (bucket, scope) if bucket is not None else (None, "none")

    def apply(self, ops, scope, runtime):
        if scope == "none":
            return tuple(ops)
        split = set(runtime.splitting_ops) if runtime.splitting_ops is not None else {
            "attention", "attention_qk", "attention_pv", "attention_softmax", "attention_merge",
        }
        segment, last_stage, last_captured = 0, None, False
        result = []
        for op in ops:
            captured = scope == "full" or (op.family not in split and op.recipe not in split)
            if op.stage != last_stage or captured != last_captured:
                segment += 1
            result.append(replace(op, metadata=op.metadata + (
                ("graph_captured", captured), ("graph_segment", segment),
            )))
            last_stage, last_captured = op.stage, captured
        return tuple(result)
