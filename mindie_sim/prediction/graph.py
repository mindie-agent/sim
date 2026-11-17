"""Graph capture as a software dispatch policy.

[Role] Select declared capture sizes and mark contiguous captured regions.
[Invariants] Capture changes padding and submissions, not model semantics.
  Measured timing rows never determine whether a capture exists.
[Timeouts & Locks] Linear traversal of one active operator graph.
[Side-Effect Policy] Returns new records, no compiler or device interaction.
"""
from __future__ import annotations

from dataclasses import replace
import math

TOLERANCE = 1e-12
SHAPE_TRAILING_DIMS = 3
ROWS_DIM = -3
COLS_DIM = -2


def annotate_geometry(ops, *, active_rows, physical_rows):
    """Capture dimensions describe the call, independently of kernel slicing.

    Full capture may pad the call's local rows to a bucket.  A declared matrix
    epilogue whose vector work still describes the active rows is scaled to the
    physical padded rows here, so the typed epilogue identity and its declared
    work stay consistent.  Integer work for the active call is unchanged.
    """
    annotated = []
    for op in ops:
        metadata = {**dict(op.metadata),
                    "graph_active_query_rows": active_rows,
                    "graph_physical_query_rows": physical_rows}
        epilogue = metadata.get("matrix_epilogue")
        shape = tuple(op.local_shape)
        if (isinstance(epilogue, dict) and len(shape) >= SHAPE_TRAILING_DIMS
                and active_rows and physical_rows
                and int(active_rows) != int(physical_rows)):
            operations = float(epilogue.get("operations_per_output", 0.0)
                               or 0.0)
            rows, cols = int(shape[ROWS_DIM]), int(shape[COLS_DIM])
            if operations > 0 and cols > 0:
                active_work = operations * int(active_rows) * cols
                physical_work = operations * int(physical_rows) * cols
                if (math.isclose(float(op.vector_ops), active_work,
                                 rel_tol=TOLERANCE, abs_tol=TOLERANCE)
                        and not math.isclose(float(op.vector_ops), physical_work,
                                             rel_tol=TOLERANCE,
                                             abs_tol=TOLERANCE)):
                    op = replace(op, vector_ops=physical_work)
        annotated.append(replace(op, metadata=tuple(metadata.items())))
    return tuple(annotated)


class GraphPolicy:
    def select(self, views, runtime):
        tokens = sum(view.q_len for view in views)
        mode = runtime.graph_mode.lower()
        if mode in {"none", "off", "eager", "disabled"} or runtime.deepep_high_throughput:
            return None, "none"
        decode = all(view.phase == "decode" for view in views)
        uniform = len({view.q_len for view in views}) == 1
        gdn_slot_capture = bool(
            isinstance(getattr(runtime, "kernels", None), dict)
            and runtime.kernels.get("lowerings", {}).get("gdn_decode")
        )
        q_unit = (views[0].q_len
                  if gdn_slot_capture and views and uniform
                  and not any(view.padding for view in views)
                  else 1)
        full = runtime.graph_capture_mode == "full" and decode and uniform and not runtime.invalid_full_modes
        if full:
            full_sizes = runtime.graph_capture_keys or runtime.graph_buckets
            bucket = next((size for size in sorted(full_sizes)
                           if size >= tokens and (q_unit <= 0 or size % q_unit == 0)), None)
            if bucket is not None:
                return bucket, "full"
        if not (decode or runtime.mixed_mode not in {"none", "off"}):
            return None, "none"
        bucket = next((size for size in sorted(runtime.piecewise_capture_keys) if size >= tokens), None)
        return (bucket, "piecewise") if bucket is not None else (None, "none")

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
