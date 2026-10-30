"""Declared grouped-GEMM source binding for integer expert assignments.

[Role] Connect ordinary MoE gate/up and down calls to the existing source plan.
[Invariants] Expected routing moments never become a fabricated histogram.
  Realized zero experts and global expert identities survive compaction.
  The selected implementation consumes unquantized BF16 ND operands.
[Timeouts & Locks] O(local experts), pure metadata; no locks or subprocesses.
[Side-Effect Policy] Returns operators without measurements or device activity.
"""
from mindie_sim.schema.common import ModelingError
from .implementations import preserve_graph
from .mechanism import PlannedRecipe
from .operators import grouped_gemm


def recipes():
    return {"grouped_expert_gemm": PlannedRecipe(grouped_gemm.build)}


def lower_op(op):
    metadata = dict(op.metadata)
    realization = metadata.get("routing_realization")
    if metadata.get("routing_basis") != "declared_integer_counts" or not realization:
        raise ModelingError("grouped_gemm_requires_explicit_integer_routing_realization")
    if (op.dtype != "bf16" or op.layout != "bf16"
            or any(metadata.get(key) not in {"bf16", "bfloat16"} for key in
                   ("weight_format", "activation_format", "output_format"))
            or metadata.get("activation_scale_bytes")):
        raise ModelingError("grouped_gemm_source_requires_unquantized_bf16_nd_operands")
    rows, n, k = op.local_shape
    counts = tuple(realization["expert_rows"])
    if sum(counts) != rows or op.flops != 2 * rows * n * k:
        raise ModelingError("grouped_gemm_realization_must_match_semantic_rows_and_flops")
    source = grouped_gemm.make_op({
        "expert_rows": counts, "group_ids": realization["expert_ids"],
        "weight_experts": len(counts), "x_rows": rows, "n": n, "k": k,
        "dtype": "bf16", "phase": op.phase,
    })
    return preserve_graph(source, op)
