"""Declared grouped-GEMM source binding for integer expert assignments.

[Role] Connect ordinary MoE gate/up and down calls to the existing source plan.
[Invariants] Expected routing moments never become a fabricated histogram.
  Realized zero experts and global expert identities survive compaction.
  A contiguous local bank prices its local valid rows and local weight slots;
  the original global ledger remains provenance and is never substituted into
  local matrix rows. The selected implementation consumes unquantized BF16 ND
  operands.
[Timeouts & Locks] O(local experts), pure metadata; no locks or subprocesses.
[Side-Effect Policy] Returns operators without measurements or device activity.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from .implementations import preserve_graph
from .mechanism import PlannedRecipe
from .operators import grouped_gemm


_BASES = {"declared_integer_counts", "forced_balanced_integer_counts"}


def recipes():
    return {"grouped_expert_gemm": PlannedRecipe(grouped_gemm.build)}


def _validated_rows(value, name):
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ModelingError(f"grouped_gemm_{name}_must_be_sequence")
    rows = tuple(value)
    if not rows or any(type(row) is not int or row < 0 for row in rows):
        raise ModelingError(f"grouped_gemm_{name}_must_be_nonempty_nonnegative_integers")
    return rows


def _local_bank(realization, rows, n, k):
    """Return local rows, weight-slot ids and physical provenance for one bank."""
    local_rows = _validated_rows(realization.get("local_expert_rows"), "local_expert_rows")
    physical_ids = tuple(realization.get("representative_expert_ids", ()))
    local_ids = tuple(realization.get("local_weight_expert_ids", range(len(local_rows))))
    global_rows = tuple(realization.get("global_expert_rows", ()))
    rank = realization.get("representative_rank")
    ranks = tuple(realization.get("represented_ep_ranks", ()))
    bank = len(local_rows)
    if len(local_ids) != bank or local_ids != tuple(range(bank)):
        raise ModelingError(
            "grouped_gemm_local_weight_ids_must_be_contiguous_bank_slots")
    if len(physical_ids) != bank:
        raise ModelingError(
            "grouped_gemm_representative_expert_ids_must_match_local_bank")
    if type(rank) is not int or not 0 <= rank < len(ranks):
        raise ModelingError("grouped_gemm_contiguous_bank_requires_valid_representative_rank")
    expected_ids = tuple(range(rank * bank, (rank + 1) * bank))
    if physical_ids != expected_ids:
        raise ModelingError(
            "grouped_gemm_representative_expert_ids_must_be_physical_contiguous_bank")
    if not global_rows or len(global_rows) != bank * len(ranks):
        raise ModelingError(
            "grouped_gemm_contiguous_bank_requires_complete_global_expert_ledger")
    if tuple(global_rows[rank * bank:(rank + 1) * bank]) != local_rows:
        raise ModelingError(
            "grouped_gemm_local_rows_must_be_selected_global_bank_slice")
    parent_rows = realization.get("expert_rows")
    if parent_rows is not None and tuple(parent_rows) != global_rows:
        raise ModelingError(
            "grouped_gemm_global_ledger_fields_must_not_disagree")
    if sum(local_rows) != rows or n < 1 or k < 1:
        raise ModelingError("grouped_gemm_local_bank_must_conserve_semantic_rows")
    return local_rows, local_ids, physical_ids, global_rows


def lower_op(op):
    metadata = dict(op.metadata)
    realization = metadata.get("routing_realization")
    basis = metadata.get("routing_basis")
    if basis not in _BASES or not realization:
        raise ModelingError("grouped_gemm_requires_explicit_integer_routing_realization")
    if realization.get("basis") != basis:
        raise ModelingError("grouped_gemm_routing_basis_must_match_realization")
    if (op.dtype != "bf16" or op.layout != "bf16"
            or any(metadata.get(key) not in {"bf16", "bfloat16"} for key in
                   ("weight_format", "activation_format", "output_format"))
            or metadata.get("activation_scale_bytes")):
        raise ModelingError("grouped_gemm_source_requires_unquantized_bf16_nd_operands")
    rows, n, k = op.local_shape
    placement = realization.get("expert_placement")
    if placement == "contiguous_ep":
        counts, group_ids, physical_ids, global_rows = _local_bank(
            realization, rows, n, k)
        if not counts or sum(counts) == 0:
            raise ModelingError("grouped_gemm_local_bank_has_no_valid_rows")
        provenance = "contiguous_local_bank"
    elif placement is None:
        counts = _validated_rows(realization.get("expert_rows"), "expert_rows")
        group_ids = tuple(realization.get("expert_ids", range(len(counts))))
        physical_ids = group_ids
        global_rows = counts
        provenance = "global_ledger"
    else:
        raise ModelingError(
            "grouped_gemm_source_requires_contiguous_or_global_integer_ledger")
    if sum(counts) != rows or op.flops != 2 * rows * n * k:
        raise ModelingError("grouped_gemm_realization_must_match_semantic_rows_and_flops")
    source = grouped_gemm.make_op({
        "expert_rows": counts, "group_ids": group_ids,
        "weight_experts": len(counts), "x_rows": rows, "n": n, "k": k,
        "dtype": "bf16", "phase": op.phase,
    })
    source = replace(source, metadata=source.metadata + (
        ("routing_basis", basis),
        ("local_expert_rows", counts),
        ("local_weight_expert_ids", group_ids),
        ("representative_expert_ids", physical_ids),
        ("global_expert_rows", global_rows),
        ("routing_rows_basis", provenance),
    ))
    return preserve_graph(source, op)
