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
from mindie_sim.prediction.implementations import preserve_graph
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators.grouped_moe import grouped_gemm


_BASES = {"declared_integer_counts", "forced_balanced_integer_counts",
          "explicit_token_routing_integer"}


def recipes():
    return {"grouped_expert_gemm": PlannedRecipe(grouped_gemm.build)}


def _validated_rows(value, name):
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ModelingError(f"grouped_gemm_{name}_must_be_sequence")
    rows = tuple(value)
    if not rows or any(type(row) is not int or row < 0 for row in rows):
        raise ModelingError(f"grouped_gemm_{name}_must_be_nonempty_nonnegative_integers")
    return rows




def native_grouped_row_contract(package):
    """Consume a NATIVE-MATRIX grouped package's actual rows and regions.

    The leaf owns tensor views and ABI decoding.  This function only keeps the
    physical row separation it already established: interface/received rows,
    capacity rows, and the valid rows of the per-expert regions.  Capacity is
    never substituted for issued rows.  The caller still supplies the selected
    matrix program; this function does not price or align row totals.
    """
    if not isinstance(package, dict):
        raise ModelingError("native_grouped_package_must_be_mapping")
    geometry = package.get("geometry")
    regions = package.get("regions")
    if not isinstance(geometry, dict) or not isinstance(regions, (list, tuple)):
        raise ModelingError("native_grouped_package_requires_geometry_and_regions")
    rows = []
    issued_macs = 0
    for region in regions:
        if not isinstance(region, dict):
            raise ModelingError("native_grouped_region_must_be_mapping")
        count = region.get("rows")
        if type(count) is not int or count < 0:
            raise ModelingError("native_grouped_region_rows_must_be_nonnegative_integer")
        macs = region.get("issued_macs", region.get("macs", 0))
        if type(macs) is not int or macs < 0:
            raise ModelingError("native_grouped_region_issued_macs_invalid")
        rows.append(count)
        issued_macs += macs
    valid = int(geometry.get("valid_rows", sum(rows)))
    interface = int(geometry.get("interface_rows", geometry.get("received_rows", valid)))
    capacity = int(geometry.get("capacity_rows", interface))
    padding = int(geometry.get("padding_rows", 0))
    slack = int(geometry.get("capacity_slack", max(0, capacity - interface)))
    if sum(rows) != valid:
        raise ModelingError(
            "native_grouped_regions_must_sum_to_valid_rows:"
            f"{sum(rows)}!={valid}")
    if interface < valid or capacity < interface or padding < 0 or slack < 0:
        raise ModelingError("native_grouped_row_extents_inconsistent")
    cube = (package.get("work") or {}).get("cube") or {}
    declared_issued = cube.get("issued_macs")
    if declared_issued is not None and int(declared_issued) != issued_macs:
        raise ModelingError(
            "native_grouped_region_issued_macs_must_match_work:"
            f"{issued_macs}!={declared_issued}")
    return {
        "group_rows": tuple(rows), "valid_rows": valid,
        "interface_rows": interface, "received_rows": interface,
        "capacity_rows": capacity, "padding_rows": padding,
        "capacity_slack": slack, "issued_macs": issued_macs,
        "active_regions": tuple(index for index, count in enumerate(rows) if count),
        "nnz_rows": sum(rows),
        "regions": tuple(dict(region) for region in regions),
    }

def _explicit_bank(realization, rows, n, k):
    """Validate an explicit destination bank and its global ledger provenance.

    ``local_expert_rows`` is bank-slot ordered for the selected destination
    rank; ``representative_expert_ids`` carries the logical expert ids in the
    same order; ``global_expert_rows`` is the complete logical-expert ledger.
    The local destination rows are not the global row slice, so they are kept
    separate rather than substituted into matrix rows.  A rank whose assigned
    bank slots are not 0..bank-1 is rejected explicitly.
    """
    local_rows = _validated_rows(realization.get("local_expert_rows"), "local_expert_rows")
    local_ids = tuple(realization.get("local_weight_expert_ids", range(len(local_rows))))
    logical_ids = _validated_rows(
        realization.get("representative_expert_ids"), "representative_expert_ids")
    global_rows = _validated_rows(realization.get("global_expert_rows"), "global_expert_rows")
    rank = realization.get("representative_rank")
    placement = realization.get("expert_placement")
    bank = len(local_rows)
    if len(local_ids) != bank or local_ids != tuple(range(bank)):
        raise ModelingError(
            "grouped_gemm_explicit_local_weight_ids_must_be_contiguous_bank_slots")
    if len(logical_ids) != bank:
        raise ModelingError(
            "grouped_gemm_explicit_representative_ids_must_match_local_bank")
    if type(rank) is not int or rank < 0:
        raise ModelingError("grouped_gemm_explicit_requires_valid_representative_rank")
    if isinstance(placement, str) and placement == "contiguous_ep":
        if bank < 1 or len(global_rows) % bank:
            raise ModelingError(
                "grouped_gemm_explicit_requires_aligned_global_expert_ledger")
        ep = len(global_rows) // bank
        if not 0 <= rank < ep:
            raise ModelingError(
                "grouped_gemm_explicit_representative_rank_outside_ep")
        expected_ids = tuple(range(rank * bank, (rank + 1) * bank))
        if logical_ids != expected_ids:
            raise ModelingError(
                "grouped_gemm_explicit_ids_must_match_physical_contiguous_bank")
        global_slice = global_rows[rank * bank:(rank + 1) * bank]
    elif isinstance(placement, (list, tuple)):
        if len(placement) != len(global_rows):
            raise ModelingError(
                "grouped_gemm_explicit_expert_placement_must_cover_global_ledger")
        ep = 0
        by_slot = {}
        owners_seen = set()
        for expert, item in enumerate(placement):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ModelingError(
                    "grouped_gemm_explicit_expert_placement_requires_rank_bank")
            owner, slot = item
            if type(owner) is not int or type(slot) is not int or owner < 0 or slot < 0:
                raise ModelingError(
                    "grouped_gemm_explicit_expert_placement_requires_nonnegative_integers")
            ep = max(ep, owner + 1)
            owners_seen.add(owner)
            if owner == rank:
                if slot >= bank:
                    raise ModelingError(
                        "grouped_gemm_explicit_local_bank_slot_out_of_range")
                if slot in by_slot:
                    raise ModelingError(
                        "grouped_gemm_explicit_duplicate_local_bank_slot")
                by_slot[slot] = int(expert)
        if owners_seen != set(range(ep)):
            raise ModelingError(
                "grouped_gemm_explicit_physical_ranks_must_be_contiguous")
        if not 0 <= rank < ep:
            raise ModelingError(
                "grouped_gemm_explicit_representative_rank_outside_ep")
        if sorted(by_slot) != list(range(bank)):
            raise ModelingError(
                "grouped_gemm_explicit_requires_contiguous_local_bank")
        expected_ids = tuple(by_slot[slot] for slot in range(bank))
        if logical_ids != expected_ids:
            raise ModelingError(
                "grouped_gemm_explicit_ids_must_follow_expert_placement")
        global_slice = tuple(global_rows[expert] for expert in logical_ids)
    else:
        raise ModelingError(
            "grouped_gemm_explicit_requires_contiguous_ep_or_expert_placement")
    if sum(local_rows) != rows:
        raise ModelingError(
            "grouped_gemm_explicit_local_bank_must_conserve_semantic_rows")
    if n < 1 or k < 1:
        raise ModelingError("grouped_gemm_explicit_requires_positive_n_k")
    return local_rows, tuple(range(bank)), logical_ids, global_rows, global_slice


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
    if basis == "explicit_token_routing_integer":
        counts, group_ids, physical_ids, global_rows, global_slice = _explicit_bank(
            realization, rows, n, k)
        if not counts or sum(counts) == 0:
            raise ModelingError("grouped_gemm_explicit_bank_has_no_valid_rows")
        provenance = "explicit_destination_bank"
    elif placement == "contiguous_ep":
        counts, group_ids, physical_ids, global_rows = _local_bank(
            realization, rows, n, k)
        if not counts or sum(counts) == 0:
            raise ModelingError("grouped_gemm_local_bank_has_no_valid_rows")
        rank = realization.get("representative_rank")
        global_slice = tuple(global_rows[rank * len(counts):(rank + 1) * len(counts)])
        provenance = "contiguous_local_bank"
    elif placement is None:
        counts = _validated_rows(realization.get("expert_rows"), "expert_rows")
        group_ids = tuple(realization.get("expert_ids", range(len(counts))))
        physical_ids = group_ids
        global_rows = counts
        global_slice = counts
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
    routing_extras = tuple(
        (f"routing_{key}", realization[key]) for key in
        ("valid_rows", "received_rows", "interface_rows", "issued_rows",
         "capacity_rows", "capacity_slack", "padding_rows")
        if key in realization)
    source = replace(source, metadata=source.metadata + (
        ("routing_basis", basis),
        ("local_expert_rows", counts),
        ("local_weight_expert_ids", group_ids),
        ("representative_expert_ids", physical_ids),
        ("global_expert_rows", global_rows),
        ("global_expert_slice", global_slice),
        ("routing_rows_basis", provenance),
        ("routing_representative_rank", realization.get("representative_rank")),
        ("routing_expert_placement", placement),
    ) + routing_extras)
    return preserve_graph(source, op)
