"""Typed boundary, HC native-call metadata and W4A8 descriptors for native fusions.

[Role]
  Construct typed boundary records, ``aclnnHcPre``/``aclnnHcPost`` BIND1 call
  descriptors, and ``w4a8_msd`` metadata dictionaries for ``lower_native_fusions``.

[Invariants]
  * Every native fusion boundary carries explicit ``replaced``, ``replacement``,
    ``member_occurrences`` and typed operand/output dispositions.
  * Source/dispatch provenance never fabricates deployed binary proof.

[Timeouts & Locks]
  Bounded by the members of one declared fusion group; no locks or I/O.

[Side-Effect Policy]
  Pure dictionary synthesis; no device state, file I/O or global mutation.
"""
from __future__ import annotations

_HC_ACLNN = {"hc_pre": ("aclnnHcPre", "1369"), "hc_post": ("aclnnHcPost", "1285")}


def hc_native_call(declaration, traffic, view):
    """BIND1 producer declaration for one selected HC native fusion."""
    role = declaration["role"]
    entry, line = _HC_ACLNN[role]
    dispatch = declaration.get("source") or {}
    kernel = traffic.get("source") or {}
    if not isinstance(kernel, dict) or not kernel.get("path"):
        return None
    shapes = {
        key: [int(view[key])]
        for key in ("tokens", "hidden", "hc", "mixed")
        if isinstance(view.get(key), int) and view[key] > 0
    }
    arch35_pre = role == "hc_pre" and str(kernel.get("sha256", "")).startswith("284443c5")
    attrs = {
        "dispatch": f"torch_binding.cpp@1bf7a547:{line} EXEC_NPU_CMD({entry})",
        "dispatch_path": "csrc/moe/torch_binding.cpp",
        "dispatch_revision": "1bf7a547",
        "source_execution_domain": kernel.get("source_domain") or "unknown",
        "parameter_dtype": view.get("parameter_dtype") or "fp32",
    }
    return {
        "api": {
            "namespace": "aclnn",
            "entry": entry,
            "aclnn_symbol": entry,
            "version": "vllm-ascend@175ef4d4",
        },
        "source": {
            "repository": dispatch.get("repository") or "vllm-ascend",
            "path": dispatch.get("path"),
            "sha256": dispatch.get("sha256"),
            "symbol": dispatch.get("symbol"),
            "revision": dispatch.get("base_revision") or dispatch.get("revision") or "175ef4d4",
            "state": dispatch.get("source_state") or "committed",
        },
        "typed": {
            "dtype": view.get("dtype") or "bf16",
            "layout": "ND",
            "shapes": shapes,
            "attrs": attrs,
        },
        "kernels": [
            {
                "name": "HcPre" if role == "hc_pre" else "HcPost",
                "role": "mix" if role == "hc_pre" else "vector",
                "source_path": kernel.get("path"),
                "source_sha256": kernel.get("sha256"),
                "evidence": "source_declared",
            }
        ],
        "execution_core": "mix" if role == "hc_pre" else "aiv",
        "unresolved": [
            "installed_selected_binary_not_established",
            "dispatch_source_sha256_unresolved",
            "hc_optional_B4_tensor_abi_unresolved",
        ],
    }


def _fusion_field_dispositions(declaration, members, view, external_refs):
    from mindie_sim.prediction.implementations import (
        field_disposition,
        produced_values,
        typed_roles,
    )

    dispositions = []
    for role, records in typed_roles(view).items():
        for record in records:
            dispositions.append(
                field_disposition(
                    f"tensor_view.{record['field']}",
                    "preserve",
                    declaration["key"],
                    f"declared {role} representation retained",
                )
            )
    for member in members:
        dispositions.append(
            field_disposition(
                f"{member.op_id}.semantic",
                "preserve",
                member.recipe,
                "semantic member and recipe remain inspectable",
            )
        )
        dispositions.append(
            field_disposition(
                f"{member.op_id}.work",
                "rewrite",
                declaration["key"],
                "member work is replaced by the fused plan",
            )
        )
        for value in produced_values(member):
            pair = (member.op_id, value["output"])
            action = "preserve" if pair in external_refs else "consume"
            dispositions.append(
                field_disposition(
                    f"{member.op_id}.{value['output']}",
                    action,
                    declaration["key"],
                    (
                        "externally visible value is materialized on the fused port"
                        if action == "preserve"
                        else "internal value remains inside the selected fusion plan"
                    ),
                )
            )
    return dispositions


def fusion_typed_context(
    declaration,
    members,
    external_plan,
    adjusted,
    total,
    selection,
    roles,
    replacement,
    values_by_id=None,
):
    """Typed I/O, occurrence replacement and allocation accounting for one fusion."""
    from mindie_sim.prediction.implementations import (
        capture_fusion_overlap,
        merge_typed_roles,
        plan_record,
        produced_values,
        producer_input_values,
        replaced_occurrences,
        typed_roles,
    )

    view = declaration.get("tensor_view") or {}
    external_refs = set(external_plan["references"])
    typed_inputs = []
    for role, records in typed_roles(view).items():
        for record in records:
            typed_inputs.append({
                "role": role,
                **record,
                "source": f"{declaration['key']}.tensor_view",
                "materialized": True,
            })
    member_id_set = {item.op_id for item in members}
    operand_inputs = []
    for member in members:
        for value in producer_input_values(member, values_by_id):
            if value.get("producer") not in member_id_set:
                record = {**value, "semantic_op": member.op_id}
                if record not in operand_inputs:
                    operand_inputs.append(record)
    external_values, internal_values = [], []
    phases = {member.op_id: phase for member, phase in zip(members, adjusted)}
    for value in external_plan["values"]:
        member_id = value["member"]
        phase = phases.get(member_id)
        materialized = value.get("materialized")
        if materialized is None:
            materialized = bool(value.get("write_bytes"))
        external_values.append({
            **value,
            "semantic_op": member_id,
            "materialized": bool(materialized),
            "hbm_materialized": bool(
                materialized and phase is not None and phase.mte3_bytes > 0
            ),
        })
    for member, phase in zip(members, adjusted):
        for value in produced_values(member, materialized=phase.mte3_bytes > 0):
            pair = (member.op_id, value["output"])
            if pair in external_refs:
                continue
            internal_values.append({
                **value,
                "semantic_op": member.op_id,
                "hbm_materialized": phase.mte3_bytes > 0,
            })
    values = {
        "roles": merge_typed_roles(view, *members),
        "inputs": tuple(typed_inputs),
        "operands": tuple(operand_inputs),
        "internal": tuple(internal_values),
        "external_outputs": tuple(external_values),
        "external_ports": tuple(value["port"] for value in external_values),
    }
    dispositions = _fusion_field_dispositions(declaration, members, view, external_refs)
    capture = capture_fusion_overlap(members, kind="native_fusion")
    removed = replaced_occurrences(*members)
    internal_ids = tuple(
        member.op_id
        for member in members
        if member.op_id not in set(external_plan["member_ids"])
    )
    return {
        "replaces_work": True,
        "replaces_allocation": True,
        "occurrences": {
            "kind": "fusion",
            "replaced": removed,
            "replacement": replacement,
            "member_occurrences": tuple(member.op_id for member in members),
        },
        "work": {
            "matrix": dict(total.matrix),
            "resources": dict(total.resources),
            "binding": (
                "fused_plan"
                if total.matrix or total.resources
                else "source_program_in_prepare"
            ),
        },
        "allocations": {
            "replaced": removed,
            "external_materialized": tuple(
                dict.fromkeys(
                    value["port"] for value in external_values if value.get("materialized")
                )
            ),
            "external_logical_exports": tuple(
                dict.fromkeys(value["port"] for value in external_values)
            ),
            "external_aliases": tuple(
                {"port": value["port"], "alias_of": value.get("alias_of", "")}
                for value in external_values
                if value.get("alias_of")
            ),
            "internal_values": internal_ids,
            "external_output_ids": tuple(external_plan["member_ids"]),
            "external_value_count": len(external_values),
        },
        "events": {
            "deps": list(
                dict.fromkeys(
                    dep
                    for member in members
                    for dep in member.deps
                    if dep not in member_id_set
                )
            ),
            "outputs": list(values["external_ports"]),
            "stream": members[-1].stream or "main",
            "capture_segment": capture["capture"]["graph_segment"],
        },
        "capture": capture["capture"],
        "overlap": capture["overlap"],
        "selection": {
            "basis": "selected_software_plan",
            "implementation": selection.name,
            "plan": plan_record(selection),
            "declared_roles": tuple(sorted(roles)),
            "selected_role": declaration["role"],
            "member_count": len(members),
            "external_output_count": len(external_values),
            "external_member_count": len(external_plan["member_ids"]),
        },
        "field_dispositions": tuple(dispositions),
        "typed": values,
        "capture_fusion_overlap_separate": True,
    }


def w4a8_metadata(declaration, members, outputs, configuration, selection, described):
    """Build ``w4a8_msd`` native fusion metadata dictionary."""
    anchor = members[-1]
    producer = dict(members[0].metadata)
    caller_tensors = {
        key: producer[key]
        for key in (
            "activation_payload_producer",
            "activation_scale_producer",
            "native_input_capacity_shape",
            "native_input_scale_capacity_shape",
        )
        if key in producer
    }
    if "native_input_capacity_shape" in producer:
        caller_tensors["native_output_capacity_shape"] = (
            producer["native_input_capacity_shape"][0],
            anchor.local_shape[-1],
        )
    return {
        **caller_tensors,
        "fusion_group_key": declaration["key"],
        "fusion_semantic_ids": [member.op_id for member in members],
        "fusion_semantic_recipes": [
            {
                "op_id": member.op_id,
                "family": member.family,
                "recipe": member.recipe,
                "local_shape": list(member.local_shape),
            }
            for member in members
        ],
        "fusion_external_outputs": outputs,
        "fusion_role": declaration["role"],
        "fusion_tensor_view": declaration["tensor_view"],
        "fusion_program_profile": described["profile"],
        "w4a8_msd": described["config"],
        "w4a8_msd_ledger": described["ledger"],
        "fusion_recipe_configuration": configuration,
        "fusion_boundary": {
            "implementation": selection.name,
            "native": declaration["implementation"],
            "source": declaration.get("source"),
            "program_profile": described["profile"],
            "traffic_profile": None,
            "evidence": {
                "semantic_boundary": declaration.get("source"),
                "program_source": described["config"]["source"],
                "capture_binary_identity": "not_established",
                "routing_basis": "explicit_integer_rows_only",
            },
            "submitted_kernels": 1,
            "semantic_phase_count": len(members),
            "traffic": {
                "profile": None,
                "eliminated_reads": 0.0,
                "eliminated_writes": 0.0,
                "reused_edges": [],
                "source": None,
                "limitations": ["member_recipe_work_and_traffic_profiles_not_applied"],
            },
            "timing_basis": "explicit_source_phase_ledger",
            "fidelity": {
                "kind": "declared_source_phase_ledger",
                "compute_and_workspaces": "w4a8_msd_pre_matrix_post",
                "local_edges": "source_pre_matrix_post_dependencies",
                "native_dispatch_reconstructed": False,
                "pipeline_overlap": False,
                "limitations": described["limitations"],
            },
        },
        **{
            key: value
            for key, value in anchor.metadata
            if key
            in {"structure_block", "structure_role", "graph_captured", "graph_segment"}
        },
    }


__all__ = ["_HC_ACLNN", "fusion_typed_context", "hc_native_call", "w4a8_metadata"]
