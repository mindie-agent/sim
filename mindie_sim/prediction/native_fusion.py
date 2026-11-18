"""Explicit native fusion resources on the existing semantic DAG and recipes.

[Role] Fold selected complete groups into one submitted multi-resource kernel.
[Invariants] Child resource programs retain dependencies and compute precision.
  Only a selected, source-verified traffic profile removes HBM transfers.
  Unknown multi-launch stages cannot become one complete kernel.
[Timeouts & Locks] Bounded by one active graph and its selected phases; no I/O.
[Side-Effect Policy] Returns records/programs; no hardware contexts or captures.
"""
from collections import defaultdict
from typing import Mapping
from dataclasses import replace
import heapq

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

from .mechanism import OperatorPlan, PlannedRecipe, Work
from .native_fusion_traffic import apply_traffic_profile
from .program import KernelProgram, RecipeBook, Stage, TransferRecipe


RECIPE = "declared_native_fusion"

_PRODUCER_FIELDS = {
    "activation_scale_producer": "activation_scales",
    "input_scale_producer": "activation_scales",
    "activation_payload_producer": None,
    "payload_producer": None,
    "topk_ids_producer": "topk_ids",
    "topk_weights_producer": "topk_weights",
    "inverse_index_producer": "inverse_index",
}

_HC_ACLNN = {"hc_pre": ("aclnnHcPre", "1369"), "hc_post": ("aclnnHcPost", "1285")}


def _hc_native_call(declaration, traffic, view):
    """BIND1 producer declaration for one selected HC native fusion.

    Source, dispatch and typed arguments are declared independently.  The
    installed selected binary remains explicit: this is source/dispatch pairing,
    not proof of the deployed image's chosen kernel.
    """
    role = declaration["role"]
    entry, line = _HC_ACLNN[role]
    dispatch = declaration.get("source") or {}
    kernel = traffic.get("source") or {}
    if not isinstance(kernel, dict) or not kernel.get("path"):
        return None
    shapes = {
        key: [int(view[key])] for key in ("tokens", "hidden", "hc", "mixed")
        if isinstance(view.get(key), int) and view[key] > 0
    }
    arch35_pre = role == "hc_pre" and str(kernel.get("sha256", "")).startswith(
        "284443c5")
    attrs = {
        "dispatch": (f"torch_binding.cpp@1bf7a547:{line} "
                     f"EXEC_NPU_CMD({entry})"),
        "dispatch_path": "csrc/moe/torch_binding.cpp",
        "dispatch_revision": "1bf7a547",
        "source_execution_domain": kernel.get("source_domain") or "unknown",
        "parameter_dtype": view.get("parameter_dtype") or "fp32",
    }
    return {
        "api": {"namespace": "aclnn", "entry": entry, "aclnn_symbol": entry,
                "version": "vllm-ascend@175ef4d4"},
        "source": {
            "repository": dispatch.get("repository") or "vllm-ascend",
            "path": dispatch.get("path"),
            "sha256": dispatch.get("sha256"),
            "symbol": dispatch.get("symbol"),
            "revision": (dispatch.get("base_revision")
                         or dispatch.get("revision") or "175ef4d4"),
            "state": dispatch.get("source_state") or "committed",
        },
        "typed": {"dtype": view.get("dtype") or "bf16", "layout": "ND",
                  "shapes": shapes, "attrs": attrs},
        "kernels": [{
            "name": "HcPre" if role == "hc_pre" else "HcPost",
            "role": "mix" if arch35_pre else "vector",
            "source_path": kernel.get("path"),
            "source_sha256": kernel.get("sha256"),
            "evidence": "source_declared",
        }],
        "execution_core": "mix" if arch35_pre else None,
        "unresolved": [
            "installed_selected_binary_not_established",
            "dispatch_source_sha256_unresolved",
            "hc_optional_B4_tensor_abi_unresolved",
            *(() if arch35_pre else ("execution_core_not_declared",)),
        ],
    }


def recipes():
    from .family_situ_mx import RECIPE as SITU_RECIPE, build as build_situ
    from .w4a8 import RECIPE as W4A8_RECIPE, build as build_w4a8
    from .mxfp8_swiglu import RECIPE as MXFP8_RECIPE, build as build_mxfp8_swiglu
    return {RECIPE: NativeFusionRecipe(build), W4A8_RECIPE: PlannedRecipe(build_w4a8),
            MXFP8_RECIPE: PlannedRecipe(build_mxfp8_swiglu),
            SITU_RECIPE: PlannedRecipe(build_situ)}


class NativeFusionRecipe(PlannedRecipe):
    def prepare(self, op, geometry):
        prepared, plan = super().prepare(op, geometry)
        from .mla_prolog_arch22 import PROFILE, describe, geometry as source_geometry
        metadata = dict(prepared.metadata)
        if metadata.get("fusion_source_profile") == PROFILE:
            view = metadata["fusion_tensor_view"]
            ledger = describe(view, source_geometry(view, geometry))
            boundary = dict(metadata["fusion_boundary"])
            source = dict(boundary["source_program"])
            source["program"] = {
                **source["program"], "work": ledger["total"],
                "work_binding": "physical_device_geometry",
                "execution_ledger": ledger,
            }
            boundary["source_program"] = source
            metadata["fusion_boundary"] = boundary
            prepared = replace(prepared, metadata=tuple(metadata.items()))
        return prepared, plan


def _work(op):
    meta = dict(op.metadata)
    matrix = dict(op.matrix_flops) if op.matrix_flops else (
        {op.dtype: op.flops} if op.flops else {})
    resources = {
        "hbm": op.hbm_bytes, "mte2": op.mte2_bytes, "mte3": op.mte3_bytes,
        "vector": op.vector_ops, "link": op.link_bytes,
    }
    for role, amount in meta.get("resource_work", {}).items():
        resources[role] = resources.get(role, 0) + amount
    for role, field in (("mte1", "mte1_bytes"), ("fixpipe", "fixpipe_bytes")):
        if meta.get(field):
            resources[role] = resources.get(role, 0) + meta[field]
    return Work(matrix, {key: value for key, value in resources.items() if value})


def _phase_record(op, ordinal, indices):
    # Physical phases use local identities. Layer/request identities live in
    # fusion_semantic_ids and do not split otherwise identical cost cohorts.
    meta = {
        key: value for key, value in op.metadata
        if not (key.startswith("structure_") or key.startswith("fusion_")
                or key in {"fused_origin_scopes", "input_conversion_subgraphs", "graph_segment"})}
    return replace(
        op, op_id=str(ordinal), layer=-1, stage=0, requests=(), stream="", notes=(),
        deps=tuple(str(indices[dep]) for dep in op.deps if dep in indices),
        metadata=tuple(meta.items()),
    ).to_dict()


def _phase_op(record):
    value = dict(record)
    for field in ("requests", "deps", "semantic_shape", "local_shape", "padded_shape",
                  "launch_shape", "notes", "returns"):
        if field in value:
            value[field] = tuple(value[field])
    value["metadata"] = tuple(value["metadata"].items())
    if "matrix_flops" in value:
        value["matrix_flops"] = tuple(value["matrix_flops"].items())
    return OpSpec(**value)


def build(op, geometry):
    from .mechanism import PlannedRecipe
    from .family_hc import selected_plan
    from .mla_prolog_arch22 import selected_plan as selected_mla_arch22

    from .family_hc_arch35 import selected_plan as selected_hc_arch35
    selected = selected_hc_arch35(op, geometry)
    if selected is not None:
        return selected
    selected = selected_plan(op, geometry)
    if selected is not None:
        return selected
    selected = selected_mla_arch22(op, geometry)
    if selected is not None:
        return selected
    metadata = dict(op.metadata)
    book = RecipeBook.from_mapping(metadata["fusion_recipe_configuration"])
    if any(dict(record.get("metadata", {})).get("mxfp4_gate_config")
           for record in metadata["fusion_phases"]):
        from .family_gate_up_situ_mxfp4 import RECIPE as GATE_RECIPE, build as build_gate
        book.recipes[GATE_RECIPE] = PlannedRecipe(build_gate)
    prepared = []
    total = Work()
    source_arrays = {}
    for record in metadata["fusion_phases"]:
        phase, recipe = book.prepare(_phase_op(record), geometry)
        work = _work(phase)
        total += work
        plan_meta = getattr(recipe, "metadata", {}) or {}
        if plan_meta.get("physical_cube_flops") is not None:
            source_arrays.update({
                key: value for key, value in plan_meta.items()
                if key in {"logical_valid_flops", "source_api_flops",
                           "physical_cube_flops", "physical_cube_flops_derived",
                           "logical_fixpipe_bytes", "source_api_fixpipe_bytes",
                           "physical_cube_m_by_bank"}
            })
        prepared.append((phase, recipe, work))

    def program(context):
        stages = []
        for phase, recipe, work in prepared:
            phase_meta = dict(phase.metadata)
            bound = context.for_phase(
                work, kernel=phase_meta.get("kernel", phase_meta.get("kernel_name", phase.recipe)),
                dtype=phase.dtype, layout=phase.layout, axes=phase_meta.get("service_axes", {}))
            bound = replace(bound, op=phase, service_identity={
                **bound.service_identity, "implementation": phase_meta.get("implementation", {})})
            stages.append(Stage(phase.op_id, after=phase.deps, program=recipe.lower(bound)))
        return KernelProgram(tuple(stages))

    return OperatorPlan(total, program,
                        {"physical_source_arrays": source_arrays} if source_arrays else {})


def _fusion_typed_context(declaration, members, external_plan, adjusted, total,
                          selection, roles, replacement, values_by_id=None):
    """Typed I/O, occurrence replacement and allocation accounting for one fusion."""
    from .implementations import (capture_fusion_overlap, field_disposition,
                                  merge_typed_roles, plan_record, produced_values,
                                  producer_input_values, replaced_occurrences,
                                  typed_roles)

    view = declaration.get("tensor_view") or {}
    external_refs = set(external_plan["references"])
    typed_inputs = []
    for role, records in typed_roles(view).items():
        for record in records:
            typed_inputs.append({
                "role": role, **record,
                "source": f"{declaration['key']}.tensor_view",
                "materialized": True,
            })
    # Operand representations are those of the actual external producers, not
    # the consuming region's model activation dtype.
    operand_inputs = []
    for member in members:
        for value in producer_input_values(member, values_by_id):
            if value.get("producer") not in {item.op_id for item in members}:
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
            "hbm_materialized": bool(materialized and phase is not None
                                     and phase.mte3_bytes > 0),
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
    dispositions = []
    for role, records in typed_roles(view).items():
        for record in records:
            dispositions.append(field_disposition(
                f"tensor_view.{record['field']}", "preserve",
                declaration["key"], f"declared {role} representation retained"))
    for member in members:
        dispositions.append(field_disposition(
            f"{member.op_id}.semantic", "preserve", member.recipe,
            "semantic member and recipe remain inspectable"))
        dispositions.append(field_disposition(
            f"{member.op_id}.work", "rewrite", declaration["key"],
            "member work is replaced by the fused plan"))
        for value in produced_values(member):
            pair = (member.op_id, value["output"])
            action = "preserve" if pair in external_refs else "consume"
            dispositions.append(field_disposition(
                f"{member.op_id}.{value['output']}", action, declaration["key"],
                ("externally visible value is materialized on the fused port"
                 if action == "preserve"
                 else "internal value remains inside the selected fusion plan")))
    capture = capture_fusion_overlap(members, kind="native_fusion")
    removed = replaced_occurrences(*members)
    internal_ids = tuple(member.op_id for member in members
                         if member.op_id not in set(external_plan["member_ids"]))
    boundary = {
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
            "binding": ("fused_plan" if total.matrix or total.resources
                        else "source_program_in_prepare"),
        },
        "allocations": {
            "replaced": removed,
            "external_materialized": tuple(
                dict.fromkeys(value["port"] for value in external_values
                              if value.get("materialized"))),
            "external_logical_exports": tuple(
                dict.fromkeys(value["port"] for value in external_values)),
            "external_aliases": tuple(
                {"port": value["port"], "alias_of": value.get("alias_of", "")}
                for value in external_values if value.get("alias_of")),
            "internal_values": internal_ids,
            "external_output_ids": tuple(external_plan["member_ids"]),
            "external_value_count": len(external_values),
        },
        "events": {
            "deps": list(dict.fromkeys(dep for member in members for dep in member.deps
                                       if dep not in {item.op_id for item in members})),
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
    return boundary


def _groups(ops):
    declarations, seeds, inputs = {}, defaultdict(set), defaultdict(set)
    by_id = {op.op_id: op for op in ops}
    positions = {op.op_id: index for index, op in enumerate(ops)}
    for op in ops:
        meta = dict(op.metadata)
        for declaration in meta.get("structure_subgraphs", ()):
            key = declaration["key"]
            previous = declarations.get(key)
            if previous is not None and previous != declaration:
                raise ModelingError("native_fusion_group_declaration_conflict:" + key)
            declarations[key] = declaration
        for key in meta.get("structure_subgraph_memberships", ()):
            seeds[key].add(op.op_id)
        for key in meta.get("structure_subgraph_entries", ()):
            inputs[key].update(op.deps)
    for key, declaration in declarations.items():
        members, pending = set(seeds[key]), list(seeds[key])
        while pending:
            current = pending.pop()
            for dependency in by_id[current].deps:
                if dependency in by_id and dependency not in inputs[key] and dependency not in members:
                    members.add(dependency)
                    pending.append(dependency)
        yield declaration, [by_id[key] for key in sorted(members, key=positions.__getitem__)]


def _w4a8_metadata(declaration, members, outputs, configuration, selection, described):
    anchor = members[-1]
    producer = dict(members[0].metadata)
    caller_tensors = {key: producer[key] for key in (
        "activation_payload_producer", "activation_scale_producer",
        "native_input_capacity_shape", "native_input_scale_capacity_shape") if key in producer}
    if "native_input_capacity_shape" in producer:
        caller_tensors["native_output_capacity_shape"] = (
            producer["native_input_capacity_shape"][0], anchor.local_shape[-1])
    metadata = {
        **caller_tensors,
        "fusion_group_key": declaration["key"],
        "fusion_semantic_ids": [member.op_id for member in members],
        "fusion_semantic_recipes": [
            {"op_id": member.op_id, "family": member.family,
             "recipe": member.recipe, "local_shape": list(member.local_shape)}
            for member in members],
        "fusion_external_outputs": outputs,
        "fusion_role": declaration["role"],
        "fusion_tensor_view": declaration["tensor_view"],
        "fusion_program_profile": described["profile"],
        "w4a8_msd": described["config"],
        "w4a8_msd_ledger": described["ledger"],
        "fusion_recipe_configuration": configuration,
        "fusion_boundary": {
            "implementation": selection.name,
            "native": declaration["implementation"], "source": declaration.get("source"),
            "program_profile": described["profile"],
            "traffic_profile": None,
            "evidence": {
                "semantic_boundary": declaration.get("source"),
                "program_source": described["config"]["source"],
                "capture_binary_identity": "not_established",
                "routing_basis": "explicit_integer_rows_only",
            },
            "submitted_kernels": 1, "semantic_phase_count": len(members),
            "traffic": {
                "profile": None, "eliminated_reads": 0.0,
                "eliminated_writes": 0.0, "reused_edges": [],
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
        **{key: value for key, value in anchor.metadata
           if key in {"structure_block", "structure_role", "graph_captured",
                      "graph_segment"}},
    }
    return metadata


def _ordered(ops):
    """A fused producer may move past an independent branch but not a dependency."""
    positions = {op.op_id: index for index, op in enumerate(ops)}
    by_id = {op.op_id: op for op in ops}
    waiting, users = {}, defaultdict(list)
    for op in ops:
        internal = {dep for dep in op.deps if dep in by_id}
        waiting[op.op_id] = len(internal)
        for dep in internal:
            users[dep].append(op.op_id)
    ready = [positions[key] for key, count in waiting.items() if count == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        op = ops[heapq.heappop(ready)]
        result.append(op)
        for consumer in users[op.op_id]:
            waiting[consumer] -= 1
            if waiting[consumer] == 0:
                heapq.heappush(ready, positions[consumer])
    if len(result) != len(ops):
        raise ModelingError("native_fusion_external_dependency_cycle")
    return tuple(result)


def lower_native_fusions(ops, runtime, *, local_tokens=None, public_outputs=()):
    """Apply the one optional resource projection after normal semantic lowering."""
    from .implementations import (add_input_ports, consumer_index,
                                  consumer_value_uses, external_value_plan,
                                  input_ports, produced_values, select,
                                  set_input_ports)

    selection = select(runtime.kernels.get("lowerings", {}), "native_fusion")
    if selection is None:
        return tuple(ops)
    roles = set(selection.options.get("roles", ()))
    if not roles:
        raise ModelingError("native_fusion_requires_explicit_roles")
    profile_selection = selection.options.get("traffic_profile", "preserve_workspaces")
    program_profiles = selection.options.get("program_profiles", {})
    if not isinstance(program_profiles, dict):
        raise ModelingError("native_fusion_program_profiles_must_be_role_mapping")
    if program_profiles.keys() - roles:
        raise ModelingError("native_fusion_program_profiles_require_selected_roles")
    from .family_situ_mx import PROFILE as SITU_PROFILE
    from .family_gate_up_situ_mxfp4 import PROFILE as GATE_PROFILE
    from .w4a8 import PROFILE as W4A8_PROFILE
    from .mxfp8_swiglu import PROFILE as MXFP8_PROFILE
    program_options = selection.options.get('program_options', {})
    if not isinstance(program_options, dict) or program_options.keys() - program_profiles.keys():
        raise ModelingError('native_fusion_program_options_require_selected_program_profiles')
    if any(program_profiles[role] != MXFP8_PROFILE for role in program_options):
        raise ModelingError('native_fusion_program_options_unsupported_profile')
    unknown_profiles = set(program_profiles.values()) - {W4A8_PROFILE, SITU_PROFILE, MXFP8_PROFILE}
    if unknown_profiles:
        raise ModelingError("native_fusion_program_profile_unknown:" +
                            ",".join(sorted(unknown_profiles)))
    configuration = {key: value for key, value in runtime.kernels.items() if key != "lowerings"}
    book = RecipeBook.from_mapping(configuration)
    original = tuple(ops)
    users = defaultdict(set)
    for op in original:
        for dep in op.deps:
            users[dep].add(op.op_id)
    values_by_id = {}
    for op in original:
        try:
            values_by_id[op.op_id] = produced_values(op)
        except ModelingError:
            values_by_id[op.op_id] = ()
    aliases, replacements, matched_roles = {}, {}, set()
    port_map, primary_ports, value_port_map = {}, {}, {}
    consumer_map = consumer_index(original)
    groups = list(_groups(original))
    for declaration, members in groups:
        if declaration["role"] not in roles:
            continue
        view = declaration.get("tensor_view", {})
        gate_members = [op for op in members
                        if dict(op.metadata).get("program_kind") == "matmul"]
        gate_rows = tuple(dict(gate_members[0].metadata).get("matrix_group_rows", ())) if len(gate_members) == 1 else ()
        if (declaration["role"] == "grouped_gate_up_situ_quant"
                and str(view.get("weight_dtype", "")) == "mxfp4"
                and members and str(members[0].phase) == "decode"
                and gate_rows and max(gate_rows) < 34):
            from . import mxfp4_supply
            from . import family_gate_up_situ_mxfp4
            from .mechanism import PlannedRecipe
            config = mxfp4_supply.config_from_group(declaration, members)
            gate = gate_members[0]
            from .gmmsq_quant_typed import work as gmmsq_work
            groups = (sum(config.group_rows) * 2 * config.output_width
                      * config.k // 2) // 256
            typed_work, _, _ = gmmsq_work(groups)
            typed_resources = dict(typed_work.resources)
            gate = replace(
                gate, recipe=family_gate_up_situ_mxfp4.RECIPE,
                flops=0, matrix_flops=(), vector_ops=0,
                hbm_bytes=0, mte2_bytes=0, mte3_bytes=0,
                metadata=gate.metadata + (
                    ("mxfp4_gate_config", {
                        "group_rows": config.group_rows,
                        "output_width": config.output_width,
                        "k": config.k,
                        "allocated_rows": config.allocated_rows,
                    }),
                    ("resource_work", typed_resources),
                ))
            members = tuple(gate if op.op_id == gate.op_id else op
                            for op in members)
            book.recipes[family_gate_up_situ_mxfp4.RECIPE] = PlannedRecipe(
                family_gate_up_situ_mxfp4.build)
        ids = {op.op_id for op in members}
        if ids & aliases.keys():
            raise ModelingError("native_fusion_selected_groups_overlap")
        placements = {(op.stage, op.phase, op.stream) for op in members}
        if len(placements) != 1:
            raise ModelingError("native_fusion_crosses_stage_phase_or_stream")
        capture_scopes = {
            (dict(op.metadata).get("graph_captured"), dict(op.metadata).get("graph_segment"))
            for op in members}
        if len(capture_scopes) != 1:
            raise ModelingError("native_fusion_crosses_graph_capture_boundary")
        if any(op.source_boundary or op.link_bytes for op in members):
            raise ModelingError("native_fusion_source_call_or_collective_requires_own_plan")
        if (declaration.get("coverage") or {}).get("kind") in {
                "multi_launch_stage_unknown", "native_boundary_unverified"}:
            raise ModelingError("native_fusion_requires_complete_single_kernel")
        external = external_value_plan(original, members, values_by_id,
                                       public_outputs=public_outputs,
                                       consumer_map=consumer_map)
        legacy_outputs = [member.op_id for member in members
                          if users[member.op_id] - ids
                          or not users[member.op_id] & ids]
        outputs = legacy_outputs
        for member in members:
            if member.op_id not in external["member_ids"]:
                continue
            primary = (values_by_id.get(member.op_id) or [{}])[0].get("output")
            for value in external["values"]:
                if value["member"] == member.op_id and value["output"] == primary:
                    primary_ports[member.op_id] = value["port"]
        profile = (profile_selection.get(declaration["role"], "preserve_workspaces")
                   if isinstance(profile_selection, dict) else profile_selection)
        program_profile = program_profiles.get(declaration["role"])
        if program_profile is not None:
            # A physical program profile owns the complete group's work. The
            # member recipes and traffic-profile savings are not applied again.
            if profile != "preserve_workspaces":
                raise ModelingError("native_fusion_program_profile_conflicts_with_traffic_profile")
            anchor = members[-1]
            op_id = declaration["key"] + ".native"
            inputs = tuple(dict.fromkeys(
                dep for member in members for dep in member.deps if dep not in ids))
            if program_profile == MXFP8_PROFILE:
                from .family_mxfp8_swiglu import bind, RECIPE as fused_recipe
                metadata = bind(declaration, members, outputs, selection,
                                program_options.get(declaration['role'], {}))
                total = Work()
                coverage = 'declared_mxfp8_gate_up_complete_source_program'
            elif program_profile == SITU_PROFILE:
                from . import family_situ_mx
                config = family_situ_mx.config_from_group(declaration, members, outputs)
                metadata = family_situ_mx.fusion_metadata(
                    config, declaration, members, outputs, selection)
                # Work, ownership, buffers and stages are bound from the actual
                # device geometry by the selected source recipe in prepare().
                total = Work()
                fused_recipe = family_situ_mx.RECIPE
                coverage = "declared_situ_mx_quant_complete_source_program"
            else:
                from . import w4a8
                config = w4a8.config_from_group(declaration, members)
                if set(outputs) != {members[-1].op_id}:
                    raise ModelingError("w4a8_msd_intermediate_external_outputs_require_materialization")
                total = w4a8.work_from_config(config)
                metadata = _w4a8_metadata(
                    declaration, members, outputs, configuration, selection,
                    w4a8.describe(config))
                fused_recipe = w4a8.RECIPE
                coverage = "declared_w4a8_msd_program"
            context = _fusion_typed_context(
                declaration, members, external, members, total, selection, roles,
                op_id, values_by_id)
            from .implementations import append_audit
            append_audit(metadata, kind="native_fusion_boundary", op_id=op_id,
                         boundary=context, layer=anchor.layer, stage=anchor.stage,
                         block=metadata.get("structure_block"),
                         role=metadata.get("structure_role"))
            boundary = dict(metadata.get("fusion_boundary") or {})
            boundary.update({key: context[key] for key in (
                "capture", "overlap", "selection", "work",
                "capture_fusion_overlap_separate")})
            metadata["fusion_boundary"] = boundary
            fused = total.apply(replace(
                anchor, op_id=op_id, family="native_fusion", recipe=fused_recipe,
                deps=inputs, metadata=tuple(metadata.items()), matrix_flops=(),
                source_boundary="kernel", coverage=coverage,
                returns=external["returns"]))
            origin_ports = tuple(
                {"op": value["producer"], "output": value["output"],
                 "source_op": value["producer"]}
                for value in context["typed"]["operands"]
                if value.get("producer"))
            fused = add_input_ports(fused, origin_ports)
            context["selection"].update({
                "actual_recipe": fused.recipe,
                "actual_program_profile": program_profile,
                "actual_phase_count": len(dict(fused.metadata).get("fusion_phases", ())),
                "actual_external_outputs": tuple(
                    value.get("port") for value in context["typed"]["external_outputs"]),
            })
            aliases.update({key: op_id for key in ids})
            port_map.update({(value["member"], value["output"]):
                             (op_id, value["port"]) for value in external["values"]})
            value_port_map.update({(value["member"], value["output"]): value
                                   for value in external["values"]})
            matched_roles.add(declaration["role"])
            replacements[members[0].op_id] = fused
            continue
        adjusted, traffic = apply_traffic_profile(
            members, declaration, profile,
            external_outputs=external["references"])
        for before, after in zip(members, adjusted):
            selected = book.select(after)
            if (before.hbm_bytes != after.hbm_bytes
                    and not isinstance(selected, TransferRecipe)
                    and not getattr(selected, "traffic_rewrite_capable", False)):
                # Shape-generated memory programs can reconstruct transfers
                # independently of OpSpec bytes; subtracting only the envelope
                # would claim a saving that their executed program does not make.
                raise ModelingError(
                    "native_fusion_local_reuse_requires_work_preserving_phase_recipe:" + after.recipe)
        indices = {op.op_id: index for index, op in enumerate(members)}
        anchor = members[-1]
        op_id = declaration["key"] + ".native"
        inputs = tuple(dict.fromkeys(dep for member in members for dep in member.deps if dep not in ids))
        outputs = list(external["member_ids"])
        metadata = {
            "fusion_group_key": declaration["key"],
            "fusion_semantic_ids": [member.op_id for member in members],
            "fusion_external_outputs": outputs,
            "fusion_external_output_names": tuple(
                value["port"] for value in external["values"]),
            "fusion_role": declaration["role"],
            "fusion_tensor_view": declaration["tensor_view"],
            "fusion_phases": [_phase_record(member, index, indices)
                              for index, member in enumerate(adjusted)],
            "fusion_recipe_configuration": configuration,
            "fusion_boundary": {
                "implementation": selection.name,
                "native": declaration["implementation"], "source": declaration.get("source"),
                "traffic_profile": profile,
                "evidence": {
                    "semantic_boundary": declaration.get("source"),
                    "local_reuse": traffic["source"],
                    "capture_binary_identity": "not_established",
                },
                "submitted_kernels": 1, "semantic_phase_count": len(members),
                "traffic": traffic, "timing_basis": "resource_DAG_not_sum_of_kernel_durations",
                "fidelity": {
                    "kind": "declared_resource_phase_dag",
                    "compute_and_workspaces": "existing_member_recipes",
                    "local_edges": "only_explicit_source_profile",
                    "native_dispatch_reconstructed": False,
                },
            },
            **{key: value for key, value in anchor.metadata
               if key in {"structure_block", "structure_role", "graph_captured", "graph_segment"}},
        }
        if declaration["role"] in _HC_ACLNN:
            hc_call = _hc_native_call(
                declaration, traffic, declaration.get("tensor_view") or {})
            if hc_call is not None:
                metadata["native_call"] = hc_call
        from .mla_prolog_arch22 import PROFILE as MLA_ARCH22_PROFILE
        source_work = None
        if profile == MLA_ARCH22_PROFILE:
            from .mla_prolog_arch22 import boundary_metadata
            source = boundary_metadata(declaration, members, outputs)
            metadata["fusion_semantic_recipes"] = source["program"]["fusion_semantic_recipes"]
            metadata["fusion_source_profile"] = MLA_ARCH22_PROFILE
            metadata["fusion_boundary"] = {
                **metadata["fusion_boundary"],
                "source_program": source,
                "timing_basis": source["timing_basis"],
                "fidelity": source["fidelity"],
                "limitations": source["limitations"],
            }
            work = source["program"]["work"]
            source_work = Work(work["matrix"],
                               {"hbm": work["hbm"], "vector": work["vector"]})
        total = source_work if source_work is not None else Work()
        if source_work is None:
            for member in adjusted:
                total += _work(member)
        context = _fusion_typed_context(
            declaration, members, external, adjusted, total, selection, roles,
            op_id, values_by_id)
        from .implementations import append_audit
        append_audit(metadata, kind="native_fusion_boundary", op_id=op_id,
                     boundary=context, layer=anchor.layer, stage=anchor.stage,
                     block=metadata.get("structure_block"),
                     role=metadata.get("structure_role"))
        boundary = dict(metadata.get("fusion_boundary") or {})
        boundary.update({key: context[key] for key in (
            "capture", "overlap", "selection", "work",
            "capture_fusion_overlap_separate")})
        metadata["fusion_boundary"] = boundary
        fused = total.apply(replace(
            anchor, op_id=op_id, family="native_fusion", recipe=RECIPE,
            deps=inputs, metadata=tuple(metadata.items()),
            matrix_flops=tuple(sorted(total.matrix.items())),
            source_boundary="kernel", coverage="declared_native_resource_phases",
            returns=external["returns"]))
        origin_ports = tuple(
            {"op": value["producer"], "output": value["output"],
             "source_op": value["producer"]}
            for value in context["typed"]["operands"]
            if value.get("producer"))
        fused = add_input_ports(fused, origin_ports)
        context["selection"].update({
            "actual_recipe": fused.recipe,
            "actual_program_profile": None,
            "actual_phase_count": len(dict(fused.metadata).get("fusion_phases", ())),
            "actual_external_outputs": tuple(
                value.get("port") for value in context["typed"]["external_outputs"]),
        })
        aliases.update({key: op_id for key in ids})
        port_map.update({(value["member"], value["output"]):
                         (op_id, value["port"]) for value in external["values"]})
        value_port_map.update({(value["member"], value["output"]): value
                               for value in external["values"]})
        matched_roles.add(declaration["role"])
        replacements[members[0].op_id] = fused
    # A receive-only EP rank still executes expert kernels and transport but
    # has no local residual, dense or shared-expert activation stream. The
    # frontend establishes those calls are absent; routed expert fusion stays
    # required because received rows do not depend on local query ownership.
    local_roles = {"hc_pre", "hc_post", "situ_quant"}
    required_roles = roles - local_roles if local_tokens == 0 else roles
    if original and all(
            op.phase in {"draft", "draft_prefill"}
            or dict(op.metadata).get("structure_model_scope") == "draft"
            or op.op_id.startswith("draft.")
            for op in original):
        absent = {role for role in ("hc_pre", "hc_post", "mla_prolog")
                  if not any(d["role"] == role for d, _ in groups)}
        required_roles = required_roles - absent
    realized_rows = None
    for op in original:
        if not any(k in {"routing_realization", "source_token_mask"} for k, _ in op.metadata):
            continue
        metadata = dict(op.metadata)
        record = metadata.get("routing_realization", metadata.get("source_token_mask"))
        if isinstance(record, Mapping) and "rows" in record:
            value = record.get("rows")
            if isinstance(value, int) and not isinstance(value, bool):
                realized_rows = value if realized_rows is None else max(
                    realized_rows, value)
    if (realized_rows == 0
            and not any(op.family == "grouped_gemm" for op in original)):
        # A source-only representative has no local received F3 rows; it still
        # executes transport/count/combine, but no grouped expert matrix.
        required_roles = required_roles - {"grouped_gate_up_situ_quant"}
    if required_roles - matched_roles:
        raise ModelingError("native_fusion_selected_roles_unmatched:" +
                            ",".join(sorted(required_roles - matched_roles)))
    from .implementations import rewire_value_graph
    port_map3 = {
        key: (target[0], target[1], value_port_map.get(key, {}))
        for key, target in port_map.items()
    }
    result = rewire_value_graph(
        original, replacements=replacements, aliases=aliases,
        removed=set(aliases), port_map=port_map3, values_by_id=values_by_id)
    return _ordered(result)
