"""Explicit native fusion resources on the existing semantic DAG and recipes.

[Role] Fold selected complete groups into one submitted multi-resource kernel.
[Invariants] Child resource programs retain dependencies and compute precision.
  Only a selected, source-verified traffic profile removes HBM transfers.
  Unknown multi-launch stages cannot become one complete kernel.
[Timeouts & Locks] Bounded by one active graph and its selected phases; no I/O.
[Side-Effect Policy] Returns records/programs; no hardware contexts or captures.
"""
from collections import defaultdict
from typing import Literal, Mapping
from dataclasses import replace
import heapq

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from mindie_sim.prediction.fusion_regions import resolve_subgraph_groups
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from .native_fusion_boundary import (
    _HC_ACLNN,
    fusion_typed_context as _fusion_typed_context,
    hc_native_call as _hc_native_call,
    w4a8_metadata as _w4a8_metadata,
)
from .native_fusion_traffic import apply_traffic_profile
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage, TransferRecipe

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCE_REFS: tuple[SourceRef, ...] = (
    SourceRef(
        "vllm-project/vllm-ascend",
        "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        "csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp",
        "29bd7606b274d6a9289ce02b6e6cc03ff947c3a163850263cfea622b53b5d26c",
    ),
    SourceRef(
        "vllm-project/vllm-ascend",
        "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        "csrc/moe/hc_post/op_kernel/hc_post.cpp",
        "c735e1e4534bc2d701fbd06636e42082567c81ea3f0ed24e4e10b9a71867eda4",
    ),
    SourceRef(
        "cann/ops-transformer",
        "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        "attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp",
        "d3c82f11d11de218b21687efa80ab20e64a9038d06da8e42061b7ec02a66f0cb",
    ),
)


RECIPE = "declared_native_fusion"

_MXFP4_DECODE_GROUP_ROWS_LIMIT = 34
_MXFP4_PACKED_GROUP_BYTES = 256

_PRODUCER_FIELDS = {
    "activation_scale_producer": "activation_scales",
    "input_scale_producer": "activation_scales",
    "activation_payload_producer": None,
    "payload_producer": None,
    "topk_ids_producer": "topk_ids",
    "topk_weights_producer": "topk_weights",
    "inverse_index_producer": "inverse_index",
}


def recipes():
    from mindie_sim.prediction.operators.mx_quant_matrix.family_situ_mx import RECIPE as SITU_RECIPE, build as build_situ
    from mindie_sim.prediction.operators.mx_quant_matrix.w4a8 import RECIPE as W4A8_RECIPE, build as build_w4a8
    from mindie_sim.prediction.operators.mx_quant_matrix.mxfp8_swiglu import RECIPE as MXFP8_RECIPE, build as build_mxfp8_swiglu
    return {RECIPE: NativeFusionRecipe(build), W4A8_RECIPE: PlannedRecipe(build_w4a8),
            MXFP8_RECIPE: PlannedRecipe(build_mxfp8_swiglu),
            SITU_RECIPE: PlannedRecipe(build_situ)}


class NativeFusionRecipe(PlannedRecipe):
    def prepare(self, op, geometry):
        prepared, plan = super().prepare(op, geometry)
        from .native_fusion_mla_membase import PROFILE, describe, geometry as source_geometry
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


def _require_profile_arch(op, geometry) -> None:
    from .family_hc import ALLOWED_ARCHS as HC_ARCHS, POST_PROFILE, PRE_PROFILE
    from .native_fusion_mla_membase import ALLOWED_ARCHS as MLA_ARCHS, PROFILE as MLA_PROFILE

    metadata = dict(op.metadata)
    profile = metadata.get("fusion_boundary", {}).get("traffic_profile")
    allowed = {
        PRE_PROFILE: HC_ARCHS,
        POST_PROFILE: HC_ARCHS,
        MLA_PROFILE: MLA_ARCHS,
    }.get(profile)
    if allowed is None:
        return
    dp_arch = getattr(getattr(geometry, "datapath", None), "compute_arch", None)
    compute_arch = dp_arch or getattr(geometry, "compute_arch", None)
    arch_id = str(
        getattr(compute_arch, "arch_id", "")
        or getattr(compute_arch, "preset", "")
        or ""
    ).strip().lower()
    if arch_id and arch_id not in allowed:
        raise ModelingError(f"{profile}_unsupported_arch:{arch_id}")


def build(op, geometry):
    from mindie_sim.prediction.mechanism import PlannedRecipe
    from .family_hc import selected_plan
    from .native_fusion_mla_membase import selected_plan as selected_mla_membase

    _require_profile_arch(op, geometry)

    from .native_fusion_hc_regbase import selected_plan as selected_hc_regbase
    selected = selected_hc_regbase(op, geometry)
    if selected is not None:
        return selected
    selected = selected_plan(op, geometry)
    if selected is not None:
        return selected
    selected = selected_mla_membase(op, geometry)
    if selected is not None:
        return selected
    metadata = dict(op.metadata)
    book = RecipeBook.from_mapping(metadata["fusion_recipe_configuration"])
    if any(dict(record.get("metadata", {})).get("mxfp4_gate_config")
           for record in metadata["fusion_phases"]):
        from mindie_sim.prediction.operators.mx_quant_matrix.family_gate_up_situ_mxfp4 import RECIPE as GATE_RECIPE, build as build_gate
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


def _groups(ops):
    yield from resolve_subgraph_groups(ops)


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


def _lower_program_profile(
    declaration, members, ids, external, outputs, selection, roles,
    profile, program_profile, program_options, configuration, values_by_id,
):
    from mindie_sim.prediction.implementations import add_input_ports
    from mindie_sim.prediction.operators.mx_quant_matrix.family_situ_mx import PROFILE as SITU_PROFILE
    from mindie_sim.prediction.operators.mx_quant_matrix.mxfp8_swiglu import PROFILE as MXFP8_PROFILE

    # A physical program profile owns the complete group's work. The
    # member recipes and traffic-profile savings are not applied again.
    if profile != "preserve_workspaces":
        raise ModelingError("native_fusion_program_profile_conflicts_with_traffic_profile")
    anchor = members[-1]
    op_id = declaration["key"] + ".native"
    inputs = tuple(dict.fromkeys(
        dep for member in members for dep in member.deps if dep not in ids))
    if program_profile == MXFP8_PROFILE:
        from mindie_sim.prediction.operators.mx_quant_matrix.family_mxfp8_swiglu import bind, RECIPE as fused_recipe
        metadata = bind(declaration, members, outputs, selection,
                        program_options.get(declaration['role'], {}))
        total = Work()
        coverage = 'declared_mxfp8_gate_up_complete_source_program'
    elif program_profile == SITU_PROFILE:
        from mindie_sim.prediction.operators.mx_quant_matrix import family_situ_mx
        config = family_situ_mx.config_from_group(declaration, members, outputs)
        metadata = family_situ_mx.fusion_metadata(
            config, declaration, members, outputs, selection)
        # Work, ownership, buffers and stages are bound from the actual
        # device geometry by the selected source recipe in prepare().
        total = Work()
        fused_recipe = family_situ_mx.RECIPE
        coverage = "declared_situ_mx_quant_complete_source_program"
    else:
        from mindie_sim.prediction.operators.mx_quant_matrix import w4a8
        config = w4a8.config_from_group(declaration, members)
        if set(outputs) != {members[-1].op_id}:
            raise ModelingError("w4a8_msd_intermediate_external_outputs_require_materialization")
        total = w4a8.work_from_config(config)
        metadata = _w4a8_metadata(
            declaration, members, outputs, configuration, selection,
            w4a8.describe(config),
            external_output_names=tuple(value["port"] for value in external["values"]),
        )
        fused_recipe = w4a8.RECIPE
        coverage = "declared_w4a8_msd_program"
    context = _fusion_typed_context(
        declaration, members, external, members, total, selection, roles,
        op_id, values_by_id)
    from mindie_sim.prediction.implementations import append_audit
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
        deps=inputs, metadata=tuple(metadata.items()), flops=0.0, matrix_flops=(),
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
    return fused, op_id


def _lower_traffic_profile(
    declaration, members, ids, external, selection, roles, profile,
    configuration, book, values_by_id,
):
    from mindie_sim.prediction.implementations import add_input_ports

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
    from .native_fusion_mla_membase import PROFILE as MLA_ARCH22_PROFILE
    source_work = None
    if profile == MLA_ARCH22_PROFILE:
        from .native_fusion_mla_membase import boundary_metadata
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
    from mindie_sim.prediction.implementations import append_audit
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
        flops=float(sum(total.matrix.values())),
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
    return fused, op_id


def _check_required_roles(roles, matched_roles, original, groups, local_tokens, *, pp=1):
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
                  if not any(d["role"] == role and d.get("native_fusion_eligible", True) is not False
                             for d, _ in groups)}
        required_roles = required_roles - absent
    if (pp > 1 and "mla_prolog" in required_roles
            and "mla_prolog" not in matched_roles
            and not any(op.recipe in {"q_a", "kv_a", "q_b", "q_absorb", "latent_attention0"}
                        for op in original)):
        # Hybrid PP stages (e.g. KDA-only stages) own zero MLA layers while
        # peer stages own and fuse MLA prolog normally.
        required_roles = required_roles - {"mla_prolog"}
    if ("mla_prolog" in required_roles
            and "mla_prolog" not in matched_roles
            and original and all(op.phase == "prefill" for op in original)
            and any(d["role"] == "mla_prolog" for d, _ in groups)
            and all(d.get("native_fusion_eligible") is False
                    and d.get("fusion_reason") == "unabsorbed_mla_prefill"
                    for d, _ in groups if d["role"] == "mla_prolog")):
        required_roles = required_roles - {"mla_prolog"}
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


def lower_native_fusions(ops, runtime, *, local_tokens=None, public_outputs=()):
    """Apply the one optional resource projection after normal semantic lowering."""
    from mindie_sim.prediction.implementations import (
        consumer_index, external_value_plan, produced_values, select,
    )

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
    from mindie_sim.prediction.operators.mx_quant_matrix.family_situ_mx import PROFILE as SITU_PROFILE
    from mindie_sim.prediction.operators.mx_quant_matrix.family_gate_up_situ_mxfp4 import PROFILE as GATE_PROFILE
    from mindie_sim.prediction.operators.mx_quant_matrix.w4a8 import PROFILE as W4A8_PROFILE
    from mindie_sim.prediction.operators.mx_quant_matrix.mxfp8_swiglu import PROFILE as MXFP8_PROFILE
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
    from mindie_sim.prediction.program.implementation_audit import _audit_outputs
    groups = list(_groups(original))
    group_member_ids = {op.op_id for _, members in groups for op in members}
    values_by_id = {
        op.op_id: (
            produced_values(op)
            if (
                op.op_id in group_member_ids
                or bool(_audit_outputs(op))
                or bool(op.dtype or dict(op.metadata).get("output_format") or dict(op.metadata).get("output_dtype"))
            )
            else ()
        )
        for op in original
    }
    aliases, replacements, matched_roles = {}, {}, set()
    port_map, primary_ports, value_port_map = {}, {}, {}
    consumer_map = consumer_index(original)
    for declaration, members in groups:
        if declaration["role"] not in roles or declaration.get("native_fusion_eligible", True) is False:
            continue
        view = declaration.get("tensor_view", {})
        gate_members = [op for op in members
                        if dict(op.metadata).get("program_kind") == "matmul"]
        gate_rows = tuple(dict(gate_members[0].metadata).get("matrix_group_rows", ())) if len(gate_members) == 1 else ()
        if (declaration["role"] == "grouped_gate_up_situ_quant"
                and str(view.get("weight_dtype", "")) == "mxfp4"
                and members and str(members[0].phase) == "decode"
                and gate_rows and max(gate_rows) < _MXFP4_DECODE_GROUP_ROWS_LIMIT):
            from mindie_sim.prediction.operators.mx_quant_matrix import mxfp4_supply
            from mindie_sim.prediction.operators.mx_quant_matrix import family_gate_up_situ_mxfp4
            from mindie_sim.prediction.mechanism import PlannedRecipe
            config = mxfp4_supply.config_from_group(declaration, members)
            gate = gate_members[0]
            from mindie_sim.prediction.operators.grouped_moe.gmmsq_quant_typed import work as gmmsq_work
            packed_groups = (sum(config.group_rows) * 2 * config.output_width
                             * config.k // 2) // _MXFP4_PACKED_GROUP_BYTES
            typed_work, _, _ = gmmsq_work(packed_groups)
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
        outputs = list(external["member_ids"])
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
            fused, op_id = _lower_program_profile(
                declaration, members, ids, external, outputs, selection, roles,
                profile, program_profile, program_options, configuration, values_by_id,
            )
        else:
            fused, op_id = _lower_traffic_profile(
                declaration, members, ids, external, selection, roles, profile,
                configuration, book, values_by_id,
            )
        aliases.update({key: op_id for key in ids})
        port_map.update({(value["member"], value["output"]):
                         (op_id, value["port"]) for value in external["values"]})
        value_port_map.update({(value["member"], value["output"]): value
                               for value in external["values"]})
        matched_roles.add(declaration["role"])
        replacements[members[0].op_id] = fused
    pp_stages = max(1, int(getattr(getattr(runtime, "parallel", None), "pp", 1) or 1))
    _check_required_roles(roles, matched_roles, original, groups, local_tokens, pp=pp_stages)
    from mindie_sim.prediction.implementations import rewire_value_graph
    port_map3 = {
        key: (target[0], target[1], value_port_map.get(key, {}))
        for key, target in port_map.items()
    }
    result = rewire_value_graph(
        original, replacements=replacements, aliases=aliases,
        removed=set(aliases), port_map=port_map3, values_by_id=values_by_id)
    return _ordered(result)
