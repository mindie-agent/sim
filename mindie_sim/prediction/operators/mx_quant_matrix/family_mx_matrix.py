"""Explicit prepared MX32 matrix representation and AIC output boundary.

[Role] Bind selected ordinary projections to prepared E8M0 scales and Fixpipe.
[Invariants] Checkpoint bytes remain separate from runtime residency. Input and
  output permutations are strides, never invented GM copies. One matrix keeps
  one invocation and the existing finite owner/K pipeline; no timing fitting.
[Timeouts & Locks] O(selected projections/layers), no I/O or locks.
[Side-Effect Policy] Creates immutable metadata, weight views and programs only.
"""
from dataclasses import replace
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.values.typed import rebind_op_precision
from .mxfp8_grouped import MX_SCALE_GROUP, MX_SCALE_GROUP_PAIR, NZ_N_GRANULE

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCE_REFS: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py",
        source_sha256="f9d03f16ae93fb678da5ff0135848226b585960dc1abe45252719ce7eb29e65a",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/attention/dsa_v1.py",
        source_sha256="8683fcf83e4b38890035af373e185c001cc874eae055b11554c88e98148edc65",
    ),
    SourceRef(
        repository="cann/ops-nn",
        revision="238f76a0",
        relative_path="samples/matmul/quant_matmul/quant_batch_matmul_mx.py",
        source_sha256="37c9d7f034ae249d2d8fcfe5b310ec6197a6df711736e2ab04ae7e2cdb3ed74e",
    ),
)

IMPLEMENTATION = "ascend_prepared_mx32_matrix"
PROFILE = "prepared_mx32_aic_matrix"
ROLES = {"q_b": "q_b", "grouped_output_a": "wo_a", "grouped_output_b": "wo_b"}
SOURCE = {
    "preparation": {
        "path": "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py",
        "lines": "493-555",
        "sha256": "9f01afd72f299e3617d4e89059840102fd7df93358f82f372daede7a723aaa16",
        "scope": "retained ds_linear expands checkpoint scales and replaces .data; "
                 "the retained hardcoded128 path does not establish captured32 preparation",
    },
    "caller": {"path": "vllm_ascend/attention/dsa_v1.py", "lines": "1464-1487",
               "sha256": "3fb4e94672e6adb931cb2e2a164201000822fe89250fd3bfd3585ba397acf52e",
               "scope": "dynamic MX quant, strided grouped matmul, BF16 output"},
    "matrix": {"path": "samples/matmul/quant_matmul/quant_batch_matmul_mx.py",
               "sha256": "37c9d7f034ae249d2d8fcfe5b310ec6197a6df711736e2ab04ae7e2cdb3ed74e",
               "lines": "1560-1627", "scope": "independent MX scale consumption and Fixpipe GM publication"},
    "identity": "source_semantics_and_explicit_runtime_representation; not_capture_binary_dispatch",
}


def _roles(selection):
    roles = tuple(selection.options.get("projection_recipes", ()))
    if not roles or len(set(roles)) != len(roles) or set(roles) - ROLES.keys():
        raise ModelingError("prepared_mx32_requires_unique_supported_projection_recipes")
    if selection.options.get("scale_storage") != "replace_checkpoint_scales":
        raise ModelingError("prepared_mx32_requires_explicit_scale_storage_replacement")
    return roles


def runtime_quant(quant, category):
    resolved = quant.for_category(category)
    if (resolved.format_for(category) != "fp8_e4m3"
            or resolved.activation_format != "fp8_e4m3"
            or resolved.activation_group_size != MX_SCALE_GROUP
            or resolved.activation_scale_format != "fp8_e8m0"):
        raise ModelingError("prepared_mx32_requires_e4m3_operands_e8m0_activation_scales")
    return replace(resolved, weight_block=(1, MX_SCALE_GROUP), scale_format="fp8_e8m0", scale_bytes=1)


def prepare_model(model, runtime, selection):
    roles = _roles(selection)
    from mindie_sim.catalog.quant import projection_quant
    layers = {}
    for plan in model.layer_plans:
        if plan.attention != "mla":
            continue
        supported = [r for r in roles if (r == "q_b" and plan.q_lora_rank)
                     or (r != "q_b" and plan.o_groups > 0)]
        for role in supported:
            category = {"q_b": "mla.q_b", "grouped_output_a": "mla.o_a",
                        "grouped_output_b": "mla.o_b"}[role]
            runtime_quant(projection_quant(model.quant, plan), category)
        layers[str(plan.index)] = tuple(ROLES[r] for r in supported)
    metadata = {**dict(model.metadata), "prepared_mx32_matrix": {
        "layers": layers, "projection_recipes": roles,
        "scale_storage": "replace_checkpoint_scales", "source": SOURCE,
        "preparation_boundary": "declared_post_load_residency; preparation_latency_outside_body",
    }}
    return replace(model, metadata=tuple(sorted(metadata.items())))


def apply(ops, selection, *, local_tokens=None):
    roles = set(_roles(selection))
    result, seen = [], set()
    for op in ops:
        if op.recipe not in roles:
            result.append(op)
            continue
        seen.add(op.recipe)
        md = dict(op.metadata)
        if (len(op.local_shape) != 3 or md.get("program_kind") != "matmul"
                or op.plan_required or md.get("implementation_binding")
                or md.get("matrix_epilogue") or md.get("bias")):
            raise ModelingError("prepared_mx32_requires_unbound_bias_free_matrix:" + op.op_id)
        m, n, k = op.local_shape
        groups = int(md.get("matrix_groups", 1))
        if k % MX_SCALE_GROUP_PAIR or n % NZ_N_GRANULE or m % groups:
            raise ModelingError("prepared_mx32_requires_K64_N32_integer_groups:" + op.op_id)
        if ((md.get("activation_format"), md.get("weight_format")) != ("fp8_e4m3", "fp8_e4m3")
                or md.get("activation_group_size") != MX_SCALE_GROUP
                or md.get("activation_scale_format") != "fp8_e8m0"
                or md.get("output_format") != "bf16"):
            raise ModelingError("prepared_mx32_requires_e4m3_mx32_bf16_output:" + op.op_id)
        original = md["packed_weight_bytes"]
        payload = groups * n * k if m else 0
        scales = groups * n * (k // MX_SCALE_GROUP_PAIR) * 2 if m else 0
        delta = payload + scales - original
        md.update(semantic_recipe=op.recipe, prepared_weight_original_bytes=original,
                  packed_weight_bytes=payload + scales, weight_bytes=payload + scales,
                  weight_representation="fp8_block1x32_fp8_e8m0",
                  prepared_weight_scale_bytes=scales,
                  prepared_scale_storage="replace_checkpoint_scales",
                  matrix_output_contract="aic_fixpipe_bf16_to_gm",
                  fixpipe_bytes=op.mte3_bytes,
                  implementation_binding={"selector": "mx_matrix", "implementation": IMPLEMENTATION},
                  source=SOURCE,
                  limitations=("generic_finite_mapping_not_captured_host_dispatch",
                               "MX_scale_load_pooled_with_operands_no_instruction_issue_model",
                               "transposed_ND_stride_declared_no_transaction_granule_model"))
        if "native_tensor_view" in md:
            view = dict(md["native_tensor_view"])
            view.update(input_stride=(groups*k, k, 1), output_stride=(groups*n, n, 1),
                        weight_stride=(k*n, n, 1),
                        permutation_scope="native_strided_addressing; no_GM_transpose_copy")
            md["native_tensor_view"] = view
        if "weight_block" in md:
            md["prepared_checkpoint_weight_block"] = md["weight_block"]
        if "weight_scale_format" in md:
            md["prepared_checkpoint_weight_scale_format"] = md["weight_scale_format"]
        if "weight_scale_bytes" in md:
            md["prepared_checkpoint_weight_scale_bytes"] = md["weight_scale_bytes"]
        md["weight_block"] = (1, MX_SCALE_GROUP)
        md["weight_scale_format"] = "fp8_e8m0"
        md["weight_scale_bytes"] = 1
        updated = replace(op, recipe=PROFILE, layout=md["weight_representation"], mte3_bytes=0,
                          hbm_bytes=op.hbm_bytes + delta, mte2_bytes=op.mte2_bytes + delta,
                          metadata=tuple(md.items()))
        if op.precision_abi is not None:
            md.setdefault("prepared_checkpoint_weight_block", op.precision_abi.weight_block)
            md.setdefault("prepared_checkpoint_weight_scale_format", op.precision_abi.weight_scale_dtype)
            md.setdefault("prepared_checkpoint_weight_scale_bytes", op.precision_abi.weight_scale_bytes)
            md["prepared_checkpoint_weight_layout"] = op.precision_abi.weight_layout
            new_abi = replace(
                op.precision_abi,
                weight_layout=md["weight_representation"],
                weight_block=(1, MX_SCALE_GROUP),
                weight_scale_dtype="fp8_e8m0",
                weight_scale_bytes=1,
            )
            updated = rebind_op_precision(updated, new_abi, metadata=md)
        result.append(updated)
    if roles - seen and local_tokens != 0:
        raise ModelingError("prepared_mx32_projection_unmatched:" + ",".join(sorted(roles-seen)))
    return tuple(result)


class FixpipeOutput:
    def lower(self, context, *, core, rows, columns, **kwargs):
        if "fixpipe" not in context.resource_names:
            raise ModelingError("prepared_mx32_fixpipe_rate_required")
        names = context.resource_names
        volume = 2 * rows * columns
        work = {names["fixpipe"]: volume}
        work[names["hbm"]] = work.get(names["hbm"], 0) + volume
        return KernelProgram((Stage("fixpipe_output", work,
            placement=context.owned(core, context.core_count, ("fixpipe",))),))


class MXMatrix:
    def lower(self, context):
        # The common matrix geometry uses the output byte count for tile
        # scoring; this private description does not submit MTE3 work. The
        # callback is the sole publisher of the declared Fixpipe demand.
        meta = dict(context.op.metadata)
        fixpipe = context.resource_names.get("fixpipe")
        if fixpipe is None:
            raise ModelingError("prepared_mx32_fixpipe_rate_required")
        # Reuse the generic output-cost slot for scoring and Stream-K final_store,
        # mapped to the actual physical Fixpipe pipe and its owner count; no AIV
        # MTE3 resource is submitted. SteadyMatmul excludes names["mte3"] from
        # compute on non-Split-K paths so FixpipeOutput remains the sole publisher.
        fixpipe_bytes = float(meta["fixpipe_bytes"])
        inner_work = {k: v for k, v in context.work.items() if k != fixpipe}
        if fixpipe_bytes > 0:
            inner_work[fixpipe] = fixpipe_bytes
        inner = replace(context, op=replace(context.op, mte3_bytes=meta["fixpipe_bytes"]),
                        resource_names={**context.resource_names, "mte3": fixpipe},
                        resource_units={**context.resource_units, "mte3":
                                        context.resource_units.get("fixpipe", context.core_count)},
                        work=inner_work)
        return SteadyMatmul().lower(inner, output=FixpipeOutput())


def recipes():
    return {PROFILE: MXMatrix()}
