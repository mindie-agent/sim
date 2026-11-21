"""Ordinary local-expert routing and MX activation producer boundary.

[Role] Replace selected route packing with one source initialization kernel.
[Invariants] Preserve router, expert matrices, down quantization and inverse
  mapping consumers. Only the matching gate-up input quantizer is replaced.
[Timeouts & Locks] Linear in the finite frontend graph; no external I/O.
[Side-Effect Policy] Returns immutable operators and recipe declarations.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.implementations import select
from mindie_sim.prediction.mechanism import PlannedRecipe
from .moe_init_mxfp8 import (
    MX_SCALE_GROUP_SIZE, MX_WIDTH_ALIGNMENT, RECIPE, SOURCE, build,
)

CAPABILITIES = frozenset({
    "moe_init_routing", "local_experts", "bf16_input", "fp8_e4m3_output",
    "mx32_scales", "full_local_expert_range", "no_transport_quantization",
})


def capabilities():
    """Machine-readable producer boundary and dynamic attributes."""
    return {
        "selector": "moe_init_routing",
        "implementation": RECIPE,
        "native_kind": "MoeInitRoutingV3",
        "source_boundary": "kernel",
        "input_dtypes": ("bf16",),
        "output_dtypes": ("fp8_e4m3",),
        "scale_format": "fp8_e8m0",
        "scale_group_size": MX_SCALE_GROUP_SIZE,
        "dynamic_attributes": (
            "tokens", "topk", "width", "experts", "local_rows",
            "activation_scale_shape", "output_contracts",
        ),
        "requires": (
            "parallel.ep == 1", "no_moe_transport_quantization",
            "direct_unique_grouped_gate_up_consumer",
            "complete_mxfp8_activation_assignment",
        ),
        "source": SOURCE,
    }


def recipes():
    return {RECIPE: PlannedRecipe(build)}


def tensor_contract(op):
    cfg = op['metadata']['moe_init_config']
    tokens, topk, width, experts = (cfg[k] for k in ('tokens', 'topk', 'width', 'experts'))
    rows = tokens * topk
    return ('moe_init_routing_mxfp8_active', 'MoeInitRoutingV3',
            (((tokens, width), 'bf16', 'ND'), ((tokens, topk), 'int32', 'ND')),
            (((rows, width), 'fp8_e4m3', 'ND'), ((rows,), 'int32', 'ND'),
             ((experts,), 'int64', 'ND'), ((rows, width // MX_SCALE_GROUP_SIZE), 'fp8_e8m0', 'ND')))


def apply(ops, bindings, runtime):
    selected = select(bindings, "moe_init_routing")
    if not selected:
        return ops
    if runtime.parallel.ep != 1 or runtime.kernels.get("moe_transport"):
        raise ModelingError("moe_init_mxfp8_requires_local_experts_without_transport_quantization")
    replacements = {}
    absorbed = set()
    for pack in ops:
        if pack.recipe != "route_pack":
            continue
        if pack.plan_required or "implementation_binding" in dict(pack.metadata):
            raise ModelingError("moe_init_mxfp8_conflicts_with_explicit_route_pack_implementation")
        tokens, topk = pack.local_shape
        if type(topk) is not int or topk < 1:
            raise ModelingError("moe_init_mxfp8_requires_positive_topk")
        if not tokens:
            continue
        matrices = [op for op in ops if op.recipe in {"grouped_gate_up", "balanced_grouped_gate_up"}
                    and pack.op_id in op.deps]
        if not matrices:
            # A preactivated frontend has already materialized the gate-input
            # quantizer between the route pack and the grouped gate-up matrix
            # (``pack -> dynamic_quantize -> expert_up``).  That quantizer is
            # exactly the ``fp8_e4m3``/MX32 producer this kernel replaces, so
            # the unique intervening producer is resolved and retired together
            # with the pack instead of being counted as a second consumer.
            quantizers = [op for op in ops
                          if pack.op_id in op.deps
                          and op.recipe in {"dynamic_quantize", "quantize"}]
            if len(quantizers) == 1:
                producer = quantizers[0]
                matrices = [op for op in ops
                            if op.recipe in {"grouped_gate_up",
                                             "balanced_grouped_gate_up"}
                            and producer.op_id in op.deps]
                if len(matrices) == 1 and tuple(matrices[0].deps) != (
                        producer.op_id,):
                    matrices = []
                elif len(matrices) == 1:
                    absorbed.add(producer.op_id)
        if len(matrices) != 1:
            raise ModelingError("moe_init_mxfp8_requires_direct_unique_grouped_gate_up_consumer")
        matrix = matrices[0]
        meta = dict(matrix.metadata)
        width, rows = matrix.local_shape[2], matrix.local_shape[0]
        if type(width) is not int or width < MX_WIDTH_ALIGNMENT or width % MX_WIDTH_ALIGNMENT:
            raise ModelingError("moe_init_mxfp8_requires_width_multiple_of_256")
        scale_shape = (rows, width // MX_SCALE_GROUP_SIZE)
        if (pack.dtype != "bf16" or meta.get("activation_format") != "fp8_e4m3"
                or meta.get("activation_scale_format") != "fp8_e8m0"
                or meta.get("activation_group_size") != MX_SCALE_GROUP_SIZE
                or rows != tokens * topk or meta.get("expected_rows", rows) != rows
                or meta.get("activation_scale_bytes") != rows * width // MX_SCALE_GROUP_SIZE):
            raise ModelingError("moe_init_mxfp8_requires_complete_BF16_to_E4M3_MX32_assignments")
        pm = dict(pack.metadata)
        for producer in (pm.get("topk_ids_producer"), pm.get("payload_producer")):
            if producer not in pack.deps:
                raise ModelingError("moe_init_mxfp8_requires_actual_ids_and_payload_producer")
        experts = pm["num_experts"]
        if type(experts) is not int or experts < 1:
            raise ModelingError("moe_init_mxfp8_requires_positive_local_experts")
        pm.update({
            "implementation": RECIPE, "implementation_binding": {"selector": "moe_init_routing",
                                                                   "implementation": RECIPE},
            "native_kind": "MoeInitRoutingV3", "invocation_class": "vector",
            "moe_init_config": {"tokens": tokens, "topk": topk, "width": width, "experts": experts},
            "source": SOURCE, "input_format": "bf16", "output_format": "fp8_e4m3",
            "activation_scale_format": "fp8_e8m0", "activation_scale_group_size": MX_SCALE_GROUP_SIZE,
            "activation_scale_shape": scale_shape,
            "output_contracts": {"expanded_x": (rows, width), "expanded_row_idx": (rows,),
                                 "expert_counts": (experts,), "expanded_scale": scale_shape},
            "expert_ids": "declared_valid_full_local_range; no_sampled_topk_ids_invented",
            "row_idx_type": "GATHER", "expert_tokens_num_type": "count",
        })
        replacements[pack.op_id] = replace(pack, recipe=RECIPE, family="moe_init_routing",
            source_boundary="kernel", returns=("activation_scales", "expert_counts", "inverse_indices"),
            metadata=tuple(pm.items()))
        # Existing activation lowering recognizes this complete producer. It
        # still creates down/input conversions for every other matrix.
        meta.update(activation_producer="explicit_routing_quantization",
                    activation_producer_kind="local_routing_mxfp8_quantization",
                    activation_payload_producer=pack.op_id, activation_scale_producer=pack.op_id,
                    activation_scale_producer_shape=scale_shape,
                    activation_scale_consumer_view="contiguous_reshape_same_E8M0_payload")
        deps = (pack.op_id,) if any(
            dep in absorbed for dep in matrix.deps) else matrix.deps
        replacements[matrix.op_id] = replace(
            matrix, deps=deps, metadata=tuple(meta.items()))
    return tuple(op for op in (replacements.get(op.op_id, op) for op in ops)
                 if op.op_id not in absorbed)


__all__ = ["CAPABILITIES", "RECIPE", "SOURCE", "apply", "capabilities",
           "recipes", "tensor_contract"]
