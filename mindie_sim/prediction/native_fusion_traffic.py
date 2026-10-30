"""Verified local tensor reuse within explicitly selected native fusion sources.

[Role] Remove only materializations that inspected kernel source keeps local.
[Invariants] Kernel fusion does not imply that every intermediate leaves HBM.
  Global workspaces and externally visible outputs retain their transfers.
  The selected source profile is not evidence of a capture's binary or tiling.
[Timeouts & Locks] Finite per-group arithmetic; no I/O, observations or locks.
[Side-Effect Policy] Returns immutable phase records and an accounting ledger.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError

from .mla_prolog_packed import PROFILE as MLA_PACKED_PROFILE
from .mla_prolog_arch22 import PROFILE as MLA_ARCH22_PROFILE
from .mla_prolog_arch22 import SOURCE as MLA_ARCH22_SOURCE
from .mla_prolog_arch22 import validate_declared_view as validate_mla_arch22_view


PROFILE_ROLES = {
    "preserve_workspaces": None,
    MLA_ARCH22_PROFILE: "mla_prolog",
    MLA_PACKED_PROFILE: "mla_prolog",
    "mla_prolog_arch35_split_m_n_ae923905": "mla_prolog",
    "hc_pre_arch35_mk_175ef4d4": "hc_pre",
    "hc_pre_arch35_mk_compact_175ef4d4": "hc_pre",
    "hc_post_arch35_bf16_175ef4d4": "hc_post",
    "hc_pre_m_k_split_175ef4d4": "hc_pre",
    "hc_post_d_split_175ef4d4": "hc_post",
    "grouped_swiglu_quant_int4_post_ae923905": "grouped_gate_up_activation_quant",
    "grouped_swiglu_quant_int8_split_workspace_ae923905": "grouped_gate_up_activation_quant",
    "situ_mx_quant_arch35_2a9bd1b0": "situ_quant",
}
_MLA_SOURCE = {
    "repository": "vllm-ascend", "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
    "path": "csrc/attention/mla_prolog_v3/op_kernel/arch35/kernel_mla_prolog_split_m.h",
    "sha256": "c0d49290db61a0bfc418415255a2ee96d2bff4cbfdcd49be4f1a65b57d541ea3",
    "lines": "1298-1307,1460-1469", "source_domain": "arch35_split_m_and_split_n",
}
_HC_SOURCE = {
    "repository": "vllm-ascend", "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "path": "csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h",
    "sha256": "f14eeb0888a27f80a17708fd6f5e57ec0a217176cfe97da512b6bac7218e7fac",
    "lines": "393-428,438-490", "source_domain": "m_k_split_local_vector_phase",
}
_GMM_SOURCE = {
    "repository": "vllm-ascend", "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
    "path": "csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/"
            "grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h",
    "sha256": "6b39b80ca4dc18eb7f99fac4c559462021012ee98a365120e77610d0a243bd07",
    "lines": "147-156,187-229", "source_domain": "int8_int4_swiglu_quant_local_postprocess",
}
_GMM_INT8_SOURCE = {
    "repository": "vllm-ascend", "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
    "path": "csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/"
            "grouped_matmul_swiglu_quant_split_ws.h",
    "sha256": "fab1d301a017d666da7604fad90a55bc3e7c07d565bea8b7aa7abceebaae2457",
    "lines": "134-135,346,462,528-530,552-625",
    "source_domain": "int8_int8_key1_split_workspace",
}
_SITU_SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "source_state": "added_in_index",
    "path": "csrc/moe/situ_mx_quant/op_kernel/arch35/situ_mx_quant_axis_last.h",
    "sha256": "2a9bd1b0058d5c5727dc313c10f1e6920ea900390c64402d5263c8a602470284",
    "lines": "186-227,251-274", "source_domain": "arch35_axis_last_bf16_to_fp8_mx",
}


def apply_traffic_profile(ops, declaration, profile, *, external_outputs=()):
    if profile not in PROFILE_ROLES:
        raise ModelingError("native_fusion_traffic_profile_unknown")
    if PROFILE_ROLES[profile] not in (None, declaration["role"]):
        raise ModelingError("native_fusion_traffic_profile_role_mismatch")
    phases = list(ops)
    positions = {op.op_id: index for index, op in enumerate(phases)}
    ledger = {"profile": profile, "eliminated_reads": 0.0, "eliminated_writes": 0.0,
              "reused_edges": [], "limitations": [], "source": None}

    def recipe(name):
        return next(op for op in phases if op.recipe == name)

    def remove(producer, consumer, amount, *, preserve_output=False):
        source, target = positions[producer.op_id], positions[consumer.op_id]
        before, after = phases[source], phases[target]
        preserve_output = preserve_output or producer.op_id in external_outputs
        write = 0 if preserve_output else amount
        if amount > after.mte2_bytes + 1e-6 or write > before.mte3_bytes + 1e-6:
            raise ModelingError("native_fusion_reuse_exceeds_declared_tensor_transfer")
        phases[source] = replace(before, mte3_bytes=before.mte3_bytes - write,
                                 hbm_bytes=before.hbm_bytes - write)
        phases[target] = replace(after, mte2_bytes=after.mte2_bytes - amount,
                                 hbm_bytes=after.hbm_bytes - amount)
        ledger["eliminated_reads"] += amount
        ledger["eliminated_writes"] += write
        ledger["reused_edges"].append({
            "producer_phase": source, "consumer_phase": target,
            "read_bytes": amount, "write_bytes": write,
            "externally_visible_write_preserved": preserve_output,
        })

    if profile == "preserve_workspaces":
        ledger["limitations"].append("internal_tensor_placement_not_selected")
        return tuple(phases), ledger
    role, view = declaration["role"], declaration["tensor_view"]
    from .family_hc_arch35 import PRE_PROFILE, COMPACT_PRE_PROFILE, POST_PROFILE, SOURCE, LIMITATIONS
    if profile in {PRE_PROFILE, COMPACT_PRE_PROFILE, POST_PROFILE}:
        ledger["source"] = SOURCE
        ledger["limitations"].extend((
            "complete_source_replacement_owns_all_workspace_and_output_traffic",
            *LIMITATIONS,
        ))
        return tuple(phases), ledger
    if profile == MLA_PACKED_PROFILE:
        from .mla_prolog_packed import SOURCE as PACKED_SOURCE, validate_view
        validate_view(view)
        ledger["source"] = PACKED_SOURCE
        ledger["limitations"].extend([
            "analytic_member_resources_with_explicit_packed_cache_quantization",
            "projection_workspaces_retained; native_arch22_schedule_not_reused",
            "whole_native_call_observed_service_may_be_selected_once",
        ])
        return tuple(phases), ledger
    if profile == MLA_ARCH22_PROFILE:
        # The complete source program owns every scratch materialization. Do
        # not apply arch35 local-reuse savings or member epilogue work here.
        if role != "mla_prolog":
            raise ModelingError("native_fusion_traffic_profile_role_mismatch")
        validate_mla_arch22_view(view)
        ledger["source"] = MLA_ARCH22_SOURCE
        ledger["limitations"].extend([
            "complete_source_replacement_owns_all_workspace_and_cache_traffic",
            "arch35_traffic_savings_not_applied",
            "source_selection_does_not_identify_capture_binary_or_tiling",
        ])
        return tuple(phases), ledger
    if role == "mla_prolog":
        norm, rotary, store = recipe("kv_norm"), recipe("rotary"), recipe("kv_store")
        latent, rope = view["latent_rank"], view["rope_width"]
        latent_bytes = store.mte2_bytes * latent / (latent + rope)
        rope_bytes = store.mte2_bytes - latent_bytes
        remove(norm, store, latent_bytes)
        remove(rotary, store, rope_bytes)
        ledger["source"] = _MLA_SOURCE
        ledger["limitations"].extend([
            "source_arch35_selection_does_not_identify_capture_implementation",
            "query_projection_norm_and_quantization_workspaces_retained",
            "cache_writes_and_external_query_rope_outputs_retained",
        ])
    elif role == "hc_pre":
        affine, sinkhorn, mix = (
            recipe(name) for name in ("hc_normalize_scale_bias_gate", "hc_sinkhorn", "hc_pre_mix"))
        remove(affine, sinkhorn, view["tokens"] * view["mixed"] * 4)
        remove(sinkhorn, mix, view["tokens"] * view["hc"] * 4, preserve_output=True)
        ledger["source"] = _HC_SOURCE
        ledger["limitations"].extend([
            "source_plan_replaces_analytical_inverse_rms_with_shared_operand_gram",
            "cast_and_split_k_partial_global_workspaces_retained",
            "source_three_output_branch_does_not_include_optional_pre_output",
            "global_workspace_l2_residency_not_assumed",
        ])
    elif role == "hc_post":
        from .family_hc import SOURCE
        ledger["source"] = {
            "repository": "vllm-ascend", "revision": SOURCE["revision"],
            **SOURCE["post"], "source_domain": "d_split_two_queue_fp32_arithmetic",
        }
        ledger["limitations"].extend([
            "external_tensor_bytes_unchanged",
            "source_plan_accounts_input_output_casts_and_dependent_d_chunks",
            "vector_issue_barrier_latency_and_instruction_ports_not_calibrated",
        ])
    elif role == "situ_quant":
        if (view["input_dtype"] != "bf16"
                or view["output_dtype"] not in {"fp8_e4m3", "fp8_e5m2"}
                or view["scale_dtype"] != "fp8_e8m0" or view["axis"] != -1):
            raise ModelingError("native_fusion_traffic_profile_representation_mismatch")
        activation = next(op for op in phases if op.family == "situ")
        quant = next(op for op in phases if op.family == "quantize")
        remove(activation, quant, activation.mte3_bytes)
        ledger["source"] = _SITU_SOURCE
        ledger["limitations"].extend([
            "source_arch35_profile_does_not_identify_capture_binary",
            "FP8_output_and_E8M0_scale_writes_retained",
            "activation_expression_and_quantization_work_inherit_declared_member_recipes",
        ])
    elif role == "grouped_gate_up_activation_quant":
        expected = "int4" if profile == "grouped_swiglu_quant_int4_post_ae923905" else "int8"
        if (view["input_dtype"], view["weight_dtype"], view["output_dtype"]) != (
                "int8", expected, "int8"):
            raise ModelingError("native_fusion_traffic_profile_representation_mismatch")
        activation = next(op for op in phases if op.family == "swiglu")
        quant = next(op for op in phases if op.family == "quantize")
        remove(activation, quant, activation.mte3_bytes)
        ledger["source"] = _GMM_SOURCE if expected == "int4" else _GMM_INT8_SOURCE
        ledger["limitations"].extend([
            "matrix_output_global_workspace_retained",
            "native_workspace_padding_and_instruction_tiling_not_modeled",
            "workspace_representation_and_quantization_arithmetic_inherit_member_recipes",
            "int4_dual_fp16_or_int8_int32_native_workspace_not_reconstructed",
        ])
    else:
        raise ModelingError("native_fusion_traffic_profile_representation_mismatch")
    return tuple(phases), ledger
