"""Declared matrix tensor contracts without flattening grouped programs.

[Role] Preserve dense versus expert-bank matrix geometry in structure exports.
[Invariants] Physical batch groups do not become a single dense GEMM. Expert-bank
  size is independent of active experts, and fractional routing moments are not
  rounded into captured tensor rows. Contracts do not establish dispatch.
[Timeouts & Locks] Bounded metadata arithmetic; no I/O or locks.
[Side-Effect Policy] Pure records; no profiler imports or calibration changes.
"""


def mxfp4_reference_coverage():
    """Packed payload accounting does not supply native MXFP4 execution."""
    return {
        "kind": "partial_program",
        "reason": "mxfp4_native_execution_unmodeled",
        "missing_work": ["native_fp8_fp4_matrix_service_or_explicit_operand_conversion",
                         "microscale_application"],
        "basis": (
            "The declared program uses BF16 matrix service after an assumed fused "
            "dequantization; packed weight bytes and scale traffic are retained, "
            "but conversion and native MX execution have no physical program."),
    }


def matrix_contract(op):
    meta, shape = op["metadata"], tuple(op["local_shape"])
    if len(shape) != 3:
        return None
    precision = (meta["activation_format"], meta["weight_format"], meta["output_format"])
    scales = (tuple(meta.get("activation_scale_shape", ())),
              meta.get("activation_scale_format", ""))
    wrapper = meta.get("matrix_mapping_domain") == "single_group_dense_wrapper"
    grouped = (op["family"] == "grouped_gemm" or wrapper
               or meta.get("matrix_mapping_domain") in {"aggregate_experts", "equal_expert_groups"})
    if grouped:
        bank = meta.get("matrix_weight_bank_groups")
        if not isinstance(bank, int) or bank < 1 or meta.get("bias"):
            return None
        if wrapper:
            declaration = meta.get("matrix_wrapper", {})
            if (bank != 1 or declaration.get("input_rank") != 2
                    or declaration.get("graph_mode") != "eager"
                    or tuple(declaration.get("group_list_values", ())) != (shape[0],)):
                return None
        if "expected_rows" in meta and meta["expected_rows"] != shape[0]:
            return None
        return "grouped_matrix", (bank, *shape), *precision, *scales
    if meta.get("matrix_groups", 1) != 1:
        # The generic shape stores aggregate M; it does not encode batch axes.
        # Native TransposeBatchMatMul has its own explicit typed projection.
        return None
    if meta.get("bias"):
        if not meta.get("bias_shape") or not meta.get("bias_dtype") or scales[0]:
            return None
        return ("matrix_bias", shape, *precision,
                tuple(meta["bias_shape"]), meta["bias_dtype"])
    return "matrix", shape, *precision, *scales


def _has_w4a8_down_epilogue(op):
    meta = op["metadata"]
    phases = ("pre", "matrix", "post_dequant", "output")
    return (
        op["recipe"] == "ascend_w4a8_msd_grouped_down"
        and op.get("source_boundary") == "kernel"
        and meta.get("compute_format") == "int4"
        and bool(meta.get("w4a8_msd_down"))
        and tuple(meta.get("w4a8_msd_down_phases", ())) == phases
        and all(phase in meta.get("w4a8_msd_down_ledger", {}) for phase in phases)
    )


def matrix_comparison_scope(op):
    meta = op["metadata"]
    grouped = op["family"] == "grouped_gemm" or meta.get("matrix_mapping_domain")
    scope = {
        "kind": "grouped_matrix_active_geometry" if grouped else "matrix_active_geometry",
        "compared": ["logical_M_N_K", "activation_weight_output_precision",
                     "activation_scale_shape_dtype", *(["expert_bank_count"] if grouped else [])],
        "not_compared": ["weight_scale_storage_representation", "scale_values",
                         "native_dispatch", "tiling",
                         *(["per_expert_row_histogram", "active_weight_reads", "group_list_values"]
                           if grouped else [])],
        "basis": "independently_declared_tensor_geometry; no_occurrence_or_workload_realization_proof",
    }
    if grouped:
        scope["routing_basis"] = meta.get("routing_basis", "undeclared")
        scope["work_assumption"] = (
            "Total rows constrain GEMM FLOPs, but the expert histogram and active weight "
            "reads remain those declared by the case.")
    if meta.get("matrix_mapping_domain") == "single_group_dense_wrapper":
        declaration = meta["matrix_wrapper"]
        scope["kind"] = "source_selected_single_bank_matrix_geometry"
        scope["routing_basis"] = "source_wrapper_all_rows_in_one_group"
        scope["declared_group_list"] = list(declaration["group_list_values"])
        scope["work_assumption"] = (
            "One independently declared dense projection supplies every row to a "
            "single weight bank. The existing matrix program prices that projection; "
            "the wrapper declaration does not identify the captured occurrence.")
        scope["source_wrapper"] = declaration
    if meta.get("weight_format") in {"int4", "w4a8"} and meta.get("activation_format") == "int8":
        epilogue = ["packed_int4_output_rescale", "quantization_offset_correction"]
        if _has_w4a8_down_epilogue(op):
            scope["implemented_quantized_epilogue"] = {
                "program": op["recipe"], "work": epilogue,
                "basis": "source_pre_matrix_post_dequant_output_ledger",
            }
            scope["source_program_limitations"] = list(meta.get("w4a8_msd_down_limitations", ()))
        else:
            # Packed bytes alone do not implement MSD assist correction and
            # output rescaling. Only the selected complete source program does.
            scope["prediction_coverage"] = {
                "kind": "partial_program", "reason": "quantized_epilogue_unmodeled",
                "missing_work": epilogue,
                "basis": "declared_int4_int8_program_has_no_supported_correction_epilogue",
            }
    if meta.get("activation_scale_format") == "fp8_e8m0":
        scope["mx_execution_assumption"] = (
            "Declared matrix compute-format work and operand-scale traffic; MX scale "
            "application and native instruction scheduling are not separately resolved. "
            "This does not infer an additional output epilogue from tensor descriptors.")
    if (meta.get("weight_format") in {"mxfp4", "fp4", "fp4_e2m1"}
            and meta.get("dequant_before_compute")
            and meta.get("compute_format") == "bf16"):
        scope["prediction_coverage"] = mxfp4_reference_coverage()
    return scope
