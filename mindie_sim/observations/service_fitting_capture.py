"""Service-fitting capture I/O and feature extraction models.

[Role] Load profiling capture datasets and build structured fitting records
  for resource-service regression.
[Invariants] Explicit schema validation with zero silent record drops.
[Timeouts & Locks] Local file reads only; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

import math


def _capture_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"N/A", "None"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _capture_int(value):
    number = _capture_float(value)
    return None if number is None else int(number)


def load_native_cohort_validation(path, *, expected_sha256=None):
    """Read the T03 native cohort validation CSV and its source hash.

    The file is already the normalised T03 output; this function never rescans
    a capture and never uses a filename as evidence.
    """
    import csv
    import hashlib
    from pathlib import Path
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("native_cohort_validation_source_missing:" + str(source))
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    if expected_sha256 and sha != str(expected_sha256):
        raise ValueError(
            "native_cohort_validation_sha256_mismatch:"
            + sha + "!=" + str(expected_sha256))
    rows = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    by_cohort = {}
    for row in rows:
        cohort_id = str(row.get("cohort_id") or "").strip()
        if not cohort_id:
            raise ValueError("native_cohort_validation_row_requires_cohort_id")
        if cohort_id in by_cohort:
            raise ValueError(
                "native_cohort_validation_duplicate_cohort_id:" + cohort_id)
        by_cohort[cohort_id] = row
    counts = {}
    for row in rows:
        key = (str(row.get("calibration_relation") or ""),
               str(row.get("contract_match_status") or ""))
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema": "mindie-sim.native-cohort-validation/1",
        "path": str(source),
        "sha256": sha,
        "bytes": source.stat().st_size,
        "rows": rows,
        "by_cohort": by_cohort,
        "count": len(rows),
        "counts": {f"{role}:{status}": value
                   for (role, status), value in sorted(counts.items())},
    }


def read_capture_locators(capture_path, locators, *, with_sha256=False):
    """Stream one capture and return only the explicitly requested CSV lines."""
    import csv
    import hashlib
    from pathlib import Path
    path = Path(capture_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("capture_source_missing:" + str(path))
    requested = sorted({int(item) for item in locators})
    if not requested:
        raise ValueError("capture_locator_lines_required")
    if requested[0] < 2:
        raise ValueError("capture_locator_line_is_one_based_data_line")
    found = {}
    rows_scanned = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Task ID" not in reader.fieldnames:
            raise ValueError("capture_source_requires_kernel_details_header")
        for row in reader:
            rows_scanned += 1
            if reader.line_num in requested:
                found[reader.line_num] = dict(row)
    sha = None
    if with_sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha,
        "requested_lines": requested,
        "rows": found,
        "missing_lines": [line for line in requested if line not in found],
        "rows_scanned": rows_scanned,
    }


def capture_image_identity(capture_path):
    """Read the same-capture image identity from its retained side files."""
    import hashlib
    import re
    from pathlib import Path
    source = Path(capture_path).expanduser().resolve()
    directory = source.parent
    checks = []
    readme = directory / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"image\s+`([0-9a-fA-F]{12,64})`(?:\s+\(([^)]+)\))?", text)
        if match:
            digest = match.group(1).lower()
            return {
                "image_id": digest,
                "revision": (match.group(2) or "").strip() or None,
                "source": str(readme),
                "raw": match.group(0),
                "sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
            }
        checks.append({"source": str(readme), "status": "image_not_found"})
    env = directory / "session.env"
    if env.is_file():
        text = env.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?m)^image=sha256:([0-9a-fA-F]{12,64})\s*$", text)
        tag = re.search(r"(?m)^image_tags=(.+)$", text)
        if match:
            digest = match.group(1).lower()
            return {
                "image_id": digest,
                "revision": None,
                "image_tags": (tag.group(1).strip() if tag else None),
                "source": str(env),
                "raw": match.group(0),
                "sha256": hashlib.sha256(env.read_bytes()).hexdigest(),
            }
        checks.append({"source": str(env), "status": "image_not_found"})
    return {"image_id": None, "revision": None, "source": None,
            "checks": checks, "reason": "capture_image_identity_not_declared"}


_CAPTURE_DTYPE_BYTES = {
    "DT_BF16": 2, "DT_FLOAT16": 2, "DT_FLOAT": 4, "DT_INT8": 1,
    "DT_UINT8": 1, "DT_FLOAT8_E4M3FN": 1, "DT_FLOAT8_E8M0": 1,
    "DT_INT64": 8, "DT_INT32": 4,
}
_CAPTURE_DTYPE_ALIASES = {
    "DT_BF16": "bf16", "DT_FLOAT16": "fp16", "DT_FLOAT": "fp32",
    "DT_INT8": "int8", "DT_UINT8": "uint8", "DT_FLOAT8_E4M3FN": "fp8_e4m3",
    "DT_FLOAT8_E8M0": "fp8_e8m0", "DT_INT64": "int64", "DT_INT32": "int32",
}
_PIPE_RATIO_ABSOLUTE_TOLERANCE = 0.0005
_PIPE_RATIO_RELATIVE_TOLERANCE = 0.02
_AIC_PIPES = ("mac", "scalar", "mte1", "mte2", "mte3", "fixpipe")
_AIV_PIPES = ("vec", "scalar", "mte2", "mte3")
_PIPE_TO_OWNER_PHASE = {
    "mac": ("cube", "compute"),
    "mte1": ("mte1", "l1_to_l0"),
    "mte2": ("mte2", "load"),
    "mte3": ("mte3", "store"),
    "fixpipe": ("fixpipe", "epilogue"),
    "scalar": ("scalar", "scalar"),
    "vec": ("vector", "compute"),
}


_CAPTURE_MATMUL_ARITY = 2
_CAPTURE_QBMM_ARITY = 4
_CAPTURE_NATIVE_ARITY = {
    "MatMulV3": _CAPTURE_MATMUL_ARITY,
    "QuantBatchMatmulV3": _CAPTURE_QBMM_ARITY,
}


def _capture_binary_contract(image_id, native_type):
    """Record caller-supplied image provenance without prefix dispatch.

    A short image prefix is not a program identity: the observation contract
    keeps the declared digest and reports binary equivalence as undeclared
    unless the caller supplies an exact identity.  No image-to-adapter table
    or prefix expansion exists here.
    """
    from mindie_sim.observations.resource_services import identity_digest
    digest = identity_digest(image_id or "")
    return {
        "native_type": native_type,
        "observed_image_id": digest or None,
        "binary_match": False,
        "status": ("caller_supplied_image_identity" if digest
                   else "image_identity_not_declared"),
    }


def parse_capture_abi(row):
    """Parse raw shapes/dtypes/formats into the ABI-signature order."""
    def clean(value):
        return str(value or "").strip().strip("\"").strip("'").strip()

    def split(value):
        return [item.strip() for item in clean(value).split(";") if item.strip()]

    def shaped(values):
        parsed = []
        for item in values:
            try:
                parsed.append([int(part) for part in item.split(",") if part])
            except ValueError:
                parsed.append(None)
        return parsed

    inputs = split(row.get("Input Shapes"))
    input_dtypes = [clean(item) for item in split(row.get("Input Data Types"))]
    input_formats = [clean(item) for item in split(row.get("Input Formats"))]
    outputs = split(row.get("Output Shapes"))
    output_dtypes = [clean(item) for item in split(row.get("Output Data Types"))]
    output_formats = [clean(item) for item in split(row.get("Output Formats"))]
    signature = "|".join([
        ";".join(inputs), ";".join(input_dtypes), ";".join(input_formats),
        ";".join(outputs), ";".join(output_dtypes), ";".join(output_formats),
    ])
    return {
        "input_shapes": shaped(inputs),
        "input_dtypes": input_dtypes,
        "input_formats": input_formats,
        "output_shapes": shaped(outputs),
        "output_dtypes": output_dtypes,
        "output_formats": output_formats,
        "abi_signature": signature,
    }


def _capture_geometry(native_type, inputs, outputs):
    """Derive M/K/N from the captured ABI shapes, not from an image table."""
    first, second = list(inputs[0]), list(inputs[1])
    if len(first) < _CAPTURE_MATMUL_ARITY or len(second) < _CAPTURE_MATMUL_ARITY:
        raise ValueError("capture_abi_requires_rank_two_matmul_operands")
    rows = 1
    for dim in first[:-1]:
        rows *= int(dim)
    inner = int(first[-1])
    if native_type == "MatMulV3":
        columns = int(second[-1])
        declared_inner = int(second[0])
        if declared_inner != inner and columns == inner:
            columns, inner = declared_inner, int(second[-1])
    else:
        declared_inner = int(second[0])
        columns = (int(second[-1]) if declared_inner == inner
                   else declared_inner)
    return {
        "m": int(rows), "k": int(inner), "n": int(columns),
        "macs": int(rows) * int(inner) * int(columns),
    }


def _capture_tensor(name, shape, dtype, layout=None):
    return {
        "name": name,
        "shape": [int(item) for item in shape],
        "dtype": _CAPTURE_DTYPE_ALIASES.get(dtype, str(dtype).lower()),
        **({"layout": layout} if layout else {}),
    }


def _capture_bytes(shapes, dtypes):
    total = 0
    for shape, dtype in zip(shapes, dtypes):
        width = _CAPTURE_DTYPE_BYTES.get(dtype)
        if width is None or not shape:
            continue
        elements = 1
        for dim in shape:
            elements *= int(dim)
        total += elements * width
    return total


def bind_capture_native_plan(row, *, image_id=None):
    """Bind one captured ABI row to the canonical declared call geometry.

    Shapes and dtype widths come only from the captured row; the returned
    declaration carries no measured duration, no image-prefix program dispatch
    and no ``predict_abi`` pricing.  Missing or malformed geometry stays an
    explicit unsupported result.
    """
    native_type = str(row.get("Type") or "")
    abi = parse_capture_abi(row)
    inputs = abi["input_shapes"]
    outputs = abi["output_shapes"]
    version = ({"image_id": str(image_id)} if image_id
               else {"source": "captured_abi"})
    binary = _capture_binary_contract(image_id, native_type)
    result = {
        "native_type": native_type,
        "entry_adapter": None,
        "binary": binary,
        "version": version,
        "status": "not_attempted",
        "reasons": [],
        "source_role_status": "not_identified_from_capture",
        "dispatch_status": "not_established_from_capture",
    }
    arity = _CAPTURE_NATIVE_ARITY.get(native_type)
    parsed = list(inputs[:arity]) if arity else []
    if (arity is None or len(inputs) < arity or not outputs
            or any(shape is None for shape in parsed + list(outputs[:1]))):
        result["status"] = "unsupported"
        result["reasons"] = ["no_canonical_geometry_for_capture_family"]
        return result
    try:
        geometry = _capture_geometry(native_type, inputs, outputs)
    except (TypeError, ValueError, IndexError) as error:
        result["status"] = "adapter_error"
        result["reasons"] = [type(error).__name__ + ":" + str(error)]
        return result
    input_bytes = _capture_bytes(inputs, abi["input_dtypes"])
    output_bytes = _capture_bytes(outputs, abi["output_dtypes"])
    issued = {
        "macs": geometry["macs"],
        "mte2_bytes": input_bytes,
        "mte3_bytes": output_bytes,
    }
    tensors = [
        _capture_tensor(f"input{index}", shape, abi["input_dtypes"][index]
                        if index < len(abi["input_dtypes"]) else "")
        for index, shape in enumerate(inputs)]
    tensors.extend(
        _capture_tensor(f"output{index}", shape,
                        abi["output_dtypes"][index]
                        if index < len(abi["output_dtypes"]) else "")
        for index, shape in enumerate(outputs))
    selected_call = {
        "api": native_type,
        "arity": arity,
        "shapes": {"inputs": inputs, "outputs": outputs},
        "dtypes": {"inputs": abi["input_dtypes"],
                   "outputs": abi["output_dtypes"]},
        "tensors": tensors,
    }
    owner_phases = {
        "cube": {
            "owner": "cube", "phase": "matmul", "resource": "cube",
            "declared_logical_macs": geometry["macs"],
            "declared_padded_macs": None,
            "declared_bytes": None, "issued_macs": None, "issued_bytes": None,
            "work_basis": "declared_ABI_geometry_not_measured_dispatch",
        },
        "mte2": {
            "owner": "mte2", "phase": "load", "resource": "mte2",
            "declared_logical_macs": None, "declared_padded_macs": None,
            "declared_bytes": input_bytes, "issued_macs": None,
            "issued_bytes": None,
            "work_basis": "declared_ABI_geometry_not_measured_dispatch",
        },
        "mte3": {
            "owner": "mte3", "phase": "store", "resource": "mte3",
            "declared_logical_macs": None, "declared_padded_macs": None,
            "declared_bytes": output_bytes, "issued_macs": None,
            "issued_bytes": None,
            "work_basis": "declared_ABI_geometry_not_measured_dispatch",
        },
    }
    descriptor = {
        "selected_call": selected_call,
        "geometry": geometry,
        "issued": issued,
        "kernels": list(owner_phases.values()),
    }
    result.update({
        "status": "predicted",
        "record_status": "predicted",
        "geometry": geometry,
        "work": {
            "work_unit": "macs",
            "logical_work": geometry["macs"],
            "logical_bytes": input_bytes + output_bytes,
            "issued_work": None,
            "issued_work_status": "requires_selected_dispatch_and_tiling",
        },
        "selected_call": selected_call,
        "operator_plan": descriptor,
        "issued": issued,
        "kernels": list(owner_phases.values()),
        "owner_phases": owner_phases,
        "plan_basis": "declared_ABI_geometry_not_measured_dispatch",
        "tail_status": "not_exposed_at_descriptor_boundary",
    })
    result["source_established"] = bool(binary.get("binary_match"))
    return result


def capture_measurement(row):
    """Expose the raw measured counters with their actual unit and limits.

    The kernel_details CSV contains one row per device kernel.  The PMU pipe
    columns are busy intervals inside that same kernel, normalised by the
    AIC/AIV total time; they overlap.  The CSV does not declare whether the
    scalar is an average over active cores or a cumulative interval, and it
    has no per-owner or per-block counter.  ``Block Num`` is therefore kept as
    a block count, not an effective-owner count.
    """
    accelerator = str(row.get("Accelerator Core") or "")
    aicore = _capture_float(row.get("aicore_time(us)"))
    aiv = _capture_float(row.get("aiv_time(us)"))
    aic = {name: _capture_float(row.get(f"aic_{name}_time(us)"))
           for name in _AIC_PIPES}
    aiv_pipes = {name: _capture_float(row.get(f"aiv_{name}_time(us)"))
                 for name in _AIV_PIPES}
    duration = _capture_float(row.get("Duration(us)"))
    checks = []
    if duration is None or duration <= 0:
        raise ValueError("capture_measurement_requires_positive_duration")
    checks.append({"name": "duration_positive", "ok": duration > 0,
                   "detail": f"{duration} us"})
    ratio_checks = []
    for name, value in aic.items():
        if not value:
            continue
        ratio = _capture_float(row.get(f"aic_{name}_ratio"))
        computed = value / aicore if aicore else None
        ok = (ratio is None or computed is None
              or abs(computed - ratio) <= max(
                  _PIPE_RATIO_ABSOLUTE_TOLERANCE,
                  abs(ratio) * _PIPE_RATIO_RELATIVE_TOLERANCE))
        ratio_checks.append({"pipe": name, "time_us": value,
                             "aicore_time_us": aicore,
                             "computed_ratio": computed,
                             "declared_ratio": ratio, "ok": ok})
    for name, value in aiv_pipes.items():
        if not value:
            continue
        ratio = _capture_float(row.get(f"aiv_{name}_ratio"))
        computed = value / aiv if aiv else None
        ok = (ratio is None or computed is None
              or abs(computed - ratio) <= max(
                  _PIPE_RATIO_ABSOLUTE_TOLERANCE,
                  abs(ratio) * _PIPE_RATIO_RELATIVE_TOLERANCE))
        ratio_checks.append({"pipe": "aiv_" + name, "time_us": value,
                             "aiv_time_us": aiv,
                             "computed_ratio": computed,
                             "declared_ratio": ratio, "ok": ok})
    bad_ratios = [item for item in ratio_checks if not item["ok"]]
    checks.append({"name": "pmu_ratios_match_total_time",
                   "ok": not bad_ratios,
                   "detail": f"{len(ratio_checks) - len(bad_ratios)}/"
                             f"{len(ratio_checks)} checked"})
    aic_active = ({name: value for name, value in aic.items() if value}
                  if accelerator in {"AI_CORE", "MIX_AIC", "MIX_AIV"} else {})
    aiv_active = ({name: value for name, value in aiv_pipes.items() if value}
                  if accelerator in {"AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"}
                  else {})
    total_time = max(value for value in (aicore or 0.0, aiv or 0.0))
    pipe_sum = sum(aic_active.values()) + sum(aiv_active.values())
    overlaps = bool(total_time and pipe_sum > total_time + 1e-6)
    checks.append({"name": "pipe_times_overlap_or_sum_at_most_total",
                   "ok": True,
                   "detail": ("overlapping; not additive" if overlaps
                              else "within total window")})
    block_num = _capture_int(row.get("Block Num"))
    mix_block_num = _capture_int(row.get("Mix Block Num"))
    checks.append({"name": "block_num_is_not_owner_count", "ok": True,
                   "detail": (f"Block Num={block_num}; it is a block count, "
                              "not a measured active-owner count")})
    return {
        "boundary": "device_kernel",
        "window": "one_kernel_invocation",
        "units": {"duration": "us", "pipe_times": "us",
                  "ratios": "fraction", "cycles": "count"},
        "kernel_duration_us": duration,
        "aicore_time_us": aicore,
        "aiv_time_us": aiv,
        "accelerator_core": accelerator or None,
        "aic_pipe_busy_us": aic_active,
        "aiv_pipe_busy_us": aiv_active,
        "pipe_busy_us": {
            **{f"aic_{name}": value for name, value in aic_active.items()},
            **{f"aiv_{name}": value for name, value in aiv_active.items()},
        },
        "pipe_ratio_checks": ratio_checks,
        "pipe_busy_sum_us": pipe_sum,
        "pipe_times_overlap": overlaps,
        "pipe_times_additive": False,
        "aggregation_semantics": (
            "not_declared_by_kernel_details_csv; pipe times are per-event busy "
            "intervals normalised by the AIC/AIV total time; average-vs-"
            "cumulative and active-owner aggregation are not declared"),
        "effective_owners": None,
        "owner_aggregation": "not_declared",
        "block_num": block_num,
        "mix_block_num": mix_block_num,
        "block_num_is_owner": False,
        "issued_mac": None,
        "issued_bytes": None,
        "issued_work_status": "not_measured_by_capture",
        "scheduling": {
            "device_id": row.get("Device_id"),
            "model_id": row.get("Model ID"),
            "task_id": row.get("Task ID"),
            "stream_id": row.get("Stream ID"),
            "start_us": _capture_float(row.get("Start Time(us)")),
            "end_us": (None if _capture_float(row.get("Start Time(us)")) is None
                       else _capture_float(row.get("Start Time(us)")) + duration),
            "wait_us": _capture_float(row.get("Wait Time(us)")),
            "op_state": row.get("OP State"),
            "kernel_name": row.get("Name"),
            "kernel_type": row.get("Type"),
        },
        "checks": checks,
    }


def logical_capture_work(row):
    """Derive only source-declared logical work from the raw ABI shapes.

    MACs are ``M*N*K`` for a dense matrix call.  The logical tensor footprint
    is the call-boundary operand/output byte count.  Neither is claimed to be
    issued work; issued work requires the selected native program/dispatch.
    """
    abi = parse_capture_abi(row)
    native_type = str(row.get("Type") or "")
    inputs = abi["input_shapes"]
    input_dtypes = abi["input_dtypes"]
    outputs = abi["output_shapes"]
    output_dtypes = abi["output_dtypes"]
    if native_type in {"MatMulV3", "QuantBatchMatmulV3"} and len(inputs) >= 2 \
            and inputs[0] and inputs[1] and outputs and outputs[0]:
        first, second, output = inputs[0], inputs[1], outputs[0]
        if len(first) != 2 or len(second) != 2 or len(output) != 2:
            return {"native_type": native_type, "work_unit": None,
                    "logical_work": None,
                    "reason": "logical_matrix_work_requires_rank_two_ABI"}
        m, n = int(output[0]), int(output[1])
        orientation = None
        if first[0] == m:
            k = int(first[1])
            orientation = "x1_mk"
        elif first[1] == m:
            k = int(first[0])
            orientation = "x1_km"
        else:
            return {"native_type": native_type, "work_unit": None,
                    "logical_work": None,
                    "reason": "logical_matrix_work_x1_does_not_match_output_m"}
        if second[0] == n and second[1] == k:
            b_layout = "n_k"
        elif second[0] == k and second[1] == n:
            b_layout = "k_n"
        else:
            return {"native_type": native_type, "work_unit": None,
                    "logical_work": None,
                    "reason": "logical_matrix_work_x2_does_not_match_output_n_and_k"}
        macs = m * n * k
        bytes_by_tensor = {}
        total_bytes = 0
        for index, (shape, dtype) in enumerate(zip(inputs, input_dtypes)):
            width = _CAPTURE_DTYPE_BYTES.get(dtype)
            if width is None or not shape:
                continue
            count = 1
            for dim in shape:
                count *= int(dim)
            bytes_by_tensor[f"input{index}"] = count * width
            total_bytes += count * width
        for index, (shape, dtype) in enumerate(zip(outputs, output_dtypes)):
            width = _CAPTURE_DTYPE_BYTES.get(dtype)
            if width is None or not shape:
                continue
            count = 1
            for dim in shape:
                count *= int(dim)
            bytes_by_tensor[f"output{index}"] = count * width
            total_bytes += count * width
        return {
            "native_type": native_type,
            "work_unit": "mac",
            "logical_work": int(macs),
            "formula": "M*N*K",
            "m": m, "n": n, "k": k,
            "x1_orientation": orientation,
            "x2_layout": b_layout,
            "logical_bytes": int(total_bytes),
            "logical_bytes_by_tensor": bytes_by_tensor,
            "logical_work_status": "derived_from_raw_ABI_shapes",
            "issued_work": None,
            "issued_work_status": "requires_selected_native_program_and_dispatch",
        }
    if native_type in {"RmsNorm", "SwiGlu", "DynamicMxQuant", "HcPost",
                       "Slice", "ScatterPaKvCache"} and inputs:
        shape = max(inputs, key=lambda item: (len(item or []), item or []))
        if shape:
            count = 1
            for dim in shape:
                count *= int(dim)
            total_bytes = 0
            for shapes, dtypes in ((inputs, input_dtypes),
                                   (outputs, output_dtypes)):
                for item, dtype in zip(shapes, dtypes):
                    width = _CAPTURE_DTYPE_BYTES.get(dtype)
                    if width is None or not item:
                        continue
                    elements = 1
                    for dim in item:
                        elements *= int(dim)
                    total_bytes += elements * width
            return {
                "native_type": native_type,
                "work_unit": "elements",
                "logical_work": int(count),
                "formula": "max_input_element_count",
                "logical_bytes": int(total_bytes),
                "logical_work_status": "derived_from_raw_ABI_shapes",
                "issued_work": None,
                "issued_work_status": "requires_selected_native_program_and_dispatch",
            }
    return {
        "native_type": native_type,
        "work_unit": None,
        "logical_work": None,
        "reason": "logical_work_not_derivable_from_capture_family",
        "issued_work": None,
    }


def join_capture_measurement_and_plan(measurement, plan):
    """Pair measured pipe times with the selected plan's owner/phase work."""
    if plan.get("status") != "predicted":
        return {
            "status": "unmatched",
            "reason": "native_plan_not_predicted:" + ",".join(
                str(item) for item in plan.get("reasons") or ()),
            "owner_phases": {},
            "resource_identified": False,
        }
    owner_phases = plan.get("owner_phases") or {}
    joins = []
    active = {}
    for name, value in (measurement.get("aic_pipe_busy_us") or {}).items():
        active[f"aic_{name}"] = value
    for name, value in (measurement.get("aiv_pipe_busy_us") or {}).items():
        active[f"aiv_{name}"] = value
    for pipe, measured_us in sorted(active.items()):
        owner_phase = (_PIPE_TO_OWNER_PHASE.get(pipe.removeprefix("aic_"))
                       if pipe.startswith("aic_") else None)
        if not owner_phase:
            joins.append({
                "pipe": pipe, "measured_us": measured_us,
                "owner": None, "phase": None,
                "issued_work": None, "issued_unit": None,
                "declared_work": None, "declared_unit": None,
                "status": "pipe_not_in_selected_native_plan",
            })
            continue
        owner, phase = owner_phase
        plan_owner = owner_phases.get(owner)
        if not plan_owner:
            joins.append({
                "pipe": pipe, "measured_us": measured_us,
                "owner": owner, "phase": phase,
                "issued_work": None, "issued_unit": None,
                "status": "owner_not_in_plan",
            })
            continue
        if owner == "cube":
            declared = plan_owner.get("declared_padded_macs")
            declared_unit = "mac"
            declared_basis = "declared_padded_16_alignment"
        else:
            declared = plan_owner.get("declared_bytes")
            declared_unit = "bytes"
            declared_basis = "declared_call_boundary_tensor_footprint"
        joins.append({
            "pipe": pipe, "measured_us": measured_us,
            "owner": owner, "phase": plan_owner.get("phase"),
            "issued_work": None,
            "issued_unit": None,
            "issued_status": "not_measured_by_capture",
            "declared_work": declared,
            "declared_unit": declared_unit if declared is not None else None,
            "declared_basis": declared_basis,
            "declared_source": plan.get("plan_basis"),
            "diagnostic_us_per_declared_unit": (
                measured_us / declared if declared else None),
            "training_eligible": False,
            "status": ("measured_pipe_with_declared_plan_work"
                       if declared is not None
                       else "plan_owner_has_no_declared_work"),
        })
    return {
        "status": "paired_diagnostic",
        "joins": joins,
        "unmapped_pipes": sorted(
            item["pipe"] for item in joins
            if item.get("status") == "pipe_not_in_selected_native_plan"),
        "owner_aggregation": measurement.get("owner_aggregation"),
        "resource_identified": False,
        "issued_work_measured": False,
        "reason": "measured_pipe_times_are_overlapping_joint_window_and_"
                  "owner_aggregation_is_not_declared",
        "training_eligible": False,
    }
