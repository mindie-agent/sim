"""Service-fitting capture I/O and feature extraction models.

[Role] Load profiling capture datasets and build structured fitting records
  for resource-service regression.
[Invariants] Explicit schema validation with zero silent record drops.
[Timeouts & Locks] Local file reads only; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

import math

from mindie_sim.observed_contracts.vocabulary import DTYPES as _OBSERVED_DTYPES


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


_BITS_PER_BYTE = 8
_CANONICAL_DTYPE_BITS = {
    "bf16": 16, "fp16": 16, "fp32": 32, "fp64": 64,
    "int8": 8, "uint8": 8, "int16": 16, "uint16": 16,
    "int32": 32, "uint32": 32, "int64": 64, "uint64": 64,
    "bool": 8, "fp8_e4m3": 8, "fp8_e5m2": 8, "fp8_e8m0": 8, "hif8": 8,
    "int4": 4, "uint4": 4, "fp4_e2m1": 4, "uint1": 1,
}
_SCATTER_MIN_INPUTS = 5
_SCATTER_OUTPUT_COUNT = 2
_SCATTER_SLOT_DTYPES = frozenset({"int32", "int64"})
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
_CAPTURE_QBMM_MIN_ARITY = 3
_CAPTURE_QBMM_ARITY = 4
_CAPTURE_NATIVE_MIN_ARITY = {
    "MatMulV3": _CAPTURE_MATMUL_ARITY,
    "QuantBatchMatmulV3": _CAPTURE_QBMM_MIN_ARITY,
}
_FRACTAL_NZ_RANK = 4
_FRACTAL_NZ_ROW_TILE = 16
_MX_MICROSCALE_BLOCK = 64
_MX_SCALE_MIN_RANK = 3
_FRACTAL_NZ_C0_BY_DTYPE = {
    "fp8_e4m3": frozenset({32}),
    "fp8_e5m2": frozenset({32}),
    "fp8_e8m0": frozenset({32}),
    "hif8": frozenset({32}),
    "int8": frozenset({32}),
    "uint8": frozenset({32}),
    "bf16": frozenset({16}),
    "fp16": frozenset({16}),
    "int16": frozenset({16}),
    "uint16": frozenset({16}),
    "fp32": frozenset({8}),
    "int32": frozenset({8}),
    "uint32": frozenset({8}),
    "int4": frozenset({32, 64}),
    "uint4": frozenset({32, 64}),
    "fp4_e2m1": frozenset({32, 64}),
}
_FRACTAL_NZ_VALID_C0_TILES = frozenset({8, 16, 32, 64})


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
        text = clean(value)
        if not text:
            return []
        return [item.strip() for item in text.split(";")]

    def shaped(values):
        parsed = []
        for item in values:
            if not item:
                parsed.append([])
                continue
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


def _is_fractal_nz(fmt):
    text = str(fmt or "").strip().upper()
    return text == "FRACTAL_NZ" or text.startswith("FRACTAL_NZ_")


def _expected_fractal_nz_c0(fmt, dtype):
    text = str(fmt or "").strip().upper()
    if text.startswith("FRACTAL_NZ_C0_"):
        suffix = text.removeprefix("FRACTAL_NZ_C0_")
        if suffix.isdigit() and int(suffix) > 0:
            return frozenset({int(suffix)})
        return frozenset()
    raw = str(dtype or "").strip()
    canonical = (
        _OBSERVED_DTYPES.get(raw)
        or _OBSERVED_DTYPES.get(raw.upper())
        or _OBSERVED_DTYPES.get(raw.lower())
    )
    if canonical in _FRACTAL_NZ_C0_BY_DTYPE:
        return _FRACTAL_NZ_C0_BY_DTYPE[canonical]
    return _FRACTAL_NZ_VALID_C0_TILES


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _broadcast_batch_matches(a_batch, b_batch, out_batch):
    if len(a_batch) > len(out_batch) or len(b_batch) > len(out_batch):
        return False
    pad_a = [1] * (len(out_batch) - len(a_batch)) + [int(d) for d in a_batch]
    pad_b = [1] * (len(out_batch) - len(b_batch)) + [int(d) for d in b_batch]
    for da, db, dy in zip(pad_a, pad_b, (int(d) for d in out_batch)):
        if da == db:
            expected = da
        elif da == 1:
            expected = db
        elif db == 1:
            expected = da
        else:
            return False
        if expected != dy:
            return False
    return True


def _scale_weight_orientation(scale_shape, k, n):
    if not scale_shape:
        return None
    groups = _ceil_div(k, _MX_MICROSCALE_BLOCK)
    if len(scale_shape) >= _MX_SCALE_MIN_RANK and int(scale_shape[-1]) == 2:
        tail = (int(scale_shape[-_MX_SCALE_MIN_RANK]), int(scale_shape[-2]))
        if tail == (n, groups) and tail != (groups, n):
            return "n_k"
        if tail == (groups, n) and tail != (n, groups):
            return "k_n"
    if len(scale_shape) == _CAPTURE_MATMUL_ARITY:
        dims = (int(scale_shape[0]), int(scale_shape[1]))
        if dims[0] == n and dims[1] != n and dims[1] > 1:
            return "n_k"
        if dims[1] == n and dims[0] != n and dims[0] > 1:
            return "k_n"
    return None


def _validate_matrix_weight(weight_shape, k, n, x2_format, x2_dtype, scale_shape=None):
    is_nz = _is_fractal_nz(x2_format)
    scale_hint = _scale_weight_orientation(scale_shape, k, n)
    if len(weight_shape) == _CAPTURE_MATMUL_ARITY:
        nk_match = int(weight_shape[0]) == n and int(weight_shape[1]) == k
        kn_match = int(weight_shape[0]) == k and int(weight_shape[1]) == n
        if nk_match and kn_match:
            chosen = scale_hint or "n_k"
            return (f"fractal_nz_{chosen}" if is_nz else chosen), None
        if nk_match:
            return ("fractal_nz_n_k" if is_nz else "n_k"), None
        if kn_match:
            return ("fractal_nz_k_n" if is_nz else "k_n"), None
        return None, "logical_matrix_work_x2_does_not_match_output_n_and_k"
    if len(weight_shape) == _FRACTAL_NZ_RANK and is_nz:
        d1_tiles, d0_tiles, tile_0, tile_1 = (int(dim) for dim in weight_shape)
        allowed_c0 = _expected_fractal_nz_c0(x2_format, x2_dtype)
        if tile_0 != _FRACTAL_NZ_ROW_TILE or tile_1 not in allowed_c0:
            return None, "logical_matrix_work_invalid_fractal_nz_tile"
        kn_match = (d1_tiles == _ceil_div(n, tile_1)
                    and d0_tiles == _ceil_div(k, tile_0))
        nk_match = (d1_tiles == _ceil_div(k, tile_1)
                    and d0_tiles == _ceil_div(n, tile_0))
        if kn_match and nk_match:
            return f"fractal_nz_{scale_hint or 'k_n'}", None
        if kn_match:
            return "fractal_nz_k_n", None
        if nk_match:
            return "fractal_nz_n_k", None
        return None, "logical_matrix_work_x2_does_not_match_output_n_and_k"
    return None, "logical_matrix_work_requires_rank_two_ABI"


def _capture_matrix_geometry(abi):
    """Derive logical M/K/N from activation and output, validating weight layout."""
    inputs = abi["input_shapes"]
    outputs = abi["output_shapes"]
    if (len(inputs) < _CAPTURE_MATMUL_ARITY or not inputs[0] or not inputs[1]
            or not outputs or not outputs[0]):
        return None, "logical_matrix_work_requires_rank_two_ABI"
    first, second, output = list(inputs[0]), list(inputs[1]), list(outputs[0])
    if (len(first) < _CAPTURE_MATMUL_ARITY or len(second) < _CAPTURE_MATMUL_ARITY
            or len(output) < _CAPTURE_MATMUL_ARITY):
        return None, "logical_matrix_work_requires_rank_two_ABI"
    if any(int(dim) < 0 for shape in (first, second, output) for dim in shape):
        return None, "logical_matrix_work_requires_nonnegative_dimensions"
    input_formats = abi.get("input_formats") or ()
    output_formats = abi.get("output_formats") or ()
    x1_format = input_formats[0] if len(input_formats) > 0 else ""
    x2_format = input_formats[1] if len(input_formats) > 1 else ""
    y_format = output_formats[0] if len(output_formats) > 0 else ""
    if ((_is_fractal_nz(x1_format) and len(first) != _CAPTURE_MATMUL_ARITY)
            or (_is_fractal_nz(y_format) and len(output) != _CAPTURE_MATMUL_ARITY)):
        return None, "logical_matrix_work_requires_rank_two_ABI"
    out_batch = output[:-_CAPTURE_MATMUL_ARITY]
    a_batch = first[:-_CAPTURE_MATMUL_ARITY]
    m_inner, n = int(output[-2]), int(output[-1])
    if int(first[-2]) == m_inner:
        k = int(first[-1])
        orientation = "x1_mk"
    elif int(first[-1]) == m_inner:
        k = int(first[-2])
        orientation = "x1_km"
    else:
        return None, "logical_matrix_work_x1_does_not_match_output_m"
    input_dtypes = abi.get("input_dtypes") or ()
    x2_dtype = input_dtypes[1] if len(input_dtypes) > 1 else ""
    scale_shape = inputs[2] if len(inputs) > 2 else None
    is_nz = _is_fractal_nz(x2_format)
    splits = []
    if is_nz and len(second) >= _FRACTAL_NZ_RANK:
        splits.append((second[:-_FRACTAL_NZ_RANK], second[-_FRACTAL_NZ_RANK:]))
        if len(second) <= len(out_batch) + _CAPTURE_MATMUL_ARITY:
            splits.append((second[:-_CAPTURE_MATMUL_ARITY], second[-_CAPTURE_MATMUL_ARITY:]))
    else:
        splits.append((second[:-_CAPTURE_MATMUL_ARITY], second[-_CAPTURE_MATMUL_ARITY:]))
    b_layout = None
    last_reason = "logical_matrix_work_requires_rank_two_ABI"
    for b_batch, weight_shape in splits:
        if not _broadcast_batch_matches(a_batch, b_batch, out_batch):
            last_reason = "logical_matrix_work_x1_does_not_match_output_m"
            continue
        b_layout, reason = _validate_matrix_weight(
            weight_shape, k, n, x2_format, x2_dtype, scale_shape)
        if reason is None:
            break
        last_reason = reason
    if b_layout is None:
        return None, last_reason
    batch_rows = math.prod(int(dim) for dim in out_batch) if out_batch else 1
    m = batch_rows * m_inner
    return {
        "m": m,
        "k": k,
        "n": n,
        "macs": m * k * n,
        "x1_orientation": orientation,
        "x2_layout": b_layout,
    }, None


def _canonical_capture_dtype(dtype):
    raw = str(dtype or "").strip()
    if not raw:
        return None
    return (
        _OBSERVED_DTYPES.get(raw)
        or _OBSERVED_DTYPES.get(raw.upper())
        or _OBSERVED_DTYPES.get(raw.lower())
    )


def _shape_elements(shape):
    return math.prod(int(dim) for dim in shape)


def _tensor_nbytes(shape, dtype):
    canonical = _canonical_capture_dtype(dtype)
    bits = _CANONICAL_DTYPE_BITS.get(canonical)
    if bits is None:
        return None
    return (_shape_elements(shape) * bits + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE


def _capture_tensor_bytes(shapes, dtypes, prefix):
    """Return ``(total_or_none, by_tensor, unknown_entries)`` for captured tensors."""
    total = 0
    by_tensor = {}
    unknown = []
    for index, shape in enumerate(shapes):
        if shape is None:
            unknown.append({"tensor": f"{prefix}{index}", "dtype": None})
            continue
        if not shape:
            continue
        raw_dtype = dtypes[index] if index < len(dtypes) else ""
        nbytes = _tensor_nbytes(shape, raw_dtype)
        if nbytes is None:
            unknown.append({
                "tensor": f"{prefix}{index}",
                "dtype": raw_dtype or None,
            })
            continue
        by_tensor[f"{prefix}{index}"] = nbytes
        total += nbytes
    return (None if unknown else int(total)), by_tensor, unknown


def _capture_tensor(name, shape, dtype, layout=None):
    canonical = _canonical_capture_dtype(dtype)
    return {
        "name": name,
        "shape": [int(item) for item in shape],
        "dtype": canonical if canonical is not None else str(dtype).lower(),
        **({"layout": layout} if layout else {}),
    }


def _capture_bytes(shapes, dtypes):
    total, _, _ = _capture_tensor_bytes(shapes, dtypes, "tensor")
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
    min_arity = _CAPTURE_NATIVE_MIN_ARITY.get(native_type)
    arity = min(len(inputs), _CAPTURE_QBMM_ARITY) if native_type == "QuantBatchMatmulV3" else min_arity
    parsed = list(inputs[:arity]) if min_arity and len(inputs) >= min_arity else []
    if (min_arity is None or len(inputs) < min_arity or not outputs
            or any(shape is None for shape in parsed + list(outputs[:1]))):
        result["status"] = "unsupported"
        result["reasons"] = ["no_canonical_geometry_for_capture_family"]
        return result
    geometry_info, reason = _capture_matrix_geometry(abi)
    if geometry_info is None:
        result["status"] = "adapter_error"
        result["reasons"] = ["ValueError:" + reason]
        return result
    geometry = {
        "m": geometry_info["m"],
        "k": geometry_info["k"],
        "n": geometry_info["n"],
        "macs": geometry_info["macs"],
    }
    input_bytes, _, input_unknown = _capture_tensor_bytes(
        inputs, abi["input_dtypes"], "input")
    output_bytes, _, output_unknown = _capture_tensor_bytes(
        outputs, abi["output_dtypes"], "output")
    unknown_dtypes = input_unknown + output_unknown
    logical_bytes = (
        None if (input_bytes is None or output_bytes is None)
        else input_bytes + output_bytes
    )
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
            "logical_bytes": logical_bytes,
            "logical_bytes_status": (
                "incomplete_unknown_dtype" if unknown_dtypes else "complete"),
            **({"unknown_dtype_tensors": unknown_dtypes} if unknown_dtypes else {}),
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


def _is_positive_shape(shape):
    return bool(shape) and all(int(dim) > 0 for dim in shape)


def _is_nonnegative_shape(shape):
    return bool(shape) and all(int(dim) >= 0 for dim in shape)


def _resolve_scatter_slots(inputs, outputs):
    """Identify (key, key_cache, slot_mapping, value, value_cache) slot indices."""
    if (len(inputs) < _SCATTER_MIN_INPUTS
            or not 1 <= len(outputs) <= _SCATTER_OUTPUT_COUNT):
        return None
    out_k = outputs[0]
    out_v = outputs[1] if len(outputs) > 1 else []
    if not _is_positive_shape(inputs[0]) or not _is_positive_shape(out_k):
        return None
    if out_v != [] and not _is_positive_shape(out_v):
        return None
    # CANN graph order: (key, keyCacheRef, slotMapping, value, valueCacheRef)
    # ACLNN order:      (key, value, keyCache, valueCache, slotMapping)
    candidates = (
        (0, 1, 2, 3, 4),
        (0, 2, 4, 1, 3),
    )
    for key_idx, kcache_idx, slot_idx, val_idx, vcache_idx in candidates:
        if inputs[kcache_idx] != out_k or inputs[vcache_idx] != out_v:
            continue
        if not _is_positive_shape(inputs[slot_idx]):
            continue
        val_shape = inputs[val_idx]
        if out_v == []:
            if val_shape == []:
                return key_idx, kcache_idx, slot_idx, None, None
        elif _is_positive_shape(val_shape):
            return key_idx, kcache_idx, slot_idx, val_idx, vcache_idx
    return None


def _scatter_pa_kv_cache_work(abi):
    """Derive active K/V scatter work and active transfer bytes from the contract."""
    inputs = abi["input_shapes"]
    input_dtypes = abi["input_dtypes"]
    outputs = abi["output_shapes"]
    output_dtypes = abi["output_dtypes"]
    slots = _resolve_scatter_slots(inputs, outputs)
    if slots is None:
        return {
            "native_type": "ScatterPaKvCache",
            "work_unit": None,
            "logical_work": None,
            "reason": "logical_scatter_work_requires_valid_kv_cache_contract",
            "issued_work": None,
        }
    if any(shape is None or bool(shape) for shape in inputs[_SCATTER_MIN_INPUTS:]):
        return {
            "native_type": "ScatterPaKvCache",
            "work_unit": None,
            "logical_work": None,
            "reason": "logical_scatter_work_requires_runtime_compress_lens",
            "issued_work": None,
        }
    key_idx, key_cache_idx, slot_idx, value_idx, value_cache_idx = slots
    key_shape, slot_shape = inputs[key_idx], inputs[slot_idx]
    value_shape = inputs[value_idx] if value_idx is not None else None
    if (len(key_shape) == 1 and len(slot_shape) == 1
            and int(key_shape[0]) % int(slot_shape[0]) != 0):
        return {
            "native_type": "ScatterPaKvCache",
            "work_unit": None,
            "logical_work": None,
            "reason": "logical_scatter_work_kv_not_divisible_by_slots",
            "issued_work": None,
        }
    if (value_shape is not None and len(value_shape) == 1 and len(slot_shape) == 1
            and int(value_shape[0]) % int(slot_shape[0]) != 0):
        return {
            "native_type": "ScatterPaKvCache",
            "work_unit": None,
            "logical_work": None,
            "reason": "logical_scatter_work_kv_not_divisible_by_slots",
            "issued_work": None,
        }
    slot_dtype_raw = input_dtypes[slot_idx] if slot_idx < len(input_dtypes) else ""
    slot_dtype = _canonical_capture_dtype(slot_dtype_raw)
    if slot_dtype is not None and slot_dtype not in _SCATTER_SLOT_DTYPES:
        return {
            "native_type": "ScatterPaKvCache",
            "work_unit": None,
            "logical_work": None,
            "reason": "logical_scatter_work_invalid_slot_dtype",
            "issued_work": None,
        }
    key_dtype_raw = input_dtypes[key_idx] if key_idx < len(input_dtypes) else ""
    kcache_dtype_raw = input_dtypes[key_cache_idx] if key_cache_idx < len(input_dtypes) else ""
    out_k_dtype_raw = output_dtypes[0] if len(output_dtypes) > 0 else kcache_dtype_raw
    if (_canonical_capture_dtype(key_dtype_raw) and _canonical_capture_dtype(kcache_dtype_raw)
            and _canonical_capture_dtype(key_dtype_raw) != _canonical_capture_dtype(kcache_dtype_raw)):
        return {
            "native_type": "ScatterPaKvCache",
            "work_unit": None,
            "logical_work": None,
            "reason": "logical_scatter_work_kv_cache_dtype_mismatch",
            "issued_work": None,
        }
    active_specs = [
        ("key_read", key_shape, key_dtype_raw),
        ("slot_mapping_read", slot_shape, slot_dtype_raw),
        ("key_write", key_shape, out_k_dtype_raw),
    ]
    logical_elements = _shape_elements(key_shape)
    formula = "key_elements"
    if value_shape is not None and value_cache_idx is not None:
        val_dtype_raw = input_dtypes[value_idx] if value_idx < len(input_dtypes) else ""
        vcache_dtype_raw = input_dtypes[value_cache_idx] if value_cache_idx < len(input_dtypes) else ""
        out_v_dtype_raw = output_dtypes[1] if len(output_dtypes) > 1 else vcache_dtype_raw
        if (_canonical_capture_dtype(val_dtype_raw) and _canonical_capture_dtype(vcache_dtype_raw)
                and _canonical_capture_dtype(val_dtype_raw) != _canonical_capture_dtype(vcache_dtype_raw)):
            return {
                "native_type": "ScatterPaKvCache",
                "work_unit": None,
                "logical_work": None,
                "reason": "logical_scatter_work_kv_cache_dtype_mismatch",
                "issued_work": None,
            }
        active_specs.extend([
            ("value_read", value_shape, val_dtype_raw),
            ("value_write", value_shape, out_v_dtype_raw),
        ])
        logical_elements += _shape_elements(value_shape)
        formula = "key_elements+value_elements"
    total_bytes = 0
    bytes_by_tensor = {}
    unknown = []
    for name, shape, raw_dtype in active_specs:
        nbytes = _tensor_nbytes(shape, raw_dtype)
        if nbytes is None:
            unknown.append({"tensor": name, "dtype": raw_dtype or None})
            continue
        bytes_by_tensor[name] = nbytes
        total_bytes += nbytes
    return {
        "native_type": "ScatterPaKvCache",
        "work_unit": "elements",
        "logical_work": int(logical_elements),
        "formula": formula,
        "logical_bytes": None if unknown else int(total_bytes),
        "logical_bytes_by_tensor": bytes_by_tensor,
        "logical_bytes_status": (
            "incomplete_unknown_dtype" if unknown else "complete"),
        **({"unknown_dtype_tensors": unknown} if unknown else {}),
        "logical_work_status": "derived_from_raw_ABI_shapes",
        "issued_work": None,
        "issued_work_status": "requires_selected_native_program_and_dispatch",
    }


def _slice_capture_work(abi):
    """Derive active Slice work and active transfer bytes from the output extent."""
    inputs = abi["input_shapes"]
    input_dtypes = abi["input_dtypes"]
    outputs = abi["output_shapes"]
    output_dtypes = abi["output_dtypes"]
    if (not inputs or not _is_nonnegative_shape(inputs[0])
            or len(outputs) != 1 or not _is_nonnegative_shape(outputs[0])
            or len(outputs[0]) != len(inputs[0])
            or any(int(out_dim) > int(src_dim)
                   for src_dim, out_dim in zip(inputs[0], outputs[0]))):
        return {
            "native_type": "Slice",
            "work_unit": None,
            "logical_work": None,
            "reason": "logical_slice_work_requires_valid_output_extent",
            "issued_work": None,
        }
    src_dtype_raw = input_dtypes[0] if input_dtypes else ""
    out_dtype_raw = output_dtypes[0] if output_dtypes else ""
    src_dtype = _canonical_capture_dtype(src_dtype_raw)
    out_dtype = _canonical_capture_dtype(out_dtype_raw)
    if src_dtype is not None and out_dtype is not None and src_dtype != out_dtype:
        return {
            "native_type": "Slice",
            "work_unit": None,
            "logical_work": None,
            "reason": "logical_slice_work_dtype_mismatch",
            "issued_work": None,
        }
    out_shape = outputs[0]
    read_bytes = _tensor_nbytes(out_shape, src_dtype_raw)
    write_bytes = _tensor_nbytes(out_shape, out_dtype_raw)
    desc_bytes, desc_by_tensor, desc_unknown = _capture_tensor_bytes(
        inputs[1:], input_dtypes[1:], "descriptor")
    bytes_by_tensor = {}
    unknown = list(desc_unknown)
    total_bytes = 0 if desc_bytes is None else desc_bytes
    if read_bytes is None:
        unknown.append({"tensor": "input0", "dtype": src_dtype_raw or None})
    else:
        bytes_by_tensor["input0_sliced"] = read_bytes
        total_bytes += read_bytes
    if write_bytes is None:
        unknown.append({"tensor": "output0", "dtype": out_dtype_raw or None})
    else:
        bytes_by_tensor["output0"] = write_bytes
        total_bytes += write_bytes
    bytes_by_tensor.update(desc_by_tensor)
    return {
        "native_type": "Slice",
        "work_unit": "elements",
        "logical_work": int(_shape_elements(out_shape)),
        "formula": "output_element_count",
        "logical_bytes": None if unknown else int(total_bytes),
        "logical_bytes_by_tensor": bytes_by_tensor,
        "logical_bytes_status": (
            "incomplete_unknown_dtype" if unknown else "complete"),
        **({"unknown_dtype_tensors": unknown} if unknown else {}),
        "logical_work_status": "derived_from_raw_ABI_shapes",
        "issued_work": None,
        "issued_work_status": "requires_selected_native_program_and_dispatch",
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
    min_arity = _CAPTURE_NATIVE_MIN_ARITY.get(native_type)
    if (min_arity is not None and len(inputs) >= min_arity
            and inputs[0] and inputs[1] and outputs and outputs[0]):
        geometry_info, reason = _capture_matrix_geometry(abi)
        if geometry_info is None:
            return {"native_type": native_type, "work_unit": None,
                    "logical_work": None, "reason": reason}
        in_bytes, in_by_tensor, in_unknown = _capture_tensor_bytes(
            inputs, input_dtypes, "input")
        out_bytes, out_by_tensor, out_unknown = _capture_tensor_bytes(
            outputs, output_dtypes, "output")
        unknown_dtypes = in_unknown + out_unknown
        total_bytes = (
            None if (in_bytes is None or out_bytes is None)
            else in_bytes + out_bytes
        )
        return {
            "native_type": native_type,
            "work_unit": "mac",
            "logical_work": int(geometry_info["macs"]),
            "formula": "M*N*K",
            "m": geometry_info["m"], "n": geometry_info["n"], "k": geometry_info["k"],
            "x1_orientation": geometry_info["x1_orientation"],
            "x2_layout": geometry_info["x2_layout"],
            "logical_bytes": total_bytes,
            "logical_bytes_by_tensor": {**in_by_tensor, **out_by_tensor},
            "logical_bytes_status": (
                "incomplete_unknown_dtype" if unknown_dtypes else "complete"),
            **({"unknown_dtype_tensors": unknown_dtypes} if unknown_dtypes else {}),
            "logical_work_status": "derived_from_raw_ABI_shapes",
            "issued_work": None,
            "issued_work_status": "requires_selected_native_program_and_dispatch",
        }
    if native_type == "ScatterPaKvCache":
        return _scatter_pa_kv_cache_work(abi)
    if native_type == "Slice":
        return _slice_capture_work(abi)
    if native_type in {"RmsNorm", "SwiGlu", "DynamicMxQuant", "HcPost"} and inputs:
        valid_shapes = [item for item in inputs if _is_positive_shape(item)]
        shape = max(valid_shapes, key=_shape_elements) if valid_shapes else None
        if shape:
            in_bytes, in_by_tensor, in_unknown = _capture_tensor_bytes(
                inputs, input_dtypes, "input")
            out_bytes, out_by_tensor, out_unknown = _capture_tensor_bytes(
                outputs, output_dtypes, "output")
            unknown_dtypes = in_unknown + out_unknown
            total_bytes = (
                None if (in_bytes is None or out_bytes is None)
                else in_bytes + out_bytes
            )
            return {
                "native_type": native_type,
                "work_unit": "elements",
                "logical_work": int(_shape_elements(shape)),
                "formula": "max_input_element_count",
                "logical_bytes": total_bytes,
                "logical_bytes_by_tensor": {**in_by_tensor, **out_by_tensor},
                "logical_bytes_status": (
                    "incomplete_unknown_dtype" if unknown_dtypes else "complete"),
                **({"unknown_dtype_tensors": unknown_dtypes} if unknown_dtypes else {}),
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
