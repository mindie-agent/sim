"""Authoritative SHA-256-verified ProbedAbiRegistry for CANN and non-CANN ops.

[Role] Load packaged per-SoC ABI catalogs (``data/abi/cann93_shipping_abi_{a2,a3,a5}.json``)
  covering BOTH CANN-package backends (``cann_aclnn``, ``hccl_mc2``) AND non-CANN
  backends (``vllm_ascend_csrc``, ``triton``, ``torch_npu``, ``catlass``,
  ``third_party_wheel``), verify SHA-256 seals and 4-field ``SourceRef`` provenance,
  and decide deterministic ``AbiSupportVerdict`` admissibility with zero fake fallbacks.
[Invariants]
  * Every loaded catalog snapshot passes ``AbiCatalogSnapshot.verify_seal`` against
    ``data/abi/abi_schema_manifest.json``.
  * Joint operand/output tuple checks forbid Cartesian dtype/layout/quant leakage.
  * Zero default-admit fallbacks, zero runtime uncalibrated states, zero model-name
    branches (AG014-clean), and fail-fast on ambiguous multi-entry matches.
[Timeouts & Locks] Bounded local JSON reads (<= 8 MiB per catalog); no locks,
  subprocesses, or device calls.
[Side-Effect Policy] Read-only packaged JSON inspection returning immutable objects.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
import importlib.metadata
import importlib.resources
import json
from pathlib import Path
import site
import sys
from typing import Any

from mindie_sim.schema.abi import (
    ABI_CATALOG_SCHEMA,
    HARDWARE_USABLE_ABI_BACKENDS,
    SOC_ARCHS,
    AbiCatalogSnapshot,
    AbiQueryKey,
    AbiShapeConstraints,
    AbiSupportVerdict,
    OperandPortSig,
    ProbedAbiEntry,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.stack import ManifestBinding, StackTrack, StackTrackSpec

_MAX_ABI_DOC_BYTES = 8 * 1024 * 1024
_DEFAULT_MANIFEST_FILENAME = "abi_schema_manifest.json"
_DTYPE_ALIASES: dict[str, str] = {
    "fp8_e4m3": "fp8_e4m3fn",
    "fp8_e4m3fn": "fp8_e4m3fn",
    "float8_e4m3fn": "fp8_e4m3fn",
    "hifloat8": "hifloat8",
    "hifp8": "hifloat8",
    "hif8": "hifloat8",
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float": "fp32",
    "fp4": "fp4_e2m1",
    "mxfp4": "fp4_e2m1",
    "fp4_e2m1": "fp4_e2m1",
}
_FP4_DTYPES = frozenset({"fp4_e2m1", "fp4", "mxfp4"})
_SPARSE_FLASH_MLA_IDS = frozenset({"SparseFlashMla", "aclnnSparseFlashMla"})
_MOE_ACT_AND_MUL_ABIS = frozenset({
    "aclnnGroupedMatmulSwigluQuant",
    "aclnnGroupedMatmulSwigluQuantV2",
    "aclnnDequantSwigluQuant",
    "aclnnSwiGluQuant",
    "aclnnDequantSituQuant",
    "aclnnClippedSwiglu",
    "aclnnMegaMoe",
    "dispatch_ffn_combine",
})
_PORT_SPEC_FIELDS_MAX = 6
_EVEN_DIVISOR = 2


def _norm_dtype(dtype: str) -> str:
    low = str(dtype or "").strip().lower()
    return _DTYPE_ALIASES.get(low, low)


def _norm_arch(arch: str) -> str:
    upper = str(arch or "").strip().upper()
    if upper not in SOC_ARCHS:
        raise ModelingError(f"abi_registry_unknown_arch:{arch!r}")
    return upper


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in pairs:
        if key in out:
            raise ModelingError(f"abi_json_duplicate_key:{key}")
        out[key] = val
    return out


def _read_abi_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_ABI_DOC_BYTES + 1)
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"abi_{label}_read_failed:{path.name}") from exc
    if len(raw) > _MAX_ABI_DOC_BYTES:
        raise ModelingError(f"abi_{label}_too_large:{path.name}")
    try:
        data = json.loads(raw, object_pairs_hook=_unique_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ModelingError(f"abi_{label}_json_invalid:{path.name}") from exc
    if not isinstance(data, dict):
        raise ModelingError(f"abi_{label}_mapping_required:{path.name}")
    return data


def default_abi_dir() -> Path:
    """Resolve packaged ``data/abi`` directory across checkout, wheel, and prefix installs."""
    checkout = Path(__file__).resolve().parents[2] / "data" / "abi"
    if (checkout / _DEFAULT_MANIFEST_FILENAME).is_file():
        return checkout
    try:
        pkg_root = importlib.resources.files("mindie_sim")
        candidate = Path(str(pkg_root)).parent / "data" / "abi"
        if (candidate / _DEFAULT_MANIFEST_FILENAME).is_file():
            return candidate
    except (ModuleNotFoundError, TypeError, OSError):
        pass
    try:
        dist = importlib.metadata.distribution("mindie-sim")
    except importlib.metadata.PackageNotFoundError:
        dist = None
    if dist is not None:
        for rel in dist.files or ():
            if rel.name == _DEFAULT_MANIFEST_FILENAME and "abi" in rel.parts:
                located = Path(dist.locate_file(rel)).resolve()
                if located.is_file():
                    return located.parent
        located = Path(dist.locate_file(f"data/abi/{_DEFAULT_MANIFEST_FILENAME}")).resolve()
        if located.is_file():
            return located.parent
    for base in (Path(sys.prefix), Path(site.getuserbase())):
        candidate = base / "data" / "abi"
        if (candidate / _DEFAULT_MANIFEST_FILENAME).is_file():
            return candidate
    return checkout


def _parse_source_ref(raw: Mapping[str, Any]) -> SourceRef:
    return SourceRef(
        repository=str(raw.get("repository", "")),
        revision=str(raw.get("revision", "")),
        relative_path=str(raw.get("relative_path", "")),
        source_sha256=str(raw.get("source_sha256", "")),
    )


def _parse_shape_constraints(raw: Mapping[str, Any]) -> AbiShapeConstraints:
    hr = raw.get("hidden_range", (0, 0))
    tr = raw.get("topk_range", (0, 0))
    return AbiShapeConstraints(
        k_min=int(raw.get("k_min", 0)),
        k_max_exclusive=int(raw.get("k_max_exclusive", 0)),
        n_max_inclusive=int(raw.get("n_max_inclusive", 0)),
        require_even_n=bool(raw.get("require_even_n", False)),
        require_even_k_for_fp4=bool(raw.get("require_even_k_for_fp4", False)),
        require_even_n_for_fp4_transb=bool(raw.get("require_even_n_for_fp4_transb", False)),
        require_m_divisible_by_ws=bool(raw.get("require_m_divisible_by_ws", False)),
        max_comm_bytes_formula=str(raw.get("max_comm_bytes_formula", "none")),
        hidden_allowed_set=frozenset(int(v) for v in raw.get("hidden_allowed_set", ())),
        hidden_range=(int(hr[0]), int(hr[1])),
        hidden_divisibility=int(raw.get("hidden_divisibility", 0)),
        inter_min=int(raw.get("inter_min", 0)),
        inter_max=int(raw.get("inter_max", 0)),
        inter_divisibility=int(raw.get("inter_divisibility", 0)),
        topk_range=(int(tr[0]), int(tr[1])),
        max_shared_experts_per_rank=int(raw.get("max_shared_experts_per_rank", -1)),
    )


def _parse_entry(raw: Mapping[str, Any]) -> ProbedAbiEntry:
    return ProbedAbiEntry(
        entry_id=str(raw["entry_id"]),
        abi_id=str(raw["abi_id"]),
        available_on_device=bool(raw["available_on_device"]),
        backend_kind=raw["backend_kind"],
        symbol_or_kernel=str(raw["symbol_or_kernel"]),
        device_arch=raw["device_arch"],
        stack_or_cann_version=str(raw["stack_or_cann_version"]),
        role=str(raw["role"]),
        dtype_tuple=tuple(tuple(str(tok) for tok in row) for row in raw["dtype_tuple"]),
        shape_constraints=_parse_shape_constraints(raw["shape_constraints"]),
        source_ref=_parse_source_ref(raw["source_ref"]),
        graph_motif=str(raw.get("graph_motif", "")),
        vllm_ascend_track=raw.get("vllm_ascend_track", "merged"),
        required_stack_pass_id=str(raw.get("required_stack_pass_id", "")),
        required_mechanism_id=str(raw.get("required_mechanism_id", "")),
        vllm_ascend_ref=str(raw.get("vllm_ascend_ref", "")),
        op_plugin_ref=str(raw.get("op_plugin_ref", "")),
        cann_check_params_ref=str(raw.get("cann_check_params_ref", "")),
        device_kernel_expansion=tuple(str(k) for k in raw.get("device_kernel_expansion", ())),
        allowed_output_tuples=tuple(
            tuple(str(tok) for tok in row) for row in raw.get("allowed_output_tuples", ())
        ),
        allowed_rank_placements=frozenset(raw.get("allowed_rank_placements", ("local",))),
        allowed_world_sizes=frozenset(int(w) for w in raw.get("allowed_world_sizes", (1,))),
        allowed_activations=frozenset(str(a) for a in raw.get("allowed_activations", ("none",))),
        allowed_model_scopes=frozenset(raw.get("allowed_model_scopes", ("target", "draft"))),
        allow_bias=bool(raw.get("allow_bias", True)),
        require_no_bias=bool(raw.get("require_no_bias", False)),
        probe_signature_sha256=str(raw.get("probe_signature_sha256", "")),
        probe_domain_note=str(raw.get("probe_domain_note", "")),
    )


def parse_abi_snapshot(raw: Mapping[str, Any]) -> AbiCatalogSnapshot:
    """Deserialize and seal-verify one ``AbiCatalogSnapshot`` document."""
    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list):
        raise ModelingError("abi_catalog_entries_list_required")
    entries = tuple(_parse_entry(item) for item in raw_entries)
    declared_sha = str(raw.get("payload_sha256", ""))
    snapshot = AbiCatalogSnapshot(
        device_arch=raw["device_arch"],
        sku_id=str(raw["sku_id"]),
        stack_or_cann_version=str(raw["stack_or_cann_version"]),
        entries=entries,
        schema_version=str(raw.get("schema_version", ABI_CATALOG_SCHEMA)),
        payload_sha256=declared_sha,
    )
    snapshot.verify_seal(declared_sha)
    return snapshot


def load_abi_snapshots(abi_dir: Path | None = None) -> dict[str, AbiCatalogSnapshot]:
    """Load and SHA-256-verify all sealed per-SoC ABI catalogs from ``abi_dir``."""
    directory = default_abi_dir() if abi_dir is None else Path(abi_dir)
    manifest = _read_abi_json(directory / _DEFAULT_MANIFEST_FILENAME, "manifest")
    if manifest.get("schema") != ABI_CATALOG_SCHEMA:
        raise ModelingError(f"abi_manifest_schema_mismatch:{manifest.get('schema')!r}")
    pattern = str(manifest.get("arch_manifest_pattern", "cann93_shipping_abi_{arch}.json"))
    raw_arch_files = manifest.get("arch_files")
    arch_files = raw_arch_files if isinstance(raw_arch_files, Mapping) else {}
    sealed = manifest.get("sealed_sha256")
    if not isinstance(sealed, Mapping):
        raise ModelingError("abi_manifest_sealed_sha256_missing")
    snapshots: dict[str, AbiCatalogSnapshot] = {}
    for arch in manifest.get("soc_archs", ("A2", "A3", "A5")):
        arch_key = _norm_arch(str(arch))
        expected_sha = str(sealed.get(arch_key, ""))
        rel_name = str(arch_files.get(arch_key) or pattern.format(arch=arch_key.lower()))
        if "/" in rel_name or "\\" in rel_name or rel_name in {"", ".", ".."}:
            raise ModelingError(f"abi_manifest_invalid_arch_file:{arch_key}:{rel_name!r}")
        snap_path = directory / rel_name
        snap = parse_abi_snapshot(_read_abi_json(snap_path, f"catalog_{arch_key.lower()}"))
        if snap.device_arch != arch_key:
            raise ModelingError(f"abi_catalog_arch_mismatch:{snap.device_arch}!={arch_key}")
        snap.verify_seal(expected_sha)
        snapshots[arch_key] = snap
    return snapshots


_MATMUL_ROLES = frozenset({
    "dense_matmul", "moe_grouped_matmul", "collective_gemm", "catlass_template",
})


def _match_port_token(token: str, port: OperandPortSig) -> bool:
    """Match a structured ``role:dtype:layout:quant:group:scale`` or plain dtype token."""
    parts = token.split(":")
    if len(parts) == 1:
        return _norm_dtype(parts[0]) == _norm_dtype(port.dtype)
    if len(parts) > _PORT_SPEC_FIELDS_MAX:
        return False
    # Fields: role, dtype, layout, quant_mode, group_shape, scale_dtype
    if parts[0] and port.role and parts[0].strip().lower() != port.role.strip().lower():
        return False
    if len(parts) >= 2 and parts[1] and _norm_dtype(parts[1]) != _norm_dtype(port.dtype):
        return False
    if len(parts) >= 3 and parts[2] and parts[2] != port.layout:
        return False
    if len(parts) >= 4 and parts[3] and parts[3].lower() != port.quant_mode.lower():
        return False
    if len(parts) >= 5 and parts[4]:
        expected_group = tuple(int(v) for v in parts[4].split("x") if v)
        if expected_group and expected_group != port.group_shape:
            return False
    if len(parts) >= 6 and parts[5] and _norm_dtype(parts[5]) != _norm_dtype(port.scale_dtype):
        return False
    return True


def _matches_port_row(spec_tuple: tuple[str, ...], ports: tuple[OperandPortSig, ...]) -> bool:
    if len(spec_tuple) != len(ports):
        return False
    return all(_match_port_token(tok, port) for tok, port in zip(spec_tuple, ports, strict=True))


def _matches_joint_io_signature(
    entry: ProbedAbiEntry,
    operands: tuple[OperandPortSig, ...],
    outputs: tuple[OperandPortSig, ...],
) -> tuple[bool, str]:
    """Match input AND output port tuples strictly 1:1 (zero Cartesian/broadcast leakage)."""
    in_rows = entry.dtype_tuple
    out_rows = entry.allowed_output_tuples
    if not out_rows:
        return False, f"rejected_dtype:missing_output_tuples:{entry.entry_id}"
    if len(out_rows) != len(in_rows):
        raise ModelingError(
            f"abi_registry_unpaired_io_tuples:{entry.entry_id}:in={len(in_rows)}!=out={len(out_rows)}"
        )
    matched_any_input = False
    for idx, in_spec in enumerate(in_rows):
        if not _matches_port_row(in_spec, operands):
            continue
        matched_any_input = True
        if _matches_port_row(out_rows[idx], outputs):
            return True, ""
    if not matched_any_input:
        return False, f"rejected_dtype:operand_tuple_mismatch:{entry.entry_id}"
    return False, f"rejected_dtype:output_tuple_mismatch:{entry.entry_id}"


def _matches_plain_dtype_tuple(
    allowed_tuples: tuple[tuple[str, ...], ...],
    requested: Sequence[str],
) -> bool:
    req_norm = tuple(_norm_dtype(item) for item in requested)
    for spec_tuple in allowed_tuples:
        if len(spec_tuple) != len(req_norm):
            continue
        spec_dtypes = tuple(
            _norm_dtype(tok.split(":")[1] if ":" in tok and len(tok.split(":")) >= 2 else tok)
            for tok in spec_tuple
        )
        if spec_dtypes == req_norm:
            return True
    return False


def _extract_mkn(query: AbiQueryKey) -> tuple[int, int, int]:
    """Derive ``(M, K, N)`` purely from typed operand/output shapes (zero heuristics)."""
    m_dim, k_dim, n_dim = 0, 0, 0
    if query.operands:
        first_shape = query.operands[0].shape
        if len(first_shape) >= 2:
            m_dim, k_dim = first_shape[0], first_shape[-1]
        elif len(first_shape) == 1:
            k_dim = first_shape[0]
    if len(query.operands) >= 2 and len(query.operands[1].shape) >= 2:
        w_port = query.operands[1]
        n_dim = w_port.shape[-2] if w_port.transposed else w_port.shape[-1]
    elif query.outputs and query.outputs[0].shape:
        n_dim = query.outputs[0].shape[-1]
    return m_dim, k_dim, n_dim


_ALLGATHER_MM_IDS = frozenset({"aclnnAllGatherMatmulV2", "aclnnAllGatherMatmul"})
_MMRS_IDS = frozenset({"aclnnMatmulReduceScatterV2", "aclnnMatmulReduceScatter"})
_BATCH_3D_MM_IDS = frozenset({"torch_npu.npu_transpose_batchmatmul"})
_STRICT_2D_MATMUL_ROLES = frozenset({"collective_gemm"})
_NO_TRANSA_ROLES = frozenset({"collective_gemm", "moe_grouped_matmul", "catlass_template"})
_RANK_2D = 2
_RANK_3D = 3
_INT32_MAX = (1 << 31) - 1
_CCU_MAX_COMM_BYTES = 63 * 256 * 1024 * 1024  # 15.75 GiB hardware CCU ceiling on Ascend 950
_BITS_PER_BYTE = 8
_DTYPE_BITS: dict[str, int] = {
    "fp4_e2m1": 4,
    "int4": 4,
    "fp8_e4m3fn": 8,
    "fp8_e5m2": 8,
    "hifloat8": 8,
    "int8": 8,
    "uint8": 8,
    "bool": 8,
    "bf16": 16,
    "fp16": 16,
    "int16": 16,
    "fp32": 32,
    "int32": 32,
    "fp64": 64,
    "int64": 64,
}


def _allow_empty_matmul_mn(entry: ProbedAbiEntry, a_dtype: str, b_dtype: str) -> bool:
    """Ascend 950 allows empty m/n (with k>0) on AGMMV2 and BF16/FP16 MMRSV2; A2/A3 forbid empty."""
    if entry.device_arch != "A5":
        return False
    if entry.abi_id in _ALLGATHER_MM_IDS:
        return True
    if entry.abi_id in _MMRS_IDS:
        return _norm_dtype(a_dtype) in {"bf16", "fp16"} and _norm_dtype(b_dtype) in {"bf16", "fp16"}
    return False


def _check_matmul_shape_equations(entry: ProbedAbiEntry, query: AbiQueryKey) -> str | None:
    """Enforce rank, transpose, empty-tensor, INT32 bounds, K, and output (M, N) matmul equations."""
    if entry.role not in _MATMUL_ROLES or len(query.operands) < 2:
        return None
    a_port, b_port = query.operands[0], query.operands[1]
    if entry.role in _NO_TRANSA_ROLES and a_port.transposed:
        return f"rejected_shape_or_align:transposed_x1_unsupported:{entry.abi_id}"

    is_batch_3d = entry.abi_id in _BATCH_3D_MM_IDS
    strict_a_roles = _STRICT_2D_MATMUL_ROLES | {"moe_grouped_matmul", "catlass_template"}
    allowed_a_ranks = {_RANK_3D} if is_batch_3d else ({_RANK_2D} if entry.role in strict_a_roles else {_RANK_2D, _RANK_3D})
    allowed_b_ranks = {_RANK_3D} if is_batch_3d else ({_RANK_2D} if entry.role in _STRICT_2D_MATMUL_ROLES else {_RANK_2D, _RANK_3D})
    # Reject 0-D scalars (shape == ()) and any unsupported operand rank unconditionally.
    if len(a_port.shape) not in allowed_a_ranks:
        return f"rejected_shape_or_align:x1_rank_invalid:{len(a_port.shape)}D_not_in_{sorted(allowed_a_ranks)}"
    if len(b_port.shape) not in allowed_b_ranks:
        return f"rejected_shape_or_align:x2_rank_invalid:{len(b_port.shape)}D_not_in_{sorted(allowed_b_ranks)}"

    a_m = a_port.shape[-1] if a_port.transposed else a_port.shape[-2]
    a_k = a_port.shape[-2] if a_port.transposed else a_port.shape[-1]
    b_k = b_port.shape[-1] if b_port.transposed else b_port.shape[-2]
    b_n = b_port.shape[-2] if b_port.transposed else b_port.shape[-1]
    if a_k <= 0 or b_k <= 0 or a_k != b_k:
        return f"rejected_shape_or_align:matmul_k_mismatch:a_k={a_k}!=b_k={b_k}"

    allow_empty_mn = _allow_empty_matmul_mn(entry, a_port.dtype, b_port.dtype)
    if not allow_empty_mn and (a_m <= 0 or b_n <= 0):
        return f"rejected_shape_or_align:empty_tensor_unsupported:{entry.abi_id}:m={a_m},n={b_n}"
    if entry.abi_id in (_ALLGATHER_MM_IDS | _MMRS_IDS) and (a_m > _INT32_MAX or b_n > _INT32_MAX):
        return f"rejected_shape_or_align:dim_exceeds_int32_max:m={a_m},n={b_n}"

    if not query.outputs:
        return None
    out_shape = query.outputs[0].shape
    allowed_out_ranks = {_RANK_3D} if is_batch_3d else ({_RANK_2D} if entry.role in strict_a_roles else {_RANK_2D, _RANK_3D})
    if len(out_shape) not in allowed_out_ranks:
        return f"rejected_shape_or_align:out_rank_invalid:{len(out_shape)}D_not_in_{sorted(allowed_out_ranks)}"
    out_m, out_n = out_shape[-2], out_shape[-1]
    if out_n != b_n or (not allow_empty_mn and out_n <= 0):
        return f"rejected_shape_or_align:matmul_n_mismatch:out_n={out_n}!=b_n={b_n}"
    ws = max(1, int(query.world_size))
    if entry.abi_id in _ALLGATHER_MM_IDS:
        expected_m = a_m * ws
        if out_m != expected_m or out_m > _INT32_MAX:
            return f"rejected_shape_or_align:agmm_out_m_mismatch:out_m={out_m}!=x1_m*ws={expected_m}"
        if len(query.outputs) >= 2:
            g_shape = query.outputs[1].shape
            if len(g_shape) != _RANK_2D or g_shape != (expected_m, a_k):
                return f"rejected_shape_or_align:agmm_gather_out_mismatch:{g_shape}!={(expected_m, a_k)}"
    elif entry.abi_id in _MMRS_IDS:
        if a_m % ws != 0:
            return f"rejected_shape_or_align:mmrs_m_not_divisible:m={a_m}%ws={ws}"
        expected_m = a_m // ws
        if out_m != expected_m:
            return f"rejected_shape_or_align:mmrs_out_m_mismatch:out_m={out_m}!=x1_m/ws={expected_m}"
    elif out_m != a_m:
        return f"rejected_shape_or_align:matmul_m_mismatch:out_m={out_m}!=a_m={a_m}"
    return None


def _check_comm_bytes_cap(entry: ProbedAbiEntry, query: AbiQueryKey, m_dim: int, k_dim: int, n_dim: int) -> str | None:
    """Enforce CANN CCU communication volume ceilings declared in ``max_comm_bytes_formula``."""
    formula = entry.shape_constraints.max_comm_bytes_formula
    if not formula or formula == "none" or query.rank_placement != "ub_ccu_intramachine":
        return None
    cap_bits = _CCU_MAX_COMM_BYTES * _BITS_PER_BYTE
    if formula == "ag_63x256MiB" and query.operands:
        x1_bits = _DTYPE_BITS.get(_norm_dtype(query.operands[0].dtype), 16)
        comm_bits = m_dim * k_dim * x1_bits * max(1, int(query.world_size))
        if comm_bits > cap_bits:
            return (
                f"rejected_shape_or_align:agmm_ccu_comm_bytes_exceeded:"
                f"{comm_bits // _BITS_PER_BYTE}>{_CCU_MAX_COMM_BYTES}"
            )
    elif formula == "mmrs_ccu_out_cap" and query.outputs:
        out_bits = _DTYPE_BITS.get(_norm_dtype(query.outputs[0].dtype), 16)
        comm_bits = m_dim * n_dim * out_bits
        if comm_bits > cap_bits:
            return (
                f"rejected_shape_or_align:mmrs_ccu_comm_bytes_exceeded:"
                f"{comm_bits // _BITS_PER_BYTE}>{_CCU_MAX_COMM_BYTES}"
            )
    return None


def _check_op_specific_shapes(entry: ProbedAbiEntry, query: AbiQueryKey) -> str | None:
    """Enforce operator-specific rank/shape invariants grounded in C++ CheckParams."""
    if entry.abi_id == "torch_npu.npu_moe_gating_top_k" and query.operands:
        x_shape = query.operands[0].shape
        if len(x_shape) != _RANK_2D:
            return f"rejected_shape_or_align:moe_gating_x_must_be_2d:got={len(x_shape)}D"
        if len(query.operands) >= 2:
            b_shape = query.operands[1].shape
            if len(b_shape) != 1:
                return f"rejected_shape_or_align:moe_gating_bias_must_be_1d:got={b_shape}"
            if b_shape[0] != x_shape[1]:
                return f"rejected_shape_or_align:moe_gating_bias_width_mismatch:{b_shape[0]}!={x_shape[1]}"
    if entry.abi_id == "aclnnTextGatedResidual" and len(query.operands) >= 3:
        x_s, wd_s, wu_s = query.operands[0].shape, query.operands[1].shape, query.operands[2].shape
        if len(x_s) != _RANK_3D:
            return f"rejected_shape_or_align:text_gated_residual_x_must_be_3d:got={len(x_s)}D"
        if len(wd_s) != _RANK_2D:
            return f"rejected_shape_or_align:text_gated_residual_wdown_must_be_2d:got={len(wd_s)}D"
        if len(wu_s) != _RANK_2D:
            return f"rejected_shape_or_align:text_gated_residual_wup_must_be_2d:got={len(wu_s)}D"
        if wd_s[1] != x_s[-1] or wu_s[0] != x_s[-1] or wd_s[0] != wu_s[1]:
            return f"rejected_shape_or_align:text_gated_residual_dim_mismatch:x={x_s},wd={wd_s},wu={wu_s}"
    return None


def _check_shape_constraints(
    entry: ProbedAbiEntry, query: AbiQueryKey,
) -> str | None:
    mm_err = _check_matmul_shape_equations(entry, query)
    if mm_err is not None:
        return mm_err
    op_err = _check_op_specific_shapes(entry, query)
    if op_err is not None:
        return op_err
    sc = entry.shape_constraints
    m_dim, k_dim, n_dim = _extract_mkn(query)
    comm_err = _check_comm_bytes_cap(entry, query, m_dim, k_dim, n_dim)
    if comm_err is not None:
        return comm_err
    if sc.k_min > 0 and k_dim > 0 and k_dim < sc.k_min:
        return f"rejected_shape_or_align:k={k_dim}<k_min={sc.k_min}"
    if sc.k_max_exclusive > 0 and k_dim >= sc.k_max_exclusive:
        return f"rejected_shape_or_align:k={k_dim}>=k_max={sc.k_max_exclusive}"
    if sc.n_max_inclusive > 0 and n_dim > sc.n_max_inclusive:
        return f"rejected_shape_or_align:n={n_dim}>n_max={sc.n_max_inclusive}"
    if sc.require_even_n and n_dim > 0 and (n_dim % _EVEN_DIVISOR) != 0:
        return f"rejected_shape_or_align:odd_n={n_dim}"
    has_fp4 = any(_norm_dtype(o.dtype) in _FP4_DTYPES for o in query.operands)
    if sc.require_even_k_for_fp4 and has_fp4 and k_dim > 0 and (k_dim % _EVEN_DIVISOR) != 0:
        return f"rejected_shape_or_align:fp4_odd_k={k_dim}"
    if (sc.require_even_n_for_fp4_transb and has_fp4
            and len(query.operands) >= 2 and query.operands[1].transposed
            and n_dim > 0 and (n_dim % _EVEN_DIVISOR) != 0):
        return f"rejected_shape_or_align:fp4_transb_odd_n={n_dim}"
    if sc.require_m_divisible_by_ws and m_dim > 0 and (m_dim % query.world_size) != 0:
        return f"rejected_shape_or_align:m={m_dim}_not_divisible_by_ws={query.world_size}"
    if sc.hidden_allowed_set and k_dim > 0 and k_dim not in sc.hidden_allowed_set:
        return f"rejected_shape_or_align:hidden={k_dim}_not_in_{sorted(sc.hidden_allowed_set)}"
    if sc.hidden_range != (0, 0) and k_dim > 0:
        lo, hi = sc.hidden_range
        if k_dim < lo or (hi > 0 and k_dim > hi):
            return f"rejected_shape_or_align:hidden={k_dim}_outside_{sc.hidden_range}"
    if sc.hidden_divisibility > 0 and k_dim > 0 and (k_dim % sc.hidden_divisibility) != 0:
        return f"rejected_shape_or_align:hidden={k_dim}_not_div_by_{sc.hidden_divisibility}"
    if sc.inter_min > 0 and n_dim > 0 and n_dim < sc.inter_min:
        return f"rejected_shape_or_align:inter={n_dim}<inter_min={sc.inter_min}"
    if sc.inter_max > 0 and n_dim > sc.inter_max:
        return f"rejected_shape_or_align:inter={n_dim}>inter_max={sc.inter_max}"
    if sc.inter_divisibility > 0 and n_dim > 0 and (n_dim % sc.inter_divisibility) != 0:
        return f"rejected_shape_or_align:inter={n_dim}_not_div_by_{sc.inter_divisibility}"
    if sc.topk_range != (0, 0) and query.top_k > 0:
        lo, hi = sc.topk_range
        if query.top_k < lo or (hi > 0 and query.top_k > hi):
            return f"rejected_shape_or_align:topk={query.top_k}_outside_{sc.topk_range}"
    if sc.max_shared_experts_per_rank >= 0 and query.shared_experts_per_rank > sc.max_shared_experts_per_rank:
        return (
            f"rejected_shape_or_align:shared_experts={query.shared_experts_per_rank}"
            f">max={sc.max_shared_experts_per_rank}"
        )
    return None


def _check_m10_legality(query: AbiQueryKey) -> str | None:
    """Enforce hardware-verified M10 structural legality constraints."""
    if query.has_kpool_indexer and query.dcp_size != 1:
        return f"rejected_shape_or_align:m10_kpool_requires_dcp_eq_1:dcp={query.dcp_size}"
    if (query.abi_id in _SPARSE_FLASH_MLA_IDS
            and query.soc_arch in {"A2", "A3"}
            and _norm_dtype(query.kv_cache_dtype) == "int8"):
        return f"rejected_dtype:m10_sparse_flash_mla_{query.soc_arch}_forbids_int8_kv"
    if (query.activation_kind == "relu2"
            and not query.is_act_and_mul
            and query.abi_id in _MOE_ACT_AND_MUL_ABIS):
        return "rejected_dtype:m10_moe_requires_is_act_and_mul"
    return None


def _query_dtype_summary(query: AbiQueryKey) -> tuple[str, str, str]:
    act_dtype = query.operands[0].dtype if query.operands else "bf16"
    wt_dtype = query.operands[1].dtype if len(query.operands) >= 2 else act_dtype
    quant_mode = query.operands[0].quant_mode if query.operands else "dense"
    return act_dtype, wt_dtype, quant_mode


def _find_manifest_binding(
    entry: ProbedAbiEntry, bindings: tuple[ManifestBinding, ...],
) -> ManifestBinding | None:
    for binding in bindings:
        if binding.mechanism_id == entry.required_mechanism_id:
            return binding
    return None


class ProbedAbiRegistry:
    """Read-only sealed registry of CANN and non-CANN hardware-usable Ascend ABIs."""

    def __init__(self, snapshots: Mapping[str, AbiCatalogSnapshot]) -> None:
        if not snapshots:
            raise ModelingError("abi_registry_empty_snapshots")
        self._snapshots: dict[str, AbiCatalogSnapshot] = {}
        self._by_symbol: dict[tuple[str, str], list[ProbedAbiEntry]] = {}
        self._by_device_kernel: dict[tuple[str, str], list[ProbedAbiEntry]] = {}
        for raw_arch, snap in snapshots.items():
            arch = _norm_arch(raw_arch)
            if snap.device_arch != arch:
                raise ModelingError(f"abi_registry_arch_key_mismatch:{arch}!={snap.device_arch}")
            snap.verify_seal(snap.payload_sha256)
            self._snapshots[arch] = snap
            seen_ids: set[str] = set()
            for entry in snap.entries:
                if entry.entry_id in seen_ids:
                    raise ModelingError(f"abi_registry_duplicate_entry_id:{arch}:{entry.entry_id}")
                seen_ids.add(entry.entry_id)
                if entry.available_on_device and entry.backend_kind not in HARDWARE_USABLE_ABI_BACKENDS:
                    raise ModelingError(f"abi_registry_non_hw_backend:{entry.entry_id}")
                if entry.available_on_device and not entry.allowed_output_tuples:
                    raise ModelingError(f"abi_registry_missing_output_tuples:{entry.entry_id}")
                if entry.allowed_output_tuples and len(entry.allowed_output_tuples) != len(entry.dtype_tuple):
                    raise ModelingError(
                        f"abi_registry_unpaired_io_tuples:{entry.entry_id}:"
                        f"in={len(entry.dtype_tuple)}!=out={len(entry.allowed_output_tuples)}"
                    )
                self._index_entry(arch, entry)

    def _index_entry(self, arch: str, entry: ProbedAbiEntry) -> None:
        for key in {entry.abi_id, entry.symbol_or_kernel}:
            if key:
                self._by_symbol.setdefault((arch, key), []).append(entry)
        for kern in entry.device_kernel_expansion:
            if kern:
                self._by_device_kernel.setdefault((arch, kern), []).append(entry)

    @classmethod
    def from_directory(cls, abi_dir: Path | None = None) -> ProbedAbiRegistry:
        return cls(load_abi_snapshots(abi_dir))

    @property
    def archs(self) -> frozenset[str]:
        return frozenset(self._snapshots)

    def get_snapshot(self, device_arch: str) -> AbiCatalogSnapshot:
        arch = _norm_arch(device_arch)
        snap = self._snapshots.get(arch)
        if snap is None:
            raise ModelingError(f"abi_registry_arch_not_loaded:{arch}")
        return snap

    def lookup(
        self,
        symbol: str,
        device_arch: str,
        *,
        version: str | None = None,
        role: str | None = None,
        dtype_tuple: Sequence[str] | None = None,
        available_only: bool = True,
    ) -> tuple[ProbedAbiEntry, ...]:
        """Return all entries keyed by ``(symbol, device_arch, version, role, dtype_tuple)``."""
        arch = _norm_arch(device_arch)
        candidates = list(self._by_symbol.get((arch, symbol), ()))
        out: list[ProbedAbiEntry] = []
        seen: set[str] = set()
        for entry in candidates:
            if entry.entry_id in seen:
                continue
            seen.add(entry.entry_id)
            if available_only and not entry.available_on_device:
                continue
            if version is not None and entry.stack_or_cann_version != version:
                continue
            if role is not None and entry.role != role:
                continue
            if dtype_tuple is not None and not _matches_plain_dtype_tuple(entry.dtype_tuple, dtype_tuple):
                continue
            out.append(entry)
        return tuple(out)

    def is_shipping_available(
        self,
        symbol: str,
        device_arch: str,
        *,
        role: str | None = None,
        dtype_tuple: Sequence[str] | None = None,
        activation: str | None = None,
    ) -> bool:
        """Return True iff a sealed hardware-usable merged shipping entry matches."""
        matches = self.lookup(
            symbol, device_arch, role=role, dtype_tuple=dtype_tuple, available_only=True,
        )
        act_norm = activation.strip().lower() if activation else None
        for entry in matches:
            if entry.vllm_ascend_track != "merged":
                continue
            if act_norm is not None and act_norm not in entry.allowed_activations:
                continue
            return True
        return False

    def device_kernel_expansion(self, symbol: str, device_arch: str) -> tuple[str, ...]:
        """Return sealed 1-to-many device kernel expansion for host API ``symbol`` on ``device_arch``."""
        arch = _norm_arch(device_arch)
        matches = self.lookup(symbol, arch, available_only=True)
        for entry in matches:
            if entry.symbol_or_kernel == symbol or entry.abi_id == symbol:
                return entry.device_kernel_expansion
        return matches[0].device_kernel_expansion if matches else ()

    def known_device_kernels(self, device_arch: str) -> frozenset[str]:
        """Return all available L0 device kernel names sealed for ``device_arch``."""
        snap = self.get_snapshot(device_arch)
        return frozenset(
            k for entry in snap.entries if entry.available_on_device
            for k in entry.device_kernel_expansion
        )

    def _reject(
        self,
        query: AbiQueryKey,
        snapshot: AbiCatalogSnapshot,
        reason: str,
        *,
        entry: ProbedAbiEntry | None = None,
    ) -> AbiSupportVerdict:
        act_dtype, wt_dtype, quant_mode = _query_dtype_summary(query)
        return AbiSupportVerdict(
            abi_id=query.abi_id,
            status="rejected_abi",
            available_on_device=bool(entry.available_on_device) if entry is not None else False,
            backend_kind=entry.backend_kind if entry is not None else "logical",
            device_arch=query.soc_arch,
            stack_or_cann_version=snapshot.stack_or_cann_version,
            act_dtype=act_dtype,
            weight_dtype=wt_dtype,
            quant_mode=quant_mode,
            probe_manifest_sha256=snapshot.payload_sha256,
            engine_speedup_verified=False,
            query_digest=query.canonical_digest(),
            matched_entry_id=entry.entry_id if entry is not None else "",
            symbol_or_kernel=entry.symbol_or_kernel if entry is not None else "",
            pr_track=query.pr_track,
            rejection_reason=reason,
            source_ref=entry.source_ref if entry is not None else None,
            vllm_ascend_ref=entry.vllm_ascend_ref if entry is not None else "",
            op_plugin_ref=entry.op_plugin_ref if entry is not None else "",
            cann_check_params_ref=entry.cann_check_params_ref if entry is not None else "",
            device_kernel_expansion=entry.device_kernel_expansion if entry is not None else (),
        )

    def _evaluate_entry(
        self,
        entry: ProbedAbiEntry,
        query: AbiQueryKey,
        *,
        effective_track: StackTrack,
        enabled_passes: frozenset[str],
        manifest_bindings: tuple[ManifestBinding, ...],
    ) -> tuple[str | None, str]:
        """Return ``(rejection_reason_or_none, proposed_pr_ref)`` for one entry."""
        if not entry.available_on_device:
            return f"rejected_arch:{entry.entry_id}:{entry.probe_domain_note or 'unavailable'}", ""
        if query.model_scope not in entry.allowed_model_scopes:
            return f"rejected_arch:model_scope={query.model_scope}", ""
        ok_io, io_err = _matches_joint_io_signature(entry, query.operands, query.outputs)
        if not ok_io:
            return io_err, ""
        if query.activation_kind != "none" and query.activation_kind not in entry.allowed_activations:
            return f"rejected_dtype:activation={query.activation_kind}", ""
        if query.has_bias and (entry.require_no_bias or not entry.allow_bias):
            return f"rejected_dtype:bias_forbidden:{entry.entry_id}", ""
        if query.rank_placement not in entry.allowed_rank_placements:
            return f"rejected_rank_domain:placement={query.rank_placement}", ""
        if query.world_size not in entry.allowed_world_sizes:
            return f"rejected_rank_domain:world_size={query.world_size}", ""
        shape_err = _check_shape_constraints(entry, query)
        if shape_err is not None:
            return shape_err, ""
        if entry.required_stack_pass_id:
            if effective_track == "eager":
                return f"rejected_missing_pass:eager_forbids_{entry.required_stack_pass_id}", ""
            if entry.required_stack_pass_id not in enabled_passes:
                return f"rejected_missing_pass:{entry.required_stack_pass_id}", ""
        if entry.vllm_ascend_track == "unmerged_pr":
            if effective_track not in {"proposed", "target"}:
                return f"rejected_unmerged_pr:{entry.required_mechanism_id}", ""
            binding = _find_manifest_binding(entry, manifest_bindings)
            if binding is None:
                return f"rejected_unmerged_pr:missing_manifest_binding:{entry.required_mechanism_id}", ""
            return None, binding.pr_url or binding.revision
        return None, ""

    def verify(
        self,
        query: AbiQueryKey,
        *,
        track_spec: StackTrackSpec | None = None,
        manifest_bindings: tuple[ManifestBinding, ...] = (),
        engine_speedup_verified: bool = False,
    ) -> AbiSupportVerdict:
        """Deterministically admit or reject ``query`` against the sealed per-SoC catalog."""
        snapshot = self.get_snapshot(query.soc_arch)
        effective_track: StackTrack = track_spec.track if track_spec is not None else query.pr_track
        m10_err = _check_m10_legality(query)
        if m10_err is not None:
            return self._reject(query, snapshot, m10_err)

        candidates = list(self._by_symbol.get((query.soc_arch, query.abi_id), ()))
        if not candidates:
            return self._reject(query, snapshot, f"rejected_unprobed:{query.soc_arch}:{query.abi_id}")

        enabled_passes = set(query.enabled_stack_passes)
        if track_spec is not None:
            enabled_passes.update(track_spec.enabled_passes)
        pass_set = frozenset(enabled_passes)

        admitted: list[tuple[ProbedAbiEntry, str]] = []
        rejections: list[tuple[ProbedAbiEntry, str]] = []
        seen_ids: set[str] = set()
        for entry in candidates:
            if entry.entry_id in seen_ids:
                continue
            seen_ids.add(entry.entry_id)
            err, pr_ref = self._evaluate_entry(
                entry, query,
                effective_track=effective_track,
                enabled_passes=pass_set,
                manifest_bindings=manifest_bindings,
            )
            if err is None:
                admitted.append((entry, pr_ref))
            else:
                rejections.append((entry, err))

        if len(admitted) > 1:
            ids = ",".join(sorted(item[0].entry_id for item in admitted))
            raise ModelingError(f"abi_registry_ambiguous_match:{query.soc_arch}:{query.abi_id}:{ids}")
        if not admitted:
            first_entry, first_err = rejections[0]
            return self._reject(query, snapshot, first_err, entry=first_entry)

        entry, pr_ref = admitted[0]
        if effective_track == "eager":
            return self._reject(query, snapshot, f"rejected_eager_track:{entry.entry_id}", entry=entry)

        act_dtype, wt_dtype, quant_mode = _query_dtype_summary(query)
        status = "admitted_proposed_pr" if entry.vllm_ascend_track == "unmerged_pr" else "admitted_shipping"
        return AbiSupportVerdict(
            abi_id=query.abi_id,
            status=status,
            available_on_device=True,
            backend_kind=entry.backend_kind,
            device_arch=query.soc_arch,
            stack_or_cann_version=entry.stack_or_cann_version,
            act_dtype=act_dtype,
            weight_dtype=wt_dtype,
            quant_mode=quant_mode,
            probe_manifest_sha256=snapshot.payload_sha256,
            engine_speedup_verified=engine_speedup_verified,
            query_digest=query.canonical_digest(),
            matched_entry_id=entry.entry_id,
            symbol_or_kernel=entry.symbol_or_kernel,
            pr_track=effective_track,
            proposed_pr_ref=pr_ref,
            source_ref=entry.source_ref,
            vllm_ascend_ref=entry.vllm_ascend_ref,
            op_plugin_ref=entry.op_plugin_ref,
            cann_check_params_ref=entry.cann_check_params_ref,
            device_kernel_expansion=entry.device_kernel_expansion,
        )


@lru_cache(maxsize=1)
def default_abi_registry() -> ProbedAbiRegistry:
    """Return cached read-only ``ProbedAbiRegistry`` loaded from packaged ``data/abi``."""
    return ProbedAbiRegistry.from_directory()


__all__ = [
    "ProbedAbiRegistry",
    "default_abi_dir",
    "default_abi_registry",
    "load_abi_snapshots",
    "parse_abi_snapshot",
]
