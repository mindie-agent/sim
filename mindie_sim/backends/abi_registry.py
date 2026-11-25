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
import math
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
_PAIR_LEN = 2
_SHAPE_CONSTRAINT_KEYS = frozenset({
    "k_min",
    "k_max_exclusive",
    "n_max_inclusive",
    "require_even_n",
    "require_even_k_for_fp4",
    "require_even_n_for_fp4_transb",
    "require_m_divisible_by_ws",
    "max_comm_bytes_formula",
    "hidden_allowed_set",
    "hidden_range",
    "hidden_divisibility",
    "inter_min",
    "inter_max",
    "inter_divisibility",
    "topk_range",
    "max_shared_experts_per_rank",
    "head_dim_allowed_set",
    "rope_dim_allowed_set",
    "kv_heads_allowed_set",
    "max_experts",
    "required_layout_b",
    "allow_ungated_activation",
    "last_axis_divisibility",
    "rank_range",
})
_PERM_3D_TABLE: dict[int, tuple[int, int, int]] = {
    12: (0, 1, 2),
    21: (0, 2, 1),
    102: (1, 0, 2),
    120: (1, 2, 0),
    201: (2, 0, 1),
    210: (2, 1, 0),
}
_DEFAULT_PERM_X: tuple[int, int, int] = (0, 1, 2)
_TRANSPOSED_PERM_X: tuple[int, int, int] = (0, 2, 1)
_DEFAULT_PERM_Y: tuple[int, int, int] = (1, 0, 2)
_VALID_TBMM_PERM_X1 = frozenset({(0, 1, 2), (1, 0, 2), (0, 2, 1)})
_VALID_TBMM_PERM_X2 = frozenset({(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0)})
_VALID_TBMM_PERM_Y = frozenset({(1, 0, 2), (0, 1, 2)})
_TBMM_BATCH_CENTERED_X2 = frozenset({(1, 0, 2), (1, 2, 0)})
_RANK_4D = 4
_RANK_5D = 5
_UINT32_MAX = (1 << 32) - 1
_MX_BLOCK_SIZE = 32
_MX_SCALE_ALIGN = 2
_DUAL_LEVEL_MX_SCALE1_BLOCK = 512
_KV_COMPRESS_MIN_ARCH22 = 608
_KV_COMPRESS_MIN_ARCH35 = 584
_MQSMLA_QUANT_RECORD_DIMS: dict[str, int] = {
    "packed_kv_608": 608,
    "packed_kv_584": 584,
    "int8_csa_608": 608,
    "tq4_ori": 512,
    "tq4_cmp": 258,
}
_REJECTION_PREFIX_PRIORITY: tuple[tuple[str, int], ...] = (
    ("rejected_unmerged_pr:", 6),
    ("rejected_missing_pass:", 6),
    ("rejected_shape_or_align:", 5),
    ("rejected_rank_domain:", 4),
    ("rejected_dtype:bias_", 3),
    ("rejected_dtype:activation=", 3),
    ("rejected_dtype:output_tuple_mismatch:", 2),
    ("rejected_dtype:operand_tuple_mismatch:", 1),
)


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


def _req_bool(raw: Mapping[str, Any], key: str, default: bool | None = None) -> bool:
    if key not in raw:
        if default is None:
            raise ModelingError(f"abi_bool_field_missing:{key}")
        return default
    val = raw[key]
    if not isinstance(val, bool):
        raise ModelingError(f"abi_bool_field_invalid:{key}:{val!r}")
    return val


def _parse_int_set(raw: Mapping[str, Any], key: str) -> frozenset[int]:
    val = raw.get(key, ())
    if isinstance(val, (str, bytes)) or not isinstance(val, (Sequence, set, frozenset)):
        raise ModelingError(f"abi_shape_{key}_invalid:{val!r}")
    items: list[int] = []
    for item in val:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ModelingError(f"abi_shape_{key}_invalid:{item!r}")
        items.append(item)
    return frozenset(items)


def _parse_int_pair(raw: Mapping[str, Any], key: str) -> tuple[int, int]:
    val = raw.get(key, (0, 0))
    if isinstance(val, (str, bytes)) or not isinstance(val, Sequence) or len(val) != _PAIR_LEN:
        raise ModelingError(f"abi_shape_{key}_invalid:{val!r}")
    a, b = val[0], val[1]
    if isinstance(a, bool) or not isinstance(a, int) or isinstance(b, bool) or not isinstance(b, int):
        raise ModelingError(f"abi_shape_{key}_invalid:{val!r}")
    return (a, b)


def _parse_shape_constraints(raw: Mapping[str, Any]) -> AbiShapeConstraints:
    if not isinstance(raw, Mapping):
        raise ModelingError("abi_shape_constraints_mapping_required")
    extra = sorted(set(raw) - _SHAPE_CONSTRAINT_KEYS)
    if extra:
        raise ModelingError(f"abi_shape_unknown_keys:{extra}")
    return AbiShapeConstraints(
        k_min=raw.get("k_min", 0),
        k_max_exclusive=raw.get("k_max_exclusive", 0),
        n_max_inclusive=raw.get("n_max_inclusive", 0),
        require_even_n=_req_bool(raw, "require_even_n", False),
        require_even_k_for_fp4=_req_bool(raw, "require_even_k_for_fp4", False),
        require_even_n_for_fp4_transb=_req_bool(raw, "require_even_n_for_fp4_transb", False),
        require_m_divisible_by_ws=_req_bool(raw, "require_m_divisible_by_ws", False),
        max_comm_bytes_formula=str(raw.get("max_comm_bytes_formula", "none")),
        hidden_allowed_set=_parse_int_set(raw, "hidden_allowed_set"),
        hidden_range=_parse_int_pair(raw, "hidden_range"),
        hidden_divisibility=raw.get("hidden_divisibility", 0),
        inter_min=raw.get("inter_min", 0),
        inter_max=raw.get("inter_max", 0),
        inter_divisibility=raw.get("inter_divisibility", 0),
        topk_range=_parse_int_pair(raw, "topk_range"),
        max_shared_experts_per_rank=raw.get("max_shared_experts_per_rank", -1),
        head_dim_allowed_set=_parse_int_set(raw, "head_dim_allowed_set"),
        rope_dim_allowed_set=_parse_int_set(raw, "rope_dim_allowed_set"),
        kv_heads_allowed_set=_parse_int_set(raw, "kv_heads_allowed_set"),
        max_experts=raw.get("max_experts", 0),
        required_layout_b=str(raw.get("required_layout_b", "")),
        allow_ungated_activation=_req_bool(raw, "allow_ungated_activation", True),
        last_axis_divisibility=raw.get("last_axis_divisibility", 0),
        rank_range=_parse_int_pair(raw, "rank_range"),
    )


def _parse_entry(raw: Mapping[str, Any]) -> ProbedAbiEntry:
    return ProbedAbiEntry(
        entry_id=str(raw["entry_id"]),
        abi_id=str(raw["abi_id"]),
        available_on_device=_req_bool(raw, "available_on_device"),
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
        allow_bias=_req_bool(raw, "allow_bias", True),
        require_no_bias=_req_bool(raw, "require_no_bias", False),
        require_bias=_req_bool(raw, "require_bias", False),
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


_ACTIVATION_PORT_ROLES = ("x", "x1", "hidden_states")
_WEIGHT_PORT_ROLES = ("weight1", "w", "x2", "weight")
_HEAD_DIM_PORT_ROLES = ("q", "query", "x")
_KV_HEAD_PORT_ROLES = ("ori_kv", "k_cache", "kv_cache", "key", "k", "kv", "window")
_MOE_GATING_TOPK_IDS = frozenset({"torch_npu.npu_moe_gating_top_k", "aclnnMoeGatingTopK"})
_MLA_PROLOG_WEIGHT_UK_IDS = frozenset({"aclnnMlaPrologV3WeightNz", "aclnnMlaPrologV4WeightNz"})
_MEGAMOE_DISPATCH_MODE_PASSTHROUGH = 0
_MEGAMOE_DISPATCH_MODE_INT8 = 2
_MEGAMOE_DISPATCH_MODE_MXFP = 4
_MEGAMOE_MAX_EXPERTS_A5 = 2048
_MEGAMOE_MAX_EXPERT_PER_RANK_A5 = 1024
_MEGAMOE_MAX_EXPERTS_A23 = 1024
_MEGAMOE_MAX_EXPERT_PER_RANK_A23 = 128
_MEGAMOE_A5_HIDDEN_DIM_MIN = 512
_MEGAMOE_A5_HIDDEN_DIM_MAX = 8192
_MEGAMOE_A5_HIDDEN_DIM_ALIGN = 256
_MEGAMOE_A5_URMA_PLACEMENTS = frozenset({"urma_cross_supernode", "layered_ub_roce"})
_MEGAMOE_A5_URMA_ALLOWED_ACTIVATIONS = frozenset({"swiglu", "situglu", "gelu"})
_MEGAMOE_A5_PREQUANT_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e5m2", "fp4_e2m1"})
_MEGAMOE_A5_MXFP_WEIGHT_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e5m2", "fp4_e2m1"})
_MEGAMOE_A23_INT_WEIGHT_DTYPES = frozenset({"int8", "int4"})
_MEGAMOE_FLOAT_WEIGHT_DTYPES = frozenset({"bf16", "fp16"})
_MEGAMOE_PORT_IDX_X = 1
_MEGAMOE_PORT_IDX_TOPK_IDS = 2
_MEGAMOE_PORT_IDX_TOPK_WEIGHTS = 3
_MEGAMOE_PORT_IDX_WEIGHT1 = 4
_MEGAMOE_PORT_IDX_WEIGHT2 = 5
_MEGAMOE_MX_SCALE_ALIGN = 64
_MEGAMOE_MX_SCALE_TRAILING = 2
_MEGAMOE_A5_MAX_EP = 1024


def _resolve_port_by_roles(
    operands: tuple[OperandPortSig, ...],
    roles: tuple[str, ...],
    fallback_idx: int,
    *,
    exclude: OperandPortSig | None = None,
) -> OperandPortSig | None:
    """Resolve an operand port by canonical role name, falling back to positional index."""
    for role in roles:
        for port in operands:
            if port is not exclude and port.role.strip().lower() == role:
                return port
    if 0 <= fallback_idx < len(operands) and operands[fallback_idx] is not exclude:
        return operands[fallback_idx]
    return None


def _extract_mkn(query: AbiQueryKey) -> tuple[int, int, int]:
    """Derive ``(M, K, N)`` purely from typed operand/output shapes (zero heuristics)."""
    m_dim, k_dim, n_dim = 0, 0, 0
    act_port = _resolve_port_by_roles(query.operands, _ACTIVATION_PORT_ROLES, 0)
    wt_port = _resolve_port_by_roles(query.operands, _WEIGHT_PORT_ROLES, 1, exclude=act_port)
    if act_port is not None:
        act_shape = act_port.shape
        if len(act_shape) >= _RANK_2D:
            if act_port.transposed:
                m_dim, k_dim = act_shape[-1], act_shape[-2]
            else:
                m_dim, k_dim = act_shape[-2], act_shape[-1]
        elif len(act_shape) == 1:
            k_dim = act_shape[0]
    if wt_port is not None and len(wt_port.shape) >= _RANK_2D:
        if query.abi_id == "aclnnMegaMoe":
            raw_hidden = wt_port.shape[-2] if query.soc_arch == "A5" else wt_port.shape[-1]
            act_div = 1 if query.activation_kind == "gelu" else _EVEN_DIVISOR
            n_dim = raw_hidden // act_div if raw_hidden > 0 else 0
        elif wt_port.layout.startswith("FRACTAL_NZ") and len(wt_port.shape) == _RANK_5D:
            n_dim = wt_port.shape[1] * wt_port.shape[_RANK_4D]
        elif wt_port.layout.startswith("FRACTAL_NZ") and len(wt_port.shape) == _RANK_4D:
            n_dim = wt_port.shape[0] * wt_port.shape[_RANK_3D]
        else:
            n_dim = wt_port.shape[-2] if wt_port.transposed else wt_port.shape[-1]
    elif query.outputs and query.outputs[0].shape:
        n_dim = query.outputs[0].shape[-1]
    return m_dim, k_dim, n_dim


_ALLGATHER_MM_IDS = frozenset({"aclnnAllGatherMatmulV2", "aclnnAllGatherMatmul"})
_MMRS_IDS = frozenset({"aclnnMatmulReduceScatterV2", "aclnnMatmulReduceScatter"})
_BATCH_3D_MM_IDS = frozenset({"torch_npu.npu_transpose_batchmatmul", "aclnnTransposeBatchMatMul"})
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


def _decode_perm_3d(
    params: Mapping[str, float],
    key: str,
    default: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    if key not in params:
        return default
    raw = params[key]
    if raw != int(raw):
        return None
    return _PERM_3D_TABLE.get(int(raw))


def _check_transpose_batch_matmul_equations(
    entry: ProbedAbiEntry, query: AbiQueryKey,
) -> str | None:
    """Enforce rank-3 perm, arch, bias, batch_split_factor, and output equations for TransposeBatchMatMul."""
    a_port, b_port = query.operands[0], query.operands[1]
    if len(a_port.shape) != _RANK_3D:
        return f"rejected_shape_or_align:x1_rank_invalid:{len(a_port.shape)}D_not_in_[{_RANK_3D}]"
    if len(b_port.shape) != _RANK_3D:
        return f"rejected_shape_or_align:x2_rank_invalid:{len(b_port.shape)}D_not_in_[{_RANK_3D}]"
    params = dict(query.activation_params)
    px1 = _decode_perm_3d(params, "perm_x1", _TRANSPOSED_PERM_X if a_port.transposed else _DEFAULT_PERM_X)
    px2 = _decode_perm_3d(params, "perm_x2", _TRANSPOSED_PERM_X if b_port.transposed else _DEFAULT_PERM_X)
    py = _decode_perm_3d(params, "perm_y", _DEFAULT_PERM_Y)
    if px1 not in _VALID_TBMM_PERM_X1 or px2 not in _VALID_TBMM_PERM_X2 or py not in _VALID_TBMM_PERM_Y:
        return f"rejected_shape_or_align:tbmm_perm_invalid:{entry.abi_id}:px1={px1},px2={px2},py={py}"
    raw_split = params.get("batch_split_factor", 1.0)
    if raw_split < 1 or raw_split != int(raw_split):
        return f"rejected_shape_or_align:tbmm_split_invalid:{raw_split}"
    split = int(raw_split)
    is_batch_first_y = py == _DEFAULT_PERM_X
    if is_batch_first_y and (entry.device_arch != "A5" or split > 1):
        return f"rejected_shape_or_align:tbmm_bmn_requires_a5_no_split:{entry.device_arch}:split={split}"
    if (px1 == _TRANSPOSED_PERM_X or px2 in _TBMM_BATCH_CENTERED_X2) and (entry.device_arch != "A5" or not is_batch_first_y):
        return f"rejected_shape_or_align:tbmm_transa_or_bcentered_requires_a5_bmn:{entry.device_arch}"
    if query.has_bias and (entry.device_arch != "A5" or split > 1):
        return f"rejected_dtype:tbmm_bias_requires_a5_no_split:{entry.device_arch}:split={split}"
    a_b, a_m, a_k = a_port.shape[px1[0]], a_port.shape[px1[1]], a_port.shape[px1[2]]
    b_b, b_k, b_n = b_port.shape[px2[0]], b_port.shape[px2[1]], b_port.shape[px2[2]]
    if a_b <= 0 or a_m <= 0 or a_k <= 0 or b_n <= 0:
        return f"rejected_shape_or_align:empty_tensor_unsupported:{entry.abi_id}:b={a_b},m={a_m},k={a_k},n={b_n}"
    if a_b != b_b:
        return f"rejected_shape_or_align:tbmm_batch_mismatch:a_b={a_b}!=b_b={b_b}"
    if a_k != b_k:
        return f"rejected_shape_or_align:matmul_k_mismatch:a_k={a_k}!=b_k={b_k}"
    if split > 1 and (a_b * b_n) % split != 0:
        return f"rejected_shape_or_align:tbmm_split_indivisible:b*n={a_b * b_n}%split={split}"
    if not query.outputs:
        return None
    out_shape = query.outputs[0].shape
    if len(out_shape) != _RANK_3D:
        return f"rejected_shape_or_align:out_rank_invalid:{len(out_shape)}D_not_in_[{_RANK_3D}]"
    if split > 1:
        expected_out = (split, a_m, (a_b * b_n) // split)
    else:
        logical_out = (a_b, a_m, b_n)
        expected_out = (logical_out[py[0]], logical_out[py[1]], logical_out[py[2]])
    if out_shape != expected_out:
        return f"rejected_shape_or_align:tbmm_out_shape_mismatch:{out_shape}!={expected_out}"
    return None


def _check_matmul_shape_equations(entry: ProbedAbiEntry, query: AbiQueryKey) -> str | None:
    """Enforce rank, transpose, empty-tensor, INT32 bounds, K, and output (M, N) matmul equations."""
    if entry.role not in _MATMUL_ROLES or len(query.operands) < _RANK_2D:
        return None
    if entry.abi_id in _BATCH_3D_MM_IDS:
        return _check_transpose_batch_matmul_equations(entry, query)
    a_port, b_port = query.operands[0], query.operands[1]
    if entry.role in _NO_TRANSA_ROLES and a_port.transposed:
        return f"rejected_shape_or_align:transposed_x1_unsupported:{entry.abi_id}"

    strict_a_roles = _STRICT_2D_MATMUL_ROLES | {"moe_grouped_matmul", "catlass_template"}
    allowed_a_ranks = {_RANK_2D} if entry.role in strict_a_roles else {_RANK_2D, _RANK_3D}
    allowed_b_ranks = {_RANK_2D} if entry.role in _STRICT_2D_MATMUL_ROLES else {_RANK_2D, _RANK_3D}
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
    allowed_out_ranks = {_RANK_2D} if entry.role in strict_a_roles else {_RANK_2D, _RANK_3D}
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
        if len(query.outputs) >= _RANK_2D:
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


def _check_collective_shape_equations(entry: ProbedAbiEntry, query: AbiQueryKey) -> str | None:
    """Enforce standalone HCCL collective input/output world-size shape conservation."""
    if entry.role != "collective" or not query.operands or not query.outputs:
        return None
    in_shape = query.operands[0].shape
    out_shape = query.outputs[0].shape
    if not in_shape or not out_shape:
        return f"rejected_shape_or_align:collective_scalar_unsupported:{entry.abi_id}"
    ws = max(1, int(query.world_size))
    if entry.abi_id == "hcom_allGather":
        valid_outs = {(in_shape[0] * ws, *in_shape[1:]), (ws, *in_shape)}
        if out_shape not in valid_outs:
            return f"rejected_shape_or_align:allgather_out_shape_mismatch:{out_shape}_not_in_{sorted(valid_outs)}"
    elif entry.abi_id == "hcom_reduceScatter":
        valid_ins = {(out_shape[0] * ws, *out_shape[1:]), (ws, *out_shape)}
        if in_shape not in valid_ins:
            return f"rejected_shape_or_align:reducescatter_shape_mismatch:{in_shape}_vs_{out_shape}_ws={ws}"
    elif entry.abi_id in {"hcom_allReduce", "hcom_alltoall"}:
        if out_shape != in_shape:
            return f"rejected_shape_or_align:collective_shape_mismatch:{out_shape}!={in_shape}"
    elif entry.abi_id == "hcom_alltoallv":
        if len(out_shape) != len(in_shape) or out_shape[1:] != in_shape[1:]:
            return f"rejected_shape_or_align:alltoallv_trailing_shape_mismatch:{out_shape}_vs_{in_shape}"
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


def _check_megamoe_mode_and_params(
    entry: ProbedAbiEntry, query: AbiQueryKey, x_p: OperandPortSig, w1_p: OperandPortSig,
) -> str | None:
    """Validate CANN ``aclnnMegaMoe`` dispatch mode, active mask, bias, placement, and activation params."""
    x_dt = _norm_dtype(x_p.dtype)
    w1_dt = _norm_dtype(w1_p.dtype)
    if entry.device_arch == "A5":
        if query.has_active_mask:
            return "rejected_shape_or_align:megamoe_a5_forbids_active_mask"
        if query.world_size > _MEGAMOE_A5_MAX_EP:
            return f"rejected_rank_domain:megamoe_a5_ep_exceeds_max:ep={query.world_size}"
        if x_dt in _MEGAMOE_A5_PREQUANT_DTYPES:
            if query.dispatch_quant_mode != _MEGAMOE_DISPATCH_MODE_PASSTHROUGH:
                return f"rejected_shape_or_align:megamoe_a5_prequant_requires_mode_0:got={query.dispatch_quant_mode}"
            if query.rank_placement != "ub_ccu_intramachine":
                return f"rejected_rank_domain:megamoe_a5_prequant_requires_mte:placement={query.rank_placement}"
        elif x_dt == "bf16" and w1_dt in _MEGAMOE_A5_MXFP_WEIGHT_DTYPES:
            if query.dispatch_quant_mode != _MEGAMOE_DISPATCH_MODE_MXFP:
                return f"rejected_shape_or_align:megamoe_a5_mxfp_requires_mode_4:got={query.dispatch_quant_mode}"
        elif x_dt == "bf16" and w1_dt == "bf16":
            if query.dispatch_quant_mode != _MEGAMOE_DISPATCH_MODE_PASSTHROUGH:
                return f"rejected_shape_or_align:megamoe_a5_bf16_requires_mode_0:got={query.dispatch_quant_mode}"
        else:
            return f"rejected_dtype:megamoe_a5_unsupported_xw_pair:x={x_dt},w1={w1_dt}"
        if (query.rank_placement in _MEGAMOE_A5_URMA_PLACEMENTS
                and query.activation_kind not in _MEGAMOE_A5_URMA_ALLOWED_ACTIVATIONS):
            return f"rejected_dtype:megamoe_a5_urma_forbids_activation:{query.activation_kind}"
        params = dict(query.activation_params)
        if query.activation_kind == "situglu":
            beta = params.get("beta")
            if beta is None or not math.isfinite(beta) or beta <= 0.0:
                return f"rejected_shape_or_align:megamoe_a5_situglu_beta_invalid:{beta}"
            if "linear_beta" in params:
                lb = params["linear_beta"]
                if not math.isfinite(lb) or lb <= 0.0:
                    return f"rejected_shape_or_align:megamoe_a5_situglu_linear_beta_invalid:{lb}"
        elif query.activation_kind in {"swiglu", "swiglustep"} and "clamp" in params:
            cl = params["clamp"]
            if not math.isfinite(cl) or cl < 0.0:
                return f"rejected_shape_or_align:megamoe_a5_swiglu_clamp_invalid:{cl}"
        return None

    expected_dqm = (
        _MEGAMOE_DISPATCH_MODE_INT8 if w1_dt in _MEGAMOE_A23_INT_WEIGHT_DTYPES
        else (_MEGAMOE_DISPATCH_MODE_PASSTHROUGH if w1_dt in _MEGAMOE_FLOAT_WEIGHT_DTYPES else -1)
    )
    if expected_dqm < 0 or query.dispatch_quant_mode != expected_dqm:
        return (
            f"rejected_shape_or_align:megamoe_a23_dispatch_quant_mode_mismatch:"
            f"got={query.dispatch_quant_mode},expected={expected_dqm}"
        )
    if w1_dt == "int4" and not query.has_bias:
        return "rejected_dtype:megamoe_a23_int4_requires_bias"
    if w1_dt != "int4" and query.has_bias:
        return f"rejected_dtype:megamoe_a23_bias_forbidden_for_{w1_dt}"
    return None


def _check_megamoe_a5_scales(
    query: AbiQueryKey,
    x_p: OperandPortSig,
    w1_p: OperandPortSig,
    w2_p: OperandPortSig,
    per_rank: int,
    h_dim: int,
    w1_hidden: int,
    w2_inter: int,
) -> str | None:
    """Validate A5 pre-quantized ``x`` E8M0 scale shape and MX weight scale trailing-2 geometry."""
    x_dt = _norm_dtype(x_p.dtype)
    if x_dt in _MEGAMOE_A5_PREQUANT_DTYPES:
        exp_x_scale = (x_p.shape[0], (h_dim + _MX_BLOCK_SIZE - 1) // _MX_BLOCK_SIZE)
        scales_p = _resolve_port_by_roles(query.operands, ("scales",), -1)
        actual_x_scale = scales_p.shape if scales_p is not None else x_p.scale_shape
        if actual_x_scale != exp_x_scale:
            return f"rejected_shape_or_align:megamoe_a5_x_scale_shape_mismatch:{actual_x_scale}!={exp_x_scale}"
    if _norm_dtype(w1_p.dtype) in _MEGAMOE_A5_MXFP_WEIGHT_DTYPES:
        prefix = (per_rank,) if len(w1_p.shape) == _RANK_3D else ()
        exp_w1_s = (
            *prefix, w1_hidden,
            (h_dim + _MEGAMOE_MX_SCALE_ALIGN - 1) // _MEGAMOE_MX_SCALE_ALIGN,
            _MEGAMOE_MX_SCALE_TRAILING,
        )
        exp_w2_s = (
            *prefix, h_dim,
            (w2_inter + _MEGAMOE_MX_SCALE_ALIGN - 1) // _MEGAMOE_MX_SCALE_ALIGN,
            _MEGAMOE_MX_SCALE_TRAILING,
        )
        if w1_p.scale_shape != exp_w1_s:
            return f"rejected_shape_or_align:megamoe_a5_w1_scale_shape_mismatch:{w1_p.scale_shape}!={exp_w1_s}"
        if w2_p.scale_shape != exp_w2_s:
            return f"rejected_shape_or_align:megamoe_a5_w2_scale_shape_mismatch:{w2_p.scale_shape}!={exp_w2_s}"
    return None


def _check_megamoe_constraints(entry: ProbedAbiEntry, query: AbiQueryKey) -> str | None:
    """Enforce CANN 9.3 ``aclnnMegaMoe`` coupled dispatch/mask/placement/expert/weight invariants."""
    ws = max(1, int(query.world_size))
    is_a5 = entry.device_arch == "A5"
    max_e = _MEGAMOE_MAX_EXPERTS_A5 if is_a5 else _MEGAMOE_MAX_EXPERTS_A23
    max_per_rank = _MEGAMOE_MAX_EXPERT_PER_RANK_A5 if is_a5 else _MEGAMOE_MAX_EXPERT_PER_RANK_A23
    if query.moe_expert_num <= 0:
        return "rejected_shape_or_align:megamoe_expert_num_missing"
    if query.moe_expert_num < ws or query.moe_expert_num > max_e or (query.moe_expert_num % ws) != 0:
        return f"rejected_shape_or_align:megamoe_experts_indivisible:E={query.moe_expert_num},ep={ws}"
    per_rank = query.moe_expert_num // ws
    if per_rank < 1 or per_rank > max_per_rank:
        return f"rejected_shape_or_align:megamoe_experts_per_rank_out_of_range:per_rank={per_rank},max={max_per_rank}"

    x_p = _resolve_port_by_roles(query.operands, ("x",), _MEGAMOE_PORT_IDX_X)
    tk_ids_p = _resolve_port_by_roles(query.operands, ("topk_ids",), _MEGAMOE_PORT_IDX_TOPK_IDS)
    tk_wt_p = _resolve_port_by_roles(query.operands, ("topk_weights",), _MEGAMOE_PORT_IDX_TOPK_WEIGHTS)
    w1_p = _resolve_port_by_roles(query.operands, ("weight1",), _MEGAMOE_PORT_IDX_WEIGHT1)
    w2_p = _resolve_port_by_roles(query.operands, ("weight2",), _MEGAMOE_PORT_IDX_WEIGHT2)
    if x_p is None or w1_p is None or w2_p is None:
        return "rejected_shape_or_align:megamoe_required_ports_missing"
    mode_err = _check_megamoe_mode_and_params(entry, query, x_p, w1_p)
    if mode_err is not None:
        return mode_err

    if len(x_p.shape) != _RANK_2D or x_p.shape[0] <= 0 or x_p.shape[1] <= 0:
        return f"rejected_shape_or_align:megamoe_x_must_be_2d:got={x_p.shape}"
    h_dim = x_p.shape[1]
    act_div = 1 if query.activation_kind == "gelu" else _EVEN_DIVISOR
    if is_a5:
        if len(w1_p.shape) not in {_RANK_2D, _RANK_3D} or len(w1_p.shape) != len(w2_p.shape):
            return f"rejected_shape_or_align:megamoe_a5_weight_rank_invalid:w1={w1_p.shape},w2={w2_p.shape}"
        if len(w1_p.shape) == _RANK_3D and (w1_p.shape[0] != per_rank or w2_p.shape[0] != per_rank):
            return (
                f"rejected_shape_or_align:megamoe_weight_local_experts_mismatch:"
                f"w1={w1_p.shape[0]},w2={w2_p.shape[0]},expected={per_rank}"
            )
        w1_hidden, w1_h = w1_p.shape[-2], w1_p.shape[-1]
        w2_h, w2_inter = w2_p.shape[-2], w2_p.shape[-1]
        if w1_h != h_dim or w2_h != h_dim:
            return f"rejected_shape_or_align:megamoe_a5_hidden_mismatch:x_h={h_dim},w1={w1_p.shape},w2={w2_p.shape}"
        if (w1_hidden < _MEGAMOE_A5_HIDDEN_DIM_MIN
                or w1_hidden > _MEGAMOE_A5_HIDDEN_DIM_MAX
                or (w1_hidden % _MEGAMOE_A5_HIDDEN_DIM_ALIGN) != 0):
            return f"rejected_shape_or_align:megamoe_a5_hidden_dim_invalid:hidden_dim={w1_hidden}"
        if (w1_hidden % act_div) != 0 or w2_inter != w1_hidden // act_div:
            return f"rejected_shape_or_align:megamoe_a5_inter_mismatch:w1={w1_p.shape},w2={w2_p.shape},act={query.activation_kind}"
        scale_err = _check_megamoe_a5_scales(query, x_p, w1_p, w2_p, per_rank, h_dim, w1_hidden, w2_inter)
        if scale_err is not None:
            return scale_err
    else:
        if len(w1_p.shape) != _RANK_2D or len(w2_p.shape) != _RANK_2D:
            return f"rejected_shape_or_align:megamoe_a23_weights_must_be_2d:w1={w1_p.shape},w2={w2_p.shape}"
        w1_h, w1_hidden = w1_p.shape[0], w1_p.shape[1]
        w2_inter, w2_h = w2_p.shape[0], w2_p.shape[1]
        if w1_h != h_dim or w2_h != h_dim:
            return f"rejected_shape_or_align:megamoe_a23_hidden_mismatch:x_h={h_dim},w1={w1_p.shape},w2={w2_p.shape}"
        if (w1_hidden % act_div) != 0 or w2_inter != w1_hidden // act_div:
            return f"rejected_shape_or_align:megamoe_a23_inter_mismatch:w1={w1_p.shape},w2={w2_p.shape},act={query.activation_kind}"
    if query.top_k > 0 and tk_ids_p is not None and tk_ids_p.shape != (x_p.shape[0], query.top_k):
        return f"rejected_shape_or_align:megamoe_topk_ids_shape_mismatch:{tk_ids_p.shape}!={(x_p.shape[0], query.top_k)}"
    if tk_ids_p is not None and tk_wt_p is not None and tk_wt_p.shape != tk_ids_p.shape:
        return f"rejected_shape_or_align:megamoe_topk_weights_shape_mismatch:{tk_wt_p.shape}!={tk_ids_p.shape}"
    if query.outputs:
        if query.outputs[0].shape != x_p.shape:
            return f"rejected_shape_or_align:megamoe_y_shape_mismatch:{query.outputs[0].shape}!={x_p.shape}"
        if len(query.outputs) >= _RANK_2D and query.outputs[1].shape != (per_rank,):
            return f"rejected_shape_or_align:megamoe_expert_token_nums_shape_mismatch:{query.outputs[1].shape}!={(per_rank,)}"
    return None


def _check_dispatch_v2_shapes(query: AbiQueryKey) -> str | None:
    """Enforce ``aclnnMoeDistributeDispatchV2`` worst-case receive capacity conservation."""
    if not query.operands or not query.outputs:
        return None
    x_s = query.operands[0].shape
    out_s = query.outputs[0].shape
    if len(x_s) != _RANK_2D or len(out_s) != _RANK_2D or x_s[0] <= 0 or x_s[1] <= 0:
        return f"rejected_shape_or_align:dispatch_v2_rank_invalid:x={x_s},expand_x={out_s}"
    if out_s[1] != x_s[1]:
        return f"rejected_shape_or_align:dispatch_v2_hidden_mismatch:x={x_s},expand_x={out_s}"
    ws = max(1, int(query.world_size))
    if ws > 1 and query.moe_expert_num > 0 and query.top_k > 0:
        local_experts = (query.moe_expert_num + ws - 1) // ws
        min_capacity = x_s[0] * ws * min(local_experts, query.top_k)
        if out_s[0] < min_capacity:
            return f"rejected_shape_or_align:dispatch_v2_expand_x_capacity_underflow:{out_s[0]}<{min_capacity}"
    return None


def _check_op_specific_shapes(entry: ProbedAbiEntry, query: AbiQueryKey) -> str | None:
    """Enforce operator-specific rank/shape invariants grounded in C++ CheckParams."""
    if entry.abi_id == "aclnnMegaMoe":
        return _check_megamoe_constraints(entry, query)
    if entry.abi_id == "aclnnMoeDistributeDispatchV2":
        return _check_dispatch_v2_shapes(query)
    if entry.abi_id in _MOE_GATING_TOPK_IDS and query.operands:
        x_shape = query.operands[0].shape
        if len(x_shape) != _RANK_2D:
            return f"rejected_shape_or_align:moe_gating_x_must_be_2d:got={len(x_shape)}D"
        if x_shape[1] <= 0:
            return f"rejected_shape_or_align:moe_gating_experts_nonpositive:got={x_shape[1]}"
        if query.moe_expert_num > 0 and query.moe_expert_num != x_shape[1]:
            return (
                f"rejected_shape_or_align:moe_gating_expert_num_mismatch:"
                f"moe_expert_num={query.moe_expert_num}!=x_cols={x_shape[1]}"
            )
        if query.top_k > x_shape[1]:
            return f"rejected_shape_or_align:moe_gating_topk_exceeds_experts:k={query.top_k}>E={x_shape[1]}"
        if query.outputs and query.outputs[0].shape and query.outputs[0].shape[-1] > x_shape[1]:
            return f"rejected_shape_or_align:moe_gating_out_k_exceeds_experts:out_k={query.outputs[0].shape[-1]}>E={x_shape[1]}"
        if len(query.operands) >= _RANK_2D:
            b_shape = query.operands[1].shape
            if len(b_shape) != 1 or b_shape[0] != x_shape[1]:
                return f"rejected_shape_or_align:moe_gating_bias_invalid:{b_shape}_vs_E={x_shape[1]}"
    if entry.abi_id == "aclnnMoeTokenPermute" and len(query.operands) >= _RANK_2D and len(query.outputs) >= _RANK_2D:
        tok_s, idx_s = query.operands[0].shape, query.operands[1].shape
        out_tok_s, sorted_idx_s = query.outputs[0].shape, query.outputs[1].shape
        if len(tok_s) != _RANK_2D or len(idx_s) not in {1, _RANK_2D} or idx_s[0] != tok_s[0]:
            return f"rejected_shape_or_align:moe_token_permute_in_invalid:tokens={tok_s},indices={idx_s}"
        total = tok_s[0] * (idx_s[1] if len(idx_s) == _RANK_2D else 1)
        if len(sorted_idx_s) != 1 or sorted_idx_s[0] != total:
            return f"rejected_shape_or_align:moe_token_permute_sorted_idx_mismatch:{sorted_idx_s}!=({total},)"
        if len(out_tok_s) != _RANK_2D or out_tok_s[1] != tok_s[1] or (
            "num_out_tokens" not in dict(query.activation_params) and out_tok_s[0] != total
        ):
            return f"rejected_shape_or_align:moe_token_permute_out_mismatch:{out_tok_s}_vs_total={total}"
    if entry.abi_id == "aclnnTextGatedResidual" and len(query.operands) >= _RANK_3D:
        x_s, wd_s, wu_s = query.operands[0].shape, query.operands[1].shape, query.operands[2].shape
        if len(x_s) != _RANK_3D or len(wd_s) != _RANK_2D or len(wu_s) != _RANK_2D:
            return f"rejected_shape_or_align:text_gated_residual_rank_invalid:x={x_s},wd={wd_s},wu={wu_s}"
        b_dim, s_dim, d_dim = x_s[0], x_s[1], x_s[2]
        r_dim = wd_s[0]
        if d_dim <= 0 or r_dim <= 0 or wd_s[1] != d_dim or wu_s[0] != d_dim or wu_s[1] != r_dim:
            return f"rejected_shape_or_align:text_gated_residual_dim_mismatch:x={x_s},wd={wd_s},wu={wu_s}"
        if (b_dim > _UINT32_MAX or s_dim > _UINT32_MAX or d_dim > _UINT32_MAX
                or r_dim > _UINT32_MAX or b_dim * s_dim > _UINT32_MAX):
            return f"rejected_shape_or_align:text_gated_residual_uint32_overflow:b={b_dim},s={s_dim},d={d_dim},r={r_dim}"
        if query.outputs:
            m_s = query.outputs[0].shape
            if len(m_s) != _RANK_3D or m_s[0] != b_dim or m_s[1] != s_dim or m_s[2] <= 0:
                return f"rejected_shape_or_align:text_gated_residual_mixed_shape_invalid:x={x_s},mixed={m_s}"
            params = dict(query.activation_params)
            if "channels" in params:
                raw_ch = params["channels"]
                if raw_ch <= 0 or raw_ch != int(raw_ch) or int(raw_ch) > _UINT32_MAX:
                    return f"rejected_shape_or_align:text_gated_residual_channels_invalid:{raw_ch}"
                channels = int(raw_ch)
                if d_dim % channels != 0 or m_s[2] != d_dim // channels:
                    return (
                        f"rejected_shape_or_align:text_gated_residual_mixed_channels_mismatch:"
                        f"d={d_dim},channels={channels},mixed={m_s}"
                    )
            elif d_dim % m_s[2] != 0:
                return f"rejected_shape_or_align:text_gated_residual_mixed_not_divisor:d={d_dim},mixed={m_s}"
    return _check_attention_and_fusion_shapes(entry, query)


def _check_attention_and_fusion_shapes(entry: ProbedAbiEntry, query: AbiQueryKey) -> str | None:
    """Enforce MLA, MQSMLA, MX quantizer, rotary, and MindIE-SD fusion shape equations."""
    if entry.abi_id in _MLA_PROLOG_WEIGHT_UK_IDS:
        w_uk = _resolve_port_by_roles(query.operands, ("weight_uk",), -1)
        if w_uk is not None and (len(w_uk.shape) != _RANK_3D or any(dim <= 0 for dim in w_uk.shape)):
            return f"rejected_shape_or_align:mla_prolog_weight_uk_must_be_3d_nd:got={w_uk.shape}"
        kv_ref = _resolve_port_by_roles(query.operands, ("kv_cache_ref",), -1)
        if kv_ref is not None and len(kv_ref.shape) not in {_RANK_3D, _RANK_4D}:
            return f"rejected_shape_or_align:mla_prolog_kv_cache_must_be_3d_or_4d:got={len(kv_ref.shape)}D"
        kr_ref = _resolve_port_by_roles(query.operands, ("kr_cache_ref",), -1)
        if kr_ref is not None and len(kr_ref.shape) not in {1, _RANK_3D, _RANK_4D}:
            return f"rejected_shape_or_align:mla_prolog_kr_cache_rank_invalid:got={len(kr_ref.shape)}D"
        tok_x = _resolve_port_by_roles(query.operands, ("token_x",), 0)
        if tok_x is not None and w_uk is not None and query.outputs and len(tok_x.shape) in {_RANK_2D, _RANK_3D}:
            exp_q = (*tok_x.shape[:-1], w_uk.shape[0], w_uk.shape[_RANK_2D])
            if query.outputs[0].shape != exp_q:
                return f"rejected_shape_or_align:mla_prolog_query_out_mismatch:{query.outputs[0].shape}!={exp_q}"
    if entry.abi_id == "aclnnMixedQuantSparseFlashMla":
        q_p = _resolve_port_by_roles(query.operands, ("q",), 0)
        if q_p is not None and query.outputs and query.outputs[0].shape != q_p.shape:
            return f"rejected_shape_or_align:mqsmla_out_shape_mismatch:{query.outputs[0].shape}!={q_p.shape}"
        for role_name in ("ori_kv", "cmp_kv"):
            kv_p = _resolve_port_by_roles(query.operands, (role_name,), -1)
            if kv_p is None:
                continue
            exp_dim = _MQSMLA_QUANT_RECORD_DIMS.get(kv_p.quant_mode.lower())
            if exp_dim is not None and (not kv_p.shape or kv_p.shape[-1] != exp_dim):
                return f"rejected_shape_or_align:mqsmla_{role_name}_width_mismatch:{kv_p.shape}_expected_last={exp_dim}"
    if entry.abi_id == "QuantFourOverSixA5" and query.operands and len(query.outputs) >= _RANK_2D:
        x_s = query.operands[0].shape
        sc_s = query.outputs[1].shape
        if x_s:
            blocks = (x_s[-1] + _MX_BLOCK_SIZE - 1) // _MX_BLOCK_SIZE
            aligned = (blocks + _MX_SCALE_ALIGN - 1) // _MX_SCALE_ALIGN
            exp_sc = (*x_s[:-1], aligned, _MX_SCALE_ALIGN)
            if sc_s != exp_sc:
                return f"rejected_shape_or_align:quant_46_mxscale_shape_mismatch:{sc_s}!={exp_sc}"
    if entry.abi_id == "catlass_epilogue_dual_level_quant_mx" and query.operands and len(query.outputs) >= _RANK_3D:
        x_s = query.operands[0].shape
        y_s, sc1_s, sc2_s = query.outputs[0].shape, query.outputs[1].shape, query.outputs[_RANK_2D].shape
        if x_s:
            n_dim = x_s[-1]
            if n_dim <= 0:
                return f"rejected_shape_or_align:catlass_mx_x_width_invalid:x={x_s}"
            if y_s != x_s:
                return f"rejected_shape_or_align:catlass_mx_y_shape_mismatch:{y_s}!={x_s}"
            exp_sc1_cols = max(1, (n_dim + _DUAL_LEVEL_MX_SCALE1_BLOCK - 1) // _DUAL_LEVEL_MX_SCALE1_BLOCK)
            exp_sc1 = (*x_s[:-1], exp_sc1_cols)
            if sc1_s != exp_sc1:
                return f"rejected_shape_or_align:catlass_mx_scale1_mismatch:{sc1_s}!={exp_sc1}_for_x={x_s}"
            sc2_blocks = max(1, (n_dim + _MX_BLOCK_SIZE - 1) // _MX_BLOCK_SIZE)
            exp_sc2_cols = ((sc2_blocks + _MX_SCALE_ALIGN - 1) // _MX_SCALE_ALIGN) * _MX_SCALE_ALIGN
            exp_sc2 = (*x_s[:-1], exp_sc2_cols)
            if sc2_s != exp_sc2:
                return f"rejected_shape_or_align:catlass_mx_scale2_mismatch:{sc2_s}!={exp_sc2}_for_x={x_s}"
    if entry.abi_id == "aclnnKvCompressEpilog" and query.operands:
        cache_s = query.operands[0].shape
        min_w = _KV_COMPRESS_MIN_ARCH35 if entry.device_arch == "A5" else _KV_COMPRESS_MIN_ARCH22
        if len(cache_s) != _RANK_4D or cache_s[_RANK_3D] < min_w:
            return f"rejected_shape_or_align:kv_compress_cache_invalid:{cache_s}:min_width={min_w}"
        if query.outputs and query.outputs[0].shape != cache_s:
            return f"rejected_shape_or_align:kv_compress_out_mismatch:{query.outputs[0].shape}!={cache_s}"
    if entry.abi_id == "InplacePartialRotaryMul" and query.operands and query.outputs:
        if query.outputs[0].shape != query.operands[0].shape:
            return f"rejected_shape_or_align:inplace_rotary_shape_mismatch:{query.outputs[0].shape}!={query.operands[0].shape}"
    if entry.abi_id == "RmsNormRopeConcatFused" and len(query.operands) >= _RANK_3D:
        q_s, k_s, v_s = query.operands[0].shape, query.operands[1].shape, query.operands[_RANK_2D].shape
        if len(q_s) != _RANK_4D or len(k_s) != _RANK_4D or len(v_s) != _RANK_4D:
            return f"rejected_shape_or_align:norm_rope_concat_must_be_4d:q={q_s},k={k_s},v={v_s}"
        if not (q_s[_RANK_2D] == k_s[_RANK_2D] == v_s[_RANK_2D]):
            return f"rejected_shape_or_align:norm_rope_concat_head_mismatch:q={q_s[_RANK_2D]},k={k_s[_RANK_2D]},v={v_s[_RANK_2D]}"
    if entry.abi_id == "CatlassMatmulAddFfn" and len(query.operands) >= _RANK_3D:
        x_s, w1_s, w2_s = query.operands[0].shape, query.operands[1].shape, query.operands[_RANK_2D].shape
        if len(x_s) < _RANK_2D or len(w1_s) != _RANK_2D or len(w2_s) != _RANK_2D:
            return f"rejected_shape_or_align:eagle_ffn_rank_invalid:x={x_s},w1={w1_s},w2={w2_s}"
        is_swiglu = query.activation_kind == "swiglu"
        canon_ok = (
            x_s[-1] == w1_s[0]
            and ((w1_s[1] % _EVEN_DIVISOR == 0 and w1_s[1] // _EVEN_DIVISOR == w2_s[0]) if is_swiglu else (w1_s[1] == w2_s[0]))
            and (not query.outputs or query.outputs[0].shape == (*x_s[:-1], w2_s[1]))
        )
        linear_ok = (
            x_s[-1] == w1_s[1]
            and ((w1_s[0] % _EVEN_DIVISOR == 0 and w1_s[0] // _EVEN_DIVISOR == w2_s[1]) if is_swiglu else (w1_s[0] == w2_s[1]))
            and (not query.outputs or query.outputs[0].shape == (*x_s[:-1], w2_s[0]))
        )
        if not (canon_ok or linear_ok):
            return f"rejected_shape_or_align:eagle_ffn_dim_mismatch:x={x_s},w1={w1_s},w2={w2_s},act={query.activation_kind}"
    return None


def _check_extended_shape_constraints(
    entry: ProbedAbiEntry,
    query: AbiQueryKey,
    act_port: OperandPortSig | None,
    wt_port: OperandPortSig | None,
) -> str | None:
    """Enforce extended typed ``AbiShapeConstraints`` (head/rope/kv_heads/experts/layout/rank)."""
    sc = entry.shape_constraints
    if sc.head_dim_allowed_set:
        q_port = _resolve_port_by_roles(query.operands, _HEAD_DIM_PORT_ROLES, 0)
        if q_port is None or not q_port.shape or q_port.shape[-1] not in sc.head_dim_allowed_set:
            got = q_port.shape[-1] if (q_port is not None and q_port.shape) else 0
            return f"rejected_shape_or_align:head_dim={got}_not_in_{sorted(sc.head_dim_allowed_set)}"
    if sc.rope_dim_allowed_set and query.qk_rope_head_dim not in sc.rope_dim_allowed_set:
        return (
            f"rejected_shape_or_align:rope_dim={query.qk_rope_head_dim}"
            f"_not_in_{sorted(sc.rope_dim_allowed_set)}"
        )
    if sc.kv_heads_allowed_set:
        kv_port = _resolve_port_by_roles(query.operands, _KV_HEAD_PORT_ROLES, -1)
        if kv_port is not None:
            if len(kv_port.shape) < _RANK_3D or kv_port.shape[-2] not in sc.kv_heads_allowed_set:
                got = kv_port.shape[-2] if len(kv_port.shape) >= _RANK_3D else 0
                return f"rejected_shape_or_align:kv_heads={got}_not_in_{sorted(sc.kv_heads_allowed_set)}"
    if sc.max_experts > 0:
        if query.moe_expert_num > sc.max_experts:
            return f"rejected_shape_or_align:experts={query.moe_expert_num}>max_experts={sc.max_experts}"
        if (entry.abi_id in _MOE_GATING_TOPK_IDS
                and query.operands
                and len(query.operands[0].shape) == _RANK_2D
                and query.operands[0].shape[1] > sc.max_experts):
            return f"rejected_shape_or_align:experts={query.operands[0].shape[1]}>max_experts={sc.max_experts}"
        if (entry.role == "moe_grouped_matmul"
                and wt_port is not None
                and len(wt_port.shape) >= _RANK_3D
                and wt_port.shape[0] > sc.max_experts):
            return f"rejected_shape_or_align:experts={wt_port.shape[0]}>max_experts={sc.max_experts}"
    if sc.required_layout_b and wt_port is not None and wt_port.layout != sc.required_layout_b:
        return f"rejected_shape_or_align:layout_b={wt_port.layout}!=required={sc.required_layout_b}"
    if not sc.allow_ungated_activation and not query.is_act_and_mul:
        return f"rejected_dtype:ungated_activation_forbidden:{entry.entry_id}"
    if sc.last_axis_divisibility > 0 and act_port is not None and act_port.shape:
        if (act_port.shape[-1] % sc.last_axis_divisibility) != 0:
            return (
                f"rejected_shape_or_align:last_axis={act_port.shape[-1]}"
                f"_not_div_by_{sc.last_axis_divisibility}"
            )
    if sc.rank_range != (0, 0) and act_port is not None:
        lo, hi = sc.rank_range
        rank = len(act_port.shape)
        if rank < lo or (hi > 0 and rank > hi):
            return f"rejected_shape_or_align:rank={rank}_outside_{sc.rank_range}"
    return None


def _check_shape_constraints(
    entry: ProbedAbiEntry, query: AbiQueryKey,
) -> str | None:
    mm_err = _check_matmul_shape_equations(entry, query)
    if mm_err is not None:
        return mm_err
    coll_err = _check_collective_shape_equations(entry, query)
    if coll_err is not None:
        return coll_err
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
    act_port = _resolve_port_by_roles(query.operands, _ACTIVATION_PORT_ROLES, 0)
    wt_port = _resolve_port_by_roles(query.operands, _WEIGHT_PORT_ROLES, 1, exclude=act_port)
    if (sc.require_even_n_for_fp4_transb and has_fp4
            and wt_port is not None and wt_port.transposed
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
    return _check_extended_shape_constraints(entry, query, act_port, wt_port)


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
    act_port = _resolve_port_by_roles(query.operands, _ACTIVATION_PORT_ROLES, 0)
    wt_port = _resolve_port_by_roles(query.operands, _WEIGHT_PORT_ROLES, 1, exclude=act_port)
    act_dtype = act_port.dtype if act_port is not None else "bf16"
    wt_dtype = wt_port.dtype if wt_port is not None else act_dtype
    quant_mode = act_port.quant_mode if act_port is not None else "dense"
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

    def device_kernel_expansion(
        self,
        symbol: str,
        device_arch: str,
        *,
        dtype_tuple: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        """Return sealed 1-to-many device kernel expansion for host API ``symbol`` on ``device_arch``."""
        arch = _norm_arch(device_arch)
        matches = self.lookup(symbol, arch, dtype_tuple=dtype_tuple, available_only=True)
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
        if entry.require_bias and not query.has_bias:
            return f"rejected_dtype:bias_required:{entry.entry_id}", ""
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
            prov = getattr(binding, "provenance", None)
            pr_ref = (
                str(getattr(binding, "ref", "") or "").strip()
                or str(getattr(prov, "revision", "") or "").strip()
                or str(getattr(binding, "pr_url", "") or "").strip()
                or str(getattr(binding, "revision", "") or "").strip()
                or f"{binding.mechanism_id}:{getattr(binding, 'state', 'proposed_pr')}"
            )
            return None, pr_ref
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

        eff_bindings = (
            tuple(manifest_bindings)
            if manifest_bindings
            else (tuple(getattr(track_spec, "active_bindings", ()) or ()) if track_spec is not None else ())
        )
        enabled_passes = set(query.enabled_stack_passes)
        for binding in eff_bindings:
            lp_id = str(getattr(binding, "lowering_pass_id", "") or "").strip()
            if lp_id:
                enabled_passes.add(lp_id)
        if track_spec is not None:
            for pass_id in getattr(track_spec, "enabled_passes", ()) or ():
                if str(pass_id).strip():
                    enabled_passes.add(str(pass_id).strip())
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
                manifest_bindings=eff_bindings,
            )
            if err is None:
                admitted.append((entry, pr_ref))
            else:
                rejections.append((entry, err))

        if len(admitted) > 1:
            ids = ",".join(sorted(item[0].entry_id for item in admitted))
            raise ModelingError(f"abi_registry_ambiguous_match:{query.soc_arch}:{query.abi_id}:{ids}")
        if not admitted:
            best_entry, best_err = max(
                rejections,
                key=lambda item: next(
                    (prio for prefix, prio in _REJECTION_PREFIX_PRIORITY if item[1].startswith(prefix)),
                    0,
                ),
            )
            return self._reject(query, snapshot, best_err, entry=best_entry)

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
