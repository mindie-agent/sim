#!/usr/bin/env python3
"""Offline Plane-1 TileLang-Ascend silicon calibration producer and sealer.

[Role] Assemble, privacy-scrub, and SHA-256 seal ``data/primitives/*_tilelang.json``
  multi-level TileLang API calibration envelopes from raw L0-L5 silicon probe
  captures paired against a sealed AscendC L0 ``PrimitiveTable``.
[Invariants] Strictly offline Plane-1 tool outside ``mindie_sim/``. Zero
  hardcoded hostnames, IP addresses, usernames, or private filesystem paths.
  Never imports ``torch_npu`` or initializes device contexts during sealing,
  planning, or verification. Enforces canonical SHA-256 integrity, privacy
  hygiene, UB hardware ceiling ordering, and DMA stage attainment monotonicity.
[Timeouts & Locks] Pure local file reads/writes; no background locks or network.
[Side-Effect Policy] Writes only the explicitly requested ``--out`` JSON file.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

_PARENTS = Path(__file__).resolve().parents
REPO_ROOT = _PARENTS[1] if len(_PARENTS) > 1 else _PARENTS[0]
if (REPO_ROOT / "mindie_sim").is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mindie_sim.primitives.tilelang import (
    TILELANG_CALIBRATION_SCHEMA_VERSION,
    compute_tilelang_envelope_sha256,
    parse_tilelang_calibration,
)

JSON_INDENT = 2
AIC_COUNT_DAV3510 = 32
GIGA_SCALE = 1e9
TERA_SCALE = 1e12
NS_PER_US = 1e3
RATIO_ROUND_DIGITS = 4
REF_ROUND_DIGITS = 2
NS_ROUND_DIGITS = 4

DEFAULT_ARCH_PRESET = "npu_arch_3510"
DEFAULT_OBSERVED_SILICON = "Ascend950DT_9582"
DEFAULT_TILELANG_VERSION = "0.1.15"
DEFAULT_BISHENG_TARGET = "dav-3510"

UB_TOTAL_PER_AIV_BYTES = 262_144
UB_SIMD_USABLE_BYTES = 253_952
UB_SIMT_USABLE_BYTES = 221_184
UB_SIMT_RECOMMENDED_USER_BYTES = 212_992
UB_SIMT_EMPIRICAL_FAIL_BYTES = 259_072

PRIVATE_TEXT_RE = re.compile(
    r"(?<![\d.])(?:\d{1,3}[._-]){3}\d{1,3}(?![\d.])|@|/(?:Users|home|root)/[^\s\"']*|\$\{?HOME\}?",
    re.IGNORECASE,
)


def assert_tilelang_privacy_clean(text: str, *, label: str = "tilelang") -> None:
    """Fail closed if any IPv4, email/host alias, or private home path appears in ``text``."""
    match = PRIVATE_TEXT_RE.search(text)
    if match is not None:
        raise ValueError(f"tilelang_privacy_leak:{label}:{match.group(0)!r}")


def describe_tilelang_probe_suite() -> dict[str, Any]:
    """Return the structured L0-L5 TileLang-Ascend silicon calibration probe protocol."""
    return {
        "schema_version": TILELANG_CALIBRATION_SCHEMA_VERSION,
        "methodology": {
            "dvfs_clock_lock": "30x 4096^3 BF16 GEMM before each probe group",
            "cold_ring_rotation": ">= 4x L2 per die for cold HBM and LLM M-curve sweeps",
            "differential_timing": "(T(K_hi) - T(K_lo)) / (K_hi - K_lo) with T_device >> T_host",
            "numerical_verification": "verified against PyTorch/CPU reference prior to timing",
        },
        "levels": {
            "L0_dispatch": [
                "jit_compile_ms",
                "eager_tvm_ffi_prealloc_us",
                "eager_tvm_ffi_callee_alloc_us",
                "aclgraph_capture.per_kernel_issue_us",
            ],
            "L1_dma_copy": [
                "cold_rw11_stages_1_gbps",
                "cold_rw11_stages_2_gbps",
                "cold_rw11_stages_3_gbps",
                "cold_rw21_vecadd_simt_gbps",
                "cold_rw21_vecadd_simd_gbps",
                "warm_l2_rw11_aggregate_gbps",
                "row_32fp32_aligned_gbps",
                "row_30fp32_data_select_gbps",
            ],
            "L2_vector_simt_vs_simd": [
                "simt_ub_fma_t256",
                "simt_ub_fma_t1024",
                "simt_ub_exp_t512",
                "simt_reg_fma_t1024",
                "simt_reg_exp_t1024",
                "simt_reducer_sum_row4096",
                "simd_alu_chain8",
                "simd_sfu_vexp_chain3",
                "simd_ub_stream_vadd",
                "simd_vcvt_bf16_roundtrip",
            ],
            "L3_cube_gemm_and_blockscaled": [
                "bf16_fixpipe_bf16",
                "bf16_dualcopy_fp32",
                "bf16_atomic_fp32",
                "fp8_e4m3_fixpipe_bf16",
                "fp8_e4m3_dualcopy_fp32",
                "fp32_ieee",
                "fp32_hf32_even",
                "bf16_llm_4096x7168_cold_m_curve",
                "mxfp8_blockscaled_8192",
            ],
            "L4_sync_primitives": ["v_mte3_set_wait_pair_ns"],
            "L5_composite_anchors": [
                "rmsnorm_simt_B64_D7168",
                "rmsnorm_simt_B256_D7168",
                "fp8_group128_quant_simt_M128_D7168",
                "fp8_group128_quant_simd_M128_D7168",
                "fp8_group128_quant_simt_M4096_D7168",
                "fp8_group128_quant_simd_M4096_D7168",
            ],
        },
    }


def extract_ascendc_l0_reference(l0_table: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    """Extract paired AscendC L0 p50 baselines and envelope SHA-256 from a sealed L0 table."""
    l0_sha = str(l0_table.get("envelope_sha256", "")).strip()
    if not l0_sha:
        raise ValueError("ascendc_l0_table_missing_envelope_sha256")
    l0 = l0_table["primitives"]
    aic = int(l0_table.get("provenance", {}).get("silicon_fingerprint", {}).get("aic_count", AIC_COUNT_DAV3510))
    ref = {
        "P01_hbm_read_gbps": round(l0["P01_datacopy"]["metrics"]["hbm_read_bytes_per_s"]["p50"] / GIGA_SCALE, REF_ROUND_DIGITS),
        "P01_hbm_rw11_gbps": round(l0["P01_datacopy"]["metrics"]["hbm_rw11_bytes_per_s"]["p50"] / GIGA_SCALE, REF_ROUND_DIGITS),
        "P01_hbm_rw21_gbps": round(l0["P01_datacopy"]["metrics"]["hbm_rw21_bytes_per_s"]["p50"] / GIGA_SCALE, REF_ROUND_DIGITS),
        "P06_bf16_cube_tflops": round(l0["P06_mmad"]["modes"]["bf16"]["cube_only_rate"]["p50"] * aic / TERA_SCALE, REF_ROUND_DIGITS),
        "P06_bf16_e2e_tflops": round(l0["P06_mmad"]["parameters"]["gemm_e2e_flops_per_s_per_aic"]["bf16"]["band"]["p50"] * aic / TERA_SCALE, REF_ROUND_DIGITS),
        "P06_fp8_cube_tflops": round(l0["P06_mmad"]["modes"]["fp8_e4m3"]["cube_only_rate"]["p50"] * aic / TERA_SCALE, REF_ROUND_DIGITS),
        "P06_mxfp8_cube_tflops": round(l0["P06_mmad"]["modes"]["mxfp8"]["cube_only_rate"]["p50"] * aic / TERA_SCALE, REF_ROUND_DIGITS),
        "P06_mxfp8_e2e_tflops": round(l0["P06_mmad"]["parameters"]["gemm_e2e_flops_per_s_per_aic"]["mxfp8"]["band"]["p50"] * aic / TERA_SCALE, REF_ROUND_DIGITS),
        "P06_fp32_cube_tflops": round(l0["P06_mmad"]["modes"]["fp32"]["cube_only_rate"]["p50"] * aic / TERA_SCALE, REF_ROUND_DIGITS),
        "P06_hf32_cube_tflops": round(l0["P06_mmad"]["modes"]["hf32"]["cube_only_rate"]["p50"] * aic / TERA_SCALE, REF_ROUND_DIGITS),
        "P07_vec_alu_gops_per_aiv": round(l0["P07_vec_alu"]["metrics"]["vec_alu_ops_per_s_per_aiv"]["p50"] / GIGA_SCALE, REF_ROUND_DIGITS),
        "P08_vec_sfu_gops_per_aiv": round(l0["P08_vec_sfu"]["metrics"]["vec_sfu_ops_per_s_per_aiv"]["p50"] / GIGA_SCALE, REF_ROUND_DIGITS),
        "P09_vec_reduce_gops_per_aiv": round(l0["P09_vec_reduce_cast"]["metrics"]["vec_reduce_ops_per_s_per_aiv"]["p50"] / GIGA_SCALE, REF_ROUND_DIGITS),
        "P09_cast_gbps_per_aiv": round(l0["P09_vec_reduce_cast"]["metrics"]["cast_bytes_per_s_per_aiv"]["p50"] / GIGA_SCALE, REF_ROUND_DIGITS),
        "P11_simt_scalar_gflops_per_aiv": round(l0["P11_simt"]["parameters"]["simt_scalar_flops_per_s_per_aiv_p50"] / GIGA_SCALE, REF_ROUND_DIGITS),
        "P12_flag_ns": round(l0["P12_intracore_flag"]["metrics"]["flag_ns"]["p50"], NS_ROUND_DIGITS),
        "P15_host_submit_us": round(l0["P15_rts_stars"]["metrics"]["host_submit_ns"]["p50"] / NS_PER_US, NS_ROUND_DIGITS),
        "P15_npugraph_issue_us": round(l0["P15_rts_stars"]["metrics"]["npugraph_issue_ns"]["p50"] / NS_PER_US, NS_ROUND_DIGITS),
    }
    return l0_sha, ref


def build_tilelang_calibration_payload(
    *,
    part1: Mapping[str, Any],
    part2: Mapping[str, Any],
    part3: Mapping[str, Any],
    part4: Mapping[str, Any],
    mxfp8: Mapping[str, Any],
    l0_table: Mapping[str, Any],
    arch_preset: str = DEFAULT_ARCH_PRESET,
    observed_silicon: str = DEFAULT_OBSERVED_SILICON,
    tilelang_version: str = DEFAULT_TILELANG_VERSION,
    bisheng_target: str = DEFAULT_BISHENG_TARGET,
    ub_ceilings: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Assemble and SHA-256 seal a ``mindie-tilelang-api-calibration/1`` envelope."""
    l0_sha, l0_ref = extract_ascendc_l0_reference(l0_table)
    dma = part1["sections"]["TL01_dma_copy"]
    cube = part2["sections"]["TL03_cube_gemm"]
    vec = part3["TL02_vector_onchip_low_cv"]
    simt_reg = part4["simt_register_resident"]
    ceilings = dict(ub_ceilings) if ub_ceilings is not None else {
        "total_ub_per_aiv": UB_TOTAL_PER_AIV_BYTES,
        "simd_usable_ub_bytes": UB_SIMD_USABLE_BYTES,
        "simt_usable_ub_bytes": UB_SIMT_USABLE_BYTES,
        "simt_recommended_user_ub_bytes": UB_SIMT_RECOMMENDED_USER_BYTES,
        "empirical_simt_launch_fail_dyn_ubuf_bytes": UB_SIMT_EMPIRICAL_FAIL_BYTES,
    }
    ratios = {
        "eta_copy_rw11_stages1": round(dma["cold_rw11_stages_1_gbps"]["p50"] / l0_ref["P01_hbm_rw11_gbps"], RATIO_ROUND_DIGITS),
        "eta_copy_rw11_stages2": round(dma["cold_rw11_stages_2_gbps"]["p50"] / l0_ref["P01_hbm_rw11_gbps"], RATIO_ROUND_DIGITS),
        "eta_copy_rw11_stages3": round(dma["cold_rw11_stages_3_gbps"]["p50"] / l0_ref["P01_hbm_rw11_gbps"], RATIO_ROUND_DIGITS),
        "eta_copy_rw21_simt": round(dma["cold_rw21_vecadd_simt_gbps"]["p50"] / l0_ref["P01_hbm_rw21_gbps"], RATIO_ROUND_DIGITS),
        "eta_copy_rw21_simd": round(dma["cold_rw21_vecadd_simd_gbps"]["p50"] / l0_ref["P01_hbm_rw21_gbps"], RATIO_ROUND_DIGITS),
        "eta_cube_bf16_fixpipe": round(cube["bf16_fixpipe_bf16"]["tflops_die"]["p50"] / l0_ref["P06_bf16_cube_tflops"], RATIO_ROUND_DIGITS),
        "eta_cube_bf16_dualcopy_fp32": round(cube["bf16_dualcopy_fp32"]["tflops_die"]["p50"] / l0_ref["P06_bf16_cube_tflops"], RATIO_ROUND_DIGITS),
        "eta_cube_bf16_atomic_fp32": round(cube["bf16_atomic_fp32"]["tflops_die"]["p50"] / l0_ref["P06_bf16_cube_tflops"], RATIO_ROUND_DIGITS),
        "eta_cube_fp8_fixpipe": round(cube["fp8_e4m3_fixpipe_bf16"]["tflops_die"]["p50"] / l0_ref["P06_fp8_cube_tflops"], RATIO_ROUND_DIGITS),
        "eta_cube_fp8_dualcopy_fp32": round(cube["fp8_e4m3_dualcopy_fp32"]["tflops_die"]["p50"] / l0_ref["P06_fp8_cube_tflops"], RATIO_ROUND_DIGITS),
        "eta_cube_fp32_ieee": round(cube["fp32_ieee"]["tflops_die"]["p50"] / l0_ref["P06_fp32_cube_tflops"], RATIO_ROUND_DIGITS),
        "eta_cube_hf32": round(cube["fp32_hf32_even"]["tflops_die"]["p50"] / l0_ref["P06_hf32_cube_tflops"], RATIO_ROUND_DIGITS),
        "eta_cube_mxfp8_blockscaled": round(mxfp8["tflops_die_p50"] / l0_ref["P06_mxfp8_cube_tflops"], RATIO_ROUND_DIGITS),
        "eta_simt_reg_fma_vs_p11": round(simt_reg["simt_reg_fma_t1024"]["gops_per_s_per_aiv"]["p50"] / l0_ref["P11_simt_scalar_gflops_per_aiv"], RATIO_ROUND_DIGITS),
        "eta_simt_ub_fma_vs_p11": round(vec["simt_ub_fma_t1024"]["gops_per_s_per_aiv"]["p50"] / l0_ref["P11_simt_scalar_gflops_per_aiv"], RATIO_ROUND_DIGITS),
        "eta_simd_sfu_exp_vs_p08": round(vec["simd_sfu_vexp_chain3"]["gops_per_s_per_aiv"]["p50"] / l0_ref["P08_vec_sfu_gops_per_aiv"], RATIO_ROUND_DIGITS),
    }
    payload: dict[str, Any] = {
        "schema_version": TILELANG_CALIBRATION_SCHEMA_VERSION,
        "arch_preset": arch_preset,
        "observed_silicon": observed_silicon,
        "tilelang_version": tilelang_version,
        "bisheng_target": bisheng_target,
        "ascendc_l0_envelope_sha256": l0_sha,
        "ascendc_l0_reference_p50": l0_ref,
        "onchip_ub_ceilings_bytes": ceilings,
        "attainment_ratios_vs_ascendc_l0": ratios,
        "levels": {
            "L0_dispatch": part1["sections"]["TL00_host_and_graph_dispatch"],
            "L1_dma_copy": dma,
            "L2_vector_simt_vs_simd": {**vec, **simt_reg},
            "L3_cube_gemm_and_blockscaled": {**cube, "mxfp8_blockscaled_8192": dict(mxfp8)},
            "L4_sync_primitives": part3["TL04_sync_low_cv"],
            "L5_composite_anchors": part2["sections"]["TL05_composite_anchors"],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert_tilelang_privacy_clean(canonical, label=f"{arch_preset}:{observed_silicon}")
    payload["envelope_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    parse_tilelang_calibration(payload)
    return payload


def verify_tilelang_calibration_file(path: Path) -> dict[str, Any]:
    """Verify privacy, canonical SHA-256, schema, and invariants of a sealed JSON file."""
    text = path.read_text(encoding="utf-8")
    assert_tilelang_privacy_clean(text, label=path.name)
    raw = json.loads(text)
    cal = parse_tilelang_calibration(raw)
    computed = compute_tilelang_envelope_sha256(raw)
    if computed != cal.envelope_sha256:
        raise ValueError("tilelang_calibration_sha256_mismatch")
    return {
        "status": "verified",
        "path": path.name,
        "schema_version": cal.schema_version,
        "arch_preset": cal.arch_preset,
        "observed_silicon": cal.observed_silicon,
        "tilelang_version": cal.tilelang_version,
        "envelope_sha256": cal.envelope_sha256,
        "simt_usable_ub_bytes": cal.simt_usable_ub_bytes,
        "simd_usable_ub_bytes": cal.simd_usable_ub_bytes,
    }


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", type=Path, help="verify an existing sealed TileLang calibration JSON")
    mode.add_argument("--seal-from-dir", type=Path, help="directory containing raw tl_part1..4 & tl_mxfp8 JSONs")
    mode.add_argument("--plan-probes", action="store_true", help="emit the L0-L5 silicon probe plan as JSON")
    parser.add_argument("--l0-table", type=Path, help="paired sealed AscendC L0 table JSON (for --seal-from-dir)")
    parser.add_argument("--out", type=Path, help="output sealed TileLang JSON path (for --seal-from-dir)")
    parser.add_argument("--arch-preset", default=DEFAULT_ARCH_PRESET)
    parser.add_argument("--observed-silicon", default=DEFAULT_OBSERVED_SILICON)
    parser.add_argument("--tilelang-version", default=DEFAULT_TILELANG_VERSION)
    parser.add_argument("--bisheng-target", default=DEFAULT_BISHENG_TARGET)
    args = parser.parse_args(argv)

    if args.plan_probes:
        print(json.dumps(describe_tilelang_probe_suite(), indent=JSON_INDENT))
        return 0
    if args.verify is not None:
        summary = verify_tilelang_calibration_file(args.verify)
        print(json.dumps(summary, indent=JSON_INDENT))
        return 0
    if args.l0_table is None or args.out is None:
        parser.error("--seal-from-dir requires both --l0-table and --out")
    raw_dir = args.seal_from_dir
    payload = build_tilelang_calibration_payload(
        part1=_read_json_file(raw_dir / "tl_part1_results.json"),
        part2=_read_json_file(raw_dir / "tl_part2_results.json"),
        part3=_read_json_file(raw_dir / "tl_part3_results.json"),
        part4=_read_json_file(raw_dir / "tl_part4_results.json"),
        mxfp8=_read_json_file(raw_dir / "tl_mxfp8_result.json"),
        l0_table=_read_json_file(args.l0_table),
        arch_preset=args.arch_preset,
        observed_silicon=args.observed_silicon,
        tilelang_version=args.tilelang_version,
        bisheng_target=args.bisheng_target,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=JSON_INDENT) + "\n", encoding="utf-8")
    print(json.dumps({"sealed": args.out.name, "envelope_sha256": payload["envelope_sha256"]}, indent=JSON_INDENT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
