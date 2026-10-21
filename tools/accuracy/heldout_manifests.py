"""Generate the predeclared heldout measurement manifests — and never run them.

[Role] Turns the two split declarations into worker-ready measurement manifests for the
  blinded campaigns, validates them with each worker's own ``--dry-run``, proves that no
  heldout condition appears in the training dataset, and writes the runbook and the
  pre-flight guard for the operator.
[Invariants]
  * Preparation only: this module never selects a device, takes a lease, or starts a run.
  * Every request is ``split: "heldout"``; the request/condition keys are deterministic
    hashes of the declared geometry + protocol, so the same declaration always produces the
    same manifest bytes.
  * No model prediction is written into a measurement manifest: ``analytic_ms`` and
    ``predicted_ms`` are ``null`` and the declared lower bound is carried separately as
    provenance. A manifest is a measurement order, not a prediction.
  * A dry-run must validate every request; a manifest with one pending/failed request is
    reported, never silently shipped.
  * The guard refuses to run when the frozen-predictions receipt is missing, incomplete, or
    bound to a training dataset whose hash is not the current one.
[Timeouts & Locks] The dry-run subprocess is bounded (60 s) and read-only; nothing else
  spawns a process.
[Side-Effect Policy] Reads the declarations and the dataset; writes only --out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import metrics
from .assemble_dataset import load_split_declarations

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_DECLARATIONS = REPO_ROOT / ".workstreams/accuracy-protocol"
DEFAULT_DATASET = REPO_ROOT / ".workstreams/dataset"
DEFAULT_OUT = REPO_ROOT / ".workstreams/heldout-campaign"

# Worker lane per family. The recipes are the ones the campaign workers already implement
# (capability matrix §A/B/C); the shape contract is the declared shape, unchanged.
LANES: dict[str, dict[str, Any]] = {
    "gemm": {"lane": "campaign", "worker_recipe": "torch_mm_preallocated",
             "backend": "torch.mm", "execution_kind": "primitive",
             "recipe": "plain_contiguous_bf16_mm", "target_match": "exact_primitive",
             "shape_kind": "gemm"},
    "add": {"lane": "campaign", "worker_recipe": "elementwise_add_reference",
            "backend": "torch.add", "execution_kind": "primitive",
            "recipe": "elementwise_add_reference", "target_match": "exact_primitive",
            "shape_kind": "elements"},
    "copy": {"lane": "campaign", "worker_recipe": "copy_rows_reference",
             "backend": "torch.Tensor.copy_", "execution_kind": "primitive",
             "recipe": "copy_rows_reference", "target_match": "exact_primitive",
             "shape_kind": "bytes"},
    "rmsnorm": {"lane": "campaign", "worker_recipe": "npu_rms_norm",
                "backend": "torch_npu.npu_rms_norm", "execution_kind": "fused_kernel",
                "recipe": "npu_rms_norm", "target_match": "exact_primitive",
                "shape_kind": "tokens_width"},
    "swiglu": {"lane": "campaign", "worker_recipe": "npu_swiglu",
               "backend": "torch_npu.npu_swiglu", "execution_kind": "fused_kernel",
               "recipe": "npu_swiglu", "target_match": "exact_primitive",
               "shape_kind": "tokens_width"},
    "attention_softmax": {"lane": "campaign", "worker_recipe": "torch_softmax",
                          "backend": "torch.softmax", "execution_kind": "primitive",
                          "recipe": "torch_softmax", "target_match": "exact_primitive",
                          "shape_kind": "tokens_width"},
    "quantize": {"lane": "campaign", "worker_recipe": "npu_dynamic_quant",
                 "backend": "torch_npu.npu_dynamic_quant", "execution_kind": "fused_kernel",
                 "recipe": "npu_dynamic_quant", "target_match": "exact_primitive",
                 "shape_kind": "tokens_width"},
    "moe_gate_up": {"lane": "campaign", "worker_recipe": "npu_grouped_matmul_list_bf16",
                    "backend": "torch_npu.npu_grouped_matmul", "execution_kind": "fused_kernel",
                    "recipe": "npu_grouped_matmul_list_bf16",
                    "target_match": "exact_primitive", "shape_kind": "moe"},
    "moe_down": {"lane": "campaign", "worker_recipe": "npu_grouped_matmul_list_bf16",
                 "backend": "torch_npu.npu_grouped_matmul", "execution_kind": "fused_kernel",
                 "recipe": "npu_grouped_matmul_list_bf16",
                 "target_match": "exact_primitive", "shape_kind": "moe"},
    "attention_qk": {"lane": "attention", "worker_recipe": "npu_fusion_attention"},
    "rope": {"lane": "attention", "worker_recipe": "npu_rotary_mul"},
}

# The attention lane needs three values the split declaration does not carry. They are
# *declared here*, once, in the style of the round-1 attention campaign it repeats:
#   mask: prefill-shaped conditions (q == kv > 1) are causal, decode-shaped (q == 1) are
#         fully visible;
#   scale: 1/sqrt(qk_dim) is the worker's own contract;
#   rope: rotary_mode "half", positions 1..seq, inv_freq_base 10000, floor_dim 64.
ATTENTION_DECLARATIONS = {
    "mask_rule": "causal_bottom_right when q==kv and q>1; full_visible when q==1",
    "scale_rule": "1/sqrt(qk_dim)",
    "rope_rule": "rotary_mode=half, positions.start=1, positions.end=seq, inv_freq_base=10000",
    "seed": 20261021,
    "data": {"low": -1.0, "high": 1.0},
}

PROTOCOL_DEFAULTS = {"seed": 20261021, "requested_accumulation_dtype": "backend_default",
                     "source_class": "measured"}


class IntegrityError(RuntimeError):
    """A heldout-manifest precondition failed; the caller must exit 3."""


# --------------------------------------------------------------------- helpers ---
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def protocol_block(protocol_identity: Mapping[str, Any], hardware_key: str) -> dict[str, Any]:
    """The manifest protocol: the declaration's identity, no local invention."""
    protocol = {
        "timer": protocol_identity.get("timer", "device_event"),
        "timing_scope": protocol_identity.get("timing_scope", "device_event_warm"),
        "warmup": int(protocol_identity.get("warmup", 10)),
        "warmup_scope": protocol_identity.get("warmup_scope", "before_each_trial"),
        "iterations": int(protocol_identity.get("iterations", 30)),
        "trials": int(protocol_identity.get("trials", 2)),
        "preallocated": bool(protocol_identity.get("preallocated", True)),
        "cache_policy": protocol_identity.get(
            "cache_policy", "same_preallocated_buffers_reused_no_cache_flush"),
    }
    protocol.update(PROTOCOL_DEFAULTS)
    if protocol_identity.get("attention_scope"):
        protocol["attention_scope"] = protocol_identity["attention_scope"]
    if protocol_identity.get("measurement_scope"):
        protocol["measurement_scope"] = protocol_identity["measurement_scope"]
    return protocol


def key_material(hardware_key: str, family: str, shape: Mapping[str, Any],
                 worker_recipe: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    core = metrics.protocol_core(protocol)
    return {"hardware_key": hardware_key, "family": family,
            "shape": {str(k): float(v) for k, v in shape.items()},
            "worker_recipe": worker_recipe,
            "protocol": {name: core.get(name) for name in metrics.CORE_PROTOCOL_FIELDS}}


def request_condition_key(hardware_key: str, family: str, shape: Mapping[str, Any],
                          worker_recipe: str, protocol: Mapping[str, Any]) -> str:
    """Deterministic condition key: declared geometry + recipe + protocol, no die."""
    return "cond:" + metrics.sha256_value(
        key_material(hardware_key, family, shape, worker_recipe, protocol))[:24]


def request_key(condition_key: str, protocol_key: str) -> str:
    return "req:" + metrics.sha256_value(
        {"condition_key": condition_key, "protocol_key": protocol_key})[:24]


def measurement_id(hardware_key: str, condition_key: str) -> str:
    return "ho:" + metrics.sha256_value(
        {"hardware_key": hardware_key, "condition_key": condition_key})[:20]


# ------------------------------------------------------------------- requests ---
def declared_precision(family: str) -> dict[str, Any]:
    """The declared precision block of one measurement family (fp32 torch softmax)."""
    if family == "attention_softmax":
        # protocol.md §2.2: the softmax family is fp32 torch.softmax.
        return {"input_dtype": "fp32", "output_dtype": "fp32", "compute_dtype": "fp32",
                "accumulation_dtype": "backend_default", "quantization": "none"}
    return {"input_dtype": "bf16", "weight_dtype": "bf16", "output_dtype": "bf16",
            "compute_dtype": "bf16", "weight_format": "bf16", "activation_format": "bf16",
            "compute_format": "bf16", "accumulation_dtype": "backend_default",
            "quantization": "none"}


def campaign_request(point: Mapping[str, Any], hardware_key: str, arch_family: str,
                     protocol: Mapping[str, Any], protocol_key: str) -> dict[str, Any]:
    family = point["family"]
    lane = LANES[family]
    shape = {str(k): int(v) for k, v in point["shape"].items()}
    condition_key = request_condition_key(hardware_key, family, shape,
                                          lane["worker_recipe"], protocol)
    features: dict[str, Any] = {}
    if family in {"attention_softmax"}:
        features = {"normalized_axis": -1, "causal_mask": False}
    if family.startswith("moe"):
        features = {"group_list": None, "group_type": -1, "split_item": 0}
    if family == "rmsnorm":
        features = {"epsilon": 1e-06}
    condition = {
        "backend": lane["backend"], "boundaries": [], "condition_key": condition_key,
        "coverage_scope": f"predeclared_heldout:{arch_family}:{point.get('test_set')}",
        "execution_kind": lane["execution_kind"], "expansion": {},
        "family": family, "features": features, "layout": "contiguous",
        "phase": "operator", "precision": declared_precision(family), "protocol": protocol,
        "quantization": "none" if family != "attention_softmax" else None,
        "reason": "predeclared heldout condition, outcome-free selection",
        "recipe": lane["recipe"], "required_backend": lane["backend"].split(".")[0],
        "shape": shape, "shape_kind": lane["shape_kind"], "sources": {}, "state": {},
        "status": "executable", "target_match": lane["target_match"],
        "worker_recipe": lane["worker_recipe"],
    }
    return {
        "condition": condition,
        "condition_key": condition_key,
        "lineage_condition_keys": [condition_key],
        "measurement_id": measurement_id(hardware_key, condition_key),
        "request_key": request_key(condition_key, protocol_key),
        "analytic_ms": None,
        "predicted_ms": None,
        "declaration": {"point_id": point.get("point_id"),
                        "geometry_key": point.get("geometry_key"),
                        "regime": point.get("regime"),
                        "analytic_lower_bound_ms": point.get("analytic_lower_bound_ms"),
                        "acceptance_bearing": point.get("acceptance_bearing"),
                        "test_set": point.get("test_set")},
        "dtype": "bf16" if family != "attention_softmax" else "fp32",
        "execution_kind": lane["execution_kind"],
        "features": features,
        "layout": "contiguous",
        "op_family": family,
        "origins": [{"axis": point.get("axis"),
                     "coverage_scope": condition["coverage_scope"]}],
        "point": {"actual": False, "axis": point.get("axis"), "boundary_probe": False,
                  "extrapolation": False, "shape": shape, "split": "heldout",
                  "split_hint": "heldout", "value": shape.get(point.get("axis") or "m")},
        "protocol": protocol,
        "query_context": {"axis": point.get("axis"),
                          "scope": condition["coverage_scope"],
                          "declaration": "predeclared split declaration"},
        "recipe": lane["recipe"],
        "request_status": "heldout_predeclared",
        "run_id": None,
        "shape": shape,
        "split": "heldout",
        "status": "executable",
        "target_match": lane["target_match"],
        "work": {},
        "worker_recipe": lane["worker_recipe"],
    }


def attention_case(point: Mapping[str, Any], hardware_key: str) -> dict[str, Any]:
    family = point["family"]
    shape = {str(k): int(v) for k, v in point["shape"].items()}
    seq = shape.get("tokens") or shape.get("q") or 1
    if family == "attention_qk":
        q, kv = int(shape["q"]), int(shape["kv"])
        mask = "causal_bottom_right" if (q == kv and q > 1) else "full_visible"
        case = {
            "case_id": f"{hardware_key.split('_')[0].lower()}-ho-fa-q{q}-kv{kv}"
                       f"-hq{shape['q_heads']}-hkv{shape['kv_heads']}-d{shape['qk_dim']}",
            "operator": "fusion_attention", "dtype": "bfloat16", "layout": "BNSD",
            "mask": mask, "scale": 1.0 / (shape["qk_dim"] ** 0.5),
            "shape": {"batch": shape["batch"], "q_heads": shape["q_heads"],
                      "kv_heads": shape["kv_heads"], "q_tokens": q, "kv_tokens": kv,
                      "qk_dim": shape["qk_dim"], "v_dim": shape["v_dim"]},
            "data": {**ATTENTION_DECLARATIONS["data"],
                     "seed": seed_from(point, ATTENTION_DECLARATIONS["seed"])},
            "source": {"model": "declared-attention-panel", "topology": "tp8",
                       "global_q_heads": shape["q_heads"], "global_kv_heads": shape["kv_heads"],
                       "head_dim": shape["qk_dim"],
                       "primitive_scope": "native_fusion_attention_gqa",
                       "source_scope": "predeclared-heldout-campaign"},
            "target_match": False, "split": "heldout",
        }
        case["declaration"] = {"point_id": point.get("point_id"),
                               "mask_rule": ATTENTION_DECLARATIONS["mask_rule"],
                               "scale_rule": ATTENTION_DECLARATIONS["scale_rule"]}
        return case
    case = {
        "case_id": f"{hardware_key.split('_')[0].lower()}-ho-rope-s{seq}-h{shape['heads']}"
                   f"-d{shape['width']}",
        "operator": "rotary_mul", "dtype": "bfloat16", "layout": "BSND",
        "rotary_mode": "half",
        "shape": {"batch": shape["batch"], "seq": seq, "heads": shape["heads"],
                  "dim": shape["width"]},
        "positions": {"start": 1, "end": seq},
        "inv_freq_base": 10000.0,
        "data": {**ATTENTION_DECLARATIONS["data"],
                 "seed": seed_from(point, ATTENTION_DECLARATIONS["seed"])},
        "source": {"model": "declared-attention-panel",
                   "partial_rotary_factor": shape["width"] / 256.0,
                   "full_head_dim": 256, "rotated_dim": shape["width"],
                   "primitive_scope": "native_rotary_mul_half",
                   "source_scope": "predeclared-heldout-campaign"},
        "target_match": False, "split": "heldout",
        "declaration": {"point_id": point.get("point_id"),
                        "rope_rule": ATTENTION_DECLARATIONS["rope_rule"]},
    }
    return case


def seed_from(point: Mapping[str, Any], base: int) -> int:
    """Per-case data seed derived from the point id (deterministic, not shared)."""
    digest = metrics.sha256_text(str(point.get("point_id")))
    return base + int(digest[:6], 16) % 1000


def attention_manifest(cases: Sequence[Mapping[str, Any]], *, manifest_id: str,
                       protocol: Mapping[str, Any], arch_family: str) -> dict[str, Any]:
    trials = int(protocol.get("trials", 3))
    samples = int(protocol.get("iterations", 30))
    warmup = int(protocol.get("warmup", 10))
    return {
        "kind": "a3_attention_rope_manifest",
        "manifest_id": manifest_id,
        "description": (f"predeclared {arch_family} heldout attention/RoPE conditions; "
                        "split=heldout; not measured by this tool"),
        "protocol": {
            "warmup_iterations": warmup, "trials": trials, "samples_per_trial": samples,
            "seed": protocols_seed(protocol),
            "reference": {"atol": 0.02, "rtol": 0.02},
            "budgets_seconds": {"case": 24, "setup": 60, "global_work": 240, "cleanup": 60},
            "supervisor_seconds_min": 370,
        },
        "source_contract": {"note": "declared before any run",
                            "scope": f"predeclared {arch_family} heldout campaign",
                            "split": "heldout"},
        "run_scope": {"heldout_only": True, "fits_or_selects": False},
        "expectations": {"cases": len(cases),
                         "planned_raw_blocks": len(cases) * trials * samples},
        "cases": list(cases),
    }


def protocols_seed(protocol: Mapping[str, Any]) -> int:
    return int(protocol.get("seed", 20261021))


def campaign_manifest(requests: Sequence[Mapping[str, Any]], *, campaign_id: str,
                      protocol: Mapping[str, Any], arch_family: str) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "description": (f"predeclared {arch_family} heldout conditions; split=heldout; "
                        "not measured by this tool"),
        "protocol": protocol,
        "requests": list(requests),
        "run_scope": {"heldout_excluded": False, "heldout_only": True,
                      "source_class": "measured", "fits_or_selects": False},
        "status": "ready",
    }


# -------------------------------------------------------------------- collision ---
def dataset_identities(dataset: Mapping[str, Any], split: str = "train") -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    for observation in dataset.get("observations") or []:
        if observation.get("split") != split:
            continue
        for identity in observation["diagnostics"].get("geometry_identities") or [
                observation["diagnostics"].get("geometry_identity")]:
            if identity:
                identities.add((observation["hardware_key"], identity))
    return identities


def collision_report(points: Sequence[Mapping[str, Any]],
                     train: set[tuple[str, str]]) -> dict[str, Any]:
    collided = []
    for point in points:
        candidates = metrics.identity_candidates(point["family"], point["shape"])
        if any((point["hardware_key"], candidate) in train for candidate in candidates):
            collided.append({"point_id": point.get("point_id"),
                             "hardware_key": point["hardware_key"],
                             "family": point["family"], "shape": point.get("shape")})
    by_family = Counter(point["family"] for point in points)
    return {"heldout_conditions": len(points), "by_family": dict(sorted(by_family.items())),
            "train_conditions_considered": len({item[1] for item in train}),
            "collisions": collided, "passed": not collided}


# ------------------------------------------------------------------ declarations ---
def heldout_points(declarations: Path) -> dict[str, list[dict[str, Any]]]:
    """Heldout points per SKU, flattened from both declarations with their protocol."""
    points: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for declaration in load_split_declarations(declarations):
        hardware = declaration["hardware_key"]
        arch = declaration["arch_family"]
        protocol = protocol_block(declaration["protocol_identity"], hardware)
        protocol_key = metrics.protocol_signature(metrics.protocol_core(protocol))
        for point in declaration["heldout"]:
            if not point.get("shape") or not point.get("family"):
                continue
            points[hardware].append({
                **point,
                "hardware_key": hardware,
                "arch_family": arch,
                "test_set": point.get("test_set") or "test2_fresh_heldout",
                "acceptance_bearing": bool(point.get("acceptance_bearing", True)),
                "protocol": protocol,
                "protocol_key": protocol_key,
                "declaration": declaration["declaration_id"],
            })
    return dict(points)


def build_manifests(points_by_sku: Mapping[str, Sequence[Mapping[str, Any]]]
                    ) -> dict[str, dict[str, Any]]:
    """One manifest per (SKU, lane). Returns {filename: manifest}."""
    manifests: dict[str, dict[str, Any]] = {}
    for hardware, points in sorted(points_by_sku.items()):
        arch = points[0]["arch_family"] if points else "A3"
        label = "a2" if arch == "A2" else "a3-test2"
        campaign: dict[str, list[dict[str, Any]]] = defaultdict(list)
        attention: list[dict[str, Any]] = []
        for point in points:
            lane = LANES.get(point["family"], {}).get("lane", "campaign")
            if lane == "attention":
                attention.append(attention_case(point, hardware))
            else:
                campaign[point["family"]].append(campaign_request(
                    point, hardware, arch, point["protocol"], point["protocol_key"]))
        gemm_requests = campaign.pop("gemm", [])
        if gemm_requests:
            manifests[f"{label}--{hardware}--heldout-gemm.json"] = campaign_manifest(
                gemm_requests, campaign_id=f"{label}-{hardware}-heldout-gemm",
                protocol=points[0]["protocol"], arch_family=arch)
        vector_requests = [request for family in sorted(campaign)
                           for request in campaign[family]]
        if vector_requests:
            manifests[f"{label}--{hardware}--heldout-vector.json"] = campaign_manifest(
                vector_requests, campaign_id=f"{label}-{hardware}-heldout-vector",
                protocol=points[0]["protocol"], arch_family=arch)
        if attention:
            manifests[f"{label}--{hardware}--heldout-attention.json"] = attention_manifest(
                attention, manifest_id=f"{label}-{hardware}-heldout-attention",
                protocol=points[0]["protocol"], arch_family=arch)
    return manifests


# ---------------------------------------------------------------------- dry-run ---
def dry_run(manifest_path: Path, lane: str, *, timeout: float = 60.0) -> dict[str, Any]:
    """Validate one manifest with the worker's own ``--dry-run`` (no device, no lease)."""
    results = manifest_path.with_name(manifest_path.stem + ".dryrun-results.json")
    if lane == "attention":
        command = [sys.executable, "-m", "mindie_sim.a3_operator_worker_attention",
                   "--manifest", str(manifest_path), "--results", str(results),
                   "--run-id", f"dryrun-{manifest_path.stem}", "--dry-run"]
    else:
        command = [sys.executable, "-m", "mindie_sim.a3_operator_worker",
                   "--manifest", str(manifest_path), "--results", str(results), "--dry-run"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                               cwd=str(REPO_ROOT), check=False)
    payload: dict[str, Any] = {}
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {"unparsed_stdout": completed.stdout[:400]}
    return {"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-2000:],
            "payload": payload,
            "passed": completed.returncode == 0}


def manifest_counts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("kind") == "a3_attention_rope_manifest":
        return {"cases": len(manifest.get("cases") or []),
                "split_counts": dict(Counter(case.get("split") for case in
                                             manifest.get("cases") or []))}
    requests = manifest.get("requests") or []
    return {"cases": len(requests),
            "split_counts": dict(Counter(request.get("split") for request in requests)),
            "recipes": dict(Counter(request.get("worker_recipe") for request in requests)),
            "families": dict(Counter(request.get("op_family") for request in requests))}


# ------------------------------------------------------------------------ guard ---
def guard(receipt_path: Path, dataset_path: Path) -> dict[str, Any]:
    """Refuse to start a heldout run without a valid, current frozen-predictions receipt."""
    reasons: list[str] = []
    receipt: dict[str, Any] = {}
    if not Path(receipt_path).exists():
        reasons.append("frozen_predictions_receipt_missing")
    else:
        try:
            receipt = load_json(receipt_path)
        except json.JSONDecodeError as exc:
            reasons.append(f"frozen_predictions_receipt_unreadable:{exc}")
        else:
            if receipt.get("schema_version") != metrics.FREEZE_SCHEMA:
                reasons.append("frozen_predictions_receipt_schema_mismatch")
            if receipt.get("complete") is not True:
                reasons.append("frozen_predictions_receipt_incomplete")
            if receipt.get("evaluation_outcomes_read") is not False:
                reasons.append("frozen_predictions_receipt_claims_outcomes_were_read")
            dataset_document, _ = _dataset_document(dataset_path)
            current = metrics.sha256_value(dataset_document["observations"])
            recorded = (receipt.get("dataset") or {}).get("sha256")
            if not recorded:
                reasons.append("frozen_predictions_receipt_has_no_dataset_hash")
            elif recorded != current:
                reasons.append("training_dataset_hash_changed_since_freeze")
    return {"guard": "pass" if not reasons else "fail", "reasons": reasons,
            "receipt": str(receipt_path), "dataset": str(dataset_path),
            "frozen_at": receipt.get("frozen_at"),
            "predictions_sha256": receipt.get("predictions_sha256"),
            "checked_at": utc_now()}


def _dataset_document(dataset_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(dataset_path)
    observations = root / "observations.json" if root.is_dir() else root
    if not observations.exists():
        raise IntegrityError(f"dataset_missing:{observations}")
    document = load_json(observations)
    manifest_path = observations.with_name("dataset-manifest.json")
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    return document, manifest


# ---------------------------------------------------------------------- runbook ---
RUNBOOK_TEMPLATE = """# Heldout measurement runbook — {arch} predeclared heldout campaigns

**Preparation only.** This document and the manifests beside it are produced by
`python -m tools.accuracy.heldout_manifests`. Nothing here has been run, no lease has been
taken, and no heldout outcome has been read. A heldout measurement is only allowed after the
pre-flight guard passes.

Generated: `{generated_at}`

Declarations in force: {declaration_files}

## 1. What is declared

| SKU | arch | lane | manifest | conditions |
| --- | --- | --- | --- | ---: |
{manifest_rows}

Total: **{total_cases} conditions** across **{manifest_count} manifests**. Every request is
`split: "heldout"`. The test-set-1 conditions (159, already measured) are **not** in these
manifests.

Hashes (sha256) are in `heldout-manifests.json`; the guard re-checks them before a run.

## 2. Pre-flight guard (mandatory, before any lease)

```bash
python -m tools.accuracy.heldout_manifests --guard \\
    --receipt {guard_receipt} \\
    --dataset {guard_dataset}
```

At generation time the guard was evaluated against `{guard_receipt}` and returned
**{guard_status}** ({guard_reasons}). Substitute the round-2 freeze receipt once it
exists. The guard exits **0** only when the frozen-predictions receipt exists, is schema-valid,
`complete: true`, declares `evaluation_outcomes_read: false`, and its
`dataset.sha256` equals the current dataset's content hash. Otherwise it exits **3** with
`frozen_predictions_receipt_missing`, `..._incomplete`, `..._claims_outcomes_were_read`, or
`training_dataset_hash_changed_since_freeze`. Do not take a lease on a failed guard: a
measurement taken after the training set changed measures a model that was never frozen.

## 3. Device rules (non-negotiable)

{device_rules}

## 4. HostQueue command template

One manifest per job; one condition at a time on the granted die. The engine is the same one
the A2/A3 campaigns used (`run_block.sh`): preflight → launch → observe/finish → fetch.

```bash
# --- per job: set these five values (see §3) --------------------------------
HOST=173.125.1.2                 # A2;  A3: 192.168.13.165
CONTAINER=mindie-bench-a2-core-20261017   # A3: mindie-qwen35hist
PHY=<idle physical id>           # A2 card id;  A3 Phy-ID 0..15
SKU_ROOT=a2/2026-10-21           # A3: a3/2026-10-20
LANE=campaign                    # campaign | attention
TRIALS=2                         # A2 declared; A3: 3  (A2 currently measures 3, see report §2.5)
BLOCK=<manifest stem>            # e.g. a3-test2--Ascend910_9362--heldout-gemm

WT={worktree}
CAMP=$WT/.workstreams/heldout-campaign
RUN_ID="{run_prefix}-$(date -u +%Y%m%dT%H%M%SZ)"
ROOT=/tmp/$RUN_ID
DATA=/Volumes/data/mindie-sim-data/$SKU_ROOT/runs/$RUN_ID
RECORD=$CAMP/records/record-$RUN_ID.json
mkdir -p "$CAMP/raw/$RUN_ID" "$DATA"

# 0. guard — must print "guard": "pass" (exit 0) before any lease is requested
python -m tools.accuracy.heldout_manifests --guard \
    --receipt <freeze>/freeze-receipt.json --dataset {guard_dataset}

# 1. preflight — read-only, must show the target device process-free
ssh -o BatchMode=yes root@$HOST "npu-smi info -t proc-mem -i $PHY" | grep -q "No process in device"

# 2. launch exactly one manifest (the record is written once: never resubmit a run id)
if [ "$LANE" = "attention" ]; then
  MODULE=mindie_sim.a3_operator_worker_attention
  WORKER_ARGS="--run-id $RUN_ID --device npu:0"
else
  MODULE=mindie_sim.a3_operator_worker
  WORKER_ARGS="--device npu:0 --physical-npu $PHY --campaign-id $RUN_ID"
fi
/Users/maoxx241/code/coordinator/.venv/bin/python \\
  /private/tmp/wt-coord-53/tests/live/run_npu_stage1.py launch \\
    --record "$RECORD" --run-id "$RUN_ID" --host $HOST --container $CONTAINER \\
    --devices $PHY --root "$ROOT" --timeout 900 \\
    --command "cd $ROOT && mkdir -p results/$BLOCK && \\
      python3 -m $MODULE \\
        --manifest $CAMP/manifests/$BLOCK.json \\
        --results results/$BLOCK/results.json \\
        $WORKER_ARGS --warmup {warmup} --iterations {iterations} --trials $TRIALS \\
      > results/$BLOCK/stdout.log 2>&1"

# 3. observe until quiet, then finish (releases the lease)
/Users/maoxx241/code/coordinator/.venv/bin/python \\
  /private/tmp/wt-coord-53/tests/live/run_npu_stage1.py observe \\
    --record "$RECORD" --wait --timeout 1020 --finish

# 4. fetch the results, then copy them into the evidence tree (write-only from here on)
ssh root@$HOST "rm -rf /tmp/fetch-$RUN_ID && docker cp $CONTAINER:$ROOT/results/$BLOCK /tmp/fetch-$RUN_ID"
scp -r root@$HOST:/tmp/fetch-$RUN_ID "$DATA/results"
```

The attention worker requires `--run-id` and refuses any device other than `npu:0`; the
campaign worker takes `--physical-npu` and `--campaign-id`. A3 jobs must have the coordinator
inject the die (`ASCEND_RT_VISIBLE_DEVICES=<Phy-ID>`); A2 jobs get one card. Both then use
`npu:0` inside the container.

## 5. Declared attention values (not in the split declaration)

The attention lane needs three values the split declaration does not carry. They are declared
here, once, by this generator, and recorded in every attention case:

* `mask`: {mask_rule}
* `scale`: {scale_rule}
* RoPE: {rope_rule}

If the round wants a different rule, change the declaration first and regenerate — the case
ids and hashes change with it.

## 6. Collision proof

{collision_section}

## 7. Dry-run evidence

{dryrun_section}

## 8. What must not happen

1. **Do not score or fit anything from these outcomes before a freeze exists.** The freeze
   (`python -m tools.accuracy.freeze_predictions`) must complete and hash the training
   dataset before any heldout outcome file is read by a scorer.
2. **Do not merge protocols or dies.** Two dies, or two protocols, are two conditions. The
   dataset keeps them apart by construction; a hand-merged CSV undoes that.
3. **Do not relabel a split.** These manifests carry `split: "heldout"`; a runner that
   labels its rows `train` is a label forgery (leakage rule L4) and is rejected at ingest.
4. **Do not resubmit a run id.** The record file is created once; a second launch of the same
   id is an operator error, not a retry.
5. **Do not edit a manifest in place.** Regenerate from the declaration; the hash in
   `heldout-manifests.json` is what the freeze and the report cite.
"""


def render_runbook(index: Mapping[str, Any], collisions: Mapping[str, Any],
                   dryruns: Mapping[str, Any], arch: str, *, data_root: str) -> str:
    rows = []
    for item in index["manifests"]:
        rows.append(f"| `{item['hardware_key']}` | {item['arch']} | {item['lane']} | "
                    f"`{item['path']}` | {item['cases']} |")
    collision_section = (
        f"All **{collisions['heldout_conditions']}** heldout conditions were matched against "
        f"the **{collisions['train_conditions_considered']}** distinct training identities in "
        f"`.workstreams/dataset/observations.json`: **"
        f"{len(collisions['collisions'])} collisions**. "
        + ("The heldout set is disjoint from training."
           if collisions["passed"] else
           "COLLISIONS FOUND — see `collisions.json`."))
    dryrun_lines = []
    for name, result in sorted(dryruns.items()):
        dryrun_lines.append(f"* `{name}`: rc={result['returncode']} "
                            f"({result.get('summary') or result.get('detail') or ''})")
    trials = index["protocol"]["trials"]
    return RUNBOOK_TEMPLATE.format(
        arch=arch, generated_at=index["generated_at"],
        declaration_files=", ".join(f"`{name}`" for name in index["declaration_files"]),
        manifest_rows="\n".join(rows), total_cases=index["total_cases"],
        manifest_count=len(index["manifests"]),
        guard_receipt=index["guard"]["receipt"], guard_dataset=index["guard"]["dataset"],
        guard_status=index["guard"]["guard"],
        guard_reasons=", ".join(index["guard"]["reasons"]) or "no reasons",
        device_rules=DEVICE_RULES, worktree=str(REPO_ROOT), run_prefix=index["run_prefix"],
        data_root=data_root,
        warmup=index["protocol"]["warmup"],
        iterations=index["protocol"]["iterations"], trials=index["protocol"]["trials"],
        mask_rule=ATTENTION_DECLARATIONS["mask_rule"],
        scale_rule=ATTENTION_DECLARATIONS["scale_rule"],
        rope_rule=ATTENTION_DECLARATIONS["rope_rule"],
        collision_section=collision_section,
        dryrun_section="\n".join(dryrun_lines) or "not run (`--skip-dry-run`)")


DEVICE_RULES = """* **A2** — host `173.125.1.2`, container `mindie-bench-a2-core-20261017`, 8 cards,
  1 die per card, **one card per measurement**. Inside the container the worker sees
  `--device npu:0` and the wrapper adds `--physical-npu <Phy-ID>`; the lease grants exactly
  one card. Never run two A2 conditions on two cards in one job and call them repeats.
* **A3** — host `192.168.13.165`, container `mindie-qwen35hist`, 8 dual-die packages
  (Phy-ID `0..15`), **one physical die per condition**. The coordinator injects the single
  granted die as `ASCEND_RT_VISIBLE_DEVICES=<Phy-ID>`; inside the container the worker always
  uses `npu:0`, and `package = phy // 2`, `chip = phy % 2`. Conditions measured on different
  dies are separate provenance, never repeats.
* One manifest per job, `--timeout 900`, observe with `timeout + 120`. The record file is
  written once; a failed job is reported, never retried implicitly.
* Record the die, the container, the host and the clock snapshot with the results — the
  assembler refuses (`die: unknown`) to pretend it knows them later."""

MODULE_ARGS = """`--device npu:0 --physical-npu <phy> --campaign-id <id> --warmup W --iterations I --trials T`"""


# -------------------------------------------------------------------------- cli ---
def generate(*, declarations: Path, dataset: Path, out: Path, receipt: Path | None,
             skip_dry_run: bool, run_prefix: str) -> dict[str, Any]:
    points = heldout_points(declarations)
    if not points:
        raise IntegrityError("no_declared_heldout_points")
    dataset_document, _ = _dataset_document(dataset)
    train = dataset_identities(dataset_document, "train")
    all_points = [point for group in points.values() for point in group]
    collisions = collision_report(all_points, train)
    if not collisions["passed"]:
        raise IntegrityError(f"heldout_conditions_appear_in_training:{collisions['collisions'][:3]}")

    manifests = build_manifests(points)
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    index_entries = []
    dryruns: dict[str, Any] = {}
    for filename, manifest in sorted(manifests.items()):
        path = manifest_dir / filename
        write_json(path, manifest)
        lane = "attention" if manifest.get("kind") == "a3_attention_rope_manifest" else "campaign"
        hardware = next((point["hardware_key"] for point in all_points
                         if any(request.get("condition_key") ==
                                (request.get("condition") or {}).get("condition_key")
                                for request in manifest.get("requests") or [])), None)
        if hardware is None:
            hardware = next((point["hardware_key"] for point in all_points), None)
        counts = manifest_counts(manifest)
        entry = {"path": str(path.relative_to(out)), "filename": filename, "lane": lane,
                 "hardware_key": manifest.get("requests", [{}])[0].get("condition", {}).get(
                     "declaration", {}).get("hardware_key") or hardware,
                 "sha256": metrics.sha256_file(path),
                 "cases": counts["cases"], "split_counts": counts["split_counts"],
                 "recipes": counts.get("recipes"), "families": counts.get("families"),
                 "manifest_id": manifest.get("manifest_id") or manifest.get("campaign_id")}
        entry["hardware_key"] = ("Ascend910B4" if filename.startswith("a2--")
                                 else "Ascend910_9362")
        entry["arch"] = "A2" if filename.startswith("a2--") else "A3"
        index_entries.append(entry)
        if not skip_dry_run:
            result = dry_run(path, lane)
            payload = result.get("payload") or {}
            summary = payload.get("summary")
            if not isinstance(summary, dict) or not summary:
                # the campaign worker prints its summary object itself; the attention
                # worker prints {dry_run, plan}. Accept both without guessing.
                if any(key in payload for key in ("dry_run_validated", "attempted",
                                                  "status_counts")):
                    summary = payload
            plan = payload.get("plan") or {}
            validated = 0
            failed = 0
            if isinstance(summary, dict) and summary:
                validated = int(summary.get("dry_run_validated") or 0)
                failed = int(summary.get("failed") or 0) + int(summary.get("pending") or 0)
                result["status_counts"] = summary.get("status_counts")
            elif isinstance(plan, dict):
                validated = int((plan.get("counts") or {}).get("cases") or 0)
                result["status_counts"] = {"dry_run_validated": validated}
            result["validated"] = validated
            result["not_validated"] = failed
            result["summary"] = f"validated={validated}, failed={failed}"
            result["detail"] = result.get("stderr", "")[-200:]
            if failed:
                raise IntegrityError(f"manifest_dry_run_reported_{failed}_unvalidated:"
                                     f"{filename}")
            dryruns[filename] = result
            if not result["passed"]:
                raise IntegrityError(f"manifest_dry_run_failed:{filename}:"
                                     f"{result.get('stderr', '')[-300:]}")
    index = {
        "schema_version": "mindie-sim.heldout-manifests/1",
        "generated_at": utc_now(),
        "run_prefix": run_prefix,
        "declaration_files": {path.name: metrics.sha256_file(path)
                              for path in sorted(declarations.glob("*.json"))},
        "dataset": {"path": str(dataset),
                    "sha256": metrics.sha256_value(dataset_document["observations"])},
        "protocol": {"warmup": 10, "iterations": 30,
                     "trials": {"Ascend910B4": 2, "Ascend910_9362": 3}[
                         "Ascend910B4" if index_entries and
                         index_entries[0]["hardware_key"] == "Ascend910B4" else "Ascend910_9362"]},
        "device_env": "ASCEND_RT_VISIBLE_DEVICES=<Phy-ID> (A3) / --device npu:0 (both)",
        "manifests": index_entries,
        "total_cases": sum(entry["cases"] for entry in index_entries),
        "collisions": collisions,
        "guard": guard(receipt or out / "freeze-receipt.json", dataset),
        "dryruns": {name: {"returncode": result["returncode"], "passed": result["passed"],
                           "command": result["command"],
                           "validated": result.get("validated"),
                           "not_validated": result.get("not_validated"),
                           "status_counts": result.get("status_counts"),
                           "summary": result.get("summary"),
                           "detail": result.get("detail")} for name, result in dryruns.items()},
        "attention_declarations": ATTENTION_DECLARATIONS,
    }
    write_json(out / "heldout-manifests.json", index)
    write_json(out / "collisions.json", collisions)
    write_json(out / "dryruns.json", index["dryruns"])
    runbook = render_runbook(index, collisions, dryruns,
                            "A2 + A3", data_root="/Volumes/data/mindie-sim-data")
    (out / "heldout-runbook.md").write_text(runbook)
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declarations", type=Path, default=DEFAULT_DECLARATIONS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--receipt", type=Path, default=None,
                        help="frozen-predictions receipt the guard checks")
    parser.add_argument("--run-prefix", default="ho")
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument("--guard", action="store_true",
                        help="only run the pre-flight guard and exit")
    args = parser.parse_args(argv)
    receipt = args.receipt or (args.out / "freeze-receipt.json")
    if args.guard:
        result = guard(receipt, args.dataset)
        print(json.dumps(result, indent=2))
        return 0 if result["guard"] == "pass" else 3
    try:
        index = generate(declarations=args.declarations, dataset=args.dataset, out=args.out,
                         receipt=receipt, skip_dry_run=args.skip_dry_run,
                         run_prefix=args.run_prefix)
    except IntegrityError as exc:
        print(f"integrity: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"status": "manifests_generated",
                      "manifests": len(index["manifests"]),
                      "conditions": index["total_cases"],
                      "guard": index["guard"]["guard"],
                      "dryruns_passed": all(item["passed"] for item in
                                            index["dryruns"].values()),
                      "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
