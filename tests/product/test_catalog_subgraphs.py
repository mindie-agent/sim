"""Orthogonal tests for the 39+ catalog 1-layer subgraph harness (Gate A & Gate B).

[Role] Verify physical conservation invariants, independent KV/Indexer ledgers,
  aclgraph padded IterationPlan semantics, hardware perturbation continuity,
  live runner spec export bounds, pure-CPU runner builder correctness (KV slot
  writes, multi-seq prefill, owner-routed EP, HBM peak preflight, curFreq probe),
  Gate B 1:1 topology diff + streaming I/O + unfiltered max(10%, 2.0 us) scoring,
  and anti-rubber-stamp falsifications.
[Invariants] Pure tier0 CPU tests using canonical mindie_sim types and _support
  hardware axes; no device contexts, network, or hardcoded SKU literals in
  parametrize decorators.
[Timeouts & Locks] Fast bounded in-memory evaluations and tmp_path JSON fixtures;
  zero background processes or unbounded waits.
[Side-Effect Policy] Writes only inside pytest tmp_path fixtures.
"""
from __future__ import annotations

import importlib
import json
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim.catalog.fleet import ARCHITECTURE_PRESET_CATALOG, CORE_CATALOG_ENTRIES
from mindie_sim.model_catalog import load_model
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_1layer_npu_subgraphs as live_runner  # noqa: E402
import scripts.verify_catalog_subgraphs as harness  # noqa: E402

MIN_CATALOG_MODELS = 39
EXPECTED_ACTIVE_SLOTS = 48
EXPECTED_PAD_SLOTS = 16
EXPECTED_DECODE_CTX = 4096
SMALL_KERNEL_OBS_NS = 5000.0
SMALL_KERNEL_PASS_PRED_NS = 6900.0
SMALL_KERNEL_FAIL_PRED_NS = 7100.0
LARGE_KERNEL_OBS_NS = 100_000.0
LARGE_KERNEL_PASS_PRED_NS = 109_900.0
LARGE_KERNEL_FAIL_PRED_NS = 110_500.0
LOCKED_FREQ_MHZ = 1650.0
CURRENT_RUNNER_SHA = live_runner._sha256_file(REPO_ROOT / "scripts" / "run_1layer_npu_subgraphs.py")


def _qwen32b_valid_windows(sku: str | None = None) -> tuple[dict, dict, dict]:
    """Construct complete 3-window Qwen3-32B traces whose timings match default_spine() when sku is given."""
    d = {"attn_norm": 5.0, "attn_q_proj": 40.0, "attn_rope": 3.0, "attn_core": 35.0, "attn_o_proj": 40.0,
         "attn_other": 2.0, "ffn_norm": 5.0, "ffn_gate_up": 120.0, "ffn_swiglu": 10.0, "ffn_down": 65.0, "ffn_other": 2.0}
    if sku is not None:
        spine = harness.default_spine()
        stub = {"model": "qwen3-32b", "layer_index": 0, "phase": "decode", "overlay": "1layer_bf16_decode",
                "parallel": {"dp": 1, "tp": 4, "ep": 1, "dcp": 1, "pcp": 1, "o_proj_tp": 1, "linear_proj_tp": 1},
                "workload": {"sequences": 4, "active_sequences": 4, "q_len": 1, "context": 2048, "quant_mode": "bf16"}}
        step, ev, meta = harness._lower_capture_sim_step(stub, sku, spine, 1)
        ops = [o for o in step.graph.executable[0].ops if str(o.op_id).startswith(str(meta["target_layer_prefix"]))]
        st = harness._extract_sim_block_spans_and_buckets(ops, ev, world_size=1)
        for k, v in st["sim_fine_us"].items():
            d[k] = max(0.2, float(v))
    t = 10.0
    def _row(nm: str, tp: str, dur: float, sh: str, dt: str) -> dict:
        nonlocal t
        r = {"name": nm, "type": tp, "duration_us": round(dur, 4), "start_us": round(t, 4), "wait_us": 0.0,
             "stream_id": 1, "input_shapes": sh, "input_dtypes": dt}
        t += dur
        return r
    attn_k = [
        _row("RmsNorm", "RmsNorm", d["attn_norm"], "4,5120;5120", "DT_BF16;DT_BF16"),
        _row("aclnnMatmul_qkv", "MatMulV2", d["attn_q_proj"], "4,5120;5120,2560", "DT_BF16;DT_BF16"),
        _row("aclnnRotaryMul", "RotaryMul", d["attn_rope"], "4,16,1,128;4,1,1,128", "DT_BF16;DT_BF16"),
        _row("aclnnFusedInferAttentionScoreV4", "FusedInferAttentionScore", d["attn_core"], "4,16,1,128;67,128,128;67,128,128;4;4,16", "DT_BF16;DT_BF16;DT_BF16;INT64;INT32"),
        _row("aclnnMatmul_o", "MatMulV2", d["attn_o_proj"], "4,2048;2048,5120", "DT_BF16;DT_BF16"),
        _row("aclnnAdd_res", "Add", d["attn_other"], "4,5120;4,5120", "DT_BF16;DT_BF16"),
    ]
    ffn_k = [
        _row("aclnnAddRmsNorm", "AddRmsNorm", d["ffn_norm"], "4,5120;4,5120;5120", "DT_BF16;DT_BF16;DT_BF16"),
        _row("aclnnMatmul_gate_up", "MatMulV2", d["ffn_gate_up"], "4,5120;5120,12800", "DT_BF16;DT_BF16"),
        _row("aclnnSwiGlu", "SwiGlu", d["ffn_swiglu"], "4,12800", "DT_BF16"),
        _row("aclnnMatmul_down", "MatMulV2", d["ffn_down"], "4,6400;6400,5120", "DT_BF16;DT_BF16"),
        _row("aclnnAdd_ffn", "Add", d["ffn_other"], "4,5120;4,5120", "DT_BF16;DT_BF16"),
    ]
    layer_k = attn_k + ffn_k
    def _wrap(rows: list[dict]) -> dict:
        return {
            "stream_blocked": True, "blocker_expected_count": 4, "blocker_matched_count": 4,
            "observed_aicore_freq_mhz": LOCKED_FREQ_MHZ,
            **live_runner._span_from_kernels(rows), "kernels": rows,
        }
    return _wrap(attn_k), _wrap(ffn_k), _wrap(layer_k)


class _CpuTensor:
    """Thin NumPy-backed tensor shim supporting torch methods used by live_runner builders."""

    def __init__(self, arr: np.ndarray) -> None:
        self.arr = np.asarray(arr)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self.arr.shape)

    @property
    def dtype(self) -> Any: return self.arr.dtype
    def contiguous(self) -> _CpuTensor: return _CpuTensor(np.ascontiguousarray(self.arr))
    def to(self, dtype: Any) -> _CpuTensor: return _CpuTensor(self.arr.astype(dtype))
    def float(self) -> _CpuTensor: return self.to(np.float32)
    def view(self, *shape: int) -> _CpuTensor:
        sh = shape[0] if (len(shape) == 1 and isinstance(shape[0], tuple)) else shape
        return _CpuTensor(self.arr.reshape(sh))
    reshape = view
    def transpose(self, d0: int, d1: int) -> _CpuTensor: return _CpuTensor(np.swapaxes(self.arr, d0, d1))
    def unsqueeze(self, dim: int) -> _CpuTensor: return _CpuTensor(np.expand_dims(self.arr, axis=dim))
    def repeat(self, *reps: int) -> _CpuTensor: return _CpuTensor(np.tile(self.arr, reps))
    def repeat_interleave(self, k: int) -> _CpuTensor: return _CpuTensor(np.repeat(self.arr, k, axis=0))
    def split(self, sizes: tuple[int, ...], dim: int = -1) -> tuple[_CpuTensor, ...]:
        return tuple(_CpuTensor(p) for p in np.split(self.arr, np.cumsum(sizes)[:-1], axis=dim))
    def sum(self, dim: int = 0) -> _CpuTensor: return _CpuTensor(self.arr.sum(axis=dim))
    def tolist(self) -> Any: return self.arr.tolist()
    def _unwrap(self, idx: Any) -> Any:
        if isinstance(idx, _CpuTensor):
            return idx.arr
        return tuple(x.arr if isinstance(x, _CpuTensor) else x for x in idx) if isinstance(idx, tuple) else idx
    def __getitem__(self, idx: Any) -> _CpuTensor: return _CpuTensor(self.arr[self._unwrap(idx)])
    def __setitem__(self, idx: Any, val: Any) -> None: self.arr[self._unwrap(idx)] = val.arr if isinstance(val, _CpuTensor) else val
    def __mul__(self, other: Any) -> _CpuTensor: return _CpuTensor(self.arr * (other.arr if isinstance(other, _CpuTensor) else other))
    def __add__(self, other: Any) -> _CpuTensor: return _CpuTensor(self.arr + (other.arr if isinstance(other, _CpuTensor) else other))
    def __sub__(self, other: Any) -> _CpuTensor: return _CpuTensor(self.arr - (other.arr if isinstance(other, _CpuTensor) else other))
    def __lt__(self, other: Any) -> _CpuTensor: return _CpuTensor(self.arr < (other.arr if isinstance(other, _CpuTensor) else other))
    def __gt__(self, other: Any) -> _CpuTensor: return _CpuTensor(self.arr > (other.arr if isinstance(other, _CpuTensor) else other))
    def __radd__(self, other: Any) -> _CpuTensor: return _CpuTensor((other.arr if isinstance(other, _CpuTensor) else other) + self.arr)
    def __mod__(self, mod_val: int) -> _CpuTensor: return _CpuTensor(np.mod(self.arr, mod_val))
    def __floordiv__(self, div_val: int) -> _CpuTensor: return _CpuTensor(np.floor_divide(self.arr, div_val))
    def __truediv__(self, div_val: Any) -> _CpuTensor: return _CpuTensor(self.arr / (div_val.arr if isinstance(div_val, _CpuTensor) else div_val))
    def __rtruediv__(self, num: Any) -> _CpuTensor: return _CpuTensor((num.arr if isinstance(num, _CpuTensor) else num) / self.arr)
    def __rpow__(self, base: Any) -> _CpuTensor: return _CpuTensor(np.power(base, self.arr))
    def expand(self, *shape: int) -> _CpuTensor: return _CpuTensor(np.broadcast_to(self.arr, shape).copy())
    def clamp(self, min: int | None = None, max: int | None = None) -> _CpuTensor: return _CpuTensor(np.clip(self.arr, min, max))
    def scatter_add_(self, dim: int, idx: _CpuTensor, src: _CpuTensor) -> _CpuTensor:
        del dim
        np.add.at(self.arr, idx.arr.astype(np.int64), src.arr)
        return self


def _make_cpu_torch_mocks(all_to_all_cb: Any = None, router_experts: list[int] | None = None, w_val: float = 1.0, qk_norm_scale: float = 1.0) -> tuple[Any, Any, dict[str, Any]]:
    """Build lightweight NumPy-backed torch + torch_npu mocks for pure-CPU runner builder tests."""
    trace: dict[str, Any] = {"gmm_calls": [], "fia_calls": [], "rope_calls": 0}
    route = router_experts or [0]

    torch_mod = SimpleNamespace(
        bfloat16=np.float32, float32=np.float32, int32=np.int32, int64=np.int64, int8=np.int8, bool=np.bool_,
        randn=lambda sh, dtype=np.float32, device="cpu": _CpuTensor(np.full(sh, w_val, dtype=dtype)),
        ones=lambda sh, dtype=np.float32, device="cpu": _CpuTensor(np.ones(sh, dtype=dtype)),
        zeros=lambda sh, dtype=np.float32, device="cpu": _CpuTensor(np.zeros(sh, dtype=dtype)),
        empty=lambda sh, dtype=np.float32, device="cpu": _CpuTensor(np.zeros(sh, dtype=dtype)),
        empty_like=lambda a: _CpuTensor(np.zeros_like(a.arr)),
        arange=lambda *a, dtype=np.int64, device="cpu": _CpuTensor(np.arange(*a, dtype=dtype)),
        outer=lambda a, b: _CpuTensor(np.outer(a.arr, b.arr)),
        cos=lambda a: _CpuTensor(np.cos(a.arr)),
        sin=lambda a: _CpuTensor(np.sin(a.arr)),
        cat=lambda seq, dim=-1: _CpuTensor(np.concatenate([x.arr for x in seq], axis=dim)),
        argsort=lambda a, stable=True: _CpuTensor(np.argsort(a.arr, kind="stable").astype(np.int64)),
        bincount=lambda a, minlength=0: _CpuTensor(np.bincount(a.arr.astype(np.int64), minlength=minlength).astype(np.int64)),
        cumsum=lambda a, dim=0: _CpuTensor(np.cumsum(a.arr, axis=dim)),
        div=lambda a, b, rounding_mode=None: _CpuTensor(np.floor_divide(a.arr, b.arr if isinstance(b, _CpuTensor) else b) if rounding_mode == "floor" else a.arr / (b.arr if isinstance(b, _CpuTensor) else b)),
        where=lambda cond, x, y: _CpuTensor(np.where(cond.arr, x.arr if isinstance(x, _CpuTensor) else x, y.arr if isinstance(y, _CpuTensor) else y)),
        repeat_interleave=lambda v, r: _CpuTensor(np.repeat(v.arr, r.arr.astype(np.int64))),
        matmul=lambda a, b: _CpuTensor(np.matmul(a.arr, b.arr)),
        triu=lambda a, diagonal=1: _CpuTensor(np.triu(a.arr, k=diagonal)),
        distributed=SimpleNamespace(all_to_all_single=all_to_all_cb, all_reduce=lambda *a, **kw: None),
        nn=SimpleNamespace(functional=SimpleNamespace(silu=lambda x: x)),
    )

    def _fia(q: _CpuTensor, kc: _CpuTensor, vc: _CpuTensor, **kw: Any) -> tuple[_CpuTensor, None]:
        trace["fia_calls"].append({"q": q, "kc": kc, "vc": vc, "kw": kw})
        return _CpuTensor(np.zeros_like(q.arr)), None

    def _gmm(xs: list[_CpuTensor], ws: list[_CpuTensor], group_list: _CpuTensor, **kw: Any) -> list[_CpuTensor]:
        del kw
        x, w = xs[0], ws[0]
        trace["gmm_calls"].append({"rows": x.shape[0], "group_list": group_list.tolist()})
        out = np.zeros((x.shape[0], w.shape[2]), dtype=x.dtype)
        off = 0
        for e_id, cnt in enumerate(group_list.tolist()):
            if cnt > 0:
                out[off:off + cnt] = np.matmul(x.arr[off:off + cnt], w.arr[e_id])
                off += cnt
        return [_CpuTensor(out)]

    def _gating(logits: _CpuTensor, k: int) -> tuple[_CpuTensor, _CpuTensor, None]:
        row = (route * ((k + len(route) - 1) // len(route)))[:k]
        idx_arr = np.tile(np.asarray(row, dtype=np.int32), (logits.shape[0], 1))
        return _CpuTensor(np.ones((logits.shape[0], k), dtype=np.float32)), _CpuTensor(idx_arr), None

    def _rms(x: _CpuTensor, w: _CpuTensor, eps: float) -> tuple[_CpuTensor, None]:
        del eps
        sc = qk_norm_scale if (len(x.shape) == 4 and w.shape == (x.shape[-1],)) else 1.0
        return _CpuTensor(x.arr * sc), None

    def _rope(x: _CpuTensor, cos: _CpuTensor, sin: _CpuTensor) -> _CpuTensor:
        del cos, sin
        trace["rope_calls"] += 1
        return x

    torch_npu_mod = SimpleNamespace(
        npu_rms_norm=_rms,
        npu_add_rms_norm=lambda x, r, w, eps: (x, None, r),
        npu_rotary_mul=_rope,
        npu_swiglu=lambda x: _CpuTensor(x.arr[:, : x.shape[1] // 2]),
        npu_fused_infer_attention_score=_fia,
        npu_fused_infer_attention_score_v2=_fia,
        npu_grouped_matmul=_gmm,
        npu_moe_gating_top_k_softmax=_gating,
    )
    return torch_mod, torch_npu_mod, trace


def test_zero_device_context_on_import() -> None:
    """Neither harness nor live runner imports torch_npu at module import time."""
    assert "torch_npu" not in sys.modules
    mod = importlib.import_module("scripts.run_1layer_npu_subgraphs")
    assert hasattr(mod, "CapabilityAbsentError")
    assert "torch_npu" not in sys.modules


def test_export_specs_covers_all_models_distinct_signatures_and_honest_budget() -> None:
    """Exported 1-layer specs cover all 43 models, distinguish ALL LayerPlan fields, and reject duplicates."""
    all_expected = set(CORE_CATALOG_ENTRIES) | set(ARCHITECTURE_PRESET_CATALOG)
    assert len(all_expected) >= MIN_CATALOG_MODELS
    with pytest.raises(ValueError, match="duplicate_models"):
        harness.export_live_runner_specs(["qwen3-32b", "qwen3-32b"])

    exported = harness.export_live_runner_specs(include_multirank=True)
    assert exported["spec_sha256"] == harness._compute_manifest_cases_digest(exported["cases"])
    assert exported["model_count"] == len(all_expected) and all_expected <= {c["model"] for c in exported["cases"]}

    dit_cases = [c for c in exported["cases"] if c["model"] == "hunyuan-video-dit"]
    assert dit_cases and dit_cases[0]["case_id"].endswith("_pre") and dit_cases[0]["overlay"] == "1layer_bf16_prefill"
    pad_cases = [c for c in exported["cases"] if "padded" in c["case_id"]]
    assert pad_cases and all(c["overlay"] == "eager_padded_b8_a6" for c in pad_cases)

    # Full LayerPlan signature distinguishes BOTH head_dim AND candidate_filter changes
    dsv41 = load_model("deepseek-v4.1-flash")
    sig_base = harness._layer_structural_signature(dsv41, dsv41.layer_plans[25])
    assert harness._layer_structural_signature(dsv41, replace(dsv41.layer_plans[25], head_dim=256)) != sig_base
    assert harness._layer_structural_signature(dsv41, replace(dsv41.layer_plans[25], candidate_filter=False)) != sig_base

    dsv41_layers = {c["layer_index"]: c for c in exported["cases"] if c["model"] == "deepseek-v4.1-flash" and c["world_size"] == 1}
    assert dsv41_layers[2]["compress_ratio"] == 2 and dsv41_layers[20]["compress_ratio"] == 1
    assert dsv41_layers[2]["owns_long_kv"] is True and dsv41_layers[3]["owns_long_kv"] is False


def test_physical_tolerance_and_non_positive_rejection() -> None:
    """Unified max(10%, 2.0 us) tolerance rejects zero/negative/NaN observations."""
    assert not harness.exceeds_physical_tolerance(SMALL_KERNEL_PASS_PRED_NS, SMALL_KERNEL_OBS_NS)
    assert harness.exceeds_physical_tolerance(SMALL_KERNEL_FAIL_PRED_NS, SMALL_KERNEL_OBS_NS)
    assert not harness.exceeds_physical_tolerance(LARGE_KERNEL_PASS_PRED_NS, LARGE_KERNEL_OBS_NS)
    assert harness.exceeds_physical_tolerance(LARGE_KERNEL_FAIL_PRED_NS, LARGE_KERNEL_OBS_NS)
    assert not harness._score_latency_pair(100.0, 0.0)["pass_physical_tol"]
    assert not harness._score_latency_pair(0.0, 100.0)["pass_physical_tol"]
    assert not harness._score_latency_pair(float("nan"), 100.0)["pass_physical_tol"]


def test_kv_ledger_independent_and_non_tautological_conservation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent KV/Indexer ledger catches zeroed runtime cache & patched index_row_bytes."""
    import mindie_sim.catalog.cache as cache_mod

    mla_spec = load_model("deepseek-v4.1-flash")
    base_audit = harness._kv_and_indexer_memory_audit(mla_spec, ParallelSpec(dp=8, tp=1, ep=8))
    assert base_audit["ledger_matches_runtime"] is True and base_audit["indexer_bytes_per_token_per_rank"] > 0.0

    # Zeroing runtime cache_mod.index_row_bytes MUST NOT zero the independent ledger!
    monkeypatch.setattr(cache_mod, "index_row_bytes", lambda *a, **kw: 0.0)
    patched = harness._kv_and_indexer_memory_audit(mla_spec, ParallelSpec(dp=8, tp=1, ep=8))
    assert patched["indexer_bytes_per_token_per_rank"] > 0.0 and patched["ledger_matches_runtime"] is False

    findings: list[str] = []
    assert harness._check_ratio_non_tautological("dp_attn_flops", 0.0, 0.0, 1e9, findings) == 0.0
    assert any("zero_work_with_positive_expected" in f for f in findings)
    mag_findings: list[str] = []
    harness._check_algebraic_magnitude("dit_attn", 0.04, 1e12, mag_findings)
    assert any("outside_algebraic_bounds" in f for f in mag_findings)


def test_aclgraph_padded_and_hardware_perturbation_continuity() -> None:
    """RequestView(padding=True) preserves zero logical KV and hardware sweeps remain monotonic."""
    active = tuple(RequestView(request_id=f"a{i}", q_len=1, context=EXPECTED_DECODE_CTX, samples=True, phase="decode") for i in range(EXPECTED_ACTIVE_SLOTS))
    padded = tuple(RequestView(request_id=f"p{i}", q_len=1, context=EXPECTED_DECODE_CTX, samples=False, phase="decode", padding=True) for i in range(EXPECTED_PAD_SLOTS))
    plan = IterationPlan(requests=active + padded, iteration=0)
    assert plan.valid_tokens == EXPECTED_ACTIVE_SLOTS and all(v.kv_len == 0 for v in padded)

    spine = harness.default_spine()
    base = harness.parse_case(harness.compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a5"], tp=4, ep=1, num_devices=4, layers=1))
    p_active = tuple(RequestView(request_id=f"a{i}", q_len=1, context=2048, samples=True, phase="decode") for i in range(6))
    p_pad = tuple(RequestView(request_id=f"p{i}", q_len=1, context=2048, samples=False, phase="decode", padding=True) for i in range(2))
    m_pad = harness._step_conservation_metrics(spine.lower(base, harness.LoweringRequest(iterations=((0, IterationPlan(requests=p_active + p_pad)),))), spine)
    m_act = harness._step_conservation_metrics(spine.lower(base, harness.LoweringRequest(iterations=((0, IterationPlan(requests=p_active)),))), spine)
    assert m_pad["attn_flops"] == pytest.approx(m_act["attn_flops"], rel=0.015) and m_pad["proj_flops"] > m_act["proj_flops"]

    pre_req = harness.LoweringRequest(step=harness.StepRequest(phase="prefill", tokens=1024, context=0, sequences=1, scope="body"))
    dec_req = harness.LoweringRequest(step=harness.StepRequest(phase="decode", tokens=1, context=2048, sequences=4, scope="body"))
    t28 = spine.run_step(spine.lower(replace(base, machine=harness.derive_hardware_sku(base.machine, sku="s-28", aic_count=28)), pre_req), harness.ReplayOptions(include_submission=False)).span_ms
    t36 = spine.run_step(spine.lower(replace(base, machine=harness.derive_hardware_sku(base.machine, sku="s-36", aic_count=36)), pre_req), harness.ReplayOptions(include_submission=False)).span_ms
    assert t28 > t36 > 0.0
    t_hi = spine.run_step(spine.lower(replace(base, machine=harness.derive_hardware_sku(base.machine, sku="s-hi", hbm_peak_bytes_per_s=4.0e12)), dec_req), harness.ReplayOptions(include_submission=False)).span_ms
    t_lo = spine.run_step(spine.lower(replace(base, machine=harness.derive_hardware_sku(base.machine, sku="s-lo", hbm_peak_bytes_per_s=1.4e12)), dec_req), harness.ReplayOptions(include_submission=False)).span_ms
    assert t_lo > t_hi > 0.0


def test_gate_b_provenance_streaming_manifest_positive_topo_and_privacy(tmp_path: Path) -> None:
    """Gate B enforces end-to-end pass_full_case, strict streaming JSON, empty/tampered manifests, tiers, and privacy."""
    sku = REPRESENTATIVE_SKUS["a3"]
    attn_w, ffn_w, layer_w = _qwen32b_valid_windows(sku=sku)
    base_case = {
        "case_id": "qwen3-32b__L0_gqa_dense_bf16_dec",
        "model": "qwen3-32b", "layer_index": 0, "attention": "gqa", "architecture": "gqa", "ffn": "dense",
        "overlay": "1layer_bf16_decode", "phase": "decode",
        "parallel": {"dp": 1, "tp": 4, "ep": 1, "dcp": 1, "pcp": 1, "o_proj_tp": 1, "linear_proj_tp": 1},
        "workload": {"sequences": 4, "active_sequences": 4, "q_len": 1, "context": 2048, "quant_mode": "bf16"},
        "coverage_tier": harness.TIER_TORCHNPU_COMPOSITE, "status": "ok",
        "attention_capture": attn_w, "ffn_capture": ffn_w, "layer_capture": layer_w,
    }
    r0_rec = live_runner._build_rank_receipt(0, base_case, freq_mhz=LOCKED_FREQ_MHZ)
    single_bundle = {
        "schema_version": live_runner.SCHEMA_VERSION, "shard_complete": True, "expected_cases": 1,
        "runner_sha256": CURRENT_RUNNER_SHA, "hardware_soc": sku, "cann_version": "9.3.0",
        "world_size": 1, "observed_aicore_freq_mhz": LOCKED_FREQ_MHZ,
        "cases": [{**base_case, "rank_receipts": [r0_rec]}],
    }

    # End-to-end positive pass_full_case=True + status=="pass" + O(1) header/case streaming
    one_dir = tmp_path / "one_case"
    one_dir.mkdir()
    bundle_p = one_dir / "s0.json"
    bundle_p.write_text(json.dumps(single_bundle), encoding="utf-8")
    hdr, n_cases = harness._stream_json_bundle_header(bundle_p)
    assert hdr["hardware_soc"] == sku and n_cases == 1 and len(list(harness._iter_json_bundle_cases(bundle_p))) == 1
    w4_res = harness.run_gate_b(one_dir, workers=4)
    assert w4_res["status"] == "pass" and w4_res["shard_integrity_errors"] == []
    assert w4_res["rows"][0]["pass_full_case"] is True and w4_res["rows"][0]["topology_isomorphic"] is True

    # Truncated JSON, trailing comma/junk, shard_complete="false", world_size="four", empty manifest all fail closed
    for tag, raw_txt in (
        ("trunc", json.dumps(single_bundle)[:-2]),
        ("junk", json.dumps(single_bundle) + " trailing"),
        ("tcomma", '{"cases":[{"a":1},],"shard_complete":true}'),
        ("str_false", json.dumps({**single_bundle, "shard_complete": "false"})),
        ("bad_ws", json.dumps({**single_bundle, "world_size": "four"})),
    ):
        d = tmp_path / f"mal_{tag}"
        d.mkdir()
        (d / "b.json").write_text(raw_txt, encoding="utf-8")
        r_m = harness.run_gate_b(d, workers=1)
        assert r_m["status"] == "fail" and r_m["shard_integrity_errors"]
    empty_m = tmp_path / "empty_m.json"
    empty_m.write_text(json.dumps({"spec_sha256": harness._compute_manifest_cases_digest([]), "cases": []}), encoding="utf-8")
    r_em = harness.run_gate_b(one_dir, workers=1, spec_manifest=empty_m)
    assert "empty_spec_manifest" in r_em["shard_integrity_errors"] and r_em["rows"][0]["status"] == "spec_mismatch"

    # Missing expected SKU + tampered inner manifest digest (with privacy redaction) + invalid tiers + peer error
    m_cases = [
        {"case_id": base_case["case_id"], "model": "qwen3-32b", "layer_index": 0, "world_size": 1, "phase": "decode",
         "attention": "gqa", "architecture": "gqa", "ffn": "dense", "overlay": "1layer_bf16_decode",
         "dp": 1, "tp": 4, "ep": 1, "dcp": 1, "pcp": 1, "o_proj_tp": 1, "linear_proj_tp": 1,
         "sequences": 4, "active_sequences": 4, "q_len": 1, "context": 2048, "quant_mode": "bf16"},
        {"case_id": "qwen3-32b__L0_mr4_pcp4_pre", "model": "qwen3-32b", "layer_index": 0, "world_size": 4, "phase": "prefill",
         "attention": "gqa", "architecture": "gqa", "ffn": "dense", "overlay": "mr4_pcp4_pre",
         "dp": 1, "tp": 1, "ep": 1, "dcp": 1, "pcp": 4, "o_proj_tp": 1, "linear_proj_tp": 1,
         "sequences": 1, "active_sequences": 1, "q_len": 1024, "context": 0, "quant_mode": "bf16"},
    ]
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(json.dumps({"spec_sha256": "fd12:3456::1%en0", "cases": m_cases}), encoding="utf-8")
    miss_all = harness.run_gate_b(one_dir, workers=1, spec_manifest=manifest_p)
    assert any("manifest_inner_spec_sha256_mismatch" in e and "fd12:3456" not in e for e in miss_all["shard_integrity_errors"])
    assert any(r["status"] == "missing_capture" and r["hardware_soc"] == harness.DEFAULT_A5_SKU for r in miss_all["rows"])
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad_bundle = {
        **single_bundle, "runner_sha256": "unrelated", "expected_cases": 4,
        "cases": [
            {**base_case, "case_id": "unknown_tier", "coverage_tier": "custom_fake_tier", "rank_receipts": [r0_rec]},
            {**base_case, "case_id": "promo_tier", "coverage_tier": harness.TIER_TORCHNPU_COMPOSITE, "native_vllm_layer_executed": True, "rank_receipts": [r0_rec]},
            {**base_case, "case_id": "relabeled_qwen", "architecture": "dit", "attention": "gdn", "ffn": "moe", "rank_receipts": [r0_rec]},
            {"case_id": "peer_fail_absent", "model": "qwen3-32b", "layer_index": 0, "coverage_tier": harness.TIER_UNMERGED_ABSENT,
             "status": "absent_custom_op", "rank_receipts": [{"rank": 0, "status": "absent_custom_op"}, {"rank": 1, "status": "runtime_error"}]},
        ],
    }
    (bad_dir / "bad.json").write_text(json.dumps(bad_bundle), encoding="utf-8")
    bad_res = harness.run_gate_b(bad_dir, workers=1)
    by_id = {r["case_id"]: r for r in bad_res["rows"]}
    assert by_id["unknown_tier"]["status"] == "invalid_capture" and by_id["promo_tier"]["status"] == "invalid_capture"
    assert by_id["relabeled_qwen"]["status"] == "invalid_capture" and by_id["peer_fail_absent"]["status"] == "peer_runtime_error"
    assert bad_res["blocked_cases"] == 0 and harness.main(["--gate-b", "--captures-dir", str(tmp_path / "none"), "--diagnostic-only"]) == 0


def test_gate_b_anti_rubber_stamp_falsifications() -> None:
    """Reproduce Astra Max R1-R4 counterexamples: 4x QKV no-FFN, q_a/q_b split, A5 hcom comms, bogus receipts."""
    # 1. Impossible receipt (span=347, busy=1, gap=0, streams=999, sum_k=-1) + kernel_count="N/A"
    bogus = harness._verify_rank_receipts(
        {"rank_receipts": [
            {"rank": 0, "status": "ok", "kernel_count": 20, "shape_digest": "0" * 16, "elapsed_span_us": 347.0,
             "union_busy_us": 1.0, "uncovered_gap_us": 0.0, "sum_kernel_us": -1.0, "stream_count": 999,
             "blocker_matched_count": 4, "stream_blocked": True, "observed_aicore_freq_mhz": 1650.0},
            {"rank": 1, "status": "ok", "kernel_count": "N/A", "shape_digest": "0" * 16, "elapsed_span_us": 347.0,
             "union_busy_us": 347.0, "uncovered_gap_us": 0.0, "stream_count": 1, "blocker_matched_count": 4,
             "stream_blocked": True, "observed_aicore_freq_mhz": 1650.0},
        ]},
        world_size=2,
    )
    assert not bogus["rank_parity_ok"]
    a_w, f_w, _ = _qwen32b_valid_windows()
    k_fwd = a_w["kernels"]
    k_rev = [{**r, "start_us": 200.0 - float(r["start_us"])} for r in reversed(k_fwd)]
    assert harness._compute_layer_shape_digest(k_fwd) != harness._compute_layer_shape_digest(k_rev)
    assert not harness._check_window_clocks_consistent({**a_w, "observed_aicore_freq_mhz": 0.0}, f_w, a_w, 1650.0)[0]
    assert not harness._check_window_clocks_consistent({}, {}, {}, 1650.0)[0]

    # 2. Duplicate gate-up in place of down, K/V=(1,), reversed layer, and 4x QKV no-FFN all fail closed
    qkv_only_k = [
        {"name": f"qkv{i}", "type": "MatMulV2", "start_us": 10.0 + i * 50.0, "duration_us": 50.0, "stream_id": 1, "input_shapes": "4,5120;5120,2560"}
        for i in range(4)
    ] + [{"name": "fia", "type": "FusedInferAttentionScore", "start_us": 210.0, "duration_us": 115.0, "stream_id": 1, "input_shapes": "4,16,1,128;67,128,128;67,128,128"}]
    qkv_only_w = {"stream_blocked": True, "blocker_expected_count": 4, "blocker_matched_count": 4, **live_runner._span_from_kernels(qkv_only_k), "kernels": qkv_only_k}
    assert not harness._reconcile_whole_layer_windows(a_w, f_w, qkv_only_w, harness._recompute_window_timing(a_w), harness._recompute_window_timing(f_w), harness._recompute_window_timing(qkv_only_w))
    d_bad: list[str] = []
    harness._check_gemm_and_attn_shapes(
        [{"type": "MatMulV2", "input_shapes": "4,5120;5120,10240", "duration_us": 5.0}, {"type": "MatMulV2", "input_shapes": "4,8192;8192,5120", "duration_us": 5.0},
         {"type": "MatMulV2", "input_shapes": "4,5120;5120,51200", "duration_us": 5.0}, {"type": "MatMulV2", "input_shapes": "4,5120;5120,51200", "duration_us": 5.0},
         {"type": "SwiGlu", "input_shapes": "4,51200", "duration_us": 2.0}, {"type": "FusedInferAttentionScoreV2", "input_shapes": "4,64,1,128;1;1", "duration_us": 9.0}],
        harness.Counter(["matmul", "attention"]), harness.Counter(["matmul", "attention"]),
        expected_tokens=4, expected_hidden=5120, expected_head_dims=frozenset({128}), expected_kv_dims=frozenset({128, 1024}),
        expected_gemm_widths=frozenset({10240, 8192, 51200, 25600}), expected_attn_gemm_widths=frozenset({10240, 8192}),
        expected_attn_in_widths=frozenset({10240}), expected_attn_o_widths=frozenset({8192}), expected_ffn_gemm_widths=frozenset({51200, 25600}),
        expected_ffn_gate_up_widths=frozenset({51200}), expected_ffn_down_widths=frozenset({25600}), is_moe=False, expected_min_gemms=4, expected_local_heads=64, topo_diffs=d_bad,
    )
    assert "missing_executed_ffn_down_gemm" in d_bad and any("observed_attention_shape_mismatch" in d for d in d_bad)

    # 3. A5 hcom_*AicpuKernel is collective (NOT helper); A3 allgatherAicpuKernel IS helper; missing/oversized payload fails
    assert harness._classify_csv_kernel({"type": "hcom_allGather_AicpuKernel", "name": "hcom_allGather_AicpuKernel_1"}) == "collective"
    assert harness._classify_csv_kernel({"type": "allgatherAicpuKernel", "name": "allgatherAicpuKernel"}) == "control_helper"
    class _FakeOp:
        def __init__(self, op_id: str, family: str, link_bytes: float = 0.0, stream: str = "main") -> None:
            self.op_id, self.family, self.link_bytes, self.metadata, self.stream = op_id, family, link_bytes, {}, stream
    sim_ops = [_FakeOp("s0.l0.qkv", "matmul"), _FakeOp("s0.l0.attn", "attention"), _FakeOp("s0.l0.ag", "allgather", link_bytes=3_145_728.0, stream="comm")]
    no_pay = [
        {"name": "mm", "type": "MatMulV2", "start_us": 1.0, "duration_us": 5.0, "stream_id": 1, "input_shapes": "4,5120;5120,2560"},
        {"name": "fia", "type": "FusedInferAttentionScore", "start_us": 2.0, "duration_us": 5.0, "stream_id": 1, "input_shapes": "4,16,1,128;64,128,128;64,128,128"},
        {"name": "ag", "type": "hcom_allGather_AicpuKernel", "start_us": 2.0, "duration_us": 5.0, "stream_id": 2},
    ]
    assert not harness._verify_topology_isomorphism(sim_ops, no_pay, world_size=4, expected_tokens=4, expected_hidden=5120, expected_head_dims=frozenset({128}), expected_local_heads=16)["collectives_verified"]
    big_pay = [{**r, "payload_bytes": 300 * 1024 * 1024, "comm_group_size": 8} if r["name"] == "ag" else r for r in no_pay]
    assert not harness._verify_topology_isomorphism(sim_ops, big_pay, world_size=4, expected_tokens=4, expected_hidden=5120, expected_head_dims=frozenset({128}), expected_local_heads=16)["topology_isomorphic"]

    # 4. q_a vs q_b split prevents cancellation; worst_l1_outlier ranks failing bucket ahead of passing small bucket
    mla_obs = harness._bucket_obs_fine_us(
        [{"name": "m0", "type": "MatMulV2", "start_us": 1.0, "duration_us": 8.1, "input_shapes": "4,7168;7168,1536"},
         {"name": "m1", "type": "MatMulV2", "start_us": 10.0, "duration_us": 143.8, "input_shapes": "4,1536;1536,6144"}],
        [], is_moe=True, kv_a_out_dim=576, q_lora_rank=1536,
    )
    assert mla_obs["attn_q_a_proj"] == pytest.approx(8.1) and mla_obs["attn_q_b_proj"] == pytest.approx(143.8)
    _, _, _, worst, _ = harness._score_l1_in_l2_buckets({"small": 1.6, "large": 120.0}, {"small": 0.1, "large": 100.0})
    assert worst is not None and worst["bucket"] == "large" and not worst["pass_physical_tol"]


def test_live_runner_cpu_builders_kv_slot_prefill_owner_ep_and_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure-CPU runner checks: multi-page KV + K-norm + RoPE, [3,6] prefill, top_k=3 EP2, KDA HBM, curFreq, blocker."""
    # 1. Validation & peak HBM preflight (Kimi-K2 > 5 GiB; Kimi-K3 KDA FP32 recurrent state > 12 GiB!)
    with pytest.raises(ValueError, match="moe_requires_positive_num_experts"):
        live_runner._validate_case_spec({"attention": "gqa", "ffn": "moe", "num_experts": 0, "quant_mode": "bf16", "phase": "decode"})
    kimi = next(c for c in harness.export_live_runner_specs(["kimi-k2"])["cases"] if c["layer_index"] == 1 and c["world_size"] == 1)
    assert live_runner._estimate_case_allocation_bytes(kimi) + live_runner.HBM_HEADROOM_BYTES > 5 * 1024**3
    kda_big = {"hidden_size": 7168, "attention": "kda", "ffn": "dense", "intermediate_size": 7168, "phase": "decode", "tp": 1, "ep": 1, "sequences": 2048, "q_len": 1, "linear_num_v_heads": 96, "linear_k_head_dim": 128, "linear_v_head_dim": 128, "quant_mode": "bf16"}
    assert live_runner._estimate_case_allocation_bytes(kda_big) > 12 * 1024**3

    # 2. GQA decode writes multi-page slots [126,127,0,1] across pages 0 & 1 (ctx=130, blk=128, q_len=4), runs RoPE & K-norm; honors small ctx & SWA window!
    tm, nm, tr = _make_cpu_torch_mocks(w_val=1.0, qk_norm_scale=3.0)
    gqa_dec = {"hidden_size": 8, "num_heads": 2, "num_kv_heads": 1, "head_dim": 4, "attention": "gqa", "phase": "decode", "tp": 1, "dcp": 1, "sequences": 2, "active_sequences": 1, "q_len": 4, "context": 130, "block_size": 128, "quant_mode": "bf16", "qk_norm": True}
    live_runner._build_gqa_attention(tm, nm, gqa_dec, "cpu")[0](_CpuTensor(np.full((8, 8), 2.0, dtype=np.float32)))
    kc = tr["fia_calls"][0]["kc"].arr
    exp_k = 48.0 / float(np.sqrt(8.0))
    assert tr["rope_calls"] == 2 and kc[0, 126, 0] == pytest.approx(exp_k) and kc[1, 1, 0] == pytest.approx(exp_k) and kc[2, 126, 0] == pytest.approx(1.0)
    tm_sw, nm_sw, tr_sw = _make_cpu_torch_mocks(w_val=1.0)
    live_runner._build_gqa_attention(tm_sw, nm_sw, {**gqa_dec, "sequences": 1, "active_sequences": 1, "q_len": 1, "context": 512, "window": 32}, "cpu")[0](_CpuTensor(np.ones((1, 8), dtype=np.float32)))
    assert tr_sw["fia_calls"][0]["kw"]["actual_seq_kvlen"] == [32]

    # 3. Multi-seq prefill passes [3, 6] + executes RoPE; sparse prefill raises RunnerUnsupportedOverlayError
    tm2, nm2, tr2 = _make_cpu_torch_mocks(w_val=1.0)
    gqa_pre = {**gqa_dec, "phase": "prefill", "sequences": 2, "active_sequences": 2, "q_len": 3, "context": 0}
    live_runner._build_gqa_attention(tm2, nm2, gqa_pre, "cpu")[0](_CpuTensor(np.ones((6, 8), dtype=np.float32)))
    assert tr2["rope_calls"] == 2 and tr2["fia_calls"][0]["kw"]["actual_seq_qlen"] == [3, 6] and tr2["fia_calls"][0]["kw"]["actual_seq_kvlen"] == [3, 6]
    with pytest.raises(live_runner.RunnerUnsupportedOverlayError, match="sparse_gqa_prefill"):
        live_runner._build_gqa_attention(tm2, nm2, {**gqa_pre, "indexer": True}, "cpu")

    # 4. 2-rank owner-routed MoE EP with top_k=3 ([0, 1, 3]) routes expert 3 to rank 1 (NOT dropped!)
    spec = {"hidden_size": 4, "ffn": "moe", "num_experts": 4, "top_k": 3, "moe_intermediate_size": 4, "ep": 2, "tp": 1, "pcp": 1, "sequences": 2, "q_len": 1, "phase": "decode", "quant_mode": "bf16"}
    bar, slots = threading.Barrier(2, timeout=5.0), {}
    def _a2a_for(rk: int) -> Any:
        def _a2a(out_t: _CpuTensor, in_t: _CpuTensor, output_split_sizes: list[int] | None = None, input_split_sizes: list[int] | None = None, group: Any = None) -> None:
            del output_split_sizes, group
            slots[rk] = (in_t.arr.copy(), input_split_sizes)
            bar.wait()
            if input_split_sizes is None:
                out_t.arr[:] = np.concatenate([np.split(slots[s][0], 2, axis=0)[rk] for s in (0, 1)], axis=0)
            else:
                chunks = [slots[s][0][sum(slots[s][1][:rk]):sum(slots[s][1][:rk]) + slots[s][1][rk]] for s in (0, 1)]
                out_t.arr[:] = np.concatenate(chunks, axis=0) if chunks else out_t.arr
            bar.wait()
        return _a2a
    r0_tm, r0_nm, r0_tr = _make_cpu_torch_mocks(_a2a_for(0), router_experts=[0, 1, 3], w_val=2.0)
    r1_tm, r1_nm, r1_tr = _make_cpu_torch_mocks(_a2a_for(1), router_experts=[0, 1, 3], w_val=5.0)
    r0_fn = live_runner._build_ffn_or_moe(r0_tm, r0_nm, spec, "cpu", dist_group="ep")[0]
    r1_fn = live_runner._build_ffn_or_moe(r1_tm, r1_nm, spec, "cpu", dist_group="ep")[0]
    errs, outs = [], {}
    x_in, res_in = _CpuTensor(np.ones((2, 4), dtype=np.float32)), _CpuTensor(np.zeros((2, 4), dtype=np.float32))
    def _exec(rk: int, fn: Any) -> None:
        try:
            outs[rk] = fn(x_in, res_in)
        except BaseException as exc:  # noqa: BLE001
            errs.append(exc)
            bar.abort()
    t0, t1 = threading.Thread(target=_exec, args=(0, r0_fn)), threading.Thread(target=_exec, args=(1, r1_fn))
    t0.start()
    t1.start()
    t0.join()
    t1.join()
    assert not errs and np.allclose(outs[0].arr, 132.0) and np.allclose(outs[1].arr, 132.0) and r0_tr["gmm_calls"] and r1_tr["gmm_calls"]

    # 5. curFreq ignores rated 1650 & maps dual-die A3 logical ID 5 -> (pkg=2, chip=1); StreamBlocker keeps 99us tail
    monkeypatch.setattr(live_runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="Aicore Freq(MHZ) : 1650\nAicore curFreq(MHZ) : 800\n"))
    assert live_runner._detect_aicore_freq_mhz(0) == 800.0
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "5")
    def _dual_die_smi(cmd: list[str], **_kw: Any) -> Any:
        pkg = cmd[-1]
        if pkg == "5":
            return SimpleNamespace(stdout="Chip ID : 0\nAicore curFreq(MHZ) : 800\nChip ID : 1\nAicore curFreq(MHZ) : 800\n")
        return SimpleNamespace(stdout="Chip ID : 0\nAicore curFreq(MHZ) : 800\nChip ID : 1\nAicore curFreq(MHZ) : 1650\n")
    monkeypatch.setattr(live_runner.subprocess, "run", _dual_die_smi)
    assert live_runner._detect_aicore_freq_mhz(0) == 1650.0
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    blk = {"type": "MatMulV3", "input_shapes": live_runner.STREAM_BLOCKER_EXACT_PREFIX, "start_us": 100.0, "duration_us": 100.0}
    tail_cp = {"type": "MemcpyAsync", "start_us": 199.0, "duration_us": 100.0}
    kept, stripped, matched = live_runner._strip_stream_blocker_rows([blk, blk, tail_cp, blk, blk])
    assert matched == 4 and stripped == 4 and len(kept) == 1
    assert live_runner._bundle_has_runtime_failures({"shard_complete": True, "cases": [{"status": "absent_custom_op", "rank_receipts": [{"status": "absent_custom_op"}, {"status": "runtime_error"}]}]})


def test_sanitization_and_tutorial_exec_mxfp4_and_mso_false(tmp_path: Path) -> None:
    """Sanitizer strips IPv6/64-hex; tutorial parser skips eagle3 & reads MkDocs tabs; Two-Tier L1 & MC2 helpers work."""
    dirty = "ValueError:fd12:3456::789a:beef%en0 abcd:efab::cdef:abcd " + "c" * 64 + " user@corp.org " + "/" + "root/my dir/secret"
    for clean in (live_runner._sanitize_text(dirty), harness._sanitize_diagnostic(dirty)):
        for tok in ("fd12:3456", "beef%en0", "abcd:efab", "c" * 32, "user@corp.org", "secret"):
            assert tok not in clean

    md = (
        "::::{tab-item} 950DT\n```bash\n"
        "exec vllm serve /models/GLM-5.3-w4a4c8-mxfp4 --tensor-parallel-size \"4\" --pipeline-parallel-size \"4\" "
        "--enable-expert-parallel --quantization ascend --kv-cache-dtype fp8 --additional-config '{\"multistream_overlap_shared_expert\": false}'\n"
        "vllm serve /models/GLM-5.3 --tensor-parallel-size \"$TP\"\n```\n::::\n"
        "=== \"950PR Products\"\n```bash\nvllm serve /models/Qwen3.8-27B --quantization ascend "
        "--speculative-config '{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": 3}'\n```\n"
    )
    (tmp_path / "GLM5.3.md").write_text(md, encoding="utf-8")
    disc = harness.audit_va_tutorial_discovery(tmp_path)
    assert disc["total_commands"] == 3 and len(disc["discovered"]) == 2 and len(disc["explicitly_skipped"]) == 1
    d0, d1 = disc["discovered"]
    assert d0["hw"] == harness.DEFAULT_A5_SKU and d0["tp"] == 4 and d0["pp"] == 4 and d0["mso"] is False
    assert isinstance(d0["quant"], dict) and d0["quant"]["weight_format"] == "mxfp4"
    assert d1["hw"] == harness.DEFAULT_A5_PR_SKU and d1["mtp"] == 3
    assert harness._parse_mtp_spec_tokens('{"method": "eagle3", "num_speculative_tokens": 3}') == 0
    assert harness._parse_mtp_spec_tokens('{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}') == 3
    assert harness._mc2_protocol_link_bytes({"mc2_phase": {"route_matrix_bytes": [[10, 40], [30, 5]]}}) == 40.0
    assert harness._l1_cliff_limit_us(10.0) == 3.0 and harness._l1_cliff_limit_us(100.0) == 15.0
    assert harness._summarize_l1_two_tier({"a": {**harness._score_latency_pair(108.004, 100.0), "pass_l1_cliff_tol": True}})["l1_two_tier_pass"] is False
    assert harness._summarize_l1_two_tier({f"p{i}": {**harness._score_latency_pair(100.0 + e, 100.0), "pass_l1_cliff_tol": True} for i, e in enumerate((1.0, 2.0, 3.0, 4.0, 12.004))})["l1_two_tier_pass"] is False
    canc = [{"l1_two_tier_pass": True, "l1_in_l2_scores": {
        f"b{i}": {"signed_err_pct": e, "pass_l1_cliff_tol": True} for i, e in enumerate((10.0, -10.0, 0.0, 0.0, 0.0))
    }}]
    assert harness._aggregate_gate_b_two_tier_stats(canc)["l1_in_l2_all_buckets_two_tier_pass"] is False
    assert harness._select_canonical_tp_width(SimpleNamespace(num_heads=16, num_kv_heads=16, layer_plans=()), is_mla=True) == 2
    assert harness._load_calibrated_primitive_table(harness.DEFAULT_A5_SKU) is not None
    assert live_runner._single_card_ep_routed_counts(4, 8, dp=1, tp=4, ep=8, pcp=1, sequences=4, active_sequences=4, q_len=1, overlay="dec") == (1, 8)
    assert live_runner._single_card_ep_routed_counts(8, 10, dp=1, tp=2, ep=8, pcp=1, sequences=8, active_sequences=6, q_len=1, overlay="eager_padded_b8_a6") == (1, 10)
    assert live_runner._single_card_ep_routed_counts(4, 8, dp=8, tp=1, ep=8, pcp=1, sequences=4, active_sequences=4, q_len=1, overlay="dec") == (4, 8)
    assert live_runner.L2_FLUSH_BYTES >= 256 * live_runner.MIB
    copied: list[int] = []
    live_runner._flush_l2_cache(("src", SimpleNamespace(copy_=lambda src: copied.append(1 if src == "src" else 0))))
    assert copied == [1]
