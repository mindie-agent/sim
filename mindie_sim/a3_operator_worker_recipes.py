"""Device recipe implementations for the A3 operator campaign worker.

[Role]
  Hold the supported primitive/reference/native recipe runners and their
  input construction, CPU-reference checks and truthful protocol labelling.

[Invariants]
  * Torch is imported only when a runner is constructed.
  * Native fused availability is probed; dispatch failure becomes pending.
  * No in-region output copy is hidden under a preallocated label.
  * Raw samples, trials and observed API/versions are preserved.
  * No lease/device selection/service mutation.

[Timeouts & Locks]
  Bounded warmup/iteration/trial loops; no lock across a device call.

[Side-Effect Policy]
  Imports are lazy; allocates device tensors only when constructed/run.
"""
from __future__ import annotations

import importlib.metadata
import math
import os
import platform
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .a3_types import A3InputError
from .a3_operator_worker_native import (
    NativeRecipesMixin, PendingBackend, _dtype_bytes, _native_softmax_geometry,
    _quant_scale_mode, _scaled_expert_counts, _shape,
)

SUPPORTED_RECIPES = {
    "torch_mm_preallocated", "fp8_block_dequant_bf16_mm", "int8_dequant_bf16_mm",
    "torch_softmax", "torch_attention_matmul", "rmsnorm_reference",
    "npu_rms_norm", "swiglu_reference", "npu_swiglu",
    "elementwise_add_reference", "copy_rows_reference",
    "npu_scaled_masked_softmax", "npu_grouped_matmul_bf16", "npu_grouped_matmul_list_bf16",
    "int8_per_row_quant_reference",
    "grouped_expert_mm_reference", "npu_dynamic_quant",
}

# Pure recipe -> builder routing.  ``None`` means intentionally unsupported;
# a dispatch-only probe never allocates a device tensor or produces timing.
RECIPE_BUILDERS = {
    "torch_mm_preallocated": "_mm_inputs",
    "fp8_block_dequant_bf16_mm": "_gemm_inputs",
    "int8_dequant_bf16_mm": "_gemm_inputs",
    "rmsnorm_reference": "_norm_inputs",
    "npu_rms_norm": "_norm_inputs",
    "swiglu_reference": "_swiglu_inputs",
    "npu_swiglu": "_swiglu_inputs",
    "torch_softmax": "_softmax_inputs",
    "npu_scaled_masked_softmax": "_masked_softmax_inputs",
    "npu_grouped_matmul_bf16": "_grouped_matmul_inputs",
    "npu_grouped_matmul_list_bf16": "_grouped_matmul_inputs",
    "grouped_expert_mm_reference": "_grouped_matmul_inputs",
    "npu_dynamic_quant": "_dynamic_quant_inputs",
    "int8_per_row_quant_reference": "_int8_per_row_quant_inputs",
    "torch_attention_matmul": "_attention_inputs",
    "elementwise_add_reference": "_elementwise_inputs",
    "copy_rows_reference": "_copy_inputs",
}


def _validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("request_key", "measurement_id", "condition_key", "point", "condition"):
        if key not in request:
            raise A3InputError(f"worker_request_missing:{key}")
    condition = request["condition"]
    recipe = request.get("worker_recipe") or condition.get("worker_recipe")
    execution_kind = request.get("execution_kind") or condition.get("execution_kind")
    status = condition.get("status") or request.get("status")
    shape = _shape(request)
    if execution_kind not in {"primitive", "fused_kernel", "reference_decomposition"}:
        raise A3InputError(f"worker_execution_kind_invalid:{execution_kind}")
    if status in {"pending_backend", "unsupported"} or not recipe:
        return {
            "status": "pending_backend",
            "reason": condition.get("reason") or "no verified worker recipe",
            "required_backend": condition.get("required_backend") or "unspecified",
        }
    if recipe not in SUPPORTED_RECIPES:
        return {
            "status": "pending_backend",
            "reason": f"worker recipe {recipe!r} is declared but not implemented",
            "required_backend": condition.get("required_backend") or recipe,
        }
    if recipe == "npu_scaled_masked_softmax":
        q, kv = int(shape.get("q", 0)), int(shape.get("kv", 0))
        if not _native_softmax_geometry(q, kv):
            return {
                "status": "pending_backend",
                "reason": f"native masked softmax shape unsupported:q={q}:kv={kv}",
                "required_backend": "q/kv in [32,4096] multiples of 32",
            }
    return {"status": "dry_run_validated", "recipe": recipe, "shape": shape,
            "execution_kind": execution_kind}


def _request_protocol(request: Mapping[str, Any]) -> dict[str, Any]:
    protocol = dict(request.get("protocol") or {})
    return {
        "timer": str(protocol.get("timer", "device_event")),
        "timing_scope": str(protocol.get("timing_scope", "device_event_warm_operator")),
        "warmup": int(protocol.get("warmup", 10)),
        "warmup_scope": str(protocol.get("warmup_scope") or "before_each_trial"),
        "iterations": int(protocol.get("iterations", 30)),
        "trials": int(protocol.get("trials", 2)),
        "preallocated": bool(protocol.get("preallocated", True)),
        "cache_policy": str(protocol.get("cache_policy", "same_preallocated_buffers_reused_no_cache_flush")),
        "seed": int(protocol.get("seed", 20261021)),
    }


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torch-npu", "torch_npu"):
        try:
            packages[name] = importlib.metadata.version(name)
        except Exception:  # noqa: BLE001 - package metadata is optional
            packages[name] = None
    return {
        "host": platform.node(),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "packages": packages,
        "worker": "mindie_sim.a3_operator_worker",
    }


class TorchNpuOperatorRunner(NativeRecipesMixin):
    """Managed-device recipe runner; imports torch only when constructed."""

    def __init__(self, *, device: str = "npu:0", warmup: int = 10,
                 iterations: int = 30, trials: int = 2, seed: int = 20261021) -> None:
        try:
            import torch  # type: ignore
            import torch_npu  # type: ignore  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - optional runtime boundary
            raise A3InputError(f"torch_npu_unavailable:{type(exc).__name__}") from exc
        self.torch = torch
        self.torch_npu = torch_npu
        self.device = device
        self.warmup = max(1, int(warmup))
        self.iterations = max(1, int(iterations))
        self.trials = max(1, int(trials))
        self.seed = int(seed)
        self.torch.manual_seed(self.seed)
        if hasattr(torch.npu, "set_device"):
            torch.npu.set_device(0)
        self.environment = _environment()
        if hasattr(torch.npu, "get_device_name"):
            try:
                self.environment["device_name"] = torch.npu.get_device_name(0)
            except Exception:  # noqa: BLE001
                self.environment["device_name"] = ""
        self.environment["torch_file"] = getattr(torch, "__file__", "")
        self.environment["torch_npu_file"] = getattr(torch_npu, "__file__", "")

    # ---- dtype/input helpers -------------------------------------------------
    def _dtype(self, name: str) -> Any:
        key = str(name or "bf16").lower()
        mapping = {
            "bf16": self.torch.bfloat16, "bfloat16": self.torch.bfloat16,
            "fp16": self.torch.float16, "float16": self.torch.float16,
            "fp32": self.torch.float32, "float32": self.torch.float32,
            "int8": self.torch.int8,
        }
        if key not in mapping:
            raise PendingBackend(f"worker_dtype_unsupported:{name}", "torch dtype backend")
        return mapping[key]

    def _rand(self, shape: Sequence[int], dtype: Any) -> Any:
        return self.torch.randn(tuple(int(v) for v in shape), dtype=dtype, device=self.device)

    def _ones(self, shape: Sequence[int], dtype: Any) -> Any:
        return self.torch.ones(tuple(int(v) for v in shape), dtype=dtype, device=self.device)

    def _f32_cpu(self, value: Any) -> Any:
        return value.detach().float().cpu()

    # ---- correctness ---------------------------------------------------------
    def _check_gemm(self, a: Any, b: Any, out: Any, *, rtol: float, atol: float) -> dict[str, Any]:
        m, n = int(out.shape[0]), int(out.shape[1])
        rows = sorted({0, m // 2, m - 1})
        cols = sorted({0, n // 2, n - 1})
        ref = self._f32_cpu(a[rows]) @ self._f32_cpu(b[:, cols])
        got = self._f32_cpu(out[rows][:, cols])
        error = float((ref - got).abs().max().item()) if ref.numel() else 0.0
        ok = bool(self.torch.allclose(ref, got, rtol=rtol, atol=atol)) if ref.numel() else True
        if not ok:
            raise A3InputError(f"worker_correctness_failed:gemm:{error:g}")
        return {"status": "passed", "reference": "cpu_sample", "rows": rows,
                "cols": cols, "max_abs_error": error, "rtol": rtol, "atol": atol}

    def _check_rows(self, reference: Any, got: Any, rows: Sequence[int], *,
                    rtol: float, atol: float, name: str) -> dict[str, Any]:
        # Callers pass the CPU reference already reduced to ``rows``; only the
        # device output is indexed here.  Re-indexing both was a row-alignment
        # bug that failed with IndexError for 3-row samples.
        ref = self._f32_cpu(reference)
        actual = self._f32_cpu(got[list(rows)])
        error = float((ref - actual).abs().max().item()) if ref.numel() else 0.0
        ok = bool(self.torch.allclose(ref, actual, rtol=rtol, atol=atol)) if ref.numel() else True
        if not ok:
            raise A3InputError(f"worker_correctness_failed:{name}:{error:g}")
        return {"status": "passed", "reference": "cpu_sample", "rows": list(rows),
                "max_abs_error": error, "rtol": rtol, "atol": atol}

    # ---- recipe builders -----------------------------------------------------
    def _gemm_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        m, n, k = int(shape.get("m", 1)), int(shape.get("n", 1)), int(shape.get("k", 1))
        precision = dict(request["condition"].get("precision") or {})
        weight_format = str(precision.get("weight_format") or "bf16").lower()
        dtype = self._dtype(str(precision.get("input_dtype") or "bf16"))
        if weight_format in {"bf16", "bfloat16", "fp16", "float16", ""}:
            a = self._rand((m, k), dtype)
            b = self._rand((k, n), dtype)
            out = torch.empty((m, n), dtype=dtype, device=self.device)
            return {
                "op": lambda: torch.mm(a, b, out=out), "out": out,
                "input": (a, b), "weight_format": weight_format,
                "preallocated": True, "kind": "mm",
            }
        if weight_format in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
            return self._fp8_gemm_inputs(request, m, n, k, precision)
        if weight_format in {"int8", "w8a8"}:
            return self._int8_gemm_inputs(request, m, n, k, precision)
        raise PendingBackend(
            f"worker_weight_format_unsupported:{weight_format}",
            "verified quantized GEMM backend",
        )

    def _fp8_gemm_inputs(self, request: Mapping[str, Any], m: int, n: int, k: int,
                         precision: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        if not hasattr(torch, "float8_e4m3fn"):
            raise PendingBackend("worker_torch_float8_missing", "torch float8_e4m3fn")
        block = tuple(precision.get("weight_block") or (0, 0))
        br, bc = max(1, int(block[0] or 128)), max(1, int(block[1] or 128))
        # Generate FP8 operands on CPU (device float8 casts vary by build) and
        # copy them to the managed device.  Dequant is inside the timed recipe.
        a8 = (torch.randn((m, k)) * 0.5).to(torch.float8_e4m3fn).to(self.device)
        w8 = (torch.randn((n, k)) * 0.5).to(torch.float8_e4m3fn).to(self.device)
        a_scale = torch.ones((m, 1), dtype=torch.float32, device=self.device)
        scale_format = str(precision.get("scale_format") or "fp32")
        if "e8m0" in scale_format:
            w_scale = torch.full((math.ceil(n / br), math.ceil(k / bc)), 127,
                                 dtype=torch.uint8, device=self.device)
        else:
            w_scale = torch.ones((math.ceil(n / br), math.ceil(k / bc)),
                                 dtype=torch.float32, device=self.device)
        a_buf = torch.empty((m, k), dtype=torch.bfloat16, device=self.device)
        b_buf = torch.empty((k, n), dtype=torch.bfloat16, device=self.device)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=self.device)

        def expand_scale() -> Any:
            if "e8m0" in scale_format:
                scale = torch.pow(2.0, w_scale.float() - 127.0)
            else:
                scale = w_scale.float()
            return scale.repeat_interleave(br, dim=0).repeat_interleave(bc, dim=1)[:n, :k]

        scale_expanded = expand_scale()
        scale_t = scale_expanded.t().contiguous()

        def dequant() -> None:
            a_buf.copy_(a8.float() * a_scale)
            b_buf.copy_(w8.t().float() * scale_t)

        def op() -> None:
            dequant()
            torch.mm(a_buf, b_buf, out=out)

        # Warm one execution before reference check so lazy metadata/allocation
        # failures surface as pending/failed, not as a fake sample.
        op()
        torch.npu.synchronize()
        check = self._check_gemm(a_buf, b_buf, out, rtol=1e-1, atol=5e-2)
        # Also verify the dequant sample formula against the FP8 source.
        rows = sorted({0, n // 2, n - 1})
        cols = sorted({0, k // 2, k - 1})
        ref_deq = self._f32_cpu(w8[rows][:, cols]) * self._f32_cpu(scale_expanded[rows][:, cols])
        got_deq = self._f32_cpu(b_buf[:, cols][rows])
        deq_error = float((ref_deq - got_deq).abs().max().item()) if ref_deq.numel() else 0.0
        if deq_error > 5e-2:
            raise A3InputError(f"worker_correctness_failed:fp8_dequant:{deq_error:g}")
        check["dequant_max_abs_error"] = deq_error
        return {
            "op": op, "out": out, "input": (a8, w8), "weight_format": "fp8_e4m3",
            "preallocated": True, "kind": "fp8_block_dequant_bf16_mm",
            "correctness": check,
            "details": {"weight_block": [br, bc], "scale_format": scale_format,
                        "dequant_in_timed_loop": True},
        }

    def _int8_gemm_inputs(self, request: Mapping[str, Any], m: int, n: int, k: int,
                          precision: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        a8 = torch.randint(-8, 8, (m, k), dtype=torch.int8, device=self.device)
        b8 = torch.randint(-8, 8, (k, n), dtype=torch.int8, device=self.device)
        a_buf = torch.empty((m, k), dtype=torch.bfloat16, device=self.device)
        b_buf = torch.empty((k, n), dtype=torch.bfloat16, device=self.device)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=self.device)

        def op() -> None:
            a_buf.copy_(a8.float())
            b_buf.copy_(b8.float())
            torch.mm(a_buf, b_buf, out=out)

        op()
        torch.npu.synchronize()
        check = self._check_gemm(a_buf, b_buf, out, rtol=1e-2, atol=1e-2)
        return {"op": op, "out": out, "input": (a8, b8), "weight_format": "int8",
                "preallocated": True, "kind": "int8_dequant_bf16_mm",
                "correctness": check,
                "details": {"dequant_in_timed_loop": True}}

    def _mm_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        m, n, k = int(shape["m"]), int(shape["n"]), int(shape["k"])
        dtype = self._dtype(str(request["condition"]["precision"].get("input_dtype") or "bf16"))
        a = self._rand((m, k), dtype)
        b = self._rand((k, n), dtype)
        out = torch.empty((m, n), dtype=dtype, device=self.device)
        def op() -> None:
            torch.mm(a, b, out=out)
        op()
        torch.npu.synchronize()
        check = self._check_gemm(a, b, out, rtol=2e-2, atol=2e-2)
        return {"op": op, "out": out, "input": (a, b), "weight_format": "",
                "preallocated": True, "kind": "torch_mm_preallocated",
                "correctness": check, "observed_api": "torch.mm",
                "details": {"m": m, "n": n, "k": k}}

    def _norm_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        if "tokens" in shape and "width" in shape:
            tokens, width = int(shape["tokens"]), int(shape["width"])
        else:
            raise PendingBackend(
                "worker_rmsnorm_width_not_exposed",
                "condition with explicit rows/width",
            )
        dtype = self._dtype(str(request["condition"]["precision"].get("input_dtype") or "bf16"))
        x = self._rand((tokens, width), dtype)
        gamma = self._ones((width,), dtype)
        out = torch.empty_like(x)
        tmp = torch.empty_like(x)
        mean = torch.empty((tokens, 1), dtype=torch.float32, device=self.device)
        inv = torch.empty_like(mean)
        eps = 1e-6

        def reference() -> None:
            torch.mul(x, x, out=tmp)
            torch.mean(tmp, dim=-1, keepdim=True, out=mean)
            torch.add(mean, eps, out=mean)
            torch.rsqrt(mean, out=inv)
            torch.mul(x, inv, out=out)
            torch.mul(out, gamma, out=out)

        def cpu_ref(rows: Sequence[int]) -> Any:
            xr = self._f32_cpu(x[list(rows)])
            return xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + eps) * self._f32_cpu(gamma)

        reference()
        torch.npu.synchronize()
        rows = sorted({0, tokens // 2, tokens - 1})
        check = self._check_rows(cpu_ref(rows), out, rows, rtol=2e-2, atol=2e-2, name="rmsnorm_reference")
        recipe = str(request.get("worker_recipe") or "")
        if recipe == "npu_rms_norm":
            if not hasattr(self.torch_npu, "npu_rms_norm"):
                raise PendingBackend("worker_npu_rms_norm_unavailable", "torch_npu.npu_rms_norm")
            def op() -> Any:
                return self.torch_npu.npu_rms_norm(x, gamma, epsilon=eps)
            try:
                result = op()
                torch.npu.synchronize()
            except Exception as exc:  # noqa: BLE001 - dispatch/pilot boundary
                raise PendingBackend(
                    f"worker_npu_rms_norm_dispatch_failed:{type(exc).__name__}:{exc}",
                    "torch_npu.npu_rms_norm on this dtype/layout",
                ) from exc
            native_out = result[0] if isinstance(result, (tuple, list)) else result
            check = self._check_rows(cpu_ref(rows), native_out, rows, rtol=2e-2, atol=2e-2,
                                     name="npu_rms_norm")
            return {"op": op, "out": native_out, "input": (x, gamma), "weight_format": "",
                    "preallocated": False, "kind": "npu_rms_norm", "correctness": check,
                    "observed_api": "torch_npu.npu_rms_norm",
                    "details": {"eps": eps, "rows": tokens, "width": width,
                                "preallocated_inputs": True,
                                "native_output_allocation": True,
                                "copy_in_timed_region": False}}
        op = reference
        return {"op": op, "out": out, "input": (x, gamma), "weight_format": "",
                "preallocated": True, "kind": "rmsnorm_reference", "correctness": check,
                "observed_api": "torch (rsqrt/mean/mul reference)",
                "details": {"eps": eps, "rows": tokens, "width": width,
                            "preallocated_inputs": True, "preallocated_output": True}}

    def _swiglu_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        if "tokens" in shape and "width" in shape:
            tokens, width = int(shape["tokens"]), int(shape["width"])
        else:
            raise PendingBackend("worker_swiglu_width_not_exposed", "condition with explicit rows/width")
        dtype = self._dtype(str(request["condition"]["precision"].get("input_dtype") or "bf16"))
        gate = self._rand((tokens, width), dtype)
        up = self._rand((tokens, width), dtype)
        out = torch.empty_like(gate)
        tmp = torch.empty_like(gate)

        def reference() -> None:
            torch.sigmoid(gate, out=tmp)
            torch.mul(tmp, gate, out=tmp)
            torch.mul(tmp, up, out=out)

        def cpu_ref(rows: Sequence[int]) -> Any:
            g = self._f32_cpu(gate[list(rows)])
            u = self._f32_cpu(up[list(rows)])
            return torch.nn.functional.silu(g) * u

        reference()
        torch.npu.synchronize()
        rows = sorted({0, tokens // 2, tokens - 1})
        check = self._check_rows(cpu_ref(rows), out, rows, rtol=2e-2, atol=2e-2, name="swiglu_reference")
        recipe = str(request.get("worker_recipe") or "")
        if recipe == "npu_swiglu":
            if not hasattr(self.torch_npu, "npu_swiglu"):
                raise PendingBackend("worker_npu_swiglu_unavailable", "torch_npu.npu_swiglu")
            # Installed schema: npu_swiglu(self, dim=-1) with input [T, 2F].
            # Build the packed input outside timing; never do a timed
            # TypeError fallback or an in-region output copy.
            packed = torch.cat((gate, up), dim=-1)
            def op() -> Any:
                return self.torch_npu.npu_swiglu(packed, -1)
            try:
                native_out = op()
                torch.npu.synchronize()
            except Exception as exc:  # noqa: BLE001 - dispatch/pilot boundary
                raise PendingBackend(
                    f"worker_npu_swiglu_dispatch_failed:{type(exc).__name__}:{exc}",
                    "torch_npu.npu_swiglu on this dtype/layout",
                ) from exc
            check = self._check_rows(cpu_ref(rows), native_out, rows, rtol=2e-2, atol=2e-2,
                                     name="npu_swiglu")
            return {"op": op, "out": native_out, "input": (packed,), "weight_format": "",
                    "preallocated": False, "kind": "npu_swiglu", "correctness": check,
                    "observed_api": "torch_npu.npu_swiglu",
                    "details": {"rows": tokens, "width": width,
                                "packed_input_width": 2 * width,
                                "preallocated_inputs": True,
                                "native_output_allocation": True,
                                "copy_in_timed_region": False}}
        op = reference
        return {"op": op, "out": out, "input": (gate, up), "weight_format": "",
                "preallocated": True, "kind": "swiglu_reference", "correctness": check,
                "observed_api": "torch (sigmoid/mul reference)",
                "details": {"rows": tokens, "width": width,
                            "preallocated_inputs": True, "preallocated_output": True}}

    def _softmax_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        kind = str(request["condition"].get("shape_kind"))
        if kind == "attention_softmax" and all(key in shape for key in ("q", "heads", "kv")):
            rows = int(shape["q"]) * int(shape["heads"])
            width = int(shape["kv"])
        elif kind == "router_softmax" and all(key in shape for key in ("tokens", "experts")):
            rows, width = int(shape["tokens"]), int(shape["experts"])
        elif "tokens" in shape and "width" in shape:
            rows, width = int(shape["tokens"]), int(shape["width"])
        else:
            rows, width = 1, int(shape.get("elements", 1))
        dtype = self._dtype(str(request["condition"]["precision"].get("input_dtype") or "bf16"))
        x = self._rand((rows, width), dtype)
        out = torch.empty_like(x)
        # Resolve the out= signature once, outside the timed interval.  A
        # fallback allocates its output honestly; no exception/copy runs in
        # every measured iteration.
        try:
            torch.softmax(x, dim=-1, out=out)
            out_mode = True
        except TypeError:
            out_mode = False
        if out_mode:
            def op() -> Any:
                return torch.softmax(x, dim=-1, out=out)
            result = out
            preallocated = True
            details = {"rows": rows, "width": width, "out_argument": True}
        else:
            def op() -> Any:
                return torch.softmax(x, dim=-1)
            result = op()
            preallocated = False
            details = {"rows": rows, "width": width, "out_argument": False,
                       "native_output_allocation": True,
                       "copy_in_timed_region": False}
        torch.npu.synchronize()
        sample = sorted({0, rows // 2, rows - 1})
        ref = torch.softmax(self._f32_cpu(x[sample]), dim=-1)
        check = self._check_rows(ref, result, sample, rtol=2e-2, atol=2e-2, name="softmax")
        return {"op": op, "out": result, "input": (x,), "weight_format": "",
                "preallocated": preallocated, "kind": "torch_softmax",
                "correctness": check, "observed_api": "torch.softmax",
                "details": details}

    def _attention_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        q, kv, heads, dim = (int(shape["q"]), int(shape["kv"]), int(shape["heads"]), int(shape["dim"]))
        dtype = self._dtype(str(request["condition"]["precision"].get("input_dtype") or "bf16"))
        if request["condition"]["family"] == "attention_qk":
            a = self._rand((q, heads, dim), dtype)
            b = self._rand((kv, heads, dim), dtype)
            out = torch.empty((heads, q, kv), dtype=dtype, device=self.device)
            def op() -> None:
                torch.bmm(a.transpose(0, 1), b.transpose(0, 1).transpose(1, 2), out=out)
            op()
            torch.npu.synchronize()
            h = 0
            qrows = sorted({0, q // 2, q - 1})
            kcols = sorted({0, kv // 2, kv - 1})
            ref = (self._f32_cpu(a[qrows, h, :]) @ self._f32_cpu(b[kcols, h, :]).t())
            got = self._f32_cpu(out[h][qrows][:, kcols])
            error = float((ref - got).abs().max().item()) if ref.numel() else 0.0
            if not bool(torch.allclose(ref, got, rtol=2e-2, atol=2e-2)):
                raise A3InputError(f"worker_correctness_failed:attention_qk:{error:g}")
            check = {"status": "passed", "reference": "cpu_sample", "head": h,
                     "q_rows": qrows, "kv_cols": kcols, "max_abs_error": error}
            return {"op": op, "out": out, "input": (a, b), "weight_format": "",
                    "preallocated": True, "kind": "attention_qk", "correctness": check,
                    "observed_api": "torch.bmm", "details": {"q": q, "kv": kv, "heads": heads, "dim": dim}}
        p = self._rand((heads, q, kv), dtype)
        v = self._rand((kv, heads, dim), dtype)
        out_bmm = torch.empty((heads, q, dim), dtype=dtype, device=self.device)
        def op() -> None:
            torch.bmm(p, v.transpose(0, 1), out=out_bmm)
        op()
        torch.npu.synchronize()
        h, row = 0, min(1, q - 1)
        dims = sorted({0, dim // 2, dim - 1})
        ref = self._f32_cpu(p[h, row, :]) @ self._f32_cpu(v[:, h, dims])
        got = self._f32_cpu(out_bmm[h, row, dims])
        error = float((ref - got).abs().max().item()) if ref.numel() else 0.0
        if not bool(torch.allclose(ref, got, rtol=2e-2, atol=2e-2)):
            raise A3InputError(f"worker_correctness_failed:attention_pv:{error:g}")
        check = {"status": "passed", "reference": "cpu_sample", "head": h,
                 "q_row": row, "dim_cols": dims, "max_abs_error": error}
        return {"op": op, "out": out_bmm.transpose(0, 1), "input": (p, v), "weight_format": "",
                "preallocated": True, "kind": "attention_pv", "correctness": check,
                "observed_api": "torch.bmm", "details": {"q": q, "kv": kv, "heads": heads, "dim": dim}}

    def _elementwise_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        if "tokens" in shape and "width" in shape:
            rows, width = int(shape["tokens"]), int(shape["width"])
        else:
            rows, width = 1, int(shape.get("elements", 1))
        dtype = self._dtype(str(request["condition"]["precision"].get("input_dtype") or "bf16"))
        a = self._rand((rows, width), dtype)
        b = self._rand((rows, width), dtype)
        out = torch.empty_like(a)
        def op() -> None:
            torch.add(a, b, out=out)
        op()
        torch.npu.synchronize()
        sample = sorted({0, rows // 2, rows - 1})
        ref = self._f32_cpu(a[sample]) + self._f32_cpu(b[sample])
        check = self._check_rows(ref, out, sample, rtol=2e-2, atol=2e-2, name="elementwise_add")
        return {"op": op, "out": out, "input": (a, b), "weight_format": "",
                "preallocated": True, "kind": "elementwise_add_reference",
                "correctness": check, "observed_api": "torch.add",
                "details": {"rows": rows, "width": width}}

    def _copy_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        rows = int(shape.get("rows", 1))
        row_bytes = int(shape.get("row_bytes", shape.get("elements", 1)))
        # row_bytes is a byte count, not a BF16 element count.  Use an explicit
        # uint8 byte view so measured read/write traffic is not doubled.
        source = torch.randint(0, 256, (rows, row_bytes), dtype=torch.uint8,
                               device=self.device)
        target = torch.empty_like(source)
        def op() -> None:
            target.copy_(source)
        op()
        torch.npu.synchronize()
        sample = sorted({0, rows // 2, rows - 1})
        error = int((source[sample].to(torch.int16) - target[sample].to(torch.int16)).abs().max().item())
        if error != 0:
            raise A3InputError(f"worker_correctness_failed:copy:{error}")
        actual_bytes = int(rows * row_bytes)
        return {"op": op, "out": target, "input": (source,), "weight_format": "",
                "preallocated": True, "kind": "copy_rows_reference",
                "correctness": {"status": "passed", "reference": "cpu_sample",
                                "rows": sample, "max_abs_error": error},
                "observed_api": "torch.Tensor.copy_",
                "details": {"rows": rows, "row_bytes": row_bytes,
                            "actual_read_bytes": actual_bytes,
                            "actual_write_bytes": actual_bytes,
                            "dtype_is_byte_view": True}}

    def _int8_per_row_quant_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        if not (shape.get("tokens") and shape.get("width")):
            raise PendingBackend("worker_int8_quant_shape_contract_missing",
                                 "condition with explicit tokens,width")
        rows, width = int(shape["tokens"]), int(shape["width"])
        precision = request["condition"]["precision"]
        dtype = self._dtype(str(precision.get("input_dtype") or "bf16"))
        scale_format = str(precision.get("scale_format") or "fp16").lower()
        scale_dtype = self._dtype(scale_format)
        if scale_dtype not in {torch.float16, torch.float32}:
            raise PendingBackend(f"worker_int8_quant_scale_dtype_unsupported:{scale_format}",
                                 "fp16/fp32 per-row scale")
        x = self._rand((rows, width), dtype)
        quantized = torch.empty((rows, width), dtype=torch.int8, device=self.device)
        scale = torch.empty((rows, 1), dtype=scale_dtype, device=self.device)
        row_max = torch.empty((rows, 1), dtype=torch.float32, device=self.device)
        work = torch.empty((rows, width), dtype=torch.float32, device=self.device)

        def op() -> None:
            work.copy_(x)
            work.abs_()
            torch.amax(work, dim=-1, keepdim=True, out=row_max)
            row_max.div_(127.0)
            scale.copy_(row_max)
            work.div_(scale.float())
            work.round_().clamp_(-127.0, 127.0)
            quantized.copy_(work)

        op()
        torch.npu.synchronize()
        sample = sorted({0, rows // 2, rows - 1})
        recon = quantized[sample].float() * scale[sample].float()
        ref = x[sample].float()
        error = float((ref - recon).abs().max().item()) if ref.numel() else 0.0
        tolerance = 0.02 * max(1e-6, float(ref.abs().max().item()))
        if error > tolerance:
            raise A3InputError(f"worker_correctness_failed:int8_per_row_quant:{error:g}>{tolerance:g}")
        check = {"status": "passed", "reference": "int8_reconstruction",
                 "max_abs_error": error, "tolerance": tolerance, "rows": sample}
        return {"op": op, "out": quantized, "input": (x,), "weight_format": "int8",
                "preallocated": True, "kind": "int8_per_row_quant_reference",
                "correctness": check,
                "observed_api": "torch (abs/amax/div/round/clamp reference)",
                "details": {"rows": rows, "width": width,
                            "scale_dtype": scale_format, "quant_dtype": "int8",
                            "semantic_family": "int8_per_row_scale",
                            "preallocated_inputs": True, "preallocated_output": True}}

    def _builder(self, request: Mapping[str, Any]) -> dict[str, Any]:
        recipe = str(request.get("worker_recipe") or "")
        method = RECIPE_BUILDERS.get(recipe)
        if method is None:
            raise PendingBackend(f"worker_recipe_unimplemented:{recipe}", recipe)
        if self is None:
            # Pure dispatch contract probe: no tensor allocation/device access.
            return {"status": "dispatch_only", "recipe": recipe, "builder": method,
                    "device_access": False}
        if method == "_mm_inputs":
            return self._mm_inputs(request)
        return getattr(self, method)(request)

    # ---- measurement ---------------------------------------------------------
    def _trial_schedule(self, *, warmup: int, trials: int, invoke: Any,
                        measure: Any) -> Iterable[tuple[int, Any]]:
        """Warm up before every independent trial, then measure that trial."""
        for trial_index in range(int(trials)):
            for _ in range(int(warmup)):
                invoke()
            yield trial_index, measure(trial_index)

    def _measure_trial(self, op: Any, iterations: int) -> list[float]:
        torch = self.torch
        raw: list[float] = []
        for _ in range(iterations):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            op()
            end.record()
            torch.npu.synchronize()
            duration = float(start.elapsed_time(end))
            if duration <= 0:
                raise A3InputError("worker_duration_invalid")
            raw.append(duration)
        return raw

    def run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        validation = _validate_request(request)
        if validation["status"] == "pending_backend":
            return {
                "status": "pending", "reason": validation["reason"],
                "required_backend": validation["required_backend"],
                "source_class": "not_measured", "raw_samples_ms": [],
                "query_context": request.get("query_context", {}),
                "observed_context": {"observed": False, "not_a3_measured": True},
            }
        torch = self.torch
        with torch.no_grad():
            built = self._builder(request)
        protocol = _request_protocol(request)
        try:
            trials = []
            def _invoke() -> None:
                built["op"]()
            def _measure(trial_index: int) -> list[float]:
                torch.npu.synchronize()
                return self._measure_trial(built["op"], self.iterations)
            for trial_index, raw in self._trial_schedule(
                    warmup=self.warmup, trials=self.trials,
                    invoke=_invoke, measure=_measure):
                trials.append({
                    "trial_index": trial_index, "raw_samples_ms": raw,
                    "median_ms": statistics.median(raw),
                    "min_ms": min(raw), "max_ms": max(raw),
                    "warmup": {
                        "iterations": self.warmup,
                        "scope": "before_each_trial",
                        "calls_before_trial": self.warmup,
                    },
                })
        except PendingBackend:
            raise
        except Exception as exc:
            raise A3InputError(f"worker_measurement_failed:{type(exc).__name__}:{exc}") from exc
        all_raw = [value for trial in trials for value in trial["raw_samples_ms"]]
        trial_medians = [trial["median_ms"] for trial in trials]
        duration = statistics.median(trial_medians)
        observed_api = built.get("observed_api", built["kind"])
        return {
            "status": "completed",
            "raw_samples_ms": all_raw,
            "duration_ms": duration,
            "analytic_ms": (float(request["analytic_ms"])
                            if request.get("analytic_ms") is not None else None),
            "source": "torch_npu_operator_worker",
            "source_class": "measured",
            "run_id": str(request.get("run_id", "")),
            "split": str(request.get("split", "train")),
            "timing_scope": "device_event_warm_operator",
            "trials": trials,
            "trial_medians_ms": trial_medians,
            "protocol": {
                **protocol,
                "execution_kind": request.get("execution_kind"),
                "worker_recipe": request.get("worker_recipe"),
                "requested_backend": request.get("backend"),
                "observed_api": observed_api,
                "timer": "device_event",
                "preallocated": bool(built.get("preallocated", True)),
                "cache_policy": protocol["cache_policy"],
                "iterations": self.iterations, "warmup": self.warmup,
                "warmup_scope": "before_each_trial",
                "warmup_iterations_per_trial": self.warmup,
                "warmup_events_total": self.warmup * self.trials,
                "trials": self.trials,
            },
            "query_context": request.get("query_context", {}),
            "observed_context": {
                "observed": True,
                "observed_api": observed_api,
                "observed_backend": observed_api,
                "execution_kind": request.get("execution_kind"),
                "worker_recipe": request.get("worker_recipe"),
                "device_name": self.environment.get("device_name", ""),
                "visible_devices": self.environment.get("visible_devices", ""),
                "packages": self.environment.get("packages", {}),
                "timer": "device_event",
                "not_an_aicpu_aiv_claim": True,
            },
            "environment": dict(self.environment),
            "correctness": built.get("correctness", {"status": "not_applicable"}),
            "details": {
                **(built.get("details") or {}),
                "kind": built["kind"],
                "shape": dict(_shape(request)),
                "preallocated": bool(built.get("preallocated", True)),
                "not_a3_measured": False,
            },
            "work": request.get("work", {}),
        }
