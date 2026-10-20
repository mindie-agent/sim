"""Native fused recipes for the A3 operator campaign worker.

[Role]
  Hold probed native recipes (masked softmax, grouped matmul list API, dynamic
  quant) and their point-shape/histogram/scale semantics as a mixin.

[Invariants]
  * Native availability/dispatch is probed; failure becomes pending.
  * Grouped matmul uses A3 group_type=-1 list API with group_list=None, not an
    invalid Tensor group_list, and uses the actual point histogram.
  * Dynamic quant broadcasts the returned [T] scale with unsqueeze(-1).
  * No in-region output copy is hidden; unsupported remains pending.

[Timeouts & Locks]
  Bounded construction/checks; no lock across device calls.

[Side-Effect Policy]
  Allocates device tensors only when a runner executes a real request.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .a3_types import A3InputError

class PendingBackend(A3InputError):
    """The exact recipe is real but no verified backend is available."""

    def __init__(self, reason: str, required_backend: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.required_backend = required_backend


def _shape(request: Mapping[str, Any]) -> dict[str, int]:
    raw = request.get("point", {}).get("shape") or request.get("shape") or {}
    shape = {str(k): int(v) for k, v in dict(raw).items()}
    if not shape or any(value < 0 for value in shape.values()):
        raise A3InputError("worker_shape_invalid")
    return shape


def _native_softmax_geometry(q: int, kv: int) -> bool:
    """Official A3 constraint: NCHW H/W in [32,4096] and multiples of 32."""
    return (32 <= q <= 4096 and 32 <= kv <= 4096
            and q % 32 == 0 and kv % 32 == 0)


def _quant_scale_mode(scale_numel: int, rows: int) -> str | None:
    if scale_numel == rows:
        return "row_unsqueeze"
    if scale_numel == 1:
        return "scalar"
    return None


def _scaled_expert_counts(base_counts: Sequence[int], target_experts: int,
                          target_routed: int) -> tuple[list[int], str]:
    """Materialize an exact point histogram with token/expert conservation.

    The returned list always has exactly ``target_experts`` entries and sums
    exactly to ``target_routed``.  Zero-token experts are valid when
    target_routed < target_experts.  Ties are distributed by expert index.
    """
    base = [max(0, int(v)) for v in base_counts]
    experts = max(1, int(target_experts))
    tokens = max(0, int(target_routed))
    if base and len(base) == experts and sum(base) == tokens:
        return list(base), "anchor_actual_histogram"
    if len(base) == experts and sum(base) > 0:
        pattern = list(base)
    elif base:
        pattern = [base[index % len(base)] for index in range(experts)]
    else:
        pattern = [1] * experts
    total_pattern = sum(pattern)
    if total_pattern <= 0:
        pattern = [1] * experts
        total_pattern = experts
    shares = [value * tokens / total_pattern for value in pattern]
    counts = [int(math.floor(value)) for value in shares]
    remainder = tokens - sum(counts)
    order = sorted(range(experts), key=lambda index: (-(shares[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    if sum(counts) != tokens or len(counts) != experts:
        raise A3InputError(
            f"worker_grouped_histogram_not_conserved:{sum(counts)}:{tokens}:{len(counts)}:{experts}")
    return counts, "actual_point_scaled_from_anchor_histogram"


def _validate_grouped_shapes(output_shapes: Sequence[Sequence[int]],
                             counts: Sequence[int], n: int) -> list[list[int]]:
    """Require complete output cardinality and exact per-expert shapes."""
    if len(output_shapes) != len(counts):
        raise A3InputError(
            f"worker_correctness_failed:grouped_matmul:output_cardinality:"
            f"{len(output_shapes)}!={len(counts)}")
    expected = []
    for expert, count in enumerate(counts):
        want = (int(count), int(n))
        got = tuple(int(v) for v in output_shapes[expert])
        expected.append(list(want))
        if got != want:
            raise A3InputError(
                f"worker_correctness_failed:grouped_matmul:shape:expert={expert}:"
                f"{list(got)}!={list(want)}")
    return expected


def _dtype_bytes(dtype: str) -> float:
    return {"bf16": 2.0, "bfloat16": 2.0, "fp16": 2.0, "float16": 2.0,
            "fp32": 4.0, "float32": 4.0, "int8": 1.0}.get(str(dtype).lower(), 2.0)




class NativeRecipesMixin:
    def _masked_softmax_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        if not all(key in shape for key in ("q", "kv", "heads")):
            raise PendingBackend("worker_attention_softmax_geometry_missing",
                                 "condition with q/heads/kv")
        q, kv, heads = int(shape["q"]), int(shape["kv"]), int(shape["heads"])
        if not _native_softmax_geometry(q, kv):
            raise PendingBackend(
                f"worker_npu_scaled_masked_softmax_shape_unsupported:q={q}:kv={kv}",
                "q/kv in [32,4096] and multiples of 32 (official A3 NCHW constraint)",
            )
        if not hasattr(self.torch_npu, "npu_scaled_masked_softmax"):
            raise PendingBackend("worker_npu_scaled_masked_softmax_unavailable",
                                 "torch_npu.npu_scaled_masked_softmax")
        dtype = self._dtype(str(request["condition"]["precision"].get("input_dtype") or "bf16"))
        # Official shape contract is 4D NCHW; H=q, W=kv.  The previous 2D
        # flatten was not a validated dispatch shape.
        x = self._rand((1, heads, q, kv), dtype)
        one_mask = torch.triu(torch.ones((1, 1, q, kv), dtype=torch.bool, device=self.device),
                              diagonal=max(0, kv - q + 1))
        mask = one_mask.expand(1, heads, q, kv)
        def op() -> Any:
            return self.torch_npu.npu_scaled_masked_softmax(
                x, mask, scale=1.0, fixed_triu_mask=False)
        try:
            native_out = op()
            torch.npu.synchronize()
        except Exception as exc:  # noqa: BLE001 - dispatch/pilot boundary
            raise PendingBackend(
                f"worker_npu_scaled_masked_softmax_dispatch_failed:{type(exc).__name__}:{exc}",
                "torch_npu.npu_scaled_masked_softmax NCHW H/W multiple-32",
            ) from exc
        flat = native_out.reshape(q * heads, kv)
        flat_mask = mask.reshape(q * heads, kv)
        sample = sorted({0, max(0, q // 2), max(0, q - 1)})
        # Independent CPU sample reference: transfer only selected x rows and
        # the mask, then compute masked softmax on the host in float32.
        x_cpu = self._f32_cpu(x.reshape(q * heads, kv)[sample])
        mask_cpu = flat_mask[sample].detach().bool().cpu()
        masked = x_cpu.masked_fill(mask_cpu, float("-inf"))
        row_max = masked.max(dim=-1, keepdim=True).values
        exponentials = (masked - row_max).exp()
        ref_rows = exponentials / exponentials.sum(dim=-1, keepdim=True)
        check = self._check_rows(ref_rows, flat, sample, rtol=2e-2, atol=2e-2,
                                 name="npu_scaled_masked_softmax")
        check["reference_device"] = "cpu_float32_masked_softmax"
        return {"op": op, "out": flat, "input": (x, mask), "weight_format": "",
                "preallocated": False, "kind": "npu_scaled_masked_softmax",
                "correctness": check, "observed_api": "torch_npu.npu_scaled_masked_softmax",
                "details": {"q": q, "kv": kv, "heads": heads,
                            "nchw_shape": [1, heads, q, kv],
                            "causal_mask": True, "native_output_allocation": True,
                            "copy_in_timed_region": False}}

    def _grouped_matmul_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        k, n = int(shape["k"]), int(shape["n"])
        target_routed = max(0, int(shape.get("routed_tokens", 1)))
        target_experts = max(1, int(shape.get("experts", 1)))
        features = request["condition"].get("features") or {}
        routing = features.get("routing") or {}
        base_counts = [int(v) for v in (routing.get("local_expert_counts") or ())]
        base_total = sum(base_counts)
        point_counts = [int(v) for v in (
            (request.get("point") or {}).get("expert_token_counts") or ())]
        if (len(point_counts) == target_experts
                and all(value >= 0 for value in point_counts)
                and sum(point_counts) == target_routed):
            counts, histogram_source = point_counts, "materialized_point_exact_histogram"
        else:
            counts, histogram_source = _scaled_expert_counts(
                base_counts, target_experts, target_routed)
        local_experts = len(counts)
        activation = str(request["condition"]["precision"].get("input_dtype") or "bf16").lower()
        if activation not in {"bf16", "bfloat16", "fp16", "float16"}:
            raise PendingBackend(f"worker_grouped_matmul_activation_unsupported:{activation}",
                                 "BF16/FP16 grouped matmul")
        dtype = self._dtype(activation)
        estimated = local_experts * (k * n + max(counts) * (k + n)) * _dtype_bytes(activation)
        if estimated > 3 * 1024 ** 3:
            raise PendingBackend("worker_grouped_matmul_working_set_above_cap",
                                 "verified grouped matmul with <=3 GiB working set")
        weights = [self._rand((k, n), dtype) for _ in range(local_experts)]
        inputs = [self._rand((count, k), dtype) if count > 0 else
                  torch.empty((0, k), dtype=dtype, device=self.device) for count in counts]
        recipe = str(request.get("worker_recipe") or "")
        if recipe in {"npu_grouped_matmul_bf16", "npu_grouped_matmul_list_bf16"}:
            if not hasattr(self.torch_npu, "npu_grouped_matmul"):
                raise PendingBackend("worker_npu_grouped_matmul_unavailable",
                                     "torch_npu.npu_grouped_matmul")
            # A3 group_type=-1 (多多多) requires group_list=None.  This is an
            # honestly named list recipe, not the packed single-X group_type=0
            # API and not a claim of packed production kernel equivalence.
            def op() -> Any:
                return self.torch_npu.npu_grouped_matmul(
                    inputs, weights, group_list=None, split_item=0, group_type=-1)
            try:
                native_out = op()
                torch.npu.synchronize()
            except Exception as exc:  # noqa: BLE001 - dispatch/pilot boundary
                raise PendingBackend(
                    f"worker_npu_grouped_matmul_list_dispatch_failed:{type(exc).__name__}:{exc}",
                    "torch_npu.npu_grouped_matmul list API (group_list=None, group_type=-1)",
                ) from exc
            outputs = list(native_out) if isinstance(native_out, (tuple, list)) else [native_out]
            observed = "torch_npu.npu_grouped_matmul"
            kind = "npu_grouped_matmul_list_bf16"
            preallocated = False
        else:
            def op() -> Any:
                return [torch.mm(x, w) for x, w in zip(inputs, weights)]
            outputs = op()
            torch.npu.synchronize()
            observed = "torch (per-expert mm reference)"
            kind = "grouped_expert_mm_reference"
            preallocated = False
        output_shapes = [tuple(int(v) for v in tuple(output.shape)) for output in outputs]
        _validate_grouped_shapes(output_shapes, counts, n)
        cols = sorted({0, max(0, n // 2), max(0, n - 1)})
        validated, max_error = [], 0.0
        for expert, count in enumerate(counts):
            if not count:
                continue
            rows = sorted({0, max(0, count // 2), max(0, count - 1)})
            ref = self._f32_cpu(inputs[expert][rows]) @ self._f32_cpu(weights[expert][:, cols])
            got = self._f32_cpu(outputs[expert][rows][:, cols])
            error = float((ref - got).abs().max().item()) if ref.numel() else 0.0
            if not bool(torch.allclose(ref, got, rtol=2e-2, atol=2e-2)):
                raise A3InputError(
                    f"worker_correctness_failed:grouped_matmul:expert={expert}:{error:g}")
            validated.append(expert)
            max_error = max(max_error, error)
        check = (
            {"status": "passed", "reference": "cpu_sample",
             "validated_experts": validated, "expert_count": local_experts,
             "output_cardinality": len(outputs), "output_shapes": output_shapes,
             "rows": "all non-empty experts", "cols": cols, "max_abs_error": max_error}
            if validated else
            {"status": "not_applicable", "reason": "no non-empty expert"}
        )
        return {"op": op, "out": outputs, "input": (inputs, weights), "weight_format": activation,
                "preallocated": preallocated, "kind": kind,
                "correctness": check, "observed_api": observed,
                "details": {"launched_expert_count": local_experts,
                            "launched_routed_tokens": sum(counts),
                            "expert_token_counts": counts,
                            "zero_token_experts": [index for index, count in enumerate(counts) if count == 0],
                            "histogram_source": histogram_source,
                            "anchor_local_experts": int(routing.get("local_experts") or len(base_counts) or 0),
                            "anchor_routed_tokens": base_total,
                            "weight_shapes": [[k, n]] * local_experts,
                            "input_shapes": [[count, k] for count in counts],
                            "group_list": None, "group_type": -1, "split_item": 0,
                            "native_output_allocation": not preallocated,
                            "copy_in_timed_region": False}}

    def _dynamic_quant_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        shape = _shape(request)
        if "tokens" in shape and "width" in shape:
            rows, width = int(shape["tokens"]), int(shape["width"])
        else:
            rows, width = 1, int(shape.get("elements", 1))
        dtype = self._dtype(str(request["condition"]["precision"].get("input_dtype") or "bf16"))
        x = self._rand((rows, width), dtype)
        if not hasattr(self.torch_npu, "npu_dynamic_quant"):
            raise PendingBackend("worker_npu_dynamic_quant_unavailable",
                                 "torch_npu.npu_dynamic_quant")
        def op() -> Any:
            return self.torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        try:
            result = op()
            torch.npu.synchronize()
        except Exception as exc:  # noqa: BLE001 - dispatch/pilot boundary
            raise PendingBackend(
                f"worker_npu_dynamic_quant_dispatch_failed:{type(exc).__name__}:{exc}",
                "torch_npu.npu_dynamic_quant on this dtype/layout",
            ) from exc
        quantized, scale = result[0], result[1]
        scale_flat = scale.reshape(-1)
        scale_broadcast = _quant_scale_mode(int(scale_flat.numel()), rows)
        if scale_broadcast == "row_unsqueeze":
            recon = quantized.float() * scale_flat.reshape(-1, 1).float()
        elif scale_broadcast == "scalar":
            recon = quantized.float() * scale.float()
        else:
            raise PendingBackend(
                f"worker_dynamic_quant_scale_shape_unexpected:{list(scale.shape)}:rows={rows}",
                "torch_npu.npu_dynamic_quant pertoken scale [T]",
            )
        sample = sorted({0, max(0, rows // 2), max(0, rows - 1)})
        x_cpu = self._f32_cpu(x)
        scale_cpu = self._f32_cpu(scale_flat)
        got_cpu = self._f32_cpu(recon)
        ref = x_cpu[sample]
        got = got_cpu[sample]
        abs_error = float((ref - got).abs().max().item())
        rel = abs_error / max(1e-6, float(ref.abs().max().item()))
        # Check the declared row-scale convention independently.  Symmetric
        # int8 per-row quantization conventionally maps max(abs(row)) to 127;
        # tolerate finite implementation rounding but not a different scale
        # convention (for example /255 or raw max without division).
        if scale_broadcast == "row_unsqueeze":
            expected_scale = x_cpu.abs().amax(dim=1) / 127.0
            actual_scale = scale_cpu.reshape(-1)
        else:
            expected_scale = x_cpu.abs().amax().reshape(1) / 127.0
            actual_scale = scale_cpu.reshape(-1)[:1]
        scale_denom = expected_scale.abs().clamp_min(1e-12)
        scale_rel = float(((actual_scale - expected_scale).abs() / scale_denom).max().item())
        abs_tol = float(actual_scale.abs().max().item()) if actual_scale.numel() else 0.0
        rel_tol = 0.02
        scale_rel_tol = 0.02
        if quantized.dtype != torch.int8:
            raise A3InputError(
                f"worker_correctness_failed:dynamic_quant:dtype:{quantized.dtype}")
        if rel > rel_tol or scale_rel > scale_rel_tol or abs_error > abs_tol:
            raise A3InputError(
                f"worker_correctness_failed:dynamic_quant:reconstruction={rel:g}:"
                f"absolute={abs_error:g}:scale={scale_rel:g}")
        check = {"status": "passed",
                 "reference": "cpu_reconstruction_with_scale_convention_check",
                 "max_relative_error": rel, "max_abs_error": abs_error,
                 "rows": sample, "scale_relative_error": scale_rel,
                 "scale_convention": ("per_row_max_abs_div_127"
                                      if scale_broadcast == "row_unsqueeze"
                                      else "scalar_max_abs_div_127"),
                 "relative_tolerance": rel_tol,
                 "scale_relative_tolerance": scale_rel_tol,
                 "absolute_tolerance": abs_tol,
                 "quantized_dtype": str(quantized.dtype).replace("torch.", ""),
                 "scale_dtype": str(scale.dtype).replace("torch.", "")}
        return {"op": op, "out": quantized, "input": (x,), "weight_format": "int8",
                "preallocated": False, "kind": "npu_dynamic_quant",
                "correctness": check, "observed_api": "torch_npu.npu_dynamic_quant",
                "details": {"rows": rows, "width": width,
                            "quantized_dtype": str(quantized.dtype).replace("torch.", ""),
                            "scale_dtype": str(scale.dtype).replace("torch.", ""),
                            "scale_shape": list(scale.shape),
                            "scale_broadcast": scale_broadcast,
                            "scale_convention": check["scale_convention"],
                            "reconstruction_relative_error": rel,
                            "scale_relative_error": scale_rel,
                            "declared_scale_format": str(
                                request["condition"]["precision"].get("scale_format") or ""),
                            "native_output_allocation": True, "copy_in_timed_region": False}}
