"""Inspect a retained implementation's compiled program on an idle device.

[Role] Preserve compiler output and bounded primitive timings for investigation.
[Invariants] Execute the original image source, with no kernel edits or weights.
  Timings are observations and never installed into simulation parameters.
[Timeouts & Locks] Caller bounds the container; four shapes and five trials.
[Side-Effect Policy] Device 0 only, <=16 MiB tensors; writes to the selected folder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource


def main(output):
    import torch
    import torch_npu  # noqa: F401
    from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_kernel
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    free, total = torch.npu.mem_get_info()
    expected = 16 * 1024**2
    if free < expected + 512 * 1024**2:
        raise RuntimeError("insufficient_free_device_memory")
    source = Path("/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fused_gdn_gating.py")
    report = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "free_bytes": free, "total_bytes": total, "tensor_budget_bytes": expected,
        "scope": "primitive investigation; no calibration or service acceptance",
        "observations": [], "assembly": [],
    }
    init_device_properties_triton()
    heads, units = 48, get_vectorcore_num()
    a_log = torch.zeros(heads, dtype=torch.float32, device="npu")
    bias = torch.zeros_like(a_log)
    for tokens in (64, 256, 4096, 16384):
        a = torch.zeros((tokens, heads), dtype=torch.bfloat16, device="npu")
        b = torch.zeros_like(a)
        g = torch.empty((1, tokens, heads), dtype=torch.float32, device="npu")
        beta_output = torch.empty((1, tokens, heads), dtype=a.dtype, device="npu")
        arguments = (g, beta_output, a_log, a, b, bias, 1, heads, tokens, 1.0, 20.0)
        options = dict(BLK_HEADS=8, BLK_BATCHES=64, ROW_ITER=(tokens + units * 64 - 1) // (units * 64))
        compiled = fused_gdn_gating_kernel[(units, 1)](*arguments, **options)
        torch.npu.synchronize()
        times = []
        for _ in range(5):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            fused_gdn_gating_kernel[(units, 1)](*arguments, **options)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000)
        # Analytic values for zeros: g=-log(2), beta=1/2. Compare on CPU.
        error = abs(g.cpu().float() + 0.6931471805599453).max().item()
        beta_error = abs(beta_output.cpu().float() - 0.5).max().item()
        if error > 1e-5 or beta_error:
            raise RuntimeError(f"gating_output_mismatch:{error}:{beta_error}")
        report["observations"].append({
            "tokens": tokens, "heads": heads, "units": units, "row_iterations": options["ROW_ITER"],
            "durations_us": times, "max_g_error": error, "max_beta_error": beta_error,
        })
        for kind, assembly in compiled.asm.items():
            payload = assembly.encode() if isinstance(assembly, str) else bytes(assembly)
            name = f"tokens{tokens}.{kind}"
            (output / name).write_bytes(payload)
            report["assembly"].append({
                "tokens": tokens, "kind": kind, "path": name, "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        del a, b, g, beta_output
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"observations": report["observations"], "assembly": report["assembly"],
                      "peak_rss_kib": report["peak_rss_kib"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    main(parser.parse_args().output)
