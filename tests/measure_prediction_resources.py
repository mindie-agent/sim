"""Isolated CPU/memory/time regression probe for prediction.

[Role] Run a real declared case with process limits and report resource usage.
[Invariants] Linux address-space and POSIX CPU limits apply before prediction;
  macOS memory is measured, not falsely described as an enforced RSS limit.
[Timeouts & Locks] Parent enforces wall time; this process has a 5s CPU budget.
[Side-Effect Policy] Reads one case on stdin, prints one report, touches no NPU.
"""
from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import time

resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
if sys.platform == "linux":
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mindie_sim.a3_framework import predict

def _peak_rss_bytes(usage: resource.struct_rusage) -> int:
    # On Linux, `getrusage(RUSAGE_SELF).ru_maxrss` inherits `signal->maxrss`
    # across fork+execve from the parent pytest runner. `/proc/self/status`
    # `VmHWM` reports `mm->hiwater_rss` for this post-execve process only.
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    return usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)


case = json.load(sys.stdin)
start = time.monotonic()
result = predict(case)
usage = resource.getrusage(resource.RUSAGE_SELF)
print(json.dumps({
    "status": result["status"], "output_tokens": result["output_tokens"],
    "elapsed_s": time.monotonic() - start,
    "cpu_s": usage.ru_utime + usage.ru_stime,
    "peak_rss_bytes": _peak_rss_bytes(usage),
    "trace_capture": result["trace_capture"],
}))
