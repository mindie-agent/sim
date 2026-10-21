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

case = json.load(sys.stdin)
start = time.monotonic()
result = predict(case)
usage = resource.getrusage(resource.RUSAGE_SELF)
print(json.dumps({
    "status": result["status"], "output_tokens": result["output_tokens"],
    "elapsed_s": time.monotonic() - start,
    "cpu_s": usage.ru_utime + usage.ru_stime,
    "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
    "trace_capture": result["trace_capture"],
}))
