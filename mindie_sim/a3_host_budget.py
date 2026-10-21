"""Host execution budget for predictive simulations.

[Role] Refuse server-sized prediction work on a desktop before DAG allocation.
[Invariants] This limits simulator work, not modeled NPU HBM or model accuracy.
  There is no local override that labels a desktop process as remote.
[Timeouts & Locks] Constant-time OS capacity queries; no processes, locks or sleeps.
[Side-Effect Policy] Read host capacity only; never submit a remote job implicitly.
"""
from __future__ import annotations

import math
import os
import sys

from .a3_types import A3InputError, CaseSpec


def desktop_memory_bytes() -> int | None:
    if sys.platform != "darwin":
        return None
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def check_prediction_host(case: CaseSpec, trace_detail: str = "bounded") -> None:
    capacity = desktop_memory_bytes()
    if capacity is None:
        return
    # A conservative work bound, not a claimed byte-accurate heap prediction.
    # Count each request separately even when batching could share a forward.
    chunk = min(case.runtime.max_num_batched_tokens,
                case.runtime.max_num_input_tokens or case.runtime.max_num_batched_tokens)
    iterations = sum(
        math.ceil(max(0, request.prompt_tokens - request.prefix_tokens) / chunk)
        + request.output_tokens
        for request in case.workload.requests
    )
    layer_iterations = max(1, case.model.layers) * iterations
    work_per_16gib = 512 if trace_detail == "full" else 4096
    limit = max(128, int(capacity / (16 * 1024**3) * work_per_16gib))
    if layer_iterations > limit:
        raise A3InputError(
            "host_capacity_requires_remote:"
            f"layer_iterations={layer_iterations}>desktop_limit={limit};"
            "run this case on a remote Linux server with remote-dev;"
            "do not retry locally or fan out local subagents"
        )
