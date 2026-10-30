"""Profiler-owned native rows for independent operator comparisons.

[Role] Expose the optional profiler parser to simulation observation consumers.
[Invariants] Tensor contracts, integer timestamps and source identities are
  produced by mindie_profiler. Simulation alone does not require that package.
[Timeouts & Locks] The shared parser owns bounded streaming reads.
[Side-Effect Policy] Reads only the caller-selected capture; no device work.
"""

PROFILER_API_REQUIREMENT = (
    "Profiling imports require the companion profiler development source: "
    "python -m pip install -e /path/to/profiler. Released mindie-profiler 0.5.0 "
    "does not provide normalized/tensor_contracts; the profiling extra alone "
    "is insufficient until a compatible release is available."
)


def kernel_rows(path, **kwargs):
    try:
        from mindie_profiler.normalized import kernel_rows as read_rows
    except ModuleNotFoundError as exc:
        if exc.name not in {"mindie_profiler", "mindie_profiler.normalized"}:
            raise
        raise ModuleNotFoundError(PROFILER_API_REQUIREMENT) from exc
    yield from read_rows(path, **kwargs)
