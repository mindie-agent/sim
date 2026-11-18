"""Shared KDA source resolution and serial-context declaration helpers.

[Role] Resolve the A5 KDA source from a runtime declaration and collapse
  context parallelism while preserving a declared PCP plan.  Used by the body
  default and the ordinary serving lowering so they cannot diverge.
[Invariants] An explicit recipe/family/source declaration wins.  KDA state is
  head-local, so a serial recurrence is valid under PCP but the parallel spec
  still requires the matching plan; ``cp_plan="none"`` with ``pcp > 1`` is an
  invalid declaration, not a supported fallback.
[Timeouts & Locks] Pure declarations; no I/O or locks.
[Side-Effect Policy] Returns immutable records only.
"""
from __future__ import annotations

from dataclasses import replace

from .implementations import select


NATIVE_KDA_MECHANISM = "recurrent_decode_native_kernels"
NATIVE_KDA_SELECTOR = "kda"


def native_kda_forbidden(runtime) -> bool:
    """True when the runtime explicitly retains the analytical KDA source.

    ``va_recommended=False`` disables every optional fused lowering, and an
    explicit ``recurrent_decode_native_kernels`` exclusion owns the ``kda``
    selector its mechanism would install.  A bare declaration without either
    policy is still allowed to resolve the structural source default.
    """
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    policy = dict(metadata.get("execution_policy") or {})
    if policy.get("automatic_optional_fusions") is False:
        return True
    report = dict(metadata.get("feature_track_report") or {})
    return NATIVE_KDA_MECHANISM in tuple(report.get("explicit_excluded") or ())


def strip_excluded_native_kda(runtime):
    """Remove an automatic ``kda`` lowering when its mechanism is excluded.

    ``compose`` already removes every declared selector owned by an explicit
    exclusion, so a binding that survives into the body was installed by the
    structural default.  Dropping it here keeps the graph analytical instead of
    re-acquiring the excluded kernel through ``body_defaults``.
    """
    if runtime is None:
        return runtime
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    report = dict(metadata.get("feature_track_report") or {})
    if NATIVE_KDA_MECHANISM not in tuple(report.get("explicit_excluded") or ()):
        return runtime
    kernels = dict(getattr(runtime, "kernels", {}) or ())
    lowerings = dict(kernels.get("lowerings", {}) or ())
    if NATIVE_KDA_SELECTOR not in lowerings:
        return runtime
    lowerings.pop(NATIVE_KDA_SELECTOR)
    kernels["lowerings"] = lowerings
    return replace(runtime, kernels=kernels)


def serving_source(runtime, model, views, plan):
    """Shared A5 KDA source selection for ordinary serving."""
    from .kda_source import resolve
    if runtime is None:
        return None
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    execution = dict(metadata.get("execution_policy") or {})
    if metadata.get("raw_eager") or execution.get("raw_eager"):
        # A raw-eager serving step has no optional fused lowering inventory.
        return None
    if native_kda_forbidden(runtime):
        # The caller disabled optional fusions or excluded this mechanism; an
        # absent binding is that decision, not permission to install one.
        return None
    if "kda" in tuple(metadata.get("analytical_implementations", ()) or ()):
        # A declared policy already chose its analytical fallback; an absent
        # native binding is that decision, not permission to install one.
        return None
    parallel = getattr(runtime, "parallel", None)
    from mindie_sim.catalog.paired_sfa import execution
    if (int(getattr(parallel, "cp", 1) or 1) > 1
            and execution(model).get("recurrent_context_sharded") is not False):
        return None
    opts = resolve(plan, arch=dict(runtime.metadata).get("arch_family"),
                   phase=(views[0].phase if views else ""), kernels=runtime.kernels,
                   activation_dtype=model.activation_dtype, views=views)
    if opts is None:
        return None
    return select({"kda": {"implementation": "vllm_ascend_recurrent_kda_arch35",
                           "options": opts}}, "kda")


def serial_context_parallel(parallel):
    """Collapse context parallelism while preserving a declared PCP plan.

    KDA state is head-local, so a serial recurrence is valid under PCP.  The
    parallel spec still requires the matching plan; passing ``cp_plan="none"``
    with ``pcp > 1`` is an invalid declaration, not a supported fallback.
    """
    next_plan = "pcp" if parallel.pcp > 1 else "none"
    return replace(parallel, cp=1, dcp=1, cp_plan=next_plan)
