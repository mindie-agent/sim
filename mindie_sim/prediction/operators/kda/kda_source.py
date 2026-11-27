"""Shared source resolution and context-parallel declarations for KDA calls.

[Role] Resolve the existing vLLM-Ascend arch35 KDA source from a typed layer
  plan, runtime declaration and phase.  Used by both the body default and the
  ordinary serving lowering so target and serving cannot diverge.
[Invariants] Explicit recipe/family/source declarations win.  A3 and foreign
  architectures do not inherit the A5 source.  A missing gate bound is not a
  hardcoded false: the optional typed plan field is the only source.  A view's
  exact context/acceptance is available for explicit state contracts, never
  inferred as a balanced sequence or a default all-row commit.
[Timeouts & Locks] Constant-time metadata resolution; no I/O or locks.
[Side-Effect Policy] Pure immutable selection metadata.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Literal

from mindie_sim.schema.datapath import SourceRef

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_VLLM_ASCEND_RECURRENT_REV = "8c30e4f491f73cba737a1dacc522894e437cdf3d"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "vllm-project/vllm-ascend",
        _VLLM_ASCEND_RECURRENT_REV,
        "csrc/attention/recurrent_kda/op_host/recurrent_kda_tiling_processor.h",
        "c9eac28bd528eb03bdeb456c4fdbed4951f716937570ee9d810e0fb5b3ef3fbd",
    ),
    SourceRef(
        "vllm-project/vllm-ascend",
        _VLLM_ASCEND_RECURRENT_REV,
        "vllm_ascend/ops/kda.py",
        "ca2f56280746f45a347578e7191decda716557c93aca69b5c3baf7b22a578803",
    ),
)

KDA_MAX_QUERY_ROWS = 8
KDA_SAFE_GATE_LOWER_BOUND_MIN = -5.0
KDA_SAFE_GATE_LOWER_BOUND_MAX = 0.0
NATIVE_KDA_MECHANISM = "recurrent_decode_native_kernels"
NATIVE_KDA_SELECTOR = "kda"


GENERIC_NAMES = {"recurrent_scan", "recurrent_vector", "short_convolution",
                 "delta_gates", "qk_l2norm"}


def explicit_generic(kernels):
    recipes = set(kernels.get("recipes", ()))
    families = set(kernels.get("families", ()))
    programs = set(kernels.get("programs", ()))
    names = recipes | families | programs
    return any(name in GENERIC_NAMES
               or (name.startswith("recurrent_scan")
                   and name[len("recurrent_scan"):].isdigit())
               for name in names)


def parse_gate_lower_bound(value):
    """Validate optional KDA safe-gate lower bound against CANN ``[-5, 0)``."""
    if value is None:
        return None, True
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        return None, False
    bound = float(value)
    if not (KDA_SAFE_GATE_LOWER_BOUND_MIN <= bound < KDA_SAFE_GATE_LOWER_BOUND_MAX):
        return None, False
    return bound, True


def gate_bound(plan):
    """Exact optional bound carried from the source linear_attn_config."""
    return parse_gate_lower_bound(getattr(plan, "kda_gate_lower_bound", None))


def view_state_declaration(views):
    """Exact caller context and acceptance pairs, never a balanced guess.

    A view's ``context`` is the committed token count before this invocation.
    The scheduler's ``draft_advanced_tokens`` is the committed advance for the
    current verification; zero means no acceptance event has been declared and
    the recurrent contract must stay pending.  Padding views are retained so
    the sequence partition keeps its physical rows.
    """
    views = tuple(views)
    contexts = tuple(int(getattr(view, "context", 0)) for view in views)
    acceptance = tuple(
        (int(getattr(view, "draft_accepted_tokens", 0) or 0),
         int(getattr(view, "draft_advanced_tokens", 0) or 0))
        for view in views)
    return contexts, acceptance


def options(plan, *, activation_dtype, views=None):
    bound, valid = gate_bound(plan)
    if not valid:
        return None
    result = {
        # Null/absent selects softplus (safe_gate=False); valid [-5, 0) selects safe_gate=True.
        "safe_gate": bound is not None,
        # A5 arch35 fuses ``SIGMOID_BETA`` (``tl.sigmoid(raw_beta)``) into the
        # KDA kernel, including the RecoverSSM verify call, so no standalone
        # FP32 ``linear_beta_sigmoid`` launch remains.  The beta operand keeps
        # the activation dtype for that in-kernel expression.
        "use_beta_sigmoid": True,
        "allow_neg_eigval": False,
        "gate_dtype": activation_dtype,
        "beta_dtype": activation_dtype,
        "has_a_log": True,
        "has_dt_bias": True,
    }
    return result


def eligible(*, plan, arch, phase, kernels, views=None):
    if getattr(plan, "attention", "") != "kda" or phase != "decode":
        return False
    if arch != "A5":
        return False
    if "kda" in kernels.get("lowerings", {}):
        return False
    if explicit_generic(kernels):
        return False
    if views is not None:
        if any(view.phase != "decode" for view in views):
            return False
        # The native recurrent source is valid only for per-sequence q 1..8.
        lengths = tuple(getattr(view, "q_len", None) for view in views)
        if (not lengths or any(type(q) is not int or not 1 <= q <= KDA_MAX_QUERY_ROWS
                               for q in lengths)):
            return False
    return True


def resolve(plan, *, arch, phase, kernels, activation_dtype, views=None):
    """One source-default resolution used by body defaults and serving."""
    if not eligible(plan=plan, arch=arch, phase=phase, kernels=kernels, views=views):
        return None
    return options(plan, activation_dtype=activation_dtype, views=views)


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
    from mindie_sim.catalog.paired_sfa import execution
    from mindie_sim.prediction.implementations import select

    if runtime is None:
        return None
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    execution_policy = dict(metadata.get("execution_policy") or {})
    if metadata.get("raw_eager") or execution_policy.get("raw_eager"):
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
