"""Shared source-default resolution for ordinary KDA recurrent calls.

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


def gate_bound(plan):
    """Exact optional bound carried from the source linear_attn_config."""
    value = getattr(plan, "kda_gate_lower_bound", None)
    if value is not None and (isinstance(value, bool)
                              or not isinstance(value, (int, float))):
        return None, False
    return value, True


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
        # Null/absent is the caller's false; zero is a valid non-None bound.
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
        if (not lengths or any(type(q) is not int or not 1 <= q <= 8
                               for q in lengths)):
            return False
    return True


def resolve(plan, *, arch, phase, kernels, activation_dtype, views=None):
    """One source-default resolution used by body defaults and serving."""
    if not eligible(plan=plan, arch=arch, phase=phase, kernels=kernels, views=views):
        return None
    return options(plan, activation_dtype=activation_dtype, views=views)
