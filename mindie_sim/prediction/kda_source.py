"""Shared source-default resolution for ordinary KDA recurrent calls.

[Role] Resolve the existing vLLM-Ascend arch35 KDA source from a typed layer
  plan, runtime declaration and phase.  Used by both the body default and the
  ordinary serving lowering so target and serving cannot diverge.
[Invariants] Explicit recipe/family/source declarations win.  A3 and foreign
  architectures do not inherit the A5 source.  A missing gate bound is not a
  hardcoded false: the optional typed plan field is the only source.
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


def options(plan, *, activation_dtype):
    bound, valid = gate_bound(plan)
    if not valid:
        return None
    return {
        # Null/absent is the caller's false; zero is a valid non-None bound.
        "safe_gate": bound is not None,
        "use_beta_sigmoid": False,
        "allow_neg_eigval": False,
        "gate_dtype": activation_dtype,
        "beta_dtype": "fp32",
        "has_a_log": True,
        "has_dt_bias": True,
    }


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
    return options(plan, activation_dtype=activation_dtype)
