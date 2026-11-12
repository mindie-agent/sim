"""Dense FlashComm sequence-parallel admission for ``qwen3_5_text``.

[Role] Decide whether a TP block's collective is all-gather plus
  reduce-scatter rather than one allreduce. The predicate is vLLM-Ascend
  ``enable_dense_sp`` at commit 7672a5b509467288a12d7a2cee3451dd939d8b9c,
  which is an ancestor of the image-window tip e8ddf634ec61.
[Invariants]
  * No duration, floor, byte table, or hardware identity is selected here.
  * The text model type must be exactly ``qwen3_5_text``. ``qwen3_5_moe_text``
    and every other type stay on the ordinary allreduce.
  * TP > 1 and PP == 1. DP is a different group and does not admit the
    protocol.
  * The switch is ``additional_config.enable_flashcomm1`` using the source
    string rule (``1`` / ``true``). A captured serving-env declaration
    ``VLLM_ASCEND_ENABLE_FLASHCOMM1`` is the only fallback. This process
    environment is not the serving process.
[Timeouts & Locks] Pure predicate. No I/O.
[Side-Effect Policy] Returns a bool.
"""
from __future__ import annotations

from typing import Any, Mapping

DENSE_SP_MODEL_TYPES = ("qwen3_5_text",)
_TRUE_STRINGS = ("1", "true")
_ENV_KEY = "VLLM_ASCEND_ENABLE_FLASHCOMM1"


def _as_map(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}
    return {}


def _text_model_type(model: Any, features: Any) -> str:
    """Flattened text ``model_type`` recorded by the catalog, not the outer wrapper."""
    facts = _as_map(getattr(model, "catalog_facts", ()))
    meta = _as_map(getattr(model, "metadata", ()))
    feat = _as_map(getattr(features, "metadata", ()))
    for value in (
        meta.get("source_model_type"),
        facts.get("source_model_type"),
        facts.get("model_type"),
        feat.get("source_model_type"),
    ):
        if value:
            return str(value)
    return ""


def _switch_string(value: str) -> bool:
    return value.strip().lower() in _TRUE_STRINGS


def _captured_env_on(meta: Mapping[str, Any]) -> bool:
    declared = meta.get(_ENV_KEY)
    env = meta.get("env")
    if declared is None and isinstance(env, (Mapping, tuple)):
        declared = _as_map(env).get(_ENV_KEY)
    if isinstance(declared, str):
        return _switch_string(declared)
    return False


def flashcomm_switch_enabled(features: Any) -> bool:
    """Source ``_flashcomm_switch_enabled``, without this process environment."""
    meta = _as_map(getattr(features, "metadata", ()))
    additional = _as_map(meta.get("additional_config"))
    if "enable_flashcomm1" in additional:
        switch = additional["enable_flashcomm1"]
        if isinstance(switch, str):
            return _switch_string(switch)
        if bool(switch):
            return True
    return _captured_env_on(meta)


def dense_flashcomm_enabled(model: Any, features: Any) -> bool:
    """True only when the v0.30 dense sequence-parallel contract admits this model."""
    tp = int(getattr(features, "tp", 1) or 1)
    pp = int(getattr(features, "pp", 1) or 1)
    if tp <= 1 or pp != 1:
        return False
    if _text_model_type(model, features) not in DENSE_SP_MODEL_TYPES:
        return False
    return flashcomm_switch_enabled(features)
