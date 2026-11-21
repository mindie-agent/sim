"""HF-config builders shared by the schema adapter tests and frontend lowering.

[Role] Single definition of the compact HF ``config(...)`` mapping, the raw
  serving-case wrapper ``run_case`` and the fixture loaders for the standard
  family / composable HF configs, so no module imports another test module.
[Invariants] Every builder returns a fresh mapping; ``config()`` is a 2-layer
  32-wide GQA body (4 heads, 2 KV heads, head_dim 8) that every standard and
  generic family adapter accepts; ``small_config`` shrinks a real fixture while
  preserving its family-specific branch declarations.
[Timeouts & Locks] Pure in-memory construction plus small fixture reads.
[Side-Effect Policy] Read-only fixture access; zero global mutation.
"""
from __future__ import annotations

from _support.captures import fixture_json
from _support.memory_cases import case

__all__ = ("case", "config", "run_case", "family_config", "small_case")

#: Standard-family fixture directory for each fixture stem.
STANDARD_DIRS = {
    **{name: "hf-standard-families" for name in (
        "deepseek-v2", "deepseek-v2-lite", "deepseek-v3", "deepseek-r1",
        "glm4-0414", "glm4-chat", "glm45", "glm45-air")},
    **{name: "standard-qwen-kimi" for name in (
        "qwen2-7b", "qwen2.5-7b-instruct", "qwen2-57b-a14b", "qwen3-8b",
        "qwen3-30b-a3b", "kimi-k2-instruct", "kimi-vl-a3b-instruct")},
}


def config(**changes) -> dict:
    """Two-layer 32-wide llama-shaped HF config with keyword overrides."""
    return dict(model_type="llama", num_hidden_layers=2, hidden_size=32,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                intermediate_size=64, vocab_size=128, **changes)


def run_case(model, **kwargs) -> dict:
    """Declared-GPU raw case (see ``memory_cases.case``) carrying ``model``."""
    raw = case(**kwargs)
    raw["model"] = model
    return raw


def family_config(name: str) -> dict:
    """Fresh copy of one standard-family HF fixture (``composable-hf`` fallback)."""
    return fixture_json(STANDARD_DIRS.get(name, "composable-hf"), f"{name}.json")


def small_case(name: str) -> dict:
    """Real standard fixture shrunk to 32-wide geometry, keeping its branches."""
    hf = family_config(name)
    text = hf.get("text_config", hf)
    text.update(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=128)
    if name.startswith("qwen3"):
        text["head_dim"] = 16  # deliberately != hidden/heads, as declared upstream
    if "num_experts" in text:
        text.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32)
    if "shared_expert_intermediate_size" in text:
        text["shared_expert_intermediate_size"] = 48
    if name.startswith("kimi"):
        text.update(n_routed_experts=4, num_experts_per_tok=2,
                    moe_intermediate_size=32, kv_lora_rank=16,
                    qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8)
        if text.get("q_lora_rank") is not None:
            text["q_lora_rank"] = 16
    raw = run_case(hf, output=2)
    raw["machine"]["device"]["compute_peaks"]["fp8_e4m3"] = 1e9
    raw["runtime"]["routing"] = {"policy": "balanced"}
    return raw
