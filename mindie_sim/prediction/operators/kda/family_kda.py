"""Explicit source domain for KDA kernels that perform gate work internally.

[Role] Select the retained KDA wrapper's native semantic boundary.
[Invariants] Selection records source behavior, not capture binary identity.
  Chunk stage identity remains unknown even with this wrapper declaration.
[Timeouts & Locks] Pure metadata validation; no I/O or locks.
[Side-Effect Policy] No device work, source reads or global mutation.
"""
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef


TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_VLLM_ASCEND_REV = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/ops/kimi_kda.py",
        source_sha256="af37687463f9b34ca4797bdcefb14c333d3b70b2956d6aa1f15217815d5fbc10",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/ops/kda.py",
        source_sha256="db94b9d9c2a11d3d7d6fcc0e4619328b01deb74f486f74787a6fd28805fa2a4c",
    ),
)
IMPLEMENTATION = "vllm_ascend_kda_in_kernel_gates"
CAPABILITIES = frozenset({"prefill", "decode", "in_kernel_gates"})
SOURCE = {
    "repository": "vllm-ascend",
    "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
    "path": "vllm_ascend/ops/kimi_kda.py",
    "sha256": "35365b3b353fc492b4dc5d75b88916431f6802c21bf51b61b4893d39d54f11f3",
    "symbols": ["KimiGatedDeltaNetAttention._run_recurrent",
                "KimiGatedDeltaNetAttention._run_prefill"],
    "lines": "434-518",
    "scope": "independent_source_declaration; not_capture_binary_identity",
}


def recipes():
    return {}


def validate(*, key_heads, value_heads, key_width, value_width, model):
    if (key_heads != value_heads or key_width != value_width
            or model.activation_dtype not in {"bf16", "fp16"}):
        raise ModelingError("kda_in_kernel_gate_source_domain_unsupported")
