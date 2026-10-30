"""Opt-in source-bounded instruction capacity declarations.

[Role] Expand an explicit preset into neutral owner/pipe service metadata.
[Invariants] Sort32 is a full 32-element AscendC API repeat, not a Sort macro
  or native ISA issue. Theoretical throughput supplies no dependency latency.
  SKU clocks are declarations; no rate is enabled merely by choosing a SKU.
[Timeouts & Locks] Bounded metadata arithmetic; no waits or locks.
[Side-Effect Policy] Return new mappings; no device, file or network access.
"""
from copy import deepcopy

from mindie_sim.schema.common import ModelingError


_REVISION = "56500ea0dc219380f50f4a17ce73ef5264f830fc"
_ROOT = "data/observations/sources/arch3510-physical-services/register-evidence/platform-units/"
_SOURCES = (
    ("vector_instruction_theoretical_perf_summary.md",
     "0d68bf9fb2584e21ca64bc3f4ab6643241186cedbf6d5e93a6f07901900c6600"),
    ("api_pipeline_type_summary.md",
     "04904132d7f803ab82a1c87297b6c9616fc6618f252ed08d92e7595872b82b02"),
)


def expand_instruction_presets(device, profile, sku):
    """Resolve selected source bounds after the final per-device override."""
    arch = device.get("compute_arch", {})
    presets = arch.get("presets", ())
    if not presets:
        return device
    data = deepcopy(device)
    arch = data["compute_arch"]
    services = arch.setdefault("instruction_services", {})
    clock = arch.pop("clock_hz", None)
    if clock is None:
        clock = profile["source"]["declaration"]["configured_clock_mhz"] * 1e6
        clock_source = {"kind": "sku_configuration", "sku": sku,
                        "field": "source.declaration.configured_clock_mhz"}
    else:
        clock_source = {"kind": "explicit_configuration", "field": "compute_arch.clock_hz"}
    for preset in arch.pop("presets"):
        if preset != "ascend_sort32_spec" or profile["architecture"] not in {"A2", "A3", "A5"}:
            raise ModelingError("instruction_service_preset_unknown:" + preset)
        declaration = {
            "unit": "blocks32", "owner_axis": "vector", "resource": "vector",
            "cycles_per_unit": 16, "clock_hz": clock,
            "provenance": {
                "kind": "isa_spec_bound",
                "spec_level": "AscendC_API_theoretical_throughput",
                "api": "Sort32", "dtypes": ["half", "float"],
                "elements_per_cycle": 2, "elements_per_unit": 32,
                "pipeline": "AIV/PIPE_V", "clock": clock_source,
                "domain": "full independent Sort32 repeats; excludes Sort macro and MrgSort",
                "interpretation": "capacity bound, not ISA issue or completion latency",
                "sources": [{"path": _ROOT + name, "sha256": digest,
                             "revision": _REVISION,
                             "url": "https://raw.githubusercontent.com/hicann/asc-devkit/"
                                    + _REVISION + "/docs/zh/api/appendix/" + name}
                            for name, digest in _SOURCES],
            },
        }
        explicit = services.get("vbitsort32", {})
        if "per_owner_rate" in explicit or "issue_ns" in explicit:
            declaration.pop("cycles_per_unit")
            declaration.pop("clock_hz")
            declaration["provenance"] = {"kind": "explicit_capacity_override",
                                       "preset": preset}
        services["vbitsort32"] = {**declaration, **explicit}
    return data
