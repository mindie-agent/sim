"""Normalize explicit device declarations before applying inheritance.

[Role] Merge profile, machine and per-device inputs in physical resource units.
[Invariants] A later declaration overrides an inherited value. Within one
  declaration, neutral fields override native aliases. Partial resource maps
  preserve other declared resources; rates never scale with a model label.
[Timeouts & Locks] Bounded metadata arithmetic; no waits or locks.
[Side-Effect Policy] Returns new mappings without changing caller inputs.
"""
from copy import deepcopy


_MAP_FIELDS = (
    "compute_peaks", "rates", "buffers", "resource_units", "buffer_units",
    "work_units", "kernel_invocations", "store_queues",
)
_RATE_FIELDS = {
    "hbm_bw_bytes_per_s": ("hbm",),
    "vector_ops_per_s": ("vector",),
    "mte_bw_bytes_per_s": ("mte2", "mte2_aic", "mte2_aiv", "mte3"),
    "mte1_bw_bytes_per_s": ("mte1",),
    "mte2_bw_bytes_per_s": ("mte2",),
    "mte2_aic_bw_bytes_per_s": ("mte2_aic",),
    "mte2_aiv_bw_bytes_per_s": ("mte2_aiv",),
    "mte3_bw_bytes_per_s": ("mte3",),
    "fixpipe_bw_bytes_per_s": ("fixpipe",),
}
_RATE_ALIASES = {"memory": "hbm", "read": "mte2", "write": "mte3"}
_POOLS = ("l1", "l0a", "l0b", "l0c", "ub")


def machine_device_defaults(raw):
    """Extract common device supply; rank/node/package remain placement fields."""
    fields = {
        *_MAP_FIELDS, "service_modes", "compute_arch", *_RATE_FIELDS,
        "memory_bytes", "hbm_gib", "hbm_bytes", "cube_flops_per_s",
        "compute_units", "aic_count", "aiv_count", "launch_us",
        *(pool + "_bytes" for pool in _POOLS),
    }
    return {key: value for key, value in raw.items() if key in fields}


def _normalize(raw):
    data = deepcopy(raw)
    rates = {}
    for field, roles in _RATE_FIELDS.items():
        if field in data:
            value = data.pop(field)
            rates.update((role, value) for role in roles)
    explicit = data.get("rates", {})
    rates.update((_RATE_ALIASES[key], value) for key, value in explicit.items()
                 if key in _RATE_ALIASES)
    rates.update((key, value) for key, value in explicit.items()
                 if key not in _RATE_ALIASES)
    if rates or "rates" in data:
        data["rates"] = rates

    if "cube_flops_per_s" in data:
        value = data.pop("cube_flops_per_s")
        data["compute_peaks"] = {
            "bf16": value, "fp16": value, **data.get("compute_peaks", {}),
        }
    memory = {}
    if "hbm_gib" in data:
        memory["memory_bytes"] = float(data.pop("hbm_gib")) * 1024**3
    if "hbm_bytes" in data:
        memory["memory_bytes"] = data.pop("hbm_bytes")
    data = {**memory, **data}
    if "aic_count" in data:
        count = data.pop("aic_count")
        data.setdefault("compute_units", count)
    if "aiv_count" in data:
        count = data.pop("aiv_count")
        data["resource_units"] = {"vector": count, **data.get("resource_units", {})}
    buffers = {pool: data.pop(pool + "_bytes")
               for pool in _POOLS if pool + "_bytes" in data}
    if buffers:
        data["buffers"] = {**buffers, **data.get("buffers", {})}
    return data


def _merge_service_modes(base, declared, ratios):
    """Keep inherited mode/base supply ratios under explicit rate overrides."""
    merged = {name: dict(mode) for name, mode in base.items()}
    for name, mode in merged.items():
        if "rate" in mode and mode.get("resource") in ratios:
            merged[name] = {**mode, "rate": mode["rate"] * ratios[mode["resource"]]}
    for name, mode in (declared or {}).items():
        # A partial record (for example only ``rate``) keeps its inherited
        # resource. Explicit records are never rescaled by this layer.
        merged[name] = {**merged.get(name, {}), **mode}
    return merged


def merge_device_inputs(*declarations):
    """Canonicalize each layer before later aliases override earlier defaults."""
    result = {}
    for raw in declarations:
        data = _normalize(raw)
        if "compute_arch" in data:
            previous_arch = result.get("compute_arch", {})
            data["compute_arch"] = {**previous_arch, **data["compute_arch"],
                "instruction_services": {
                    **previous_arch.get("instruction_services", {}),
                    **data["compute_arch"].get("instruction_services", {})}}
        previous = result.get("rates", {})
        ratios = {}
        for role, new_rate in data.get("rates", {}).items():
            old_rate = previous.get(role)
            if (isinstance(old_rate, (int, float)) and isinstance(new_rate, (int, float))
                    and not isinstance(old_rate, bool) and not isinstance(new_rate, bool)
                    and old_rate > 0 and new_rate > 0):
                ratios[role] = new_rate / old_rate
        if ratios or "service_modes" in data:
            data["service_modes"] = _merge_service_modes(
                result.get("service_modes", {}), data.get("service_modes"), ratios)
        for field in _MAP_FIELDS:
            if field in data:
                if field == "store_queues":
                    data[field] = {**result.get(field, {}), **{
                        name: {**result.get(field, {}).get(name, {}), **queue}
                        for name, queue in data[field].items()}}
                else:
                    data[field] = {**result.get(field, {}), **data[field]}
        result.update(data)
    return result
