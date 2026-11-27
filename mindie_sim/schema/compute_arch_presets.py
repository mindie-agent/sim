"""Registered compute-architecture capability presets.

[Role]
  Declare payload roles, native compute modes, casts and directional paths.
[Invariants]
  Preset data records structural capabilities with source provenance.
  A payload format alone does not declare a native compute mode.
[Timeouts & Locks]
  Static preset construction. No I/O or locks.
[Side-Effect Policy]
  Builds capability mappings without loading devices or captures.
"""
from __future__ import annotations

from .device_input import SubtileWaveSpec

_NIBBLE_FORMAT_BITS = 4
_BYTE_BITS = 8
_HALFWORD_FORMAT_BITS = 16
_WORD_FORMAT_BITS = 32
_CUBE_DIM = 16
_NIBBLE_CUBE_K = 64
_BYTE_CUBE_K = 32
_HF32_CUBE_K = 8
_DAV_2201_FP32_CUBE_K = 4
_DAV_3510_FP32_CUBE_K = 1
_BIAS_FIXPIPE_BYTES_PER_CYCLE = 32
_DAV_3510_UB_COPY_BYTES_PER_CYCLE = 256

_DIRECT_ROUTE_ALIGNMENT = {
    "gm": 1, "l1": 32, "l0a": 512, "l0b": 512, "l0c": 64, "ub": 32,
    "bt": 32, "fb": 64,
}


def _prov(kind, architecture, basis, source):
    return {"kind": kind, "architecture": architecture, "basis": basis,
            "source": source}


_2201_SOURCE = "registered A2/A3 profile npu_arch=2201 and design 4.7"
_3510_SOURCE = ("data/observations/sources/arch3510-physical-services/"
                "platform-units/npu_arch_3510.md")


def _storage_representations(architecture, source, *, fp8_operand,
                            int4_operand):
    """Payload formats every family stores/transfers; operand roles differ."""
    result = {}
    for name, bits, roles in (
        ("bf16", _HALFWORD_FORMAT_BITS, ("storage", "transfer", "operand", "accumulator")),
        ("fp16", _HALFWORD_FORMAT_BITS, ("storage", "transfer", "operand", "accumulator")),
        ("fp32", _WORD_FORMAT_BITS, ("storage", "transfer", "operand", "accumulator")),
        ("hf32", _WORD_FORMAT_BITS, ("storage", "transfer", "operand")),
        ("int8", _BYTE_BITS, ("storage", "transfer", "operand")),
        ("int4", _NIBBLE_FORMAT_BITS, (("storage", "transfer", "operand") if int4_operand
                     else ("storage", "transfer"))),
        ("int32", _WORD_FORMAT_BITS, ("storage", "transfer", "accumulator")),
        ("u8", _BYTE_BITS, ("storage", "transfer")),
        ("s8", _BYTE_BITS, ("storage", "transfer")),
        ("u16", _HALFWORD_FORMAT_BITS, ("storage", "transfer")),
        ("s16", _HALFWORD_FORMAT_BITS, ("storage", "transfer")),
        ("u32", _WORD_FORMAT_BITS, ("storage", "transfer")),
        ("s32", _WORD_FORMAT_BITS, ("storage", "transfer")),
    ):
        result[name] = {"bits": bits, "roles": list(roles),
                        "alignment_bytes": max(1, bits // _BYTE_BITS),
                        "provenance": _prov("payload_format", architecture,
                                            "payload storage/transfer role", source)}
    fp8_roles = ("storage", "transfer", "operand") if fp8_operand else (
        "storage", "transfer")
    for name in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        result[name] = {"bits": _BYTE_BITS, "roles": list(fp8_roles), "alignment_bytes": 1,
                        "provenance": _prov(
                            "payload_format", architecture,
                            ("8-bit payload may be stored/transferred; native FP8 "
                             "matrix is a separate declared mode")
                            if not fp8_operand else "native FP8 operand",
                            source)}
    if fp8_operand:
        for name, bits in (("fp4_e2m1", _NIBBLE_FORMAT_BITS), ("mxfp4", _NIBBLE_FORMAT_BITS),
                           ("hifloat8", _BYTE_BITS), ("mxfp8", _BYTE_BITS)):
            result[name] = {
                "bits": bits, "roles": list(fp8_roles), "alignment_bytes": 1,
                "provenance": _prov("payload_format", architecture, "native Cube operand", source),
            }
    return result


def _native_machine_modes(architecture, source, *, fp8, int4_operand):
    modes = {}
    formats = [
        ("bf16", "fp32"), ("fp16", "fp32"), ("fp32", "fp32"),
        ("hf32", "fp32"), ("int8", "int32"),
    ]
    if int4_operand:
        # arch2201 converts INT4 k64; arch3510 storage-only INT4 (S4 is not a
        # native Cube operand) must not become a Cube mode through a peak.
        formats.append(("int4", "int32"))
    if fp8:
        formats.extend((operand, "fp32") for operand in ("fp4_e2m1", "mxfp4", "hifloat8", "mxfp8"))
    for operand, accumulator in formats:
        name = f"{operand}_x_{operand}_acc_{accumulator}"
        modes[name] = {"kind": "matrix", "operands": [operand],
                       "accumulators": [accumulator], "unit": "cube",
                       "provenance": _prov("native_matrix_mode", architecture,
                                           "native matrix operand/accumulator",
                                           source)}
    if fp8:
        for operand in ("fp8_e4m3", "fp8_e5m2"):
            name = f"{operand}_x_{operand}_acc_fp32"
            modes[name] = {"kind": "matrix",
                           "operands": [operand, "fp8"],
                           "accumulators": ["fp32"], "unit": "cube",
                           "provenance": _prov(
                               "native_matrix_mode", architecture,
                               "native FP8 matrix operand; FP32 accumulator is "
                               "the declared profile condition (source lists "
                               "operand formats only)", source)}
    return modes


def _casts(architecture, source_ref, extra=()):
    return {
        f"{source_format}->{target}": {
            "source": source_format, "target": target,
            "lossless": lossless,
            "provenance": _prov(
                "declared_element_conversion", architecture,
                "representation conversion only; no selected kernel, runtime "
                "implementation or rate is claimed", source_ref)}
        for source_format, target, lossless in (
            ("bf16", "fp32", True), ("fp32", "bf16", False),
            ("fp16", "fp32", True), ("fp32", "fp16", False),
            ("bf16", "fp16", False), ("fp16", "bf16", False),
            ("hf32", "fp32", True), ("fp32", "hf32", False), *extra,
        )
    }


def _path(source, target, roles, *, unit="", executor="", reach=(),
          buffer="", sync="pipe", representation_provenance=None):
    source_alignment = _DIRECT_ROUTE_ALIGNMENT[source]
    target_alignment = _DIRECT_ROUTE_ALIGNMENT[target]
    item = {"source": source, "target": target, "roles": list(roles),
            "sync": sync, "alignment_bytes": max(source_alignment, target_alignment)}
    if unit:
        item["unit"] = unit
    if executor:
        item["executor"] = executor
    if reach:
        item["reach"] = list(reach)
    if buffer:
        item["buffer"] = buffer
    item["provenance"] = {
        **representation_provenance,
        "source_alignment_bytes": source_alignment, "target_alignment_bytes": target_alignment,
    }
    return item


def _paths_2201():
    source = _prov("directional_data_path", "npu_arch_2201",
                   "registration of the legacy direct-route set", _2201_SOURCE)
    paths = {}
    for name, src, dst, roles, unit, executor, reach, buffer in (
        ("gm->l1", "gm", "l1", ("storage", "transfer"),
         "mte2_aic", "aic", ("aic",), "l1"),
        ("gm->ub", "gm", "ub", ("storage", "transfer"),
         "mte2_aiv", "aiv", ("aiv",), "ub"),
        ("ub->gm", "ub", "gm", ("storage", "transfer"),
         "mte3", "aiv", ("aiv",), ""),
        ("l1->l0a", "l1", "l0a", ("operand",), "mte1", "aic", ("aic",), "l0a"),
        ("l1->l0b", "l1", "l0b", ("operand",), "mte1", "aic", ("aic",), "l0b"),
        ("l0c->gm", "l0c", "gm", ("accumulator",),
         "fixpipe", "aic", ("aic",), ""),
        ("l0c->l1", "l0c", "l1", ("accumulator",),
         "fixpipe", "aic", ("aic",), "l1"),
        # 2201 expresses C/V exchange through GM; these direct routes exist.
        ("gm->l0a", "gm", "l0a", ("operand",),
         "mte2_aic", "aic", ("aic",), "l0a"),
        ("gm->l0b", "gm", "l0b", ("operand",),
         "mte2_aic", "aic", ("aic",), "l0b"),
        ("l1->gm", "l1", "gm", ("storage", "transfer"), "mte3", "aic", ("aic",), ""),
    ):
        paths[name] = _path(src, dst, roles, unit=unit, executor=executor,
                            reach=reach, buffer=buffer,
                            representation_provenance=source)
    return paths


def _paths_3510():
    source = _prov("directional_data_path", "npu_arch_3510",
                   "sourced L0C->UB and UB->L1 additions; GM->L0A/L0B and "
                   "L1->GM removed", _3510_SOURCE)
    paths = {}
    for name, src, dst, roles, unit, executor, reach, buffer, sync in (
        ("gm->l1", "gm", "l1", ("storage", "transfer"),
         "mte2_aic", "aic", ("aic",), "l1", "pipe"),
        ("gm->ub", "gm", "ub", ("storage", "transfer"),
         "mte2_aiv", "aiv", ("aiv",), "ub", "pipe"),
        ("ub->gm", "ub", "gm", ("storage", "transfer"),
         "mte3", "aiv", ("aiv",), "", "pipe"),
        ("l1->l0a", "l1", "l0a", ("operand",),
         "mte1", "aic", ("aic",), "l0a", "pipe"),
        ("l1->l0b", "l1", "l0b", ("operand",),
         "mte1", "aic", ("aic",), "l0b", "pipe"),
        # Sourced L1->UB route; the profile binds the shared MTE1 mode at
        # 128 B/cycle. T07 confirms the final owner/sync/capacity contract.
        ("l1->ub", "l1", "ub", ("storage", "transfer"),
         "mte1_l1_ub", "aic", ("aic", "aiv"), "ub", "event"),
        ("l0c->gm", "l0c", "gm", ("accumulator",),
         "fixpipe", "aic", ("aic",), "", "pipe"),
        ("l0c->l1", "l0c", "l1", ("accumulator",),
         "fixpipe", "aic", ("aic",), "l1", "pipe"),
        ("l0c->ub", "l0c", "ub", ("accumulator",),
         "fixpipe", "aic", ("aic", "aiv"), "ub", "event"),
        ("ub->l1", "ub", "l1", ("storage", "transfer"),
         "mte3", "aiv", ("aic", "aiv"), "l1", "event"),
    ):
        paths[name] = _path(
            src, dst, roles, unit=unit, executor=executor, reach=reach,
            buffer=buffer, sync=sync,
            representation_provenance=source)
    paths["l1->ub"]["provenance"]["basis"] = (
        "sourced L1->UB PIPE_MTE1 128 B/cycle route; profile binds shared "
        "MTE1 mode mte1_l1_ub; final owner/sync/capacity remains T07")
    return paths


def _local_copy_paths(architecture, source, *, ub_rate):
    # The compute-arch interchange schema describes movement between distinct
    # storage domains. Datapath adds local copies and the dedicated BT/FB ports.
    paths = {}
    for src, dst, unit, executor, pipe, rate in (
        ("ub", "ub", "vector" if ub_rate is None else "ub_copy", "aiv", "PIPE_V", ub_rate),
        ("l1", "bt", "mte1", "aic", "PIPE_MTE1", _BIAS_FIXPIPE_BYTES_PER_CYCLE),
        ("l1", "fb", "fixpipe", "aic", "PIPE_FIX", _BIAS_FIXPIPE_BYTES_PER_CYCLE),
    ):
        name = f"{src}->{dst}"
        provenance = _prov("directional_data_path", architecture, pipe, source)
        if rate is not None:
            provenance["bytes_per_cycle"] = rate
        paths[name] = _path(src, dst, ("storage", "transfer"), unit=unit,
                            executor=executor, reach=(executor,), buffer=dst,
                            representation_provenance=provenance)
    return paths


def _subtile_waves(*, fp32_cube_k, extra):
    sizes = {"bf16": _CUBE_DIM, "fp16": _CUBE_DIM, "int8": _BYTE_CUBE_K,
             "hf32": _HF32_CUBE_K, **extra}
    sizes["fp32"] = _HF32_CUBE_K
    return tuple(
        SubtileWaveSpec.from_mapping({
            "native_mode": mode, "tile_m": _CUBE_DIM, "tile_n": _CUBE_DIM, "tile_k": tile_k,
            "independently_issuable_m": _CUBE_DIM, "owner_groups": 1,
            "cube_k": fp32_cube_k if mode == "fp32" else tile_k,
        })
        for mode, tile_k in sizes.items()
    )


CAPABILITY_PRESETS: dict[str, dict] = {
    "npu_arch_2201": {
        "representations": _storage_representations(
            "npu_arch_2201", _2201_SOURCE, fp8_operand=False,
            int4_operand=True),
        "native_compute": _native_machine_modes(
            "npu_arch_2201", _2201_SOURCE, fp8=False, int4_operand=True),
        "casts": _casts("npu_arch_2201", _2201_SOURCE),
        "paths": _paths_2201(),
        "local_copy_paths": _local_copy_paths("npu_arch_2201", _2201_SOURCE, ub_rate=None),
        "subtile_waves": _subtile_waves(fp32_cube_k=_DAV_2201_FP32_CUBE_K, extra={"int4": _NIBBLE_CUBE_K}),
    },
    "npu_arch_3510": {
        "representations": _storage_representations(
            "npu_arch_3510", _3510_SOURCE, fp8_operand=True,
            int4_operand=False),
        "native_compute": _native_machine_modes(
            "npu_arch_3510", _3510_SOURCE, fp8=True, int4_operand=False),
        "casts": _casts("npu_arch_3510", _3510_SOURCE, extra=(
            ("bf16", "hifloat8", False), ("hifloat8", "bf16", True),
            ("bf16", "fp8_e4m3", False), ("fp8_e4m3", "bf16", True),
            ("bf16", "fp8_e5m2", False), ("fp8_e5m2", "bf16", True),
        )),
        "paths": _paths_3510(),
        "local_copy_paths": _local_copy_paths(
            "npu_arch_3510", _3510_SOURCE, ub_rate=_DAV_3510_UB_COPY_BYTES_PER_CYCLE),
        "subtile_waves": _subtile_waves(fp32_cube_k=_DAV_3510_FP32_CUBE_K, extra={
            "fp4_e2m1": _NIBBLE_CUBE_K, "mxfp4": _NIBBLE_CUBE_K, "hifloat8": _BYTE_CUBE_K,
            "mxfp8": _BYTE_CUBE_K, "fp8_e4m3": _BYTE_CUBE_K, "fp8_e5m2": _BYTE_CUBE_K,
        }),
    },
}


def preset_datapath_parts(preset_name, declared_waves):
    """Resolve the preset's local paths and fill omitted wave geometry."""
    from .compute_arch import DataPath

    preset = CAPABILITY_PRESETS.get(preset_name, {})
    paths = tuple(
        DataPath(name, **{**path, "roles": tuple(path["roles"]), "reach": tuple(path["reach"])})
        for name, path in preset.get("local_copy_paths", {}).items()
    )
    waves = {wave.native_mode: wave for wave in preset.get("subtile_waves", ())}
    for wave in declared_waves:
        cube_k = getattr(waves.get(wave.native_mode), "cube_k", None)
        if (getattr(wave, "cube_k", None) is None and cube_k is not None
                and wave.tile_k % cube_k == 0):
            wave = SubtileWaveSpec.from_mapping({**wave.to_dict(), "cube_k": cube_k})
        waves[wave.native_mode] = wave
    return paths, tuple(waves.values())
