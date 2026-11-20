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


_NIBBLE_FORMAT_BITS = 4
_BYTE_BITS = 8
_HALFWORD_FORMAT_BITS = 16
_WORD_FORMAT_BITS = 32

_DIRECT_ROUTE_ALIGNMENT = {
    "gm": 1, "l1": 32, "l0a": 512, "l0b": 512, "l0c": 64, "ub": 32,
}


def _prov(kind, architecture, basis, source):
    return {"kind": kind, "architecture": architecture, "basis": basis,
            "source": source}


_2201_SOURCE = "registered A2/A3 profile npu_arch=2201 and design 4.7"
_3510_SOURCE = ("data/observations/sources/arch3510-physical-services/"
                "platform-units/npu_arch_3510.md")
_COMMON_FORMATS = ("bf16", "fp16", "fp32")


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


def _casts(architecture, source_ref):
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
            ("hf32", "fp32", True), ("fp32", "hf32", False),
        )
    }


def _path(name, source, target, roles, *, unit="", executor="", reach=(),
          buffer="", sync="pipe", alignment=1, representation_provenance=None):
    item = {"source": source, "target": target, "roles": list(roles),
            "sync": sync, "alignment_bytes": alignment}
    if unit:
        item["unit"] = unit
    if executor:
        item["executor"] = executor
    if reach:
        item["reach"] = list(reach)
    if buffer:
        item["buffer"] = buffer
    item["provenance"] = representation_provenance
    return item


def _paths_2201():
    source = _prov("directional_data_path", "npu_arch_2201",
                   "registration of the legacy direct-route set", _2201_SOURCE)
    paths = {}
    for name, src, dst, roles, unit, executor, reach, buffer, align in (
        ("gm->l1", "gm", "l1", ("storage", "transfer"),
         "mte2_aic", "aic", ("aic",), "l1", 1),
        ("gm->ub", "gm", "ub", ("storage", "transfer"),
         "mte2_aiv", "aiv", ("aiv",), "ub", 1),
        ("ub->gm", "ub", "gm", ("storage", "transfer"),
         "mte3", "aiv", ("aiv",), "", 1),
        ("l1->l0a", "l1", "l0a", ("operand",), "mte1", "aic", ("aic",), "l0a", 1),
        ("l1->l0b", "l1", "l0b", ("operand",), "mte1", "aic", ("aic",), "l0b", 1),
        ("l0c->gm", "l0c", "gm", ("accumulator",),
         "fixpipe", "aic", ("aic",), "", 1),
        ("l0c->l1", "l0c", "l1", ("accumulator",),
         "fixpipe", "aic", ("aic",), "l1", 1),
        # 2201 expresses C/V exchange through GM; these direct routes exist.
        ("gm->l0a", "gm", "l0a", ("operand",),
         "mte2_aic", "aic", ("aic",), "l0a", 1),
        ("gm->l0b", "gm", "l0b", ("operand",),
         "mte2_aic", "aic", ("aic",), "l0b", 1),
        # Existence is declared; this legacy route has no registered service.
        ("l1->gm", "l1", "gm", ("storage", "transfer"), "", "", (), "", 1),
    ):
        paths[name] = _path(name, src, dst, roles, unit=unit, executor=executor,
                            reach=reach, buffer=buffer, alignment=align,
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
            name, src, dst, roles, unit=unit, executor=executor, reach=reach,
            buffer=buffer, sync=sync,
            alignment=max(_DIRECT_ROUTE_ALIGNMENT[src],
                          _DIRECT_ROUTE_ALIGNMENT[dst]),
            representation_provenance=source)
    paths["l1->ub"]["provenance"] = _prov(
        "directional_data_path", "npu_arch_3510",
        "sourced L1->UB PIPE_MTE1 128 B/cycle route; profile binds shared "
        "MTE1 mode mte1_l1_ub; final owner/sync/capacity remains T07", _3510_SOURCE)
    return paths


CAPABILITY_PRESETS: dict[str, dict] = {
    "npu_arch_2201": {
        "representations": _storage_representations(
            "npu_arch_2201", _2201_SOURCE, fp8_operand=False,
            int4_operand=True),
        "native_compute": _native_machine_modes(
            "npu_arch_2201", _2201_SOURCE, fp8=False, int4_operand=True),
        "casts": _casts("npu_arch_2201", _2201_SOURCE),
        "paths": _paths_2201(),
    },
    "npu_arch_3510": {
        "representations": _storage_representations(
            "npu_arch_3510", _3510_SOURCE, fp8_operand=True,
            int4_operand=False),
        "native_compute": _native_machine_modes(
            "npu_arch_3510", _3510_SOURCE, fp8=True, int4_operand=False),
        "casts": _casts("npu_arch_3510", _3510_SOURCE),
        "paths": _paths_3510(),
    },
}
