"""Reusable native arch35 GDN prefill composite.

[Role] Build caller materializations, Prepare/H/O and requested exports for an
  already prepared varlen prefill batch.
[Invariants] Q/K/V are BF16 BSND; gates are FP32 BSN; beta is BF16 BSN and is
  materialized to FP32 before its BNS view. State is FP32 V,K and is read from
  the caller's initial tensor and written once to the caller's final tensor.
  Prepare/H/O are the shared source planners. QKV packing and cache
  gather/scatter are caller concerns and never appear here.
[Timeouts & Locks] O(sequence count) metadata; lazy kernel work; no locks.
[Side-Effect Policy] Appends operators to the caller's Builder; no device I/O.
"""
from dataclasses import dataclass, replace

from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from .mechanism import PlannedRecipe
from .operators import gdn_output, gdn_prepare, gdn_state


def recipes():
    """Recipes for the three native kernels; callers add their own extras."""
    params = {"lengths": (1,), "heads": 1, "key_heads": 1, "dk": 128, "dv": 128, "chunk": 64}
    return {module.make_op(params).recipe: PlannedRecipe(module.build)
            for module in (gdn_prepare, gdn_state, gdn_output)}


@dataclass(frozen=True)
class PrefillComposite:
    """Declared native prefill call plus the caller-side materializations."""

    prepare: str
    state: str
    output: str
    ops: tuple[OpSpec, ...]


def _external_copy(builder, value, name, *, reason, dtype=None, source,
                   source_stream="current"):
    """Materialize a tensor whose payload was prepared outside the timed call."""
    dtype = dtype or value.dtype
    layout = value.layout
    if layout.is_contiguous and dtype == value.dtype:
        return replace(value, producer="")
    read = layout.elements * dtype_bytes(value.dtype)
    write = layout.elements * dtype_bytes(dtype)
    producer = builder.work(
        name, (), layout.shape, read=read, write=write,
        vector=layout.elements if dtype != value.dtype else 0,
        family="tensor_copy", dtype=dtype,
        metadata=(("source_layout", layout.to_dict()),
                  ("target_layout", TensorLayout.contiguous(layout.shape).to_dict()),
                  ("input_format", value.dtype), ("output_format", dtype),
                  ("source_output", value.output),
                  ("source_input", source), ("source_stream", source_stream),
                  ("materialization_reason", reason)),
    )
    return TensorValue(producer, dtype, TensorLayout.contiguous(layout.shape))


def _materialize(builder, tensors, value, name, *, reason, dtype=None, source=None):
    if source is None:
        return tensors.contiguous(value, name, reason=reason, dtype=dtype)
    return _external_copy(builder, value, name, reason=reason,
                          dtype=dtype, source=source)


def prefill(*, builder, tensors, q, k, v, gate, beta, lengths, heads,
            key_heads, dk, dv, chunk, initial_state_dependency=None,
            external=None, native_requests=None):
    """Append one native prefill call to ``builder``.

    ``external`` maps logical input names to caller-owned source labels. Those
    inputs have no producer operator in this call, so their materializations
    carry an explicit source/stream edge in metadata instead of a fabricated
    dependency. ``initial_state_dependency`` is the serving gather op when the
    caller owns cache movement; it is absent for a standalone initial tensor.
    """
    b = builder
    external = dict(external or {})
    lengths = tuple(lengths)
    tokens = sum(lengths)
    dtype = b.model.activation_dtype
    if dtype != "bf16" or (chunk, dk, dv) != (64, 128, 128):
        raise ValueError("gdn_prefill_arch35_requires_bf16_chunk64_k128_v128")
    params = {"lengths": lengths, "heads": heads, "key_heads": key_heads,
              "dk": dk, "dv": dv, "chunk": chunk, "dtype": dtype,
              "gate_dtype": "fp32", "use_exp2": True}

    # beta is a BF16 producer value in the historical callback; the native
    # wrapper promotes it to FP32 before the BSN->BNS caller conversion.
    beta = _materialize(
        b, tensors, beta, "beta_fp32", reason="fla_common_gate_precision",
        dtype="fp32", source=external.get("beta"))
    gate = _materialize(
        b, tensors, gate.permute((0, 2, 1)), "g_head_first",
        reason="fla_BSN_to_BNS", source=external.get("g"))
    beta = _materialize(
        b, tensors, beta.permute((0, 2, 1)), "beta_head_first",
        reason="fla_BSN_to_BNS")
    inputs = [
        _materialize(b, tensors, value.permute((0, 2, 1, 3)),
                     name + "_head_first", reason="fla_BSND_to_BNSD",
                     source=external.get(name))
        for name, value in zip(("q", "k", "v"), (q, k, v))]

    def bind(module, name, dependencies, flags):
        op = module.make_op({**params, **flags})
        common = {**b.common, "phase": "prefill"}
        if native_requests is not None:
            common["requests"] = tuple(native_requests)
        return b.add(replace(op, op_id=b.prefix + "." + name,
                             deps=tuple(dep for dep in dependencies if dep), **common))

    prepare = bind(gdn_prepare, "gdn_prepare",
                   (*[value.producer for value in inputs], gate.producer, beta.producer),
                   {"use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": False,
                    "use_beta_sigmoid_in_kernel": False, "output_a": True,
                    "output_beta": False})
    state_dependencies = ((prepare, initial_state_dependency)
                          if initial_state_dependency else (prepare,))
    state = bind(gdn_state, "gdn_state", state_dependencies,
                 {"use_initial_state": True, "output_final_state": True,
                  "state_v_first": True})
    output = bind(gdn_output, "gdn_output", (prepare, state), {"state_v_first": True})
    # Internal allocations and the wrapper's requested output buffers differ.
    # TransposeContiguous may be a view, but ViewCopy still writes that buffer.
    def export(value, name):
        destination = TensorValue("", value.dtype, TensorLayout.contiguous(value.layout.shape),
                                  output="caller:" + name)
        tensors.copy_into(value, destination, name + "_caller_copy")

    for name, count, width in (("q_hat", key_heads, dk), ("k_hat", key_heads, dk)):
        value = tensors.result(prepare, (1, count, tokens, width), output=name)
        value = _materialize(b, tensors, value.permute((0, 2, 1, 3)), name + "_export",
                             reason="fla_requested_normalized_output")
        export(value, name)
    for name in ("q_rstd", "k_rstd"):
        export(tensors.result(prepare, (1, key_heads, tokens), "fp32", output=name), name)
    export(tensors.result(prepare, (1, heads, tokens, chunk), output="A"), "A")
    value = tensors.result(prepare, (1, heads, tokens), "fp32", output="g_cumsum")
    value = _materialize(b, tensors, value.permute((0, 2, 1)), "g_cumsum_export",
                         reason="fla_requested_cumsum_output")
    export(value, "g_cumsum")
    return PrefillComposite(prepare, state, output, tuple(b.ops))
