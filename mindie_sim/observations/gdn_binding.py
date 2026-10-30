"""Bind reviewed GDN historical arguments to standalone call declarations.

[Role] Interpret producer argument semantics without reading measured durations.
[Invariants] Unsupported wrappers retain concrete semantic mismatches. Source
  or binary uncertainty does not prohibit analytical simulation.
[Timeouts & Locks] One condition at a time; no locks, subprocesses or device work.
[Side-Effect Policy] Returns declarations only; never relabels measured boundaries.
"""
from mindie_sim.prediction.call_gdn import make_call


_PRODUCER_SHA256 = "1662a58957250fed9043593b62a9c5b558783850c1016f66367cb5737cb43063"
_IMPLEMENTATIONS = {
    "gating": "vllm_ascend.ops.triton.fused_gdn_gating.fused_gdn_gating_patch",
    "conv_prefill": "fla_npu.ops.ascendc.causal_conv1d_fn",
    "scan_prefill": "fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd",
}
_CONTRACTS = {
    "gating": {"a_dtype": "fp32", "b_dtype": "fp32", "a_log_dtype": "fp32",
               "dt_bias_dtype": "fp32"},
    "conv_prefill": {"activation": "silu", "bias": True, "initial_state": False,
                     "input_dtype": "bf16", "sequence_lengths": "batch equal-length sequences, tokens per sequence",
                     "state_cache_rows": "batch+1", "weight_layout": "[kernel,width]"},
    "scan_prefill": {"chunk_size": 64, "input_dtype": "bf16", "gate_dtype": "fp32",
                     "beta_dtype": "bf16", "state_dtype": "fp32",
                     "state_v_first": True, "output_final_state": True,
                     "use_qk_l2norm_in_kernel": True, "layout": "BSND",
                     "use_exp2": True, "disable_recompute": True,
                     "allow_neg_eigval": False},
}
_UNSUPPORTED = {
    "conv_update": [
        "native causal convolution runMode=1 with max_query_len=1 and caller-provided out is not implemented",
        "the prefill FN planner cannot substitute for update state movement",
    ],
    "scan_decode": [
        "complete callback includes q and k l2norm calls",
        "actual_seq_lengths is constructed and copied to the device inside every timed callback",
        "the native recurrence and its stable wrapper are not represented by the analytical serving recurrence alone",
    ],
}


def bind_observation(observation):
    """Return a call spec for compatible arguments, never a timing prediction."""
    identity = observation["identity"]
    condition, contract = identity["conditions"], identity["call_contract"]
    role = condition["role"]
    mismatches = list(_UNSUPPORTED.get(role, ()))
    if identity.get("producer_sha256") != _PRODUCER_SHA256:
        mismatches.append("producer source differs from the reviewed callback argument semantics")
    if role not in _IMPLEMENTATIONS and not mismatches:
        mismatches.append(f"unrecognized historical role:{role}")
    if role in _IMPLEMENTATIONS:
        if identity["implementation"] != _IMPLEMENTATIONS[role]:
            mismatches.append("implementation does not select the reviewed public callable")
        if condition.get("impl") not in (None, _IMPLEMENTATIONS[role]):
            mismatches.append("recorded condition names a different public callable")
        for field, expected in _CONTRACTS[role].items():
            if contract.get(field) != expected:
                mismatches.append(f"call_argument:{field}:expected={expected!r}:actual={contract.get(field)!r}")
        if role in ("gating", "conv_prefill"):
            dtype = "a fp32 / b fp32" if role == "gating" else "bf16"
            if condition.get("dtype") != dtype:
                mismatches.append(f"recorded_dtype:{condition.get('dtype')!r}:expected={dtype!r}")
        if role == "conv_prefill" and condition.get("layout") != "x[T_total,width] varlen":
            mismatches.append("recorded convolution layout is not the producer's contiguous varlen input")
        if role == "scan_prefill":
            batch, tokens = condition.get("batch"), condition.get("tokens")
            heads, keys = condition.get("v_heads"), condition.get("k_heads")
            dk, dv = condition.get("k_dim"), condition.get("v_dim")
            chunk = contract["chunk_size"]
            if condition.get("chunk") != chunk:
                mismatches.append(f"call_argument:chunk:expected={chunk!r}:actual={condition.get('chunk')!r}")
            if condition.get("phase") != "prefill":
                mismatches.append(f"recorded_phase:{condition.get('phase')!r}:expected='prefill'")
            if condition.get("dtype") not in (None, contract["input_dtype"]):
                mismatches.append(f"recorded_dtype:{condition.get('dtype')!r}:expected={contract['input_dtype']!r}")
            if condition.get("layout") not in (None, contract["layout"]):
                mismatches.append(f"recorded_layout:{condition.get('layout')!r}:expected={contract['layout']!r}")
            for name, value in (("batch", batch), ("tokens", tokens), ("k_heads", keys),
                                ("v_heads", heads), ("k_dim", dk), ("v_dim", dv)):
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    mismatches.append(f"recorded_geometry:{name}:{value!r}")
            if not mismatches:
                total = batch * tokens
                chunks = (total + chunk - 1) // chunk
                state_bytes = batch * heads * dv * dk * 4
                if condition.get("chunks") != chunks:
                    mismatches.append(f"derived_chunks:{condition.get('chunks')!r}:expected={chunks}")
                if condition.get("state_bytes") != state_bytes:
                    mismatches.append(f"derived_state_bytes:{condition.get('state_bytes')!r}:expected={state_bytes}")
    result = {
        "status": "semantic_mismatch" if mismatches else "bound",
        "role": role, "observation_id": observation["observation_id"],
        "observation_boundary": observation["boundary"], "mismatches": mismatches,
        "call": None,
        "comparison_limit": "Device work is not relabeled as an NPU-event interval; host/event gaps remain separate.",
    }
    if mismatches:
        return result
    params = {"batch": condition["batch"], "tokens": condition["tokens"],
              "phase": condition["phase"]}
    if role == "gating":
        params.update(rows=params["batch"] * params["tokens"], heads=condition["heads"])
    elif role == "conv_prefill":
        params.update(conv_width=condition["conv_width"], kernel=condition["kernel"])
    else:
        contract = _CONTRACTS[role]
        params.update(k_heads=condition["k_heads"], v_heads=condition["v_heads"],
                      k_dim=condition["k_dim"], v_dim=condition["v_dim"],
                      chunk=contract["chunk_size"], input_dtype=contract["input_dtype"],
                      gate_dtype=contract["gate_dtype"], beta_dtype=contract["beta_dtype"],
                      state_dtype=contract["state_dtype"],
                      state_v_first=contract["state_v_first"],
                      output_final_state=contract["output_final_state"],
                      use_qk_l2norm_in_kernel=contract["use_qk_l2norm_in_kernel"],
                      layout=contract["layout"], use_exp2=contract["use_exp2"],
                      disable_recompute=contract["disable_recompute"],
                      allow_neg_eigval=contract["allow_neg_eigval"])
    result["call"] = make_call(role, {**params, **_CONTRACTS[role]})
    return result
