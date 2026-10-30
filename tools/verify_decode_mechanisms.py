"""Installed frontend controls for batched decode and expert transport.

[Role] Extend wheel verification through its isolated public CLI runner.
[Invariants] No package imports from the checkout. Every selected mechanism
  must execute, retain useful work and release its programs and allocations.
[Timeouts & Locks] Each CLI call inherits the verifier's 60-second budget.
[Side-Effect Policy] Writes temporary case inputs and runs local simulation.
"""
from copy import deepcopy
import json
import math


def verify_decode_mechanisms(run, root, work):
    def graph(raw, name, *, batch=2, query=1):
        path = work / (name + ".json")
        path.write_text(json.dumps(raw) + "\n")
        result = run(["graph", path, "--scope", "body", "--phase", "decode",
                      "--tokens", query, "--sequences", batch, "--context", 128], name)
        assert result["duration_ms"] > 0
        assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
        assert all(abs(value) < 1e-6 for value in result["execution"]["live_allocated"].values())
        return result

    base = json.loads((root / "examples/source-families/kda-state-control.json").read_text())
    base["machine"] = {"backend": "ascend", "sku": "Ascend910_9362", "num_devices": 1}
    base["model"]["layer_plans"][0].update(recurrent_state_dtype="fp32", gate_rank=128)
    for architecture, sku in (("arch22", "Ascend910_9362"), ("arch35", "Ascend950DT_9582")):
        raw = deepcopy(base)
        raw["machine"]["sku"] = sku
        implementation = "vllm_ascend_recurrent_kda_" + architecture
        selection = ({"kind": "exact", "index_layout": "speculative",
                      "index_row_stride": 4, "prefix": [0, 4, 8],
                      "state_indices": [[3, 1, 5, 0], [7, 2, 4, 6]],
                      "accepted_tokens": [2, 4], "state_capacity": 8}
                     if architecture == "arch22" else {
                         "kind": "symbolic_valid_selection", "index_layout": "speculative",
                         "index_row_stride": 4, "validity": {key: True for key in
                            ("prefix", "state_indices", "accepted_tokens", "independent_sequences")}})
        raw["runtime"]["kernels"]["lowerings"]["kda"] = {
            "implementation": implementation, "options": {"state_selection": selection}}
        report = graph(raw, "graph-source-kda-decode-" + architecture, query=4)
        kernels = [row for row in report["timeline"] if row["recipe"] == implementation]
        assert len(kernels) == 1
        assert kernels[0]["source_boundary"] == "kernel"
        assert kernels[0]["program"] == "OperatorPlan"
        core, = [row["operator"] for row in report["isolated"]["operators"]
                 if row["operator"]["recipe"] == implementation]
        assert core["metadata"]["kda_recurrent_params"]["has_accepted_tokens"]
        addresses = core["metadata"]["state_selection"]
        if architecture == "arch22":
            assert addresses["initial_slots"] == [1, 6]
            assert addresses["checkpoint_slots"] == selection["state_indices"]
        else:
            assert "initial_slots" not in addresses and "checkpoint_slots" not in addresses

    raw = json.loads((root / "examples/source-families/indexed-mla-a3-control.json").read_text())
    raw["runtime"]["kernels"].pop("lowerings")
    raw["runtime"]["kernels"]["programs"]["indexed_attention"] = {
        "kind": "indexed_attention_pipeline", "scratch_policy": "offchip", "operand_modes": True}
    report = graph(raw, "graph-indexed-attention-batch", batch=8)
    kernels = [row for row in report["isolated"]["operators"]
               if row["operator"]["metadata"].get("program_kind") == "indexed_attention"]
    assert sum(row["occurrences"] for row in kernels) == 1
    assert kernels[0]["operator"]["local_shape"][0] == 8

    packed = json.loads((root /
        "examples/source-families/packed-int8-indexed-c32.json").read_text())
    # The captured packed caller has latent512/rope0; the BF16 control above
    # has rope64. Preserve the complete tensor contract, including its first
    # indexed layer at position 3, instead of replacing only the cache dtype.
    packed["model"]["layers"] = 4
    report = graph(packed, "graph-packed-int8-indexed-batch", batch=3)
    kernels = [row for row in report["isolated"]["operators"]
               if row["operator"]["metadata"].get("program_kind") == "indexed_attention"]
    assert sum(row["occurrences"] for row in kernels) == 1
    assert kernels[0]["operator"]["local_shape"][0] == 3

    raw = json.loads((root / "examples/quickstart.json").read_text())
    raw["model"].update(layers=1, ffn="moe", num_experts=8, top_k=2,
                        moe_intermediate_size=256, num_shared_experts=1)
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 4}
    raw["runtime"].update(tp=2, ep=4, dp=2, graph_mode="none")
    raw["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "steady_matmul"}},
                                  "recipes": {name: {"kind": "direct_peer_collective",
                                                     "max_inflight_per_rank": 2}
                                              for name in ("expert_dispatch", "expert_combine")}}
    report = graph(raw, "graph-expected-expert-banks-direct-peer-cohort")
    assert len(report["replicas"]) == 2
    assert any(row["program"] == "DirectPeerCollective" for row in report["timeline"])
    assert any(row["execution_abstraction"] == "expected_expert_bank_grid"
               for row in report["parameters"]["matrix_mappings"])
    transports = [row for row in report["isolated"]["operators"]
                  if row["program"] == "DirectPeerCollective"]
    assert transports and all(row["communication"]["setup_ms"] == 0 for row in transports)
    assert sum(row["communication"]["wire_bytes"] for row in transports) > 0

    bf16 = deepcopy(raw)
    bf16["runtime"]["routing"] = {"policy": "balanced"}
    bf16["runtime"]["kernels"]["families"] = {"grouped_gemm": {"kind": "ascend_bf16_grouped"}}
    report = graph(bf16, "graph-source-bf16-grouped-cohort")
    banks = [op["operator"]["metadata"]["program_execution"]["source_mapping"]
             for op in report["isolated"]["operators"]
             if "source_mapping" in op["operator"]["metadata"].get("program_execution", {})]
    assert banks and all(sum(row["group_rows"]) > 0 for row in banks)
    assert all(row["base_n"] == 256 for row in banks)

    mxfp8 = deepcopy(bf16)
    mxfp8["model"]["quantization"] = "w8a8_mxfp8"
    mxfp8["runtime"]["kernels"].pop("families")
    mxfp8["runtime"]["kernels"]["recipes"]["balanced_grouped_down"] = {
        "kind": "ascend_mxfp8_grouped_nz"}
    report = graph(mxfp8, "graph-source-mxfp8-grouped-cohort")
    matrices = [row["operator"] for row in report["isolated"]["operators"]
                if row["operator"]["recipe"] == "balanced_grouped_down"]
    assert matrices and all(op["metadata"]["compute_format"] == "fp8_e4m3"
                            for op in matrices)
    assert all(op["metadata"]["program_execution"]["source_mapping"]["group_rows"]
               for op in matrices)
    assert any(value > 0 for key, value in report["execution"]["serviced"].items()
               if key.startswith("fixpipe@"))

    raw["machine"]["num_devices"] = 2
    raw["runtime"].update(tp=1, ep=2, dp=2)
    raw["runtime"]["kernels"].pop("recipes")
    raw["runtime"]["kernels"]["native_moe_protocol"] = {
        "kind": "arch35_fullmesh", "max_inflight_per_rank": 8}
    report = graph(raw, "graph-native-mc2-complete-caller")
    phases = [row["operator"] for row in report["isolated"]["operators"]
              if row["operator"]["recipe"] == "arch35_fullmesh_resource_phase"]
    assert phases and all(op["source_boundary"] == "kernel" for op in phases)
    assert {dict(op["metadata"])["mc2_phase"]["kind"] for op in phases} == {"dispatch", "combine"}
    assert any(row["program"] == "DirectPeerCollective" for row in report["timeline"])

    raw["model"] = {"hf_config": {
        "model_type": "llama", "num_hidden_layers": 2, "hidden_size": 32,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 64, "vocab_size": 128},
        "quantization": "w8a8", "layers": 1,
        "metadata": {"model_coverage": "partial", "unmodeled_features": ["declared_custom_layout"]}}
    # This independent dense control has no expert cohort or native MC2 caller.
    raw["machine"]["num_devices"] = 1
    raw["runtime"].update(tp=1, ep=1, dp=1)
    raw["runtime"]["kernels"].pop("native_moe_protocol")
    report = graph(raw, "graph-hf-wrapper-explicit-int8")
    operators = [row["operator"] for row in report["isolated"]["operators"]]
    assert any(op["metadata"].get("weight_format") == "int8" for op in operators)
    assert any(op["family"] == "quantize" for op in operators)
    assert report["model_coverage"] == "partial"
    assert report["unmodeled_features"] == ["declared_custom_layout"]

    raw = json.loads((root / "examples/source-families/v41-mqsmla-source-control.json").read_text())
    # Three original layers retain both window-only and compressed-cache
    # attention, Engram and a complete EP cohort without a full-model canary.
    raw["model"]["layers"] = 3
    raw["runtime"]["kernels"].setdefault("families", {})["grouped_gemm"] = {
        "kind": "ascend_mxa8w4_nznk"}
    report = graph(raw, "graph-mixed-quant-cache-and-grouped-source", batch=1)
    assert len(report["replicas"]) == 8
    operators = [row["operator"] for row in report["isolated"]["operators"]]
    attention = [op for op in operators if op["recipe"] == "cannbot_mixed_quant_sparse_flash_mla"]
    assert attention and all(op["source_boundary"] == "kernel" for op in attention)
    assert {op["metadata"]["mqsmla_geometry"]["ratio"] for op in attention} == {0, 2}
    assert all(op["metadata"]["attention_tensor_view"]["compressed_cache_row_bytes"] == 320
               for op in attention)
    grouped = [op["metadata"]["program_execution"]["mxa8w4"] for op in operators
               if "mxa8w4" in op["metadata"].get("program_execution", {})]
    assert grouped and all(row["valid_rows"] == sum(row["group_rows"]) > 0 for row in grouped)
    assert any(op["recipe"] == "compressed_store" for op in operators)

    prior = report
    raw["runtime"]["kernels"]["families"]["grouped_gemm"]["producer_mode"] = "dynamic_buffer4"
    report = graph(raw, "graph-mixed-quant-dynamic-producer", batch=1)
    grouped = [row["operator"]["metadata"]["program_execution"]["mxa8w4"]
               for row in report["isolated"]["operators"]
               if "mxa8w4" in row["operator"]["metadata"].get("program_execution", {})]
    assert grouped and all(row["producer_mode"] == "dynamic_buffer4" for row in grouped)
    for name in ("work", "serviced"):
        assert report["execution"][name].keys() == prior["execution"][name].keys()
        assert all(math.isclose(value, prior["execution"][name][key], rel_tol=1e-10, abs_tol=1e-8)
                   for key, value in report["execution"][name].items())
    raw["runtime"]["kernels"]["lowerings"]["v41_rotary"] = "v41_partial_rotary"
    report = graph(raw, "graph-shared-kv-rotary-caller", batch=1)
    rotations = [row["operator"] for row in report["isolated"]["operators"]
                 if row["operator"]["recipe"] in ("q_rotary", "kv_rotary", "output_inverse_rotary")]
    assert rotations and all(op["source_boundary"] == "kernel" for op in rotations)
    assert any(op["metadata"]["native_tensor_view"]["inverse"] for op in rotations)
    assert not any(row["recipe"] == "rotary" for row in report["timeline"])

    raw = json.loads((root / "examples/cases/high-concurrency-20261030/"
        "glm-compatible-variants/glm53flash-a5-tp8-c3--attention-plus-down.json").read_text())
    raw["model"]["layers"] = 4
    role, fused = "grouped_gate_up_activation_quant", "ascend_mxfp8_grouped_swiglu_quant"
    raw["runtime"]["kernels"].setdefault("lowerings", {})["native_fusion"] = {
        "implementation": "declared_native_fusion", "options": {
            "roles": [role], "program_profiles": {role: fused},
            "program_options": {role: {"l1_k": 512}}}}
    report = graph(raw, "graph-rope0-and-mxfp8-down-body", batch=3, query=4)
    programs = [row["operator"]["metadata"].get("program_execution", {})
                for row in report["isolated"]["operators"]]
    assert any(p.get("geometry_extension") == {"latent": 512, "rope": 0} for p in programs)
    assert any(p.get("source_program", {}).get("branch", "").startswith("MX NZ FP8")
               for p in programs)
    assert any(p.get("implementation") == fused for p in programs)

    raw = json.loads((root / "examples/cases/high-concurrency-20261030/"
        "kimi-source/a5_kimi_tp8_e16_bf16_c1_p4k_source.json").read_text())
    raw["model"]["layers"] = 4
    raw["runtime"]["kernels"]["lowerings"]["causal_conv_update"] = "ascend_causal_conv1d_update"
    report = graph(raw, "graph-kimi-source-body-contract", batch=1)
    operators = [row["operator"] for row in report["isolated"]["operators"]]
    assert any(op["recipe"] == "vllm_ascend_recurrent_kda_arch35" for op in operators)
    convolutions = [op for op in operators if op["recipe"] == "ascend_causal_conv1d_update"]
    assert convolutions and all(op["source_boundary"] == "kernel" for op in convolutions)
    for op in convolutions:
        meta = op["metadata"]
        assert meta["state_write_bytes"] == (meta["kernel_width"] - 1) * op["local_shape"][1] * 2
        assert op["mte3_bytes"] == meta["state_write_bytes"] + op["local_shape"][1] * 2
    assert any(op["recipe"] == "arch35_fullmesh_resource_phase" for op in operators)
    assert any(op["metadata"].get("implementation") == "vllm_ascend_mla_head_padding"
               for op in operators)
    projections = [op for op in operators if op["recipe"] in ("linear_q", "q_a")]
    assert projections and all(op["local_shape"][0] == 1 for op in projections)
    pads = [op for op in operators if op["recipe"] == "sequence_parallel_reduce_scatter_pad"]
    assert pads and all(op["metadata"]["padding_rows"] == 7 for op in pads)
    raw["runtime"]["kernels"]["lowerings"]["paged_mla_attention"] = {
        "implementation": "ascend_paged_mla_decode_arch35", "options": {"kv_splits": 2}}
    report = graph(raw, "graph-dense-paged-mla-caller", batch=2)
    kernels = [row for row in report["isolated"]["operators"]
               if row["operator"]["recipe"] == "ascend_paged_mla_decode_arch35"]
    assert sum(row["occurrences"] for row in kernels) == 1
    assert kernels[0]["operator"]["metadata"]["program_execution"]["physical_segments"] == 4
    assert not any(row["recipe"].startswith("paged_mla_query_pad") for row in report["timeline"])
