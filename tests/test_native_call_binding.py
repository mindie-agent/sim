"""BIND1 actual-invocation identity over real emitted operators.

Evidence sources:
* I-r64/major CANN 9.2.1 ``quant_batch_matmul_v3`` (arch35) and
  ``kv_quant_sparse_flash_attention`` kernel files, exact sha256 from its
  manifest.
* COMPHY7 signed static HCCL ``all_gather`` packet: two ranks, BF16 N=5120,
  ``torch.distributed.all_gather_into_tensor``; no ACLNN wrapper.
* A real GLM53 W4A8 normal emission for the quantized matrix shape.
"""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.native_binding import (
    bind_native_call, coverage_from_records, declare_native_call,
    member_physical_view,
    join_profiler_rows, native_call_coverage, validate_invocation,
)
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure import export_op_structure, export_structure

QBMM_CPP = ("/usr/local/Ascend/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/impl/"
            "ops_nn/ascendc/quant_batch_matmul_v3/arch35/quant_batch_matmul_v3.cpp")
QBMM_SHA = "13daf996890d70e7b1dc9275ab3a464043860e67d6df730e28742393c023ec88"
KVSPARSE = (
    ("kv_quant_sparse_flash_attention_kernel_mla_arch35.h",
     "bf44789d098853c7101fc76795cfadfd571efe60af5909e1cee6ade129c158d1"),
    ("kv_quant_sparse_flash_attention_kvcache.h",
     "88de779e838630800735a11618546e30ddf0fac12d357cc45fce6d6eed0be0d5"),
    ("kv_quant_sparse_flash_attention_service_cube_mla_arch35.h",
     "225931480b6ce5d275415d32681aabf4517617769a8a80077ca4454bb9e91a9a"),
    ("kv_quant_sparse_flash_attention_service_vector_mla_arch35.h",
     "8a0c40168f4c55b9238933f354d5f0c7220a3e1e23d55e94dae42880023c2b1b"),
)


def _real_quantized_matmul():
    raw = json.load(open("examples/cases/glm53-a3-w4a8.json"))
    raw["model"]["layers"] = 1
    ops = target_graphs(parse_case(raw), phase="prefill", tokens=64, context=0,
                        sequences=1, scope="body")[0]
    return next(op for op in ops if op.recipe == "q_a")


def _qbmm_declared(op):
    return declare_native_call(
        op,
        api={"namespace": "cann_kernel", "entry": "QuantBatchMatmulV3"},
        source={"repository": "cann", "path": QBMM_CPP, "sha256": QBMM_SHA,
                "symbol": "QuantBatchMatmulV3BaseKernel", "revision": "cann-9.2.1",
                "state": "task_artifact"},
        kernels=[{"name": "QuantBatchMatmulV3", "role": "cube",
                  "source_path": QBMM_CPP, "source_sha256": QBMM_SHA,
                  "evidence": "source_declared"}],
        unresolved=["aclnn_public_symbol_not_in_evidence"],
    )


def test_real_quantized_matmul_binds_r64_qbmm_and_forwards_plan_work():
    op = _real_quantized_matmul()
    declared = _qbmm_declared(op)
    plan = OperatorPlan(Work(matrix={"int8": 1.5e9}, resources={"hbm": 4096}),
                        lambda context: None)
    prepared = replace(declared, flops=1.5e9)
    binding = bind_native_call(declared, plan=plan, prepared=prepared)
    assert binding.api["namespace"] == "cann_kernel"
    assert binding.api["entry"] == "QuantBatchMatmulV3"
    assert binding.api["aclnn_symbol"] is None
    assert binding.source["sha256"] == QBMM_SHA
    assert binding.typed["dtype"] == "int8"
    assert binding.typed["shapes"]["local"] == list(op.local_shape)
    assert binding.evidence["entry"] == "source_declared"
    assert binding.execution_core is None  # no proved class; names are not a core rule
    assert "aclnn_public_symbol_not_in_evidence" in binding.unresolved
    # Existing plan work is forwarded unchanged, never re-priced.
    assert binding.physical["matrix"] == {"int8": 1.5e9}
    assert binding.physical["resources"] == {"hbm": 4096}
    assert binding.physical["charged"] == "once_at_timing_owner"
    # Prepare-only typed invocation is validated against the same signature.
    validate_invocation(binding, {"dtype": "int8", "layout": "int8_per_channel",
                                  "shapes": binding.typed["shapes"]})
    with pytest.raises(ModelingError, match="shape_mismatch"):
        validate_invocation(binding, {"dtype": "int8", "layout": "int8_per_channel",
                                      "shapes": {**binding.typed["shapes"],
                                                 "local": [1, 2, 3]}})


def test_structure_export_carries_same_identity_and_counts_exact_denominator():
    op = _qbmm_declared(_real_quantized_matmul())
    structure = export_op_structure({0: (op,)}, model={},
                                    execution={"phase": "prefill"})
    assert structure["nodes"][0].get("native_call") is None
    assert structure["native_calls"][0]["call_id"] == bind_native_call(op).call_id
    coverage = coverage_from_records(structure["native_calls"])
    assert coverage["denominator"]["emitted_operators"] == 1
    assert coverage["denominator"]["recorded_bindings"] == 1
    assert coverage["denominator"]["unrecorded_emitted"] == 0
    assert coverage["totals"]["bound"] == 1
    assert coverage["totals"]["unresolved"] == 0


def test_kv_sparse_is_one_mix_launch_with_four_inline_source_stages():
    """I-r64: the four .h files are stage classes of one native launch."""
    op = OpSpec("sfa", "attention", "prefill", 0, 0, (), (128, 64, 576),
                (128, 64, 576), (128, 64, 576), (128, 64, 576), dtype="bf16",
                layout="PA_ND", recipe="kv_quant_sparse_flash_attention",
                source_boundary="kernel")
    stages = [f"op_kernel/arch35/{name}" for name, _ in KVSPARSE]
    declared = declare_native_call(
        op,
        api={"namespace": "torch",
             "entry": "torch_npu.npu_kv_quant_sparse_flash_attention"},
        source={"repository": "cann", "path": stages[0],
                "sha256": KVSPARSE[0][1], "symbol": "KvQuantSparseFlashAttentionMla",
                "revision": "cann-9.2.1", "state": "task_artifact"},
        kernels=[{"name": "KvQuantSparseFlashAttention", "role": "mix",
                  "source_path": stages[0], "source_sha256": KVSPARSE[0][1],
                  "source_stages": stages, "evidence": "source_declared"}],
        unresolved=["aclnn_symbol_unresolved", "triton_or_native_dispatch_unobserved"],
    )
    binding = bind_native_call(declared)
    assert len(binding.kernels) == 1
    assert binding.kernels[0]["name"] == "KvQuantSparseFlashAttention"
    assert binding.kernels[0]["role"] == "mix"
    assert len(binding.kernels[0]["source_stages"]) == 4
    assert binding.api["aclnn_symbol"] is None
    assert binding.typed["layout"] == "PA_ND"


def test_gdn_aclnn_call_expands_one_to_many_real_children():
    """R31: one aclnnChunkGatedDeltaRuleFwd call owns several device children."""
    parent = OpSpec("gdn_chunk", "gdn_prefill", "prefill", 0, 0, (),
                    (1, 2, 64, 128), (1, 2, 64, 128), (1, 2, 64, 128),
                    (1, 2, 64, 128), dtype="bf16", layout="BNSD",
                    recipe="chunk_gated_delta_rule_fwd", source_boundary="call",
                    metadata=(("source", {
                        "path": "vllm_ascend/ops/gdn.py",
                        "sha256": "5e3b0fb0d9fd71bdcf64fdd5f0e182f12041e291b0122b638613ada2f53224ed",
                        "symbol": "chunk_gated_delta_rule_fwd", "revision": "r31"}),))
    names = ("Cast", "Transpose", "Prepare", "H", "O")
    children = tuple(
        replace(parent, op_id=name.lower(), source_boundary="kernel",
                metadata=(("native_kind", name),))
        for name in names)
    recipe = PlannedRecipe(
        lambda op, geometry: OperatorPlan(Work(), lambda context: None),
        kernels=lambda op: children)
    book = RecipeBook(recipes={"chunk_gated_delta_rule_fwd": recipe})
    expanded = book.expand((parent,))
    declared = declare_native_call(
        parent,
        api={"namespace": "aclnn", "entry": "aclnnChunkGatedDeltaRuleFwd",
             "aclnn_symbol": "aclnnChunkGatedDeltaRuleFwd"},
        kernels=[{"name": name, "role": "vector", "evidence": "source_declared"}
                 for name in names],
    )
    binding = bind_native_call(declared, children=expanded)
    assert binding.api["aclnn_symbol"] == "aclnnChunkGatedDeltaRuleFwd"
    assert binding.evidence["aclnn_symbol"] == "source_declared"
    assert [kernel["name"] for kernel in binding.kernels] == list(names)
    assert list(binding.member_op_ids) == [
        child.op_id for child in expanded if child.family != "barrier"]
    assert binding.timing_owner == {"op_id": parent.op_id, "scope": "call"}
    child_plan = OperatorPlan(Work(matrix={"bf16": 1.0}), lambda context: None)
    views = [member_physical_view(binding, child_plan, child)
             for child in expanded if child.family != "barrier"]
    assert {view.call_id for view in views} == {binding.call_id}
    assert all(view.physical["basis"] == "member_view_of_shared_call_work"
               for view in views)


def test_hccl_all_gather_is_native_hccl_with_no_fabricated_aclnn_symbol():
    from mindie_sim.prediction.hccl_graph import lower

    source = OpSpec("tp_all_gather", "allgather", "decode", 0, 0, (),
                    (5120,), (5120,), (5120,), (5120,), link_bytes=10240,
                    dtype="bf16", metadata=(("group_axis", "tp"),
                                            ("collective", "allgather")))
    runtime = SimpleNamespace(
        parallel=SimpleNamespace(tp=2, pcp=1, etp=1),
        kernels={"hccl": {"allgather": {
            "expansion": "AICPU_TS", "template": "nhr", "dtype_bytes": 2}}})
    lowered = lower((source,), runtime, "bf16")
    entry = next(op for op in lowered if op.family == "hccl_boundary")
    declared = declare_native_call(
        entry,
        api={"namespace": "hccl",
             "entry": "torch.distributed.all_gather_into_tensor"},
        source={"repository": "task-evidence", "path": "COMPHY3-lab/COMPHY7_collect.py",
                "sha256": "8f4a2912fffaff1cf2d82212a25ae20596463390729fc8561bb77632f5560f5e",
                "symbol": "measured_graph_replay", "revision": "COMPHY7",
                "state": "task_artifact"},
        typed={"dtype": "bf16", "layout": "contiguous",
               "shapes": {"input": [5120], "output": [10240]},
               "attrs": {"world_size": 2, "ranks": 2}},
        execution_core="hccl",
        unresolved=["hccl_native_c_symbol_not_claimed"],
    )
    binding = bind_native_call(declared)
    assert binding.api["namespace"] == "hccl"
    assert binding.api["aclnn_symbol"] is None
    assert binding.evidence["aclnn_symbol"] == "not_applicable"
    assert binding.execution_core == "hccl"
    assert binding.typed["attrs"]["ranks"] == 2
    with pytest.raises(ModelingError, match="attr_mismatch"):
        validate_invocation(binding, {"attrs": {"ranks": 3}})
    with pytest.raises(ModelingError, match="dtype_mismatch"):
        validate_invocation(binding, {"dtype": "fp16"})
    # A genuine collective can never be relabelled with an ACLNN symbol.
    with pytest.raises(ModelingError, match="aclnn_symbol"):
        bind_native_call(declare_native_call(
            entry, api={"namespace": "hccl", "entry": "all_gather",
                        "aclnn_symbol": "aclnnAllGather"}))


def test_unknown_evidence_is_useful_but_explicit_and_unbound():
    op = OpSpec("norm", "rmsnorm", "prefill", 0, 0, (), (64, 2048), (64, 2048),
                (64, 2048), (64, 2048), dtype="bf16")
    binding = bind_native_call(op)
    assert binding is not None
    assert binding.api["entry"] is None
    assert binding.execution_core is None
    assert "exact_native_entry_unresolved" in binding.unresolved
    assert binding.physical is None
    coverage = native_call_coverage((op,))
    assert coverage["denominator"]["emitted_operators"] == 1
    assert coverage["totals"] == {
        "bound": 0, "unresolved": 1, "with_entry": 0,
        "with_source_sha256": 0, "with_kernels": 0, "with_core": 0,
        "fully_paired": 0}


def test_profiler_join_uses_the_same_call_identity_not_a_name_bucket():
    op = _qbmm_declared(_real_quantized_matmul())
    binding = bind_native_call(op)
    report = join_profiler_rows([{"name": "QuantBatchMatmulV3"},
                                 {"name": "unknown_kernel"}], [binding])
    assert report["joined"] == 1 and report["unmatched"] == 1
    assert report["records"][0]["call_ids"] == [binding.call_id]
    assert report["records"][0]["basis"] == "declared_child_kernel"
    assert report["records"][0]["tensor_match_implied"] is False


def test_real_source_family_control_binds_triton_fused_call_once():
    """The shipped MLA-decode control exercises the ordinary lowerings path."""
    raw = json.load(open("examples/source-families/mla-decode-control.json"))
    structure = export_structure(raw, phase="decode", tokens=1, context=16,
                                sequences=1, scope="body")
    records = [record for record in structure["native_calls"]
               if (record.get("api") or {}).get("entry") == "decode_attention_fwd_grouped"]
    assert len({record["call_id"] for record in records}) == 1
    call = records[0]
    assert call["api"]["namespace"] == "triton"
    assert call["api"]["entry"] == "decode_attention_fwd_grouped"
    assert call["api"]["aclnn_symbol"] is None
    assert [kernel["name"] for kernel in call["kernels"]] == [
        "_fwd_grouped_kernel_stage1", "_fwd_kernel_stage2"]
    assert {record["node_id"].split("/", 1)[1] for record in records} == set(
        call["members"]["op_ids"])
    assert len(records) == 2  # two Triton launches under one call identity
    coverage = structure["native_call_coverage"]
    assert coverage["totals"]["bound"] == len(records) == 2
    assert coverage["denominator"]["emitted_operators"] > 2
    assert coverage["totals"]["unresolved"] == (
        coverage["denominator"]["emitted_operators"] - 2)


def test_body_forward_expanded_children_share_parent_owner_not_each_child():
    """Ordinary body graph rows must carry the one fused call's timing owner."""
    from mindie_sim.prediction.body import body_forward

    raw = json.load(open("examples/source-families/mla-decode-control.json"))
    result = body_forward(raw, phase="decode", tokens=1, context=16, sequences=1)
    rows = [row for row in result["timeline"]
            if (row.get("native_call") or {}).get("api", {}).get("entry")
            == "decode_attention_fwd_grouped"]
    assert len(rows) == 2
    assert len({row["native_call"]["call_id"] for row in rows}) == 1
    assert {row["native_call"]["timing_owner"]["op_id"] for row in rows} == {
        "s0.l0.latent_attention_decode"}
    members = set(rows[0]["native_call"]["members"]["op_ids"])
    assert members == {"s0.l0.latent_attention_decode/stage1_grouped_decode",
                       "s0.l0.latent_attention_decode/stage2_split_merge"}
    assert all(set(row["native_call"]["members"]["op_ids"]) == members for row in rows)
    # The fused parent is a zero-duration join, never a separately priced
    # third native row or charge.
    join = [row for row in result["timeline"]
            if row["op_id"] == "s0.l0.latent_attention_decode"]
    assert len(join) == 1 and join[0]["family"] == "barrier"
    assert join[0]["elapsed_ms"] == 0.0 and "native_call" not in join[0]


def test_prepare_only_caller_validates_real_typed_args_and_rejects_mismatches():
    op = _real_quantized_matmul()
    binding = bind_native_call(_qbmm_declared(op))
    actual = {"dtype": binding.typed["dtype"], "layout": binding.typed["layout"],
              "shapes": binding.typed["shapes"]}
    validated = validate_invocation(binding, actual)
    assert validated["call_id"] == binding.call_id
    assert validated["shapes"] == binding.typed["shapes"]
    with pytest.raises(ModelingError, match="dtype_mismatch"):
        validate_invocation(binding, {"dtype": "bf16"})
    with pytest.raises(ModelingError, match="layout_mismatch"):
        validate_invocation(binding, {"layout": "ND"})


def test_hc_pre_native_call_pairs_source_dispatch_and_typed_args():
    """A3 HcPre: real normal native-fusion emission, not a catalogue row."""
    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.structure import parse_case

    raw = json.load(open(
        "examples/cases/longctx-decode-20261030/glm53flash-a3-native-keypool.json"))
    ops = target_graphs(parse_case(raw), phase="decode", tokens=1, context=2048,
                        sequences=1, scope="body")[0]
    op = next(op for op in ops
              if op.op_id.endswith(".attention.hc_pre.native"))
    binding = bind_native_call(op)
    assert binding.api["namespace"] == "aclnn"
    assert binding.api["entry"] == "aclnnHcPre"
    assert binding.api["aclnn_symbol"] == "aclnnHcPre"
    assert binding.source["path"] == "vllm_ascend/models/deepseek_v4/model.py"
    assert binding.source["sha256"] == (
        "257c415f820f6d203b78b3e1000b5e30c475281db1d99a270c65cba52a3f62ab")
    assert binding.kernels[0]["name"] == "HcPre"
    assert binding.kernels[0]["source_sha256"].startswith("f14eeb08")
    assert "EXEC_NPU_CMD(aclnnHcPre)" in binding.typed["attrs"]["dispatch"]
    assert "installed_selected_binary_not_established" in binding.unresolved
    # The source plan does not prove the installed binary's selected core.
    assert binding.execution_core is None


def test_real_triton_occurrence_carries_dynamic_attrs_and_rejects_empty():
    from mindie_sim.prediction.operators import mla_decode

    op = mla_decode.make_op({
        "implementation": mla_decode.IMPLEMENTATION, "contexts": [17],
        "heads": 4, "key_heads": 1, "latent_dim": 256, "rope_dim": 32,
        "page_size": 16, "num_kv_splits": 2, "dtype": "bf16",
        "cache_dtype": "bf16", "is_mla": True, "is_hip": False})
    binding = bind_native_call(op)
    with pytest.raises(ModelingError, match="complete_typed_arguments"):
        validate_invocation(binding, {})
    attrs = {
        "page_size": binding.typed["attrs"]["page_size"],
        "num_kv_splits": binding.typed["attrs"]["num_kv_splits"],
        "grouped": binding.typed["attrs"]["grouped"],
        "is_mla": True,
        "seq_lens": [17],
        "block_table": {"ref": "block_table", "dtype": "int32", "rank": 2},
        "cache_layout": "PA_ND",
    }
    call = {"dtype": "bf16", "layout": "contiguous",
            "shapes": binding.typed["shapes"], "attrs": attrs}
    assert validate_invocation(binding, call)["call_id"] == binding.call_id
    # A different occurrence length is a new occurrence of the same call.
    assert validate_invocation(
        binding, {**call, "attrs": {**attrs, "seq_lens": [65]}})["call_id"] == binding.call_id
    for broken, error in (
            ({**call, "attrs": {key: value for key, value in attrs.items()
                                if key != "seq_lens"}}, "missing_fields"),
            ({**call, "attrs": {**attrs, "page_size": 999}}, "attr_mismatch"),
            ({**call, "attrs": {**attrs, "seq_lens": [-1, 0]}},
             "not_positive"),
            ({**call, "attrs": {**attrs, "block_table": 3.14159}},
             "not_typed_ref"),
            ({**call, "attrs": {**attrs, "cache_layout": 123}},
             "const_mismatch"),
            ({**call, "shapes": {**binding.typed["shapes"], "local": [1, 4, 999]}},
             "shape_mismatch")):
        with pytest.raises(ModelingError, match=error):
            validate_invocation(binding, broken)


def test_triton_call_identity_is_stable_while_occurrence_args_change():
    """Reviewer control: seq_lens 17->65 keeps one call id but not one invocation."""
    from mindie_sim.prediction.body import body_forward
    from mindie_sim.prediction.operators import mla_decode

    raw = json.load(open("examples/source-families/mla-decode-control.json"))
    observed = {}
    for context in (16, 64):
        result = body_forward(raw, phase="decode", tokens=1, context=context,
                              sequences=1)
        row = next(row for row in result["timeline"]
                   if (row.get("native_call") or {}).get("api", {}).get("entry")
                   == "decode_attention_fwd_grouped")
        observed[context] = (row["native_call"], result["duration_ms"])
    first, second = observed[16][0], observed[64][0]
    assert first["call_id"] == second["call_id"]
    assert first["occurrence"]["seq_lens"] == [17]
    assert second["occurrence"]["seq_lens"] == [65]
    assert first["typed"]["attrs"] == second["typed"]["attrs"]
    assert first["typed"]["attrs"]["page_size"] > 0
    # Same selected call, different occurrence payload and body work.
    assert observed[16][1] != observed[64][1]
    op = mla_decode.make_op({
        "implementation": mla_decode.IMPLEMENTATION, "contexts": [17],
        "heads": 4, "key_heads": 1, "latent_dim": 256, "rope_dim": 32,
        "page_size": 16, "num_kv_splits": 2, "dtype": "bf16",
        "cache_dtype": "bf16", "is_mla": True, "is_hip": False})
    binding = bind_native_call(op)
    args = binding.prepare_args()
    assert validate_invocation(binding, args)["call_id"] == binding.call_id
    assert args["attrs"]["seq_lens"] == [17]
