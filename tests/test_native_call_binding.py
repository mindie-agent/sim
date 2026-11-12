"""BIND1 native-call identity over real emitted structure and declared sources.

[Role] Prove one declared boundary identity through bind, coverage and export.
[Invariants] Unresolved evidence never binds; derived provenance never hashes.
[Timeouts & Locks] Pure in-process metadata; no I/O, device or locks.
[Side-Effect Policy] Writes only pytest temporary HOME/config/data fixtures.
"""
import ast
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

import mindie_sim.prediction.native_coverage as native_coverage
from mindie_sim.prediction.hccl_graph import lower
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.native_binding import (
    PROVED_CORES, SCHEMA, NativeCallBinding, bind_expanded_call, bind_native_call,
    coverage_from_records, declare_native_call, family_is_hccl, join_profiler_rows,
    member_physical_view, native_call_coverage, self_non_claims, validate_invocation,
)
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.operators import mla_decode
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.subgraphs.registry import validate_combination
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure import export_op_structure

ROOT = Path(__file__).resolve().parents[1]
QBMM_CPP = ("/usr/local/Ascend/cann/opp/built-in/op_impl/ai_core/tbe/impl/"
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
PUBLIC = ("SCHEMA", "PROVED_CORES", "NativeCallBinding", "declare_native_call",
          "bind_native_call", "bind_expanded_call", "validate_invocation",
          "native_call_coverage", "coverage_from_records", "join_profiler_rows",
          "member_physical_view", "family_is_hccl", "self_non_claims")


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    for name, leaf in (("MINDIE_CONFIG_HOME", "config"),
                       ("MINDIE_DATA_HOME", "data"),
                       ("MINDIE_DIAGNOSTICS_ROOT", "diagnostics"), ("HOME", "home")):
        path = tmp_path / leaf
        path.mkdir()
        monkeypatch.setenv(name, str(path))


def _real_quantized_matmul():
    raw = json.loads((ROOT / "examples/cases/glm53-a3-w4a8.json").read_text())
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


def _mla_call_parent(observed_aliases=None):
    raw = json.loads((ROOT / "examples/source-families/mla-decode-control.json").read_text())
    parent = mla_decode.make_op({
        "implementation": mla_decode.IMPLEMENTATION, "contexts": [17], "heads": 4,
        "key_heads": 1, "latent_dim": 256, "rope_dim": 32, "page_size": 16,
        "num_kv_splits": 2, "dtype": "bf16", "cache_dtype": "bf16",
        "is_mla": True, "is_hip": False})
    declared = declare_native_call(
        parent, api={"namespace": "triton", "entry": "decode_attention_fwd_grouped"},
        source={"path": "vllm_ascend/ops/mla.py", "sha256": "a" * 64,
                "symbol": "decode_attention_fwd_grouped"},
        kernels=[{"name": "_fwd_grouped_kernel_stage1", "role": "vector",
                  "evidence": "source_declared"},
                 {"name": "_fwd_kernel_stage2", "role": "vector",
                  "evidence": "source_declared"}],
        observed_aliases=observed_aliases)
    return declared, raw["runtime"]["kernels"]


def _plain_op(op_id="norm"):
    return OpSpec(op_id, "rmsnorm", "prefill", 0, 0, (), (8, 16), (8, 16),
                  (8, 16), (8, 16), dtype="bf16",
                  metadata=(("module_type", "Linear"), ("module_id", op_id)))


def _occurrence_binding():
    op = OpSpec("occ", "attention", "decode", 0, 0, (), (4,), (4,), (4,), (4,),
                dtype="bf16")
    declared = declare_native_call(
        op, api={"namespace": "torch", "entry": "occ"},
        typed={"attrs": {"mode": "a", "axis": 1}, "occurrence_attrs": {
            "c": {"const": 7}, "i": "int", "p": "positive_int",
            "il": "int_list", "pl": "positive_int_list", "im": "int_matrix",
            "s": "string", "ref": "typed_ref", "d": "declared"}})
    return bind_native_call(declared)


def test_public_reexports_are_the_single_coverage_implementation():
    import mindie_sim.prediction.native_binding as binding

    assert binding.__all__ == PUBLIC
    assert len(PUBLIC) == 13
    assert all(getattr(binding, name) is not None for name in PUBLIC)
    for name in ("native_call_coverage", "coverage_from_records", "join_profiler_rows"):
        assert getattr(binding, name) is getattr(native_coverage, name)
    assert binding.self_non_claims() == self_non_claims()
    assert "hccl" in PROVED_CORES and "host" in PROVED_CORES


def test_real_quantized_matmul_forwards_plan_work_and_consumer_views():
    op = _real_quantized_matmul()
    declared = _qbmm_declared(op)
    plan = OperatorPlan(Work(matrix={"int8": 1.5e9}, resources={"hbm": 4096}),
                        lambda context: None)
    prepared = replace(declared, flops=1.5e9)
    binding = bind_native_call(declared, plan=plan, prepared=prepared)
    assert isinstance(binding, NativeCallBinding)
    assert binding.api["namespace"] == "cann_kernel"
    assert binding.api["entry"] == "QuantBatchMatmulV3"
    assert binding.api["aclnn_symbol"] is None
    assert binding.source["sha256"] == QBMM_SHA
    assert binding.typed["dtype"] == "int8"
    assert binding.typed["shapes"]["local"] == list(op.local_shape)
    assert binding.evidence["entry"] == "source_declared"
    assert binding.execution_core is None  # a kernel name is not a core rule
    assert "aclnn_public_symbol_not_in_evidence" in binding.unresolved
    assert binding.physical == {
        "basis": "existing_operator_plan_work", "matrix": {"int8": 1.5e9},
        "resources": {"hbm": 4096}, "charged": "once_at_timing_owner",
        "prepared_flops": 1.5e9}
    ledger = binding.to_dict()
    assert ledger["schema"] == SCHEMA
    assert ledger["members"] == {
        "op_ids": [op.op_id], "charge": "call",
        "charge_note": "priced once at timing_owner.op_id"}
    assert ledger["api"]["entry"] == "QuantBatchMatmulV3"
    assert ledger["kernels"][0]["name"] == "QuantBatchMatmulV3"
    args = binding.prepare_args()
    assert args["dtype"] == "int8" and args["layout"] == "int8_per_channel"
    assert args["shapes"] == binding.typed["shapes"]
    assert validate_invocation(binding, args)["call_id"] == binding.call_id
    forwarded = binding.forward()
    assert forwarded["call_id"] == binding.call_id
    assert forwarded["typed"] is binding.typed
    assert forwarded["physical"] is binding.physical
    assert forwarded["timing_owner"] == {"op_id": op.op_id, "scope": "call"}


def test_multistage_kernel_and_one_to_many_source_call_keep_one_identity():
    op = OpSpec("sfa", "attention", "prefill", 0, 0, (), (128, 64, 576),
                (128, 64, 576), (128, 64, 576), (128, 64, 576), dtype="bf16",
                layout="PA_ND", recipe="kv_quant_sparse_flash_attention",
                source_boundary="kernel")
    stages = [f"op_kernel/arch35/{name}" for name, _ in KVSPARSE]
    declared = declare_native_call(
        op, api={"namespace": "torch",
                 "entry": "torch_npu.npu_kv_quant_sparse_flash_attention"},
        source={"repository": "cann", "path": stages[0], "sha256": KVSPARSE[0][1],
                "symbol": "KvQuantSparseFlashAttentionMla", "revision": "cann-9.2.1"},
        kernels=[{"name": "KvQuantSparseFlashAttention", "role": "mix",
                  "source_path": stages[0], "source_sha256": KVSPARSE[0][1],
                  "source_stages": stages, "evidence": "source_declared"}])
    binding = bind_native_call(declared)
    assert len(binding.kernels) == 1
    assert binding.kernels[0]["name"] == "KvQuantSparseFlashAttention"
    assert binding.kernels[0]["role"] == "mix"
    assert binding.kernels[0]["source_stages"] == stages
    assert binding.typed["layout"] == "PA_ND"
    assert binding.typed["shapes"]["launch"] == list(op.launch_shape)
    normalized = bind_native_call(declare_native_call(
        _plain_op("normalized"), kernels=[{"name": "ExampleKernel"}]))
    assert normalized.kernels[0]["role"] == "unknown"
    assert normalized.kernels[0]["evidence"] == "source_declared"
    assert normalized.evidence["kernels"] == "source_declared"
    bound_alias = replace(_plain_op("alias"), metadata=(
        ("native_call_binding", {"kernels": [{"name": "ExampleKernel"}]}),))
    assert bind_native_call(bound_alias).evidence["kernels"] == "source_declared"

    names = ("Cast", "Transpose", "Prepare", "H", "O")
    parent = OpSpec("gdn_chunk", "gdn_prefill", "prefill", 0, 0, (),
                    (1, 2, 64, 128), (1, 2, 64, 128), (1, 2, 64, 128),
                    (1, 2, 64, 128), dtype="bf16", layout="BNSD",
                    recipe="chunk_gated_delta_rule_fwd", source_boundary="call",
                    metadata=(("source", {"path": "vllm_ascend/ops/gdn.py",
                                          "sha256": "5e" * 32,
                                          "symbol": "chunk_gated_delta_rule_fwd"}),))
    children = tuple(replace(parent, op_id=name.lower(), source_boundary="kernel",
                             metadata=(("native_kind", name),)) for name in names)
    recipe = PlannedRecipe(
        lambda op, geometry: OperatorPlan(Work(), lambda context: None),
        kernels=lambda op: children)
    book = RecipeBook(recipes={"chunk_gated_delta_rule_fwd": recipe})
    expanded = book.expand((parent,))
    source_declared = declare_native_call(
        parent, api={"namespace": "aclnn", "entry": "aclnnChunkGatedDeltaRuleFwd",
                     "aclnn_symbol": "aclnnChunkGatedDeltaRuleFwd"},
        kernels=[{"name": name, "role": "vector", "evidence": "source_declared"}
                 for name in names])
    fused = bind_expanded_call(source_declared, book)
    explicit = bind_native_call(source_declared, children=expanded)
    assert fused.call_id == explicit.call_id
    assert fused.timing_owner == {"op_id": parent.op_id, "scope": "call"}
    assert fused.member_op_ids == tuple(child.op_id for child in expanded
                                        if child.family != "barrier")
    child_plan = OperatorPlan(Work(matrix={"bf16": 1.0}), lambda context: None)
    views = [member_physical_view(fused, child_plan)
             for child in expanded if child.family != "barrier"]
    assert {view.call_id for view in views} == {fused.call_id}
    assert all(view.physical["basis"] == "member_view_of_shared_call_work"
               for view in views)
    assert all(view.physical["charged"].startswith("never_separately")
               for view in views)

    real_parent, real_kernels = _mla_call_parent()
    report = export_op_structure(
        {0: (real_parent,)}, model={}, execution={"phase": "decode"},
        kernels=real_kernels)
    records = report["native_calls"]
    assert len(records) == 2
    assert len({record["call_id"] for record in records}) == 1
    assert {record["timing_owner"]["op_id"] for record in records} == {
        real_parent.op_id}
    assert report["native_call_coverage"]["totals"]["bound"] == 2
    assert report["native_call_coverage"]["denominator"]["unrecorded_emitted"] == 1


def test_hccl_boundaries_are_declared_not_invented():
    source = OpSpec("tp_all_gather", "allgather", "decode", 0, 0, (), (5120,),
                    (5120,), (5120,), (5120,), link_bytes=10240, dtype="bf16",
                    metadata=(("group_axis", "tp"), ("collective", "allgather")))
    runtime = SimpleNamespace(
        parallel=SimpleNamespace(tp=2, pcp=1, etp=1),
        kernels={"hccl": {"allgather": {
            "expansion": "AICPU_TS", "template": "nhr", "dtype_bytes": 2}}})
    lowered = lower((source,), runtime, "bf16")
    entry = next(op for op in lowered if op.family == "hccl_boundary"
                 and dict(op.metadata)["device_kernel_group"]["entry"])
    local = next(op for op in lowered if op.family == "hccl_local")
    exit_atom = next(op for op in lowered if op.family == "hccl_boundary"
                     and not dict(op.metadata)["device_kernel_group"]["entry"])
    assert family_is_hccl(entry) and family_is_hccl(local)
    assert not family_is_hccl(_plain_op())
    assert bind_native_call(local) is None
    assert bind_native_call(exit_atom) is None
    assert bind_native_call(OpSpec("join", "barrier", "decode", 0, 0, (),
                                   (1,), (1,), (1,), (1,))) is None
    declared = declare_native_call(
        entry, api={"namespace": "hccl",
                    "entry": "torch.distributed.all_gather_into_tensor"},
        source={"repository": "task-evidence", "path": "COMPHY3-lab/COMPHY7_collect.py",
                "sha256": "8f" * 32, "symbol": "measured_graph_replay"},
        typed={"dtype": "bf16", "layout": "contiguous",
               "shapes": {"input": [5120], "output": [10240]},
               "attrs": {"world_size": 2, "ranks": 2}},
        execution_core="hccl")
    binding = bind_native_call(declared)
    assert binding.api["namespace"] == "hccl"
    assert binding.api["aclnn_symbol"] is None
    assert binding.evidence["aclnn_symbol"] == "not_applicable"
    assert binding.execution_core == "hccl"
    with pytest.raises(ModelingError, match="native_call_declaration_empty"):
        declare_native_call(entry)
    for broken, error in (
            (declare_native_call(
                entry, api={"namespace": "hccl", "entry": "all_gather",
                            "aclnn_symbol": "aclnnAllGather"}),
             "hccl_call_cannot_carry_aclnn_symbol"),
            (declare_native_call(entry, api={"namespace": "bogus",
                                             "entry": "collective"}),
             "native_call_namespace_unknown:bogus"),
            (declare_native_call(entry, execution_core="npu"),
             "native_call_core_not_proved:npu")):
        with pytest.raises(ModelingError, match=error):
            bind_native_call(broken)


def test_validate_invocation_round_trip_and_explicit_mismatches():
    binding = bind_native_call(_qbmm_declared(_real_quantized_matmul()))
    actual = binding.prepare_args()
    validated = validate_invocation(binding, actual)
    assert validated["call_id"] == binding.call_id
    assert validated["shapes"] == binding.typed["shapes"]
    for altered, error in (
            ({**actual, "dtype": "bf16"}, "dtype_mismatch"),
            ({**actual, "layout": "ND"}, "layout_mismatch"),
            ({**actual, "shapes": {**actual["shapes"], "local": [1, 2, 3]}},
             "shape_mismatch")):
        with pytest.raises(ModelingError, match=error):
            validate_invocation(binding, altered)
    with pytest.raises(ModelingError, match="complete_typed_arguments"):
        validate_invocation(binding, {})


def test_occurrence_attrs_reject_missing_and_invalid_values():
    binding = _occurrence_binding()
    base = {"mode": "a", "axis": 1, "c": 7, "i": 3, "p": 1, "il": [1], "pl": [2],
            "im": [[0, 1]], "s": "x",
            "ref": {"ref": "r", "dtype": "bf16", "rank": 2}, "d": False}
    args = {"dtype": "bf16", "layout": "contiguous",
            "shapes": binding.typed["shapes"], "attrs": base}
    assert validate_invocation(binding, args)["call_id"] == binding.call_id
    with pytest.raises(ModelingError, match="shape_mismatch:local"):
        validate_invocation(binding, {**args, "shapes": {
            **binding.typed["shapes"], "local": None}})
    with pytest.raises(ModelingError, match="attr_mismatch:axis"):
        validate_invocation(binding, {**args, "attrs": {**base, "axis": None}})
    with pytest.raises(ModelingError, match="missing_fields:s"):
        validate_invocation(binding, {"dtype": "bf16", "layout": "contiguous",
                                      "shapes": binding.typed["shapes"],
                                      "attrs": {key: value for key, value in base.items()
                                                if key != "s"}})
    for kind, value, error in (
            ("mode", "b", "attr_mismatch"),
            ("c", 8, "const_mismatch"),
            ("i", True, "not_int"),
            ("p", 0, "not_positive"),
            ("il", [], "not_int_list"),
            ("pl", [1, 0], "not_positive"),
            ("im", [[0, -1]], "not_nonnegative"),
            ("s", "", "not_string"),
            ("ref", {"ref": "r", "dtype": "bf16", "rank": 0}, "not_typed_ref"),
            ("d", None, "missing")):
        broken = {**args, "attrs": {**base, kind: value}}
        with pytest.raises(ModelingError, match=error):
            validate_invocation(binding, broken)


def test_coverage_and_profiler_join_keep_actual_denominators():
    bound = bind_native_call(_qbmm_declared(_real_quantized_matmul()))
    plain = bind_native_call(_plain_op())
    coverage = native_call_coverage((_qbmm_declared(_real_quantized_matmul()),
                                     _plain_op()))
    assert coverage["denominator"]["emitted_operators"] == 2
    assert coverage["totals"]["bound"] == 1
    assert coverage["totals"]["unresolved"] == 1
    assert coverage["totals"]["with_entry"] == 1
    assert coverage["totals"]["with_source_sha256"] == 1
    assert coverage["totals"]["with_kernels"] == 1
    assert coverage["unresolved_source_families"] == [("rmsnorm", "analytical", "-")]

    records = [bound.to_dict(), plain.to_dict()]
    measured = coverage_from_records(records, emitted=3)
    assert measured["denominator"] == {
        "emitted_operators": 3, "recorded_bindings": 2,
        "unrecorded_emitted": 1, "distinct_source_families": 2}
    assert measured["totals"]["bound"] == 1
    assert measured["totals"]["unresolved"] == 2
    assert ("__unrecorded__", "none", "no_binding_record") in measured[
        "unresolved_source_families"]
    with pytest.raises(ModelingError, match="coverage_emitted_below_recorded"):
        coverage_from_records(records, emitted=1)

    entry_binding = bind_native_call(declare_native_call(
        _plain_op("entry"), api={"namespace": "torch", "entry": "entry_kernel"}))
    joined = join_profiler_rows(
        [{"name": "QuantBatchMatmulV3"}, {"name": "entry_kernel"}, {"name": "other"}],
        [bound, entry_binding])
    assert (joined["rows"], joined["joined"], joined["unmatched"]) == (3, 2, 1)
    assert [record["basis"] for record in joined["records"]] == [
        "declared_child_kernel", "declared_entry"]
    assert joined["records"][0]["call_ids"] == [bound.call_id]
    assert joined["records"][1]["call_ids"] == [entry_binding.call_id]
    assert joined["unmatched_rows"][0]["basis"] == "unmapped"
    assert joined["unmatched_rows"][0]["call_ids"] == []
    assert joined["records"][0]["tensor_match_implied"] is False


def test_structure_attaches_native_calls_outside_graph_identity_and_interns():
    import mindie_sim.structure as structure

    ops = (_plain_op("a"), _plain_op("b"))
    report = export_op_structure({0: ops}, model={}, execution={"phase": "prefill"},
                                 source={"revision": "one"})
    assert len(report["native_calls"]) == 2
    assert report["native_call_coverage"]["denominator"]["emitted_operators"] == 2
    assert all("native_call" not in node for node in report["nodes"])
    payload = {key: value for key, value in report.items()
               if key not in {"graph_id", "source", "native_calls",
                              "native_call_coverage"}}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    assert report["graph_id"] == "sha256:" + sha256(canonical.encode("utf-8")).hexdigest()
    same = export_op_structure({0: ops}, model={}, execution={"phase": "prefill"},
                               source={"revision": "two"})
    assert same["graph_id"] == report["graph_id"]
    first, second = report["nodes"]
    first_meta, second_meta = first["operator"]["metadata"], second["operator"]["metadata"]
    first_key = next(key for key in first_meta if key == "module_type")
    second_key = next(key for key in second_meta if key == "module_type")
    assert first_key is second_key is structure._intern_tree("module_type")
    assert first_meta["module_type"] is second_meta["module_type"]
    assert first["family"] is second["family"]
    interned = structure._intern_tree({"k": "v"})
    assert next(iter(interned)) is structure._intern_tree("k")
    assert interned["k"] is structure._intern_tree("v")


def test_d0_accepts_every_attention_macro_and_rejects_plain_ffn():
    for kinds in (("D0", "A0"), ("D0", "A1"), ("D0", "A2", "K0"),
                  ("D0", "A3"), ("D0", "A4")):
        assert validate_combination(kinds) == kinds
    with pytest.raises(ModelingError, match="subgraph_combination_invalid:D0"):
        validate_combination(("D0", "F0"))


def test_piecewise_and_subgraph_sources_never_name_retired_mechanisms():
    forbidden = {"service_regimes", "fixed_us"}
    violations = []
    files = [*sorted((ROOT / "mindie_sim/prediction/piecewise").glob("*.py")),
             *sorted((ROOT / "mindie_sim/prediction/subgraphs").glob("*.py"))]
    assert len(files) >= 20
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
            elif isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names = [node.value]
            if any(token in name for name in names for token in forbidden):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert violations == []


def _counts(binding, records=None):
    totals = (coverage_from_records(records) if records is not None else native_call_coverage(
        (_plain_op("coverage"),), bindings={0: binding}))["totals"]
    return totals["bound"], totals["unresolved"], totals["fully_paired"]


@pytest.mark.parametrize("declaration, evidence, expected", (
    ({"kernels": [{"name": "UnresolvedKernel", "evidence": "unresolved"}]},
     "unresolved", (0, 1, 0)),
    ({"kernels": [{"name": "Resolved", "evidence": "source_declared"},
                  {"name": "Unresolved", "evidence": "unresolved"}]}, "partial", (1, 0, 0)),
    ({"api": {"namespace": "torch", "entry": "CallableKernel"}}, "unresolved", (1, 0, 0)),
    ({"api": {"namespace": "cann_kernel", "entry": "CompleteKernel"},
      "kernels": [{"name": "CompleteKernel", "evidence": "source_declared"}]},
     "source_declared", (1, 0, 1)),
))
def test_kernel_evidence_controls_binding(declaration, evidence, expected):
    binding = bind_native_call(declare_native_call(
        _plain_op("scenario"), source={"path": "kernel.cpp", "sha256": "9" * 64},
        **declaration))
    assert binding.evidence["kernels"] == evidence
    assert _counts(binding) == _counts(binding, [binding.to_dict()]) == expected
    if evidence == "unresolved":
        claimed = {**binding.to_dict(), "evidence": {
            **binding.evidence, "kernels": "partial"}}
        assert _counts(binding, [claimed]) == expected


def test_observed_aliases_are_evidence_not_structural_identity():
    def report(expanded, aliases):
        if not expanded:
            op = _plain_op("alias") if aliases is None else declare_native_call(
                _plain_op("alias"), observed_aliases=aliases)
            return export_op_structure({0: (op,)}, model={},
                                       execution={"phase": "prefill"})
        op, kernels = _mla_call_parent(aliases)
        return export_op_structure({0: (op,)}, model={},
                                   execution={"phase": "decode"}, kernels=kernels)
    for expanded in (False, True):
        reports = [report(expanded, alias) for alias in (None, [{"name": "a"}], [{"name": "b"}])]
        keys = ("graph_id", "nodes", "source_calls", "subgraphs")
        assert all(row[key] == reports[0][key] for row in reports for key in keys)
        assert [row["native_calls"][0]["observed_aliases"] for row in reports] == [
            [], [{"name": "a"}], [{"name": "b"}]]
