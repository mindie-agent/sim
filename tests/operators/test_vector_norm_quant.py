"""Vector normalization, quantization, copy and sort capacity laws.
[Role] Bind vector/copy/quant producers to one causal engine: fused residual
  norm and MX/INT8 scale materialization, activation call closures, AIV tensor
  copy controls, joint copy service, activation quant producers, and the
  opt-in Sort32 speculative TopK/sampling capacity preset.
[Invariants] INT8 keeps one FP32 scale per complete row; MX keeps an E8M0
  scale per 32-value group plus the even-group tail pad.  Copy and joint
  service conserve selected payload bytes, owner counts and inclusive call
  timing.  A declared rate changes time, never physical work; an unclassified
  work composition is rejected rather than silently repriced.
[Timeouts & Locks] Pure metadata arithmetic and bounded in-memory engine runs
  on small synthetic shapes; no device, weight load or repository mutation.
[Side-Effect Policy] Read-only fixture inputs, in-memory simulation only.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from math import prod
from pathlib import Path
import pytest
from _support.captures import activation_protocol
from _support.hardware import hardware_case
from _support.memory_cases import case as declared_case
from mindie_sim import graph_costs, operator_costs, predict, target_graphs
from mindie_sim.catalog.ops import vector_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.observations.calibration import bind_activation_condition
from mindie_sim.observations.captures import iter_conditions
from mindie_sim.prediction import call_builders as calls_activation, instructions
from mindie_sim.prediction.operators.vector import dynamic_quant_schedule, family_ascend_norm, mx_dynamic_quant
from mindie_sim.prediction.call_builders import call_costs
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.operators.hyper_connections.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.operators.vector import cuda_quant
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.prediction.source_program_service import SourceProgramService
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.tensors import TensorLowering
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.invocation import InvocationTiming
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.structure_contracts import predicted_contract
from mindie_sim.structure_subgraphs import subgraph_contract
pytestmark = pytest.mark.tier0
ROOT = Path(__file__).resolve().parents[2]
SORT_SKUS = (("Ascend910B4", 1.5e9), ("Ascend910_9391", 1.85e9), ("Ascend950DT_9581", 1.65e9))
PACKET = json.loads((ROOT / "data/observations/a5-copy-joint-service-20261030/parameters.json").read_text())
def _selected(op, **kwargs):
    return family_ascend_norm.lower_op(op, **kwargs)[-1]
def _band(width, rows):
    groups, padded = (width + 31) // 32, (((width + 31) // 32 + 1) // 2) * 2
    return groups, padded, (rows, padded // 2, 2)
def _norm(family, rows, width, *, quant=None, returns=(), extra=(), bias=False):
    elements = rows * width
    meta = {"norm_has_weight": True, "norm_has_bias": bias, "norm_eps": 1e-6,
            "norm_gamma_role": "ffn_norm", "norm_gamma_runtime_bytes": width * 2,
            "norm_gamma_storage_bytes": width * 2, "norm_gamma_runtime_format": "bf16",
            "norm_gamma_storage_format": "bf16"}
    payload = scale = 0
    if quant:
        _, padded, shape = _band(width, rows)
        int8 = quant == "int8"
        meta.update(output_format="int8" if int8 else "fp8_e4m3",
                    activation_group_size=width if int8 else 32,
                    activation_scale_format="fp32" if int8 else "fp8_e8m0",
                    activation_scale_shape=(rows,) if int8 else shape,
                    activation_scale_bytes=rows * 4 if int8 else rows * padded)
        payload, scale = elements, rows * 4 if int8 else rows * padded
    read = ((2 * rows + 2) if family == "add_rmsnorm_quant" else rows + 1) * width * 2
    write = (rows * width * 2 + payload + scale if family == "add_rmsnorm_quant"
             else payload + scale if quant else 2 * elements * 2)
    vector = (6 if family == "add_rmsnorm_quant" else 5) * elements + (
        4 * elements + 2 * rows * ((width + 31) // 32 if quant == "mx" else 1) if quant else 0)
    if family == "add_rmsnorm_quant":
        returns = (*returns, "residual", "activation_scales")
    elif quant:
        returns = (*returns, "activation_scales")
    return vector_op(
        op_id="body.norm", deps=("residual",), stage=0, layer=0, requests=("r",),
        elements=elements, ops_per_element=0, dtype="bf16", family=family, phase="decode",
        recipe=("add_rmsnorm_quantize" if family == "add_rmsnorm_quant"
                else "rmsnorm_quantize" if quant else "add_rmsnorm"),
        shape=(rows, width), read_bytes_override=read, write_bytes_override=write,
        vector_ops_override=vector, returns=returns,
        metadata=tuple(meta.items()) + tuple(extra))
def _rewire(op, *, vector_ops=None, **changes):
    fields = dict(op_id=op.op_id, deps=op.deps, stage=0, layer=0, requests=("r",),
                  elements=prod(op.local_shape), ops_per_element=0, dtype="bf16",
                  family=op.family, phase="decode", recipe=op.recipe, shape=op.local_shape,
                  read_bytes_override=op.mte2_bytes, write_bytes_override=op.mte3_bytes,
                  vector_ops_override=op.vector_ops if vector_ops is None else vector_ops,
                  returns=op.returns, metadata=op.metadata)
    fields.update(changes)
    return vector_op(**fields)
@pytest.mark.parametrize(("family", "rows", "width", "base"), [
    ("rmsnorm_quant", 4, 5120, 5), ("add_rmsnorm_quant", 3, 96, 6)])
def test_mx_norm_quant_bills_source_register_ledger(family, rows, width, base):
    selected = _selected(_norm(family, rows, width, quant="mx"))
    meta, elements = dict(selected.metadata), rows * width
    _, padded, shape = _band(width, rows)
    fused = family == "add_rmsnorm_quant"
    assert meta["native_norm_kind"] == (
        "AddRmsNormDynamicMxQuant" if fused else "RmsNormDynamicMxQuant")
    assert selected.vector_ops == base * elements + rows * mx_dynamic_quant.group32_work(
        width)["lane_ops"]["total"]
    assert meta["native_norm_quant_work_source"] == "mx_dynamic_quant.group32_work:r64_alg0"
    assert meta["native_norm_quant_scale_bytes"] == rows * padded
    assert selected.mte2_bytes == (2 * (rows + 1) if fused else rows + 1) * width * 2
    assert selected.mte3_bytes == (3 if fused else 1) * elements + rows * padded
    if fused:
        assert meta["native_norm_residual_bytes"] == elements * 2
    else:
        odd = _selected(_norm("rmsnorm_quant", 7, 4097, quant="mx"))
        assert odd.vector_ops == 5 * 7 * 4097 + 7 * mx_dynamic_quant.group32_work(
            4097)["lane_ops"]["total"]
def test_norm_quant_contracts_conserve_scales_and_repeats():
    int8 = _selected(_norm("add_rmsnorm_quant", 2, 6144, quant="int8", bias=True), has_bias=True)
    meta = dict(int8.metadata)
    assert meta["native_norm_kind"] == "AddRmsNormDynamicQuant"
    assert int8.mte3_bytes == 2 * 6144 * 3 + 2 * 4
    add = _selected(_norm("add_rmsnorm", 2, 5120))
    residual = 2 * 5120 * 2
    assert add.mte3_bytes == 2 * residual + 2 * 4
    three = _selected(_norm("add_rmsnorm", 2, 5120, extra=(("norm_native_api", "AddRmsNorm"),)))
    gemma = _norm("add_rmsnorm", 1, 3584, extra=(("rms_norm_source_class", "AscendGemmaRMSNorm"),))
    plain = _norm("add_rmsnorm", 1, 3584, extra=(("rms_norm_source_class", "AscendRMSNorm"),))
    assert dict(three.metadata)["native_implementation"]["conditions"]["api"] == "AddRmsNorm"
    assert (dict(_selected(gemma).metadata)["native_norm_kind"], dict(_selected(plain).metadata)["native_norm_kind"]) == ("AddRmsNorm", "AddRmsNormBias")
    op = _norm("rmsnorm_quant", 4, 5120, quant="mx")
    guard = _rewire(op, vector_ops=5 * 4 * 5120 + 4 * 5120,
                    metadata=op.metadata + (("mx_vector_ops_per_row", 1234),))
    ledger = _selected(guard)
    assert ledger.vector_ops == 5 * 4 * 5120 + 4 * 1234
    assert dict(ledger.metadata)["native_norm_quant_work_source"] == (
        "declared_mx_vector_ops_per_row")
    assert _selected(ledger).vector_ops == ledger.vector_ops
    for returns in (("norm_output",), ()):
        partial = _selected(_rewire(_norm("add_rmsnorm", 2, 5120, returns=returns),
                                    write_bytes_override=2 * 5120 * 2))
        assert "residual" not in partial.returns
        assert dict(partial.metadata)["native_norm_residual_billing"] == "absent"
        assert partial.mte3_bytes == 2 * 5120 * 2 + 2 * 4
def _bad_norm(kind):
    if kind == "block":
        op = _norm("add_rmsnorm_quant", 2, 5120, quant="int8")
        return _rewire(op, metadata=op.metadata + (("activation_group_size", 128),
                                                  ("activation_scale_shape", (2, 40)),
                                                  ("activation_scale_bytes", 320)))
    if kind == "floor":
        op = _norm("add_rmsnorm_quant", 3, 96, quant="mx")
        return _rewire(op, metadata=op.metadata + (("activation_scale_shape", (3, 1, 2)),
                                                  ("activation_scale_bytes", 6)))
    if kind in ("plain", "beta"):
        return _norm("rmsnorm_quant" if kind == "plain" else "add_rmsnorm_quant",
                     1 if kind == "plain" else 2, 5120 if kind == "plain" else 3584,
                     quant="int8" if kind == "plain" else "mx", bias=kind == "beta")
    if kind == "slot":
        return _norm("add_rmsnorm", 1, 3584,
                     extra=(("norm_native_api", "AddRmsNorm"), ("norm_has_bias", True)))
    op = _norm("rmsnorm_quant", 4, 5120, quant="mx")
    return _rewire(op, vector_ops=op.vector_ops + 7)
@pytest.mark.parametrize(("kind", "match"), (
    ("block", "complete_row_fp32_scale"), ("floor", "scale_materialization_mismatch"),
    ("plain", "plain_int8_quant_kernel_not_established"),
    ("beta", "mx_quant_beta_not_in_declared_abi"),
    ("slot", "add_rms_norm_has_no_beta_slot"), ("unknown", "work_composition_unknown")))
def test_invalid_norm_declarations_are_rejected(kind, match):
    kwargs = {"has_bias": True} if kind == "slot" else {}
    with pytest.raises(ModelingError, match=match):
        family_ascend_norm.lower_op(_bad_norm(kind), **kwargs)
    if kind == "beta":
        for family, native_kind in (("rmsnorm_quant", "RmsNormDynamicMxQuant"), ("add_rmsnorm_quant", "AddRmsNormDynamicMxQuant")):
            with pytest.raises(ModelingError, match="mx_quant_beta_not_in_declared_abi"):
                family_ascend_norm.lower_op(_norm(family, 2, 3584, quant="mx", bias=True))
            caller_beta = _norm(family, 2, 3584, quant="mx")
            with pytest.raises(ModelingError, match="mx_quant_beta_not_in_declared_abi"):
                family_ascend_norm.lower_op(caller_beta, has_bias=True)
            assert dict(_selected(caller_beta).metadata)["native_norm_kind"] == native_kind
    if kind == "plain":
        with pytest.raises(ModelingError, match="positive_width"):
            mx_dynamic_quant.producer_contract(0)
        bad = _norm("rmsnorm_quant", 4, 5120, quant="mx")
        declared = _rewire(bad, metadata=bad.metadata + (("mx_vector_ops_per_row", 0),))
        with pytest.raises(ModelingError, match="mx_vector_ops_per_row_invalid"):
            family_ascend_norm.lower_op(declared)
    if kind == "floor":
        assert mx_dynamic_quant.producer_contract(96, scale_alg=1)["scale_shape"] == (2, 2)
def test_norm_quant_declarations_flow_through_real_cases():
    raw = json.loads((ROOT / "examples/cases/qwen38-a5.json").read_text())
    options = {"implementation": "vllm_ascend_native_norm",
               "options": {"has_bias": False, "external_gamma_shift": True}}
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {}).update({
        "add_rmsnorm": options, "add_rmsnorm_quant": options, "rmsnorm_quant": options})
    ops = target_operators(parse_case(raw), phase="decode", tokens=4, context=2048)
    producers = [op for op in ops if op.family in {"rmsnorm_quant", "add_rmsnorm_quant"}
                 and op.local_shape == (4, 5120)]
    assert producers
    for producer in producers:
        meta = dict(producer.metadata)
        consumers = [op for op in ops if producer.op_id in op.deps]
        assert len(consumers) == 1
        cm = dict(consumers[0].metadata)
        assert cm["activation_scale_shape"] == meta["native_norm_quant_scale_shape"]
        assert cm["activation_scale_bytes"] == meta["native_norm_quant_scale_bytes"]
def test_activation_closures_and_retained_protocols():
    n = 2 * 7
    fused = calls_activation.build_activation_call(
        calls_activation.ActivationCallSpec("swiglu_fused", 2, 7))
    separate = calls_activation.build_activation_call(
        calls_activation.ActivationCallSpec("swiglu_separate", 2, 7))
    assert [(op.mte2_bytes, op.mte3_bytes) for op in fused.ops] == [(2 * n * 2, n * 2)]
    assert [(op.mte2_bytes, op.mte3_bytes) for op in separate.ops] == [
        (n * 2, n * 2), (2 * n * 2, n * 2)]
    assert sum(op.vector_ops for op in fused.ops) == sum(
        op.vector_ops for op in separate.ops) == 5 * n
    fused_cost = graph_costs(declared_case(), fused.ops, recipes=fused.recipes)
    separate_cost = graph_costs(declared_case(), separate.ops, recipes=separate.recipes)
    assert fused_cost["duration_ms"] == pytest.approx(0.000154)
    assert separate_cost["duration_ms"] == pytest.approx(0.00021)
    conditions = list(iter_conditions(activation_protocol()))
    single, burst = (bind_activation_condition(conditions[index]) for index in (0, 1))
    assert single.call.ops == burst.call.ops
    assert single.invocations == 1 and burst.invocations == 100
    assert single.target_identity["boundary"] == "npu_event_call" and (
        burst.target_identity["boundary"] == "npu_event_burst")
    changed = deepcopy(conditions[0])
    changed["samples"] = [{"duration_ms": 1e20, "split": "heldout"}]
    changed["metadata"] = {"context": {"hidden": 999999}, "fitted_duration_ms": 1e30}
    second = bind_activation_condition(changed)
    assert second.call.ops == single.call.ops and second.ledger["measured_values_consumed"] is False
    raw = declared_case()
    raw["machine"]["device"]["launch_us"] = 1000
    first_cost = call_costs(raw, single.call.ops, recipes=single.call.recipes)
    repeated = call_costs(raw, burst.call.ops, recipes=burst.call.recipes,
                          invocations=burst.invocations)
    assert repeated["duration_ms"] == pytest.approx(first_cost["duration_ms"] * 100)
    for field, value, match in (("api", ["another_kernel"], "api_or_layout"),
                                ("mode", "graph", "execution_differs")):
        changed = deepcopy(conditions[0])
        if field == "api":
            changed["identity"]["implementation"]["api"] = value
        else:
            changed["identity"]["execution"]["mode"] = value
        with pytest.raises(ModelingError, match=match):
            bind_activation_condition(changed)
def copy_case(*, units=4, active=1, tile=128, slots=1, read_units=None, rates=None):
    raw = hardware_case()
    raw["machine"]["device"].update({
        "compute_units": max(1, units // 2),
        "rates": {"memory": 1e15, "vector": units * 1e9, "mte2": 1000, "mte2_aic": 1000,
                  "mte2_aiv": units * 1e6, "mte3": units * 2e6, **(rates or {})},
        "resource_units": {"vector": units, "mte2": units, "mte2_aic": units,
                           "mte2_aiv": read_units or units, "mte3": units},
        "buffers": {"ub": 65536}, "buffer_units": {"ub": units}})
    raw["runtime"]["kernels"] = {"families": {"tensor_copy": {
        "kind": "aiv_tensor_copy", "tile_elements": tile, "slots": slots,
        "active_units": active}}}
    return parse_case(raw)
def tensor_ops(parsed, *, shape=(16, 24), narrow=12, output_dtype="bf16", concat=False):
    views = (RequestView("r", 1, 0, False, "decode"),)
    builder = Builder("test", parsed.model.layer_plans[0], 0, views, parsed.model, "decode")
    tensors, source = TensorLowering(builder), TensorValue(
        "", "bf16", TensorLayout.contiguous(shape))
    if concat:
        tensors.concatenate_flat(tuple(source.narrow(-1, start, narrow)
                                       for start in (0, narrow, 2 * narrow)), "concat")
    else:
        tensors.materialize(source.narrow(-1, 0, narrow), "copy",
                            reason="explicit_test_materialization", dtype=output_dtype)
    return views, builder.ops
def execute(parsed, views, ops):
    execution = Execution(parsed, CostModel(parsed), include_submission=False, trace_limit=10000)
    ready = execution.engine.submit("input", release=3)
    work = IterationWork(tuple(ops), IterationPlan(views), 1, 1, 1, None, "body", "copy_test")
    complete = execution.add(work, 0, 0, external_inputs={"": ready}).completion
    execution.engine.advance(horizon=2)
    assert not any(execution.engine.serviced.values())
    execution.engine.advance(until=(complete,))
    engine = execution.engine
    assert not engine._tasks and not engine._promises and not engine.allocations and all(row["start_ms"] >= 3 for row in engine.trace if row["demand"]) and engine.serviced["mte2@r0:s0"] == engine.serviced["mte2_aic@r0:s0"] == 0
    return engine, complete.time - 3
def test_copy_service_conserves_payload_slots_and_owners():
    parsed = copy_case(units=64, active=1, tile=256)
    views, ops = tensor_ops(parsed)
    engine, duration = execute(parsed, views, ops)
    assert duration == pytest.approx(384 / 1000 + 384 / 2000)
    assert engine.serviced["memory@r0:s0"] == 768
    assert engine.serviced["mte2_aiv@r0:s0"] == engine.serviced["mte3@r0:s0"] == 384
    transfers = [e for e in engine.trace if e["demand"].get("mte2_aiv@r0:s0") or e["demand"].get("mte3@r0:s0")]
    assert transfers[0]["end_ms"] <= transfers[1]["start_ms"] and all(p == (0,) for e in transfers for p in e["placement"].values())
    for active, slots in ((1, 1), (1, 2), (2, 1), (4, 2)):
        parsed = copy_case(active=active, tile=32, slots=slots)
        views, ops = tensor_ops(parsed, shape=(5, 27), narrow=19, output_dtype="fp32")
        engine, _ = execute(parsed, views, ops)
        assert engine.serviced["mte2_aiv@r0:s0"] == 95 * 2 and engine.serviced["mte3@r0:s0"] == 95 * 4
        assert engine.serviced["memory@r0:s0"] == 95 * 6 and engine.serviced["vector@r0:s0"] == 95
    controls = [execute(copy_case(tile=128, slots=slots, rates={"mte3": 4e6}), *tensor_ops(copy_case(tile=128, slots=slots, rates={"mte3": 4e6}), shape=(1, 512), narrow=512)) for slots in (1, 2)]
    assert controls[0][1] == pytest.approx(8 * 0.256) and controls[1][1] == pytest.approx(5 * 0.256)
    assert controls[0][0].serviced == controls[1][0].serviced and controls[1][0].peak_allocated["ub@r0:s0"] == 2 * 128 * 2
    parsed = copy_case(units=4, active=2, tile=128, read_units=2, rates={"mte3": 4e15})
    views, ops = tensor_ops(parsed, shape=(1, 256), narrow=256)
    engine, duration = execute(parsed, views, ops)
    assert duration == pytest.approx(512 / 2000, abs=1e-8)
    reads = [row for row in engine.trace if row["demand"].get("mte2_aiv@r0:s0")]
    assert len(reads) == 2 and all(row["placement"]["mte2_aiv@r0:s0"] == (0,) for row in reads)
    parsed = copy_case(active=2, tile=64, slots=2)
    views, ops = tensor_ops(parsed, shape=(16, 24), narrow=8, concat=True)
    engine, _ = execute(parsed, views, ops)
    assert engine.serviced["memory@r0:s0"] == 3 * 16 * 8 * 2 * 2
def test_copy_declarations_only_rescale_time():
    parsed = copy_case(active=2, tile=32, slots=2)
    views, ops = tensor_ops(parsed, shape=(16, 24), narrow=12)
    plain_engine, plain_duration = execute(parsed, views, ops)
    for minimum in (0, 10):
        die = replace(parsed.machine.dies[0], kernel_invocations=(("vector", InvocationTiming(
            startup_ms=2, minimum_ms=minimum)),))
        engine, duration = execute(replace(parsed, machine=replace(parsed.machine, dies=(die,))),
                                   views, ops)
        assert duration == pytest.approx(max(minimum, plain_duration + 2)) and (
            engine.serviced == plain_engine.serviced)
def norm_case(name, *, quant=False, bias=False, gamma=False):
    raw = json.loads((ROOT / "examples/cases" / (name + ".json")).read_text())
    raw["runtime"]["lowering_passes"] = [
        "activation_quantization", *(("norm_quant",) if quant else ()), "residual_norm"]
    selector = "add_rmsnorm_quant" if quant else "add_rmsnorm"
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[selector] = {
        "implementation": "vllm_ascend_native_norm",
        "options": {"has_bias": bias, "external_gamma_shift": gamma}}
    return parse_case(raw)
def norm_ops(case, tokens=1):
    return target_operators(case, phase="decode", tokens=tokens, context=16)
def norm_row(capture, kind, rows=1):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    path = (Path(mindie_profiler.__file__).resolve().parents[1] / "tests/fixtures"
            / "fusion_contracts" / (capture + ".csv"))
    return next(row for row in kernel_rows(path)
                if row.kind == kind and row.input_shapes and row.input_shapes[0][0] == rows)
def kimi_mx_case():
    raw = json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text())
    base = parse_case(raw)
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "norm_quant"]
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[
        "rmsnorm_quant"] = "vllm_ascend_native_norm"
    selected = parse_case(raw)
    quant = replace(selected.model.quant, weight_format="fp8_e4m3",
                    expert_weight_format="fp8_e4m3", weight_block=(32, 32),
                    activation_format="fp8_e4m3", activation_group_size=32,
                    activation_scale_format="fp8_e8m0", activation_scale_pack=2, exclusions=())
    return base, replace(selected, model=replace(selected.model, quant=quant))
@pytest.mark.requires_profiler
def test_observed_native_norm_contracts_and_fusions(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract
    plain, biased = norm_case("glm53-a3-w4a8"), norm_case("glm53-a3-w4a8", bias=True)
    a = next(op for op in norm_ops(plain) if op.family == "add_rmsnorm")
    b = next(op for op in norm_ops(biased) if op.op_id == a.op_id)
    rows, width = b.local_shape
    captured = observed_contract(norm_row("a3_glm53_decode", "AddRmsNormBias"))
    assert b.mte2_bytes - a.mte2_bytes == width * 2
    assert predicted_contract(b.to_dict()) == captured != predicted_contract(a.to_dict())
    shift_ops = norm_ops(norm_case("qwen35-a3-tp4", gamma=True))
    shifted = next(op for op in shift_ops if op.family == "add_rmsnorm")
    gamma = next(op for op in shift_ops if op.op_id == shifted.op_id + ".gamma_shift")
    assert gamma.op_id in shifted.deps and gamma.vector_ops == shifted.local_shape[-1]
    int8 = next(op for op in norm_ops(norm_case("glm53-a3-w4a8", quant=True, bias=True))
                if op.family == "add_rmsnorm_quant")
    assert predicted_contract(int8.to_dict()) == observed_contract(
        norm_row("a3_glm53_decode", "AddRmsNormDynamicQuant"))
    assert "residual" in int8.returns and "rstd" not in int8.returns
    base, selected = kimi_mx_case()
    mx = next(op for op in norm_ops(selected)
              if op.family == "rmsnorm_quant" and op.local_shape == (1, 3584))
    assert predicted_contract(mx.to_dict()) == observed_contract(
        norm_row("a5_kimi_k3", "RmsNormDynamicMxQuant"))
    rows, width = mx.local_shape
    declaration, members = next(
        (d, members) for d, members in _groups(norm_ops(selected))
        if d["role"] == "situ_quant" and d["tensor_view"]["width"] == 768)
    assert subgraph_contract(declaration) == observed_contract(norm_row("a5_kimi_k3", "SituMxQuant"))
    kernels = {**selected.runtime.kernels, "lowerings": {
        **selected.runtime.kernels.get("lowerings", {}),
        "native_fusion": {"implementation": "declared_native_fusion", "options": {
            "roles": ["situ_quant"], "traffic_profile": "situ_mx_quant_arch35_2a9bd1b0"}}}}
    fused_case = replace(selected, runtime=replace(selected.runtime, kernels=kernels))
    physical = lower_native_fusions(members, fused_case.runtime)
    activation = next(op for op in members if op.family == "situ")
    assert len(members) == 2 and len(physical) == 1
    assert physical[0].hbm_bytes == sum(op.hbm_bytes for op in members) - 2 * activation.mte3_bytes
    ids = {op.op_id for op in members}
    external = {dep: 0 for op in members for dep in op.deps if dep not in ids}
    before = graph_costs(selected, members, external_inputs=external, isolated=False)
    after = graph_costs(fused_case, physical, external_inputs=external, isolated=False)
    assert (before["execution"]["work"]["hbm_bytes"] - after["execution"]["work"]["hbm_bytes"]
            == pytest.approx(2 * activation.mte3_bytes))
def joint_case(scale=1):
    raw = json.loads((ROOT / "examples/source-families/glm-routing-state-control.json").read_text())
    arch = deepcopy(PACKET["compute_arch"])
    for service in arch["instruction_services"].values():
        service["per_owner_rate"] *= scale
    raw["machine"].setdefault("device", {})["compute_arch"] = arch
    raw["runtime"]["kernels"]["lowerings"]["conv_state_copy"]["options"][
        "joint_program_service"] = PACKET["joint_program_service"]
    return raw
def joint_copy(raw, batch=16, direction="gather"):
    return next(op for op in target_graphs(raw, scope="body", phase="decode", tokens=4,
                                          context=4096, sequences=batch)[0]
                if op.family == "conv_state_copy"
                and dict(op.metadata)["direction"] == direction)
@pytest.mark.parametrize("direction", ("gather", "scatter"))
def test_joint_copy_service_conserves_owners_and_call_time(direction):
    amounts = []
    for scale in (.5, 1, 2):
        raw = joint_case(scale)
        op = joint_copy(raw, direction=direction)
        row = operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]
        info = row["operator"]["metadata"]["program_execution"]
        expected = max(p["owner_busy_ns"] + 18 * p["program_busy_ns"]
                       for p in PACKET["joint_program_service"]["directions"][direction].values())
        assert row["duration_ms"] == pytest.approx(expected / 1e6 / scale)
        assert sum(info["source_programs_per_owner"]) == info["source_programs"] == 16 * 72
        assert info["payload_read_bytes"] == info["payload_write_bytes"] == 16 * 6 * 3072 * 2
        amounts.append(row["declared_service"])
    assert amounts[0] == amounts[1] == amounts[2]
    raw = joint_case()
    op = joint_copy(raw)
    raw["runtime"]["kernels"]["service_regimes"] = [{
        "id": "inclusive-copy-control", "match": {"recipe": op.recipe},
        "response": {"fixed_us": 20}}]
    assert operator_costs(raw, (op,))["operators"][0]["duration_ms"] == pytest.approx(.020)
    joint = SourceProgramService(PACKET["joint_program_service"]["directions"]["gather"])
    roles = tuple(joint.work(2).resources)
    rates = {r: 64e6 for r in roles}
    context = RecipeContext(None, {}, {r: r for r in roles}, rates, 32,
                            resource_units={r: 64 for r in roles})
    expected = max(joint.work(2).resources.values()) / 1e6
    for owners, multiplier in (((0,), 1), ((0, 1), 1), ((0, 0), 2)):
        engine = Engine([Resource(r, rates[r], units=64) for r in roles])
        ready = engine.submit("late", latency=7)
        done = [KernelProgram((Stage("reserve"), joint.stage(context, 2, owner, 64))).start(
            engine, f"call{i}", ready) for i, owner in enumerate(owners)]
        engine.advance(until=tuple(done), first=False)
        assert max(e.time for e in done) - 7 == pytest.approx(expected * multiplier)
    info = operator_costs(joint_case(), (replace(joint_copy(joint_case()),
                                                local_shape=(16, 6, 3071)),)
                          )["operators"][0]["operator"]["metadata"]["program_execution"]
    assert not info["joint_pipe_busy_ns"] and info["payload_read_bytes"] == 16 * 6 * 3071 * 2
def quant_case(*, source=True, dtype=None):
    raw = declared_case(prompt=3, output=2)
    raw["model"].update(layers=1, quant=asdict(quant_preset("w8a8_int8")))
    if dtype:
        raw["model"]["activation_dtype"] = dtype
    device = raw["machine"]["device"]
    device["compute_peaks"]["int8"] = 1e9
    device["rates"]["shared_memory"] = 1e9
    device["service_modes"] = {role: {"resource": resource, "rate": 1e9}
                               for role, resource in cuda_quant.SERVICE_ROLES.items()}
    if source:
        raw["runtime"]["kernels"] = {"lowerings": {"quantize": cuda_quant.IMPLEMENTATION}}
    return raw
def test_quant_producers_scales_and_rejections():
    analytic, source = quant_case(source=False), quant_case()
    a_ops = {op.op_id: op for op in target_operators(analytic, phase="prefill", tokens=3)}
    s_ops = {op.op_id: op for op in target_operators(source, phase="prefill", tokens=3)}
    quantized = [op for op in s_ops.values() if op.family == "quantize"]
    for op in quantized:
        original, (rows, width) = a_ops[op.op_id], op.local_shape
        assert op.plan_required and op.returns == ("activation_scales",) and (
            op.mte2_bytes == 2 * original.mte2_bytes)
        assert op.mte3_bytes == original.mte3_bytes == rows * width + rows * 4
        consumers = [consumer for consumer in s_ops.values() if op.op_id in consumer.deps]
        assert consumers and all(dict(consumer.metadata)["activation_scale_bytes"] == rows * 4
                                 for consumer in consumers)
    assert predict(source)["parameters"]["calibration"] == "none"
    raw = quant_case()
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "norm_quant", "residual_norm"]
    graphs = target_graphs(raw, phase="prefill", tokens=3)
    quantized = [op for op in graphs[0] if op.family == "quantize"]
    assert quantized and not any(op.family == "rmsnorm_quant" for op in graphs[0])
    report = graph_costs(raw, graphs, input_ready_ms=5)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    for dtype in ("fp16", "fp32"):
        raw = quant_case(source=False, dtype=dtype)
        ops = {op.op_id: op for op in target_operators(raw, phase="prefill", tokens=3)}
        assert dict(ops["s0.l0.gate_up"].metadata)["output_format"] == "bf16" and ops["s0.l0.swiglu"].dtype == dtype
        raw["runtime"]["kernels"] = {"lowerings": {"quantize": cuda_quant.IMPLEMENTATION}}
        with pytest.raises(ModelingError, match="requires_bf16_producer_chain"):
            target_operators(raw, phase="prefill", tokens=3)
    raw = quant_case()
    raw["model"]["quant"]["activation_group_size"] = 16
    with pytest.raises(ModelingError, match="integer_per_row_int8_fp32_scale"):
        target_operators(raw, phase="prefill", tokens=3)
    del raw["runtime"]["kernels"]
    assert any(len(dict(op.metadata)["activation_scale_shape"]) == 2
               for op in target_operators(raw, phase="prefill", tokens=3)
               if op.family == "quantize")
    raw = quant_case()
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32,
                        expert_routing_histogram=[.8, .1, .05, .05])
    raw["runtime"].update(tp=2, ep=2)
    raw["machine"]["num_devices"] = 2
    raw["machine"]["links"][0]["members"] = [0, 1]
    with pytest.raises(ModelingError, match="integer_per_row_int8_fp32_scale"):
        target_operators(raw, phase="prefill", tokens=3)
def sort_machine(sku="Ascend950DT_9581", clock=None, owners=4):
    arch = {"presets": ["ascend_sort32_spec"]}
    if clock is not None:
        arch["clock_hz"] = clock
    return {"backend": "ascend", "sku": sku, "num_devices": 1,
            "device": {"compute_arch": arch, "resource_units": {"vector": owners}}}
def sort_execute(device, owners=(0,), vector_work=0):
    rate, units = dict(device.rates)["vector"] / 1000, device.units("vector")
    engine = Engine([Resource("vector", rate, units=units)])
    context = RecipeContext(None, {}, {"vector": "vector"}, {"vector": rate},
                            device.compute_units, resource_units={"vector": units},
                            service_modes=dict(device.service_modes))
    work = instructions.InstructionInventory(
        {"VBitSort32": instructions.InstructionCount(1600, 51200)}).capacity_work(
            instructions.CONTROL_INSTRUCTIONS)
    priced, unpriced = instructions.PrimitiveInventory(work.resources).partition(
        Geometry.from_devices((device,)).resource_units)
    demand = context.bind(priced)
    stages = [Stage(str(index), demand, placement=context.owned(owner, units, ("vbitsort32",)))
              for index, owner in enumerate(owners)]
    if vector_work:
        stages.append(Stage("other_vector", context.bind(Work(resources={"vector": vector_work})),
                            placement=context.owned(0, units, ("vector",))))
    done = KernelProgram(tuple(stages)).start(engine, "sort", engine.submit("input", latency=7))
    engine.advance(until=(done,))
    assert engine.serviced["vector"] == pytest.approx(
        len(owners) * demand["vector"] + vector_work) and work.resources == {"vbitsort32": 1600}
    return done.time - 7
def test_sort_capacity_survives_clock_and_supply_roundtrip():
    for sku, clock in SORT_SKUS:
        baseline = Hardware.from_mapping({"backend": "ascend", "sku": sku}).dies[0]
        assert "vbitsort32" not in Geometry.from_devices((baseline,)).resource_units
        device = Hardware.from_mapping(sort_machine(sku)).dies[0]
        expected = 1600 * 16 / clock * 1000
        assert sort_execute(device) == pytest.approx(expected)
        assert sort_execute(device, (0, 0)) == pytest.approx(2 * expected)
    for owners in (2, 8):
        for scale in (.5, 1, 2):
            hardware = Hardware.from_mapping(sort_machine(clock=1.6e9 * scale, owners=owners))
            assert sort_execute(hardware.dies[0]) == pytest.approx(.016 / scale) and (
                Hardware.from_mapping(hardware.to_dict()) == hardware)
    raw = sort_machine(clock=1.6e9)
    raw["device"]["compute_arch"]["instruction_services"] = {"vbitsort32": {"per_owner_rate": 2e8}}
    assert sort_execute(Hardware.from_mapping(raw).dies[0]) == pytest.approx(.008)
    raw = sort_machine(clock=1.6e9)
    raw["device"]["service_modes"] = {"vbitsort32": {"rate": 4 * 4e8}}
    explicit = Hardware.from_mapping(raw)
    assert sort_execute(explicit.dies[0]) == pytest.approx(.004)
    raw = sort_machine(clock=1.6e9)
    raw["device"]["service_modes"] = {"vbitsort32": {"resource": "vector"}}
    raw["devices"] = [{"rates": {"vector": 2e12}}]
    hardware = Hardware.from_mapping(raw)
    assert sort_execute(hardware.dies[0]) == pytest.approx(.016)
    raw = {"device": {"memory_bytes": 1000, "compute_peaks": {"bf16": 1000},
                      "rates": {"hbm": 1000, "vector": 1e12, "mte2_aiv": 2e12},
                      "resource_units": {"vector": 4, "mte2_aiv": 4},
                      "compute_arch": {"instruction_services": {"counted_service": {
                          "unit": "blocks", "owner_axis": "vector", "resource": "vector",
                          "cycles_per_unit": 16, "clock_hz": 1.6e9}}},
                      "service_modes": {"counted_service": {"resource": "mte2_aiv"}}}}
    restored = Hardware.from_mapping(Hardware.from_mapping(raw).to_dict())
    device = restored.dies[0]
    rates = {role: value / 1000 for role, value in device.rates}
    engine = Engine([Resource(role, rate, units=device.units(role))
                     for role, rate in rates.items()])
    context = RecipeContext(None, {}, {r: r for r in rates}, rates, device.compute_units,
                            resource_units=dict(device.resource_units),
                            service_modes=dict(device.service_modes))
    demand = context.bind(Work(resources={"counted_service": 1600}))
    first = engine.submit("counted", demand, placement=context.owned(0, 4, ("counted_service",)))
    second = engine.submit("copy_competitor", {"mte2_aiv": demand["mte2_aiv"]},
                           placement={"mte2_aiv": (0,)})
    engine.advance(until=(first, second))
    assert first.time == second.time == pytest.approx(.032)
def test_dynamic_quant_engine_program_covers_tail_rows():
    for row_len, mode in ((1234, "full_db"), (200000, "large")):
        report = dynamic_quant_schedule.quant_timeline(
            rates={"vector": 1e6, "mte2": 1e6, "mte3": 1e6, "hbm": 1e6},
            units={"vector": 4, "mte2": 4, "mte3": 4}, rows=7, row_len=row_len,
            source_bytes=2, y_bytes=1, scale_bytes=4, ub_bytes=64 * 1024, aiv=4)
        assert report["mode"] == mode and report["launched"] == 4 and (
            report["rows_per_head"], report["rows_per_tail"]) == (2, 1)
        assert report["mte2_bytes"] + report["mte3_bytes"] == report["hbm_bytes"] > 0
