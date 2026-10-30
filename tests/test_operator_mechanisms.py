"""Physical contracts checked against raw kernels and independent work oracles."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.observations.kernels import kernel_rows
from mindie_sim.prediction.blocks import norm
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.prediction.pointwise import TiledPointwise
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import lower_implementations
from mindie_sim.schema.workload import RequestView
from mindie_sim.schema.tensors import TensorLayout
from test_prediction_system import case


CAPTURE = Path(__file__).parent / "fixtures/observations/operator-kernels.csv"


def quantized_case():
    parsed = parse_case(case())
    return replace(parsed, model=replace(parsed.model, quant=quant_preset("mxfp8")),
                   runtime=replace(parsed.runtime, lowering_passes=(
                       "activation_quantization", "norm_quant", "residual_norm")))


def matrix(name, deps, m=2, n=16, k=8, quant=None):
    return gemm_op(op_id=name, deps=deps, stage=0, layer=0, requests=("r",),
                   m=m, n=n, k=k, quant=quant or quant_preset("mxfp8"),
                   category="dense_mlp", phase="prefill")


def test_shared_quantized_input_is_produced_once_with_payload_and_scales():
    parsed = quantized_case()
    ops = lower_implementations([matrix("a", ("input",)), matrix("b", ("input",), n=32)],
                                parsed.runtime, "bf16")
    quant, a, b = ops
    assert a.deps == b.deps == (quant.op_id,)
    assert quant.mte2_bytes == 2 * 8 * 2
    assert quant.mte3_bytes == 2 * 8 + 2 * 2  # paired E8M0 scales include the tail
    assert a.mte2_bytes == quant.mte3_bytes + 16 * 8 + 16
    integer = matrix("integer", ("input",), quant=quant_preset("w8a8_int8"))
    q, integer = lower_implementations([integer], parsed.runtime, "bf16")
    assert q.mte3_bytes == 2 * 8 + 2 * 4  # one FP32 scale per row
    assert integer.dtype == "int8" and dict(integer.metadata)["output_format"] == "bf16"


def test_activation_grouping_is_independent_of_weight_block_geometry():
    quant = replace(quant_preset("mxfp8"), weight_block=(128, 128), scale_format="fp32",
                    scale_bytes=4, activation_group_size=64, activation_scale_format="fp32",
                    activation_scale_pack=1)
    op = matrix("block", (), m=2, n=128, k=129, quant=quant)
    meta = dict(op.metadata)
    assert meta["packed_weight_bytes"] == 128 * 129 + 2 * 4
    assert meta["activation_scale_shape"] == (2, 3)
    assert op.mte2_bytes == meta["packed_weight_bytes"] + 2 * 129 + 2 * 3 * 4


def test_fusion_preserves_live_residual_and_eliminates_only_intermediate_traffic():
    parsed = quantized_case()
    residual = vector_op(op_id="residual", deps=("x", "saved"), stage=0, layer=0,
                         requests=("r",), elements=16, ops_per_element=1, dtype="bf16",
                         family="elementwise", phase="prefill", recipe="residual_add",
                         read_bytes_override=64, write_bytes_override=32)
    normalized = norm(name="norm", deps=("residual",), stage=0, layer=0, requests=("r",),
                      tokens=2, width=8, model=parsed.model, mode="prefill")
    projection = matrix("projection", ("norm",))
    consumer = replace(residual, op_id="later_add", deps=("projection", "residual"))
    ops = lower_implementations([residual, normalized, projection, consumer], parsed.runtime, "bf16")
    fused, projection, consumer = ops
    assert fused.mte2_bytes == 64 + 8 * 2
    assert fused.mte3_bytes == 32 + 16 + 4
    assert consumer.deps == (projection.op_id, fused.op_id)
    assert fused.returns == ("activation_scales", "residual")
    assert fused.deps == ("x", "saved")


def test_fused_norm_quant_matches_captured_tensor_payloads(profiler_api):
    parsed = quantized_case()
    residual = vector_op(op_id="residual", deps=("x", "saved"), stage=0, layer=0,
                         requests=("r",), elements=16, ops_per_element=1, dtype="bf16",
                         family="elementwise", phase="prefill", recipe="residual_add",
                         read_bytes_override=64, write_bytes_override=32)
    # The payload and scale tensor contracts match a real fused-kernel row.
    actual = next(row for row in kernel_rows(CAPTURE) if row.kind == "AddRmsNormDynamicMxQuant")
    large = lower_implementations([
        replace(residual, mte2_bytes=16384 * 5120 * 4, mte3_bytes=16384 * 5120 * 2,
                local_shape=(16384 * 5120,)),
        norm(name="norm", deps=("residual",), stage=0, layer=0, requests=("r",),
             tokens=16384, width=5120, model=parsed.model, mode="prefill"),
        matrix("projection", ("norm",), m=16384, n=34816, k=5120),
    ], parsed.runtime, "bf16")[0]
    assert large.local_shape == actual.input_shapes[0]
    assert dict(large.metadata)["activation_scale_shape"] == actual.output_shapes[2]
    assert large.mte3_bytes == 16384 * 5120 * 3 + 16384 * 80 * 2


def test_unquantized_norm_consumer_prevents_removing_its_output():
    parsed = quantized_case()
    normalized = norm(name="norm", deps=("input",), stage=0, layer=0, requests=(),
                      tokens=2, width=8, model=parsed.model, mode="prefill")
    bf16 = replace(normalized, op_id="consumer", deps=("norm",), family="elementwise")
    ops = lower_implementations([normalized, matrix("projection", ("norm",)), bf16],
                                parsed.runtime, "bf16")
    assert ops[0] == normalized
    assert ops[-1].deps == ("norm",)
    assert sum(op.family == "quantize" for op in ops) == 1


def recurrent_case():
    parsed = parse_case(case())
    layer = replace(parsed.model.layer_plans[0], attention="gdn", linear_key_heads=1,
                    linear_value_heads=2, linear_heads=2, linear_key_head_dim=3,
                    linear_head_dim=2, conv_kernel=3, recurrent_state_dtype="fp32")
    return replace(parsed, model=replace(parsed.model, layers=1, layer_plans=(layer,)),
                   runtime=replace(parsed.runtime, pp_layer_partition=(1,)))


def test_decode_recurrence_uses_vector_service_and_retains_state_between_tokens():
    parsed = recurrent_case()
    ops, _ = Frontend().body(parsed.model, parsed.runtime,
                            (RequestView("r", 3, 20, False, "decode"),), mode="decode")
    scan = next(op for op in ops if op.family == "recurrent_vector")
    state = 2 * 3 * 2 * 4
    assert scan.flops == 0
    assert scan.vector_ops == 3 * 2 * (3 * 2 + (2 * 3 - 1) * 2 + 2 * 2
                                     + 2 * 3 * 2 + (2 * 3 - 1) * 2)
    assert scan.mte2_bytes == state + 3 * 2 * ((2 * 3 + 2) * 2 + 6)
    assert scan.mte3_bytes == 3 * (state + 2 * 2 * 2)
    fast = operator_costs(parsed, [scan])
    device = parsed.machine.dies[0]
    slow_vector = replace(device, rates=tuple((key, rate / 8 if key == "vector" else rate)
                                             for key, rate in device.rates))
    slow_cube = replace(device, compute_peaks=tuple((key, peak / 8) for key, peak in device.compute_peaks))
    slow = operator_costs(replace(parsed, machine=replace(parsed.machine, dies=(slow_vector,))), [scan])
    cube = operator_costs(replace(parsed, machine=replace(parsed.machine, dies=(slow_cube,))), [scan])
    assert slow["operator_sum_ms"] > fast["operator_sum_ms"] == cube["operator_sum_ms"]
    serviced = fast["operators"][0]["serviced"]
    assert serviced["vector@r0:s0"] == scan.vector_ops
    assert serviced["memory@r0:s0"] == scan.hbm_bytes


def test_captured_decode_recurrence_runs_on_vector_core(profiler_api):
    raw = next(row for row in kernel_rows(CAPTURE) if row.kind == "RecurrentGatedDeltaRule")
    assert raw.core == "AI_VECTOR_CORE" and raw.counters_us["aic_mac_time(us)"] == 0


def test_gdn_head_gates_and_output_channel_gate_have_different_producers():
    parsed = recurrent_case()
    ops, _ = Frontend().body(parsed.model, parsed.runtime,
                            (RequestView("r", 3, 20, False, "decode"),), mode="decode")
    gates = next(op for op in ops if op.family == "gdn_gating")
    output = next(op for op in ops if op.recipe == "linear_output_norm_gate")
    projection = next(op for op in ops if op.recipe == "linear_qkvz")
    assert gates.local_shape == (3, 2)
    assert gates.mte3_bytes == 3 * 2 * (4 + 2)
    copied_gate = next(op for op in ops if op.recipe == "output_gate_reshape")
    assert copied_gate.deps == (projection.op_id,)
    assert copied_gate.op_id in output.deps and gates.op_id not in output.deps
    assert output.mte2_bytes == (2 * 3 * 2 * 2 + 2) * 2


def test_tail_wave_transfer_supply_is_limited_by_participating_copy_units():
    op = matrix("m", (), m=1, n=4, k=2)
    names = {role: role for role in ("cube", "hbm", "mte2", "mte3")}
    rates = {"cube": 10000, "hbm": 100000, "mte2": 100, "mte3": 100}
    times = []
    for distributed in (False, True):
        engine = Engine([Resource(key, value, units=4) for key, value in rates.items()])
        context = RecipeContext(op, {"cube": op.flops}, names, rates, 4)
        event = TiledMatmul(1, 2, 2, distributed_transfers=distributed).lower(context).start(
            engine, "matrix", engine.submit("input"))
        engine.advance(until=(event,))
        times.append(event.time)
        assert engine.serviced["hbm"] == op.hbm_bytes
    # Two output tiles occupy two of four copy units. Both need the activation
    # and its scales in private buffers, despite their ideal shared HBM reuse.
    weights = dict(op.metadata)["packed_weight_bytes"]
    local_copies = 2 * (op.mte2_bytes - weights) + weights + op.mte3_bytes
    assert times[1] - times[0] == pytest.approx(local_copies / 100)


def test_raw_epoch_precision_and_unknown_shapes_survive_import(profiler_api):
    rows = list(kernel_rows(CAPTURE))
    matrix_row = next(row for row in rows if row.input_shapes == ((16384, 5120), (16384, 5120)))
    assert matrix_row.start_ns == 1792684744878053232
    assert matrix_row.end_ns - matrix_row.start_ns == 6381070
    unknown = next(row for row in rows if row.kind == "fused_gdn_gating_kernel" and row.input_shapes is None)
    assert unknown.duration_ns > 0


def test_strided_reshape_preserves_actual_addresses_and_detects_required_copies():
    source = np.arange(3 * 8 * 4).reshape(3, 8, 4)
    dense = TensorLayout.contiguous(source.shape)
    for actual, layout in (
        (source, dense),
        (source[:, 2:6], dense.narrow(1, 2, 4)),
        (source.transpose(1, 0, 2), dense.permute((1, 0, 2))),
        (source[:1, 2:6], dense.narrow(0, 0, 1).narrow(1, 2, 4)),
    ):
        for shape in (actual.shape, (actual.size,), (actual.size // 4, 4),
                      (1, actual.size // 4, 2, 2, 1)):
            expected = actual.reshape(shape)
            view = layout.reshape_view(shape)
            assert (view is not None) == np.shares_memory(actual, expected)
            if view:
                addresses = [view.offset + sum(i * stride for i, stride in zip(index, view.strides))
                             for index in np.ndindex(shape)]
                assert np.array_equal(source.ravel()[addresses], expected.ravel())
    assert dense.narrow(1, 2, 4).reshape_view((3, 4, 2, 2)) is not None
    assert dense.narrow(1, 2, 4).reshape_view((12, 4)) is None


def test_layout_contracts_materialize_only_selected_payloads_and_skip_single_row_views():
    parsed = recurrent_case()
    runtime = replace(parsed.runtime, tensor_layouts={
        "gdn_gating": {"beta": "contiguous", "alpha": "contiguous"},
        "short_convolution": {"input": "contiguous"},
    })
    for tokens in (1, 3):
        ops, _ = Frontend().body(parsed.model, runtime,
                                (RequestView("r", tokens, 0, False, "prefill"),))
        copies = {op.recipe: op for op in ops if op.family == "tensor_copy"}
        if tokens == 1:
            assert not copies
            continue
        assert copies["gate_alpha_contiguous"].mte2_bytes == tokens * 2 * 2
        assert copies["gate_beta_contiguous"].mte3_bytes == tokens * 2 * 2
        assert copies["convolution_input_contiguous"].mte2_bytes == tokens * (2 * 3 + 2 * 2) * 2
        assert copies["output_gate_reshape"].mte2_bytes == tokens * 2 * 2 * 2
        gate = next(op for op in ops if op.family == "gdn_gating")
        assert set(gate.deps) == {copies["gate_alpha_contiguous"].op_id,
                                 copies["gate_beta_contiguous"].op_id}
        for op in copies.values():
            assert op.hbm_bytes == op.mte2_bytes + op.mte3_bytes


def test_packed_projection_capture_requires_a_strided_slice_copy(profiler_api):
    # A real packed BA projection has two discontinuous [T,H] slices.
    raw = next(row for row in kernel_rows(CAPTURE) if row.kind == "Slice"
               and row.input_shapes[0] == (16384, 96))
    view = TensorLayout.contiguous(raw.input_shapes[0]).narrow(1, 48, 48)
    assert view.shape == raw.output_shapes[0] and not view.is_contiguous
    assert view.elements * 2 == 1572864


def test_final_normalization_and_selection_order_preserve_full_prefill_work():
    parsed = parse_case(case())
    views = (RequestView("prompt", 8, 0, True, "prefill"),
             RequestView("verify", 3, 16, True, "decode"))
    frontend = Frontend()
    ops, _ = frontend.logits(parsed.model, parsed.runtime, views, "body", mode="mixed")
    norm_op, selection, logits = ops[:3]
    assert norm_op.local_shape == (11, parsed.model.hidden_size)
    assert selection.local_shape == logits.local_shape[:1] + (parsed.model.hidden_size,)
    assert selection.local_shape[0] == 4
    assert selection.deps == (norm_op.op_id,) and logits.deps == (selection.op_id,)
    selected, _ = frontend.logits(parsed.model, replace(parsed.runtime, final_norm_scope="selected"),
                                  views, "body", mode="mixed")
    assert selected[0].family == "tensor_select"
    assert selected[1].local_shape == (4, parsed.model.hidden_size)
    assert selected[1].deps == (selected[0].op_id,)
    # A nonterminal prefill chunk still executes the model's final norm.
    unfinished = (replace(views[0], samples=False),)
    partial, _ = frontend.logits(parsed.model, parsed.runtime, unfinished, "body", mode="prefill")
    assert len(partial) == 1 and partial[0].local_shape[0] == 8


def test_batched_vector_recurrence_keeps_each_sequence_state_and_tail_length():
    parsed = recurrent_case()
    device = replace(parsed.machine.dies[0], resource_units=(("vector", 3),))
    parsed = replace(parsed, machine=replace(parsed.machine, dies=(device,)))
    views = (RequestView("a", 2, 16, False, "decode"), RequestView("b", 4, 32, False, "decode"))
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode")
    fused = [op for op in ops if op.family == "recurrent_vector"]
    assert len(fused) == 1
    assert dict(fused[0].metadata)["sequence_tokens"] == (2, 4)
    total_steps, state_bytes = 6 * 2, 3 * 2 * 4
    assert fused[0].mte2_bytes == 4 * state_bytes + total_steps * ((2 * 3 + 2) * 2 + 6)
    assert fused[0].mte3_bytes == total_steps * (state_bytes + 2 * 2)
    result = operator_costs(parsed, fused)
    assert result["operators"][0]["serviced"]["vector@r0:s0"] == 7 * total_steps * 3 * 2
    assert result["operators"][0]["serviced"]["memory@r0:s0"] == fused[0].hbm_bytes
    separate, _ = Frontend(attention_policy=AttentionPolicy(fused_batch=False)).body(
        parsed.model, parsed.runtime, views, mode="decode")
    scans = [op for op in separate if op.family == "recurrent_vector"]
    assert len(scans) == 2
    assert sum(op.hbm_bytes for op in scans) == fused[0].hbm_bytes


def test_scalar_output_path_retains_serial_unit_work_and_exact_masked_payloads():
    rates = {name: 1e20 for name in ("hbm", "mte2", "mte3", "vector")}
    rates["scalar_store"] = 64000
    names = {name: name for name in rates}
    recipe = TiledPointwise(
        4, ({"bytes_per_element": 4}, {"bytes_per_element": 2, "resource": "scalar_store",
                                     "bytes_per_work": 2}), parameter_bytes_per_col=8)
    for rows, cols in ((1, 7), (64, 48), (4096, 48), (4225, 33), (16384, 48)):
        op = vector_op(op_id="pointwise", deps=(), stage=0, layer=0, requests=(),
                       elements=rows * cols, ops_per_element=12, dtype="bf16",
                       family="pointwise", phase="prefill", recipe="tiled",
                       shape=(rows, cols))
        iterations = (rows + 4095) // 4096
        # Independent address enumeration verifies both the store count and the
        # longest worker chain, including partial rows, heads and idle workers.
        per_worker = []
        addresses = set()
        for worker in range(64):
            count = 0
            for row in range(worker * iterations * 64, (worker + 1) * iterations * 64):
                if row < rows:
                    addresses.update(range(row * cols, (row + 1) * cols))
                    count += cols
            per_worker.append(count)
        assert len(addresses) == rows * cols
        engine = Engine([Resource(key, value) for key, value in rates.items()])
        context = RecipeContext(op, {}, names, rates, 32,
                                resource_units={name: 64 for name in rates})
        complete = recipe.lower(context).start(engine, "pointwise", engine.submit("input"))
        engine.advance(until=(complete,))
        assert complete.time == pytest.approx(max(per_worker) / 1000)
        assert engine.serviced["scalar_store"] == len(addresses)
        assert engine.serviced["mte3"] == rows * cols * 4
        assert engine.serviced["mte2"] == rows * cols * 4 + 64 * iterations * cols * 8
        assert engine.serviced["hbm"] == engine.serviced["mte2"] + rows * cols * 6
        assert engine.serviced["vector"] == 64 * iterations * ((cols + 7) // 8) * 64 * 8 * 12
