"""Native convolution reuses the stateful program and real caller payloads."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.catalog.weights import stage_weight_bytes, stage_weight_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.prediction.gdn_decode_programs import CONV, conv_plan
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]


def case(batch, *, native=True):
    path = ROOT / ('examples/cases/high-concurrency-20261030/kimi-source-paged-host/'
                   f'a5_kimi_tp8_e16_bf16_c{batch}_p4k_source.json')
    raw = json.loads(path.read_text())
    if native:
        raw['runtime']['kernels']['lowerings']['causal_conv_update'] = CONV
    return raw


def graph(raw, batch):
    return target_graphs(raw, scope='body', phase='decode', tokens=1,
                         context=4096, sequences=batch, apply_body_defaults=True)[0]


@pytest.mark.parametrize('batch', [1, 4, 32])
def test_complete_frontend_binds_one_active_stateful_kernel_per_kda_layer(batch):
    raw = case(batch)
    ops = graph(raw, batch)
    convs = [op for op in ops if op.recipe == CONV]
    assert len(convs) == 69
    for conv in convs:
        meta = dict(conv.metadata)
        assert conv.local_shape == (batch, 4608)
        assert meta['sequence_tokens'] == (1,) * batch
        assert meta['state_write_bytes'] == batch * 3 * 4608 * 2
        assert (meta['state_dtype'], meta['bias'], meta['speculative']) == ('bf16', False, False)
        assert conv.source_boundary == 'kernel'
        assert conv.deps == (f's0.l{conv.layer}.convolution_input_pack',)
        core = next(op for op in ops if op.layer == conv.layer and op.recipe == 'vllm_ascend_recurrent_kda_arch35')
        assert conv.op_id in core.deps
        assert dict(core.metadata)['state_dtype'] == 'fp32'
    assert not any(op.recipe == 'short_convolution' for op in ops)
    # One source launch replaces the old envelope, not a parallel second call.
    result = operator_costs(raw, [convs[0]], input_ready_ms=9)['operators'][0]
    assert not result['service_before_input']
    assert result['live_tasks'] == 0
    assert all(value == 0 for value in result['remaining_buffers'].values())
    actual = result['operator']
    assert actual['mte3_bytes'] == batch * 4 * 4608 * 2  # output + W-1 state
    assert actual['hbm_bytes'] == batch * 12 * 4608 * 2  # 8 reads + 4 writes


@pytest.mark.parametrize('batch', [1, 4])
def test_packed_projection_is_a_view_or_exact_selected_payload_copy(batch):
    parsed = parse_case(case(batch))
    parsed = replace(parsed, model=replace(parsed.model, layer_plans=tuple(
        replace(layer, packed_input_projection=True) if layer.attention == 'kda' else layer
        for layer in parsed.model.layer_plans)))
    ops = graph(parsed, batch)
    by_id = {op.op_id: op for op in ops}
    conv = by_id['s0.l0.short_convolution']
    assert not any(op.recipe == 'convolution_input_pack' for op in ops)
    if batch == 1:
        assert conv.deps == ('s0.l0.linear_qkvbfg_a',)
        assert 's0.l0.convolution_input_contiguous' not in by_id
    else:
        copied = by_id['s0.l0.convolution_input_contiguous']
        assert conv.deps == (copied.op_id,)
        assert copied.mte2_bytes == copied.mte3_bytes == batch * 4608 * 2
        assert dict(copied.metadata)['source_layout']['shape'] == (batch, 4608)
    assert by_id['s0.l0.linear_qkvbfg_a'].local_shape[0] == batch


def test_prepared_weights_are_resident_not_source_checkpoint_or_per_call_cast():
    base = parse_case(case(1, native=False))
    selected = parse_case(case(1))
    prepared = [term for terms in stage_weight_terms(selected.model, selected.runtime.parallel)
                for term in terms if 'kda_prepared_conv' in term.name]
    assert len(prepared) == 69
    extra = 69 * 4 * 4608 * 2
    old = sum(stage_weight_bytes(base.model, base.runtime.parallel))
    new = sum(stage_weight_bytes(selected.model, selected.runtime.parallel))
    assert new - old == extra
    assert all(not term.source_owned and term.runtime_dtype == 'bf16' for term in prepared)
    assert sum(term_bytes(term, selected.model.quant, selected.runtime.parallel, source=True)
               for term in prepared) == 0
    # Removing a selector from an already parsed case removes derived storage.
    runtime = replace(selected.runtime, kernels=deepcopy(dict(selected.runtime.kernels)))
    runtime.kernels['lowerings'].pop('causal_conv_update')
    restored = parse_case(replace(selected, runtime=runtime))
    assert sum(stage_weight_bytes(restored.model, restored.runtime.parallel)) == old


def test_source_state_write_and_row_pack_conserve_engine_service_after_late_input():
    raw = case(4)
    ops = graph(raw, 4)
    wanted = [op for op in ops if op.layer == 0 and op.recipe in ('convolution_input_pack', CONV)]
    external = {dep: 11 for op in wanted for dep in op.deps
                if dep not in {item.op_id for item in wanted}}
    result = graph_costs(raw, wanted, external_inputs=external)
    assert min(row['ready_ms'] for row in result['timeline']) >= 11
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert all(value == 0 for value in result['execution']['live_allocated'].values())
    conv = next(op for op in wanted if op.recipe == CONV)
    geometry = Geometry.from_devices(parse_case(raw).machine.dies)
    native = conv_plan(conv, geometry).work.resources
    expected = native['hbm'] + 2 * 4 * 4608 * 2
    assert result['execution']['work']['hbm_bytes'] == expected


def test_unselected_and_prefill_callers_keep_existing_semantic_convolution():
    raw = case(1, native=False)
    assert sum(op.recipe == 'short_convolution' for op in graph(raw, 1)) == 69
    raw = case(1)
    # Only the convolution update selector is relevant to this prefill control.
    raw['runtime']['kernels']['lowerings'].pop('paged_mla_attention')
    raw['runtime']['kernels']['lowerings'].pop('mla_head_padding')
    raw['runtime']['kernels']['lowerings'].pop('kda')
    prefill = target_graphs(raw, scope='body', phase='prefill', tokens=8, sequences=1,
                            context=0, apply_body_defaults=True)[0]
    assert not any(op.recipe == CONV for op in prefill)
    assert sum(op.recipe == 'short_convolution' for op in prefill) == 69


def test_prepared_weight_capacity_does_not_ceil_fractional_source_heads_as_channels():
    parsed = parse_case(case(1))
    # Five heads / TP8 would execute one whole head/rank in the generic
    # builder, while channel-wise out_tp would hold only 5/8 of that weight.
    # The selected source asserts integral head shards and cannot claim this
    # mismatched prepared inventory as an exact source-compatible caller.
    model = replace(parsed.model, layer_plans=tuple(
        replace(layer, linear_heads=5, linear_key_heads=5, linear_value_heads=5)
        if layer.attention == 'kda' else layer for layer in parsed.model.layer_plans))
    with pytest.raises(ValueError, match='source_requires_integral_local_heads'):
        parse_case(replace(parsed, model=model))


@pytest.mark.parametrize('field', ['draft_proposal_tokens', 'draft_accepted_tokens', 'draft_advanced_tokens'])
def test_q1_with_explicit_speculative_state_is_not_silently_non_speculative(field):
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.schema.workload import RequestView
    parsed = parse_case(case(1))
    kernels = deepcopy(dict(parsed.runtime.kernels))
    kernels.pop('sequence_parallel')
    runtime = replace(parsed.runtime, kernels=kernels)
    view = RequestView('spec', 1, 4096, False, 'decode', **{field: 1})
    with pytest.raises(ValueError, match='causal_conv_update_requires_q1_non_speculative_decode'):
        Frontend().body(parsed.model, runtime, (view,), ready='input', tokens=1,
                        mode='decode', global_tokens=1)
