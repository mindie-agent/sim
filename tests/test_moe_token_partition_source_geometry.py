"""MC2WithTP source geometry: TP pad/split, one gather, unpad.

[Role]
  Pin the source-selected TP4 prep padding of hidden and router logits, the
  post-combine rank-local gather (4096 B -> 16384 B BF16 for q1/TP4), the
  prefix-rank truth for q<TP, and the negative DP/alternate layouts.

[Invariants]
  * TP1 emits no pad/gather.  TP>1 with right_to_tp padding pads both hidden
    (BF16) and logits (FLOAT), views the local prefix, gathers once and trims.
  * q<TP has valid rows on rank 0 only; q divisible has no pad; uneven q pads
    the tail and keeps prefix valid rows.
  * DP != 1 or EP != TP is rejected by the source caller, not silently
    generalized to another MC2 layout.  No scatter/payload network is added.

[Side-Effect Policy] Immutable graph descriptions only; isolated homes.
"""
from types import SimpleNamespace

import pytest

from mindie_sim.prediction.moe_token_partition import KIND, select
from mindie_sim.schema.common import ModelingError
from test_moe_token_partition import case, graph


def _by_id(ops):
    return {op.op_id: op for op in ops}


def _select(**parallel):
    from dataclasses import replace
    base = dict(tp=4, ep=4, dp=1, etp=1, cp=1, pcp=1)
    raw = SimpleNamespace(kernels={
        "moe_token_partition": {"kind": KIND, "padding": "right_to_tp",
                                "padding_implementation": "pad_v3"}},
        moe_recipe="alltoall")
    plan = SimpleNamespace(scope="body", expert_latent_size=0)
    model = SimpleNamespace(hidden_size=2048)
    return select(raw, parallel=SimpleNamespace(**{**base, **parallel}),
                  plan=plan, model=model, tokens=1)


def test_tp1_emits_no_gather_or_padding():
    assert _select(tp=1, ep=1) is None
    raw = case(tp=1)
    raw['runtime']['kernels'].pop('native_moe_protocol', None)
    ops = graph(raw, rows=1)
    assert not any('mc2_tp_gather' in op.op_id or 'pad' in op.op_id
                   or 'token_view' in op.op_id for op in ops)


def test_q1_lt_tp_pads_hidden_and_logits_then_gathers_prefix_rank():
    raw = case(tp=4)
    raw['runtime']['kernels']['moe_token_partition'].update(
        padding='right_to_tp', padding_implementation='concat',
        logits_dtype='float32')
    ops = graph(raw, rows=1)
    by_id = _by_id(ops)
    hidden_pad = by_id['s0.l0.mc2_input_pad']
    logits_pad = by_id['s0.l0.mc2_logits_pad']
    assert hidden_pad.local_shape == (4, 32) and hidden_pad.dtype == 'bf16'
    assert logits_pad.local_shape == (4, 16) and logits_pad.dtype == 'float32'
    assert hidden_pad.mte2_bytes == 4 * 32 * 2
    assert logits_pad.mte2_bytes == 4 * 16 * 4
    assert hidden_pad.hbm_bytes == 2 * 4 * 32 * 2
    assert logits_pad.hbm_bytes == 2 * 4 * 16 * 4
    assert hidden_pad.metadata and 'ConcatD' in str(hidden_pad.metadata)
    assert logits_pad.metadata and 'ConcatD' in str(logits_pad.metadata)
    hidden_view = by_id['s0.l0.mc2_input_view']
    logits_view = by_id['s0.l0.mc2_logits_view']
    assert hidden_view.local_shape == (1, 32)
    assert logits_view.local_shape == (1, 16)
    meta = dict(by_id['s0.l0.mc2_tp_gather'].metadata)
    assert meta['moe_token_partition']['valid_rows_per_rank'] == (1, 0, 0, 0)
    gather = by_id['s0.l0.mc2_tp_gather']
    assert gather.family == 'allgather'
    assert gather.local_shape == (4, 32)
    assert gather.link_bytes == 3 * 1 * 32 * 2
    trim = by_id['s0.l0.mc2_output_trim']
    assert trim.local_shape == (1, 32)
    assert trim.deps == ('s0.l0.mc2_tp_gather',)
    assert 's0.l0.mc2_output_trim' in by_id['s0.l0.join'].deps
    # No second scatter/payload network and one gather only.
    assert sum(1 for op in ops if op.family == 'allgather') == 1
    assert not any(op.op_id.endswith('.mc2_tp_gather.payload_network') for op in ops)


@pytest.mark.parametrize("rows,expected_allocated,expected_valid", (
    (8, 8, (2, 2, 2, 2)),
    (5, 8, (2, 2, 1, 0)),
    (4, 4, (1, 1, 1, 1)),
))
def test_divisible_and_uneven_q_keep_prefix_valid_rows(
        rows, expected_allocated, expected_valid):
    raw = case(tp=4)
    raw['runtime']['kernels']['moe_token_partition'].update(
        padding='right_to_tp', padding_implementation='concat')
    ops = graph(raw, rows=rows)
    by_id = _by_id(ops)
    meta = dict(by_id['s0.l0.mc2_tp_gather'].metadata)['moe_token_partition']
    assert meta['valid_rows_per_rank'] == expected_valid
    assert meta['allocated_rows'] == expected_allocated
    gather = by_id['s0.l0.mc2_tp_gather']
    assert gather.local_shape == (expected_allocated, 32)
    if expected_allocated != rows:
        assert 's0.l0.mc2_input_pad' in by_id
        assert by_id['s0.l0.mc2_input_pad'].local_shape == (expected_allocated, 32)
    else:
        assert 's0.l0.mc2_input_pad' not in by_id


@pytest.mark.parametrize("parallel", (
    {"dp": 2},
    {"ep": 8},
))
def test_dp_and_alternate_ep_layouts_are_rejected(parallel):
    with pytest.raises(ModelingError, match="dp1_ep_equals_tp"):
        _select(**parallel)


def test_tp_divisible_without_padding_still_gathers_once():
    raw = case(tp=4)
    raw['runtime']['kernels']['moe_token_partition']['padding'] = 'none'
    ops = graph(raw, rows=4)
    by_id = _by_id(ops)
    assert 's0.l0.mc2_input_pad' not in by_id
    assert by_id['s0.l0.mc2_tp_gather'].local_shape == (4, 32)


@pytest.mark.parametrize("rows", (1, 3, 5, 8))
def test_activation_capacity_does_not_mutate_active_expert_or_network_work(rows):
    base = case(tp=4)
    base['runtime']['kernels']['moe_token_partition'].update(
        padding='right_to_tp', padding_implementation='concat',
        logits_dtype='float32')
    capped = case(tp=4)
    capped['runtime']['kernels']['moe_token_partition'].update(
        padding='right_to_tp', padding_implementation='concat',
        logits_dtype='float32', activation_tensor_rows=32,
        activation_owners=4)
    base_ops = _by_id(graph(base, rows=rows))
    capped_ops = _by_id(graph(capped, rows=rows))
    base_act, capped_act = base_ops['s0.l0.expert_act'], capped_ops['s0.l0.expert_act']
    assert capped_act.local_shape[0] == 32
    assert dict(capped_act.metadata)['activation_owners'] == 4
    assert dict(capped_act.metadata)['activation_tensor_rows'] == 32
    # Active expert/GMM work, weight reads, receive counts and payload do not
    # change with the unmasked activation tensor capacity.
    for name, op in base_ops.items():
        if op.family not in {'grouped_gemm', 'alltoall', 'alltoallv', 'mc2_protocol',
                             'routing', 'tensor_materialization'}:
            continue
        other = capped_ops[name]
        assert (op.flops, op.hbm_bytes, op.mte2_bytes, op.mte3_bytes,
                op.link_bytes, op.local_shape) == (
                other.flops, other.hbm_bytes, other.mte2_bytes, other.mte3_bytes,
                other.link_bytes, other.local_shape), name
    phase = dict(dict(capped_ops['s0.l0.dispatch'].metadata)['mc2_phase'])
    base_phase = dict(dict(base_ops['s0.l0.dispatch'].metadata)['mc2_phase'])
    assert phase == base_phase
    # Only the activation output capacity changes.
    assert base_act.local_shape[0] != capped_act.local_shape[0]
