"""Source flag predicates and per-output top-k dependency controls."""
import pytest

from mindie_sim.prediction.operators.mc2_fullmesh import combine_layout, combine_plan
from test_mc2_owner_and_prefix import run_phase


@pytest.mark.parametrize('tokens,top_k', [(1, 1), (3, 8), (65, 3)])
def test_one_successful_flag_check_prices_vector_and_exposes_unpriced_scalar(tokens, top_k):
    elapsed, engine, plan, _ = run_phase('receive_flags', kind='combine', tokens=tokens, top_k=top_k)
    # Test width240 is exactly one480B return block. Flag arrays have eight
    # FP32 elements/slot, rounded to64 for Duplicate and CompareScalar.
    compared = ((top_k * 8 + 63) // 64) * 64
    assert engine.serviced['vector'] == tokens * 2 * compared
    assert engine.serviced['mte2_aiv'] == pytest.approx(tokens * top_k * 32)
    details = plan.metadata['source_service_details']
    assert details['scalar_instruction_counts']['scalar_sff_calls'] == tokens * (compared // 64)
    assert details['scalar_instruction_counts']['ub_uint64_mask_reads'] == tokens * (compared // 64)
    assert details['additional_spin_polls'] == 'unmodeled'
    # Independent low-rate serial initialize -> DMA -> compare oracle.
    assert elapsed == pytest.approx(((tokens + 63) // 64) * (2 * compared + top_k * 32) / 1000)
    own = [r for r in engine.trace if 'owner0:' in r['task']]
    init, read, compare = (next(r for r in own if r['task'].endswith(':'+s))
                           for s in ('initialize_flags','read_flags','compare_flags'))
    assert init['end_ms'] <= read['start_ms'] <= read['end_ms'] <= compare['start_ms']


def test_width3584_successful_poll_and_payload_are_not_double_counted():
    plan = combine_plan(combine_layout(3584), source_tokens=1, top_k=8,
                        source_assignments=8, received_assignments=8,
                        experts=16, ep=8, aiv_count=64)
    phases = {p.name:p for p in plan.phases}
    flags = dict(phases['receive_flags'].work)
    assert flags == {'vector': 1920, 'hbm': 3840, 'mte2_aiv': 3840}
    assert not phases['receive_slots'].work
    reduction = dict(phases['weighted_reduce'].work)
    assert reduction['mte2_aiv'] == 8 * (7200 + 4)
    assert reduction['vector'] == (3 * 8 + 1) * 3584
    assert dict(phases['receive_flags'].details)['scalar_instruction_counts']['scalar_sff_calls'] == 15


def test_two_tokens_on_same_owner_keep_slot_and_accumulator_order():
    _, engine, _, _ = run_phase('weighted_reduce', kind='combine', tokens=3, top_k=3, units=2)
    def stage(suffix):
        return next(r for r in engine.trace if r['task'].endswith(suffix))
    for token in (0,1):
        for slot in range(3):
            read=stage(f'owner0:tokens:program:tile{token}:slot{slot}_read')
            cast=stage(f'owner0:tokens:program:tile{token}:slot{slot}_cast')
            acc=stage(f'owner0:tokens:program:tile{token}:slot{slot}_accumulate')
            assert read['end_ms'] <= cast['start_ms'] <= acc['start_ms']
            if slot:
                assert stage(f'owner0:tokens:program:tile{token}:slot{slot-1}_accumulate')['end_ms'] <= read['start_ms']
    assert stage('owner0:tokens:program:tile0:slot2_accumulate')['end_ms'] <= stage('owner0:tokens:program:tile1:initialize_accumulator')['start_ms']


def test_fractional_expected_output_retains_framed_work_and_no_extra_whole_token():
    _, engine, plan, _ = run_phase('weighted_reduce', kind='combine', tokens=1.25, top_k=8, units=2)
    assert engine.serviced['hbm'] == pytest.approx(1.25 * 8 * (480 + 4))
    assert engine.serviced['vector'] == pytest.approx(1.25 * 25 * 240)
