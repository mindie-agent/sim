"""A5 KDA and causal-conv state-version/valid-row contracts.

The fixtures exercise the production lowering and the common ``Work``/``Stage``
programs.  They do not read a measurement, device state or source runtime.
Expected work is derived from declared payloads and query rows so a new shape
is priced by the same formulas.
"""
from dataclasses import replace

import pytest

from mindie_sim.catalog.quant import quant_preset
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.gdn_decode_programs import CONV, conv_plan
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.operators import kda_recurrent
from mindie_sim.prediction.operators.kda_state_selection import (
    resolve, version_transition,
)
from mindie_sim.prediction.recurrent import linear
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Software
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView
from test_kda_state_pipeline import hardware_case


def kda_op(lengths=(4,), *, heads=4, key_heads=4, **params):
    values = {"lengths": lengths, "heads": heads, "key_heads": key_heads,
              "dk": 128, "dv": 128, **params}
    return kda_recurrent.make_op(values, requests=tuple(f"r{i}" for i in range(len(lengths))))


def kda_plan(op, units=4, ub=256 * 1024):
    counts = {"hbm": 1, "mte2_aiv": units, "mte2_aic": units, "mte2": units,
              "mte3": units, "vector": units}
    rates = {"hbm": 1e12, "mte2_aiv": 1e12, "mte2_aic": 1e12, "mte2": 1e12,
             "mte3": 1e12, "vector": 1e12}
    geometry = Geometry(max(1, units // 2), counts, {"ub": ub}, {"ub": units})
    return kda_recurrent.build(op, geometry, compact=True)


def test_verification_checkpoints_commit_prefix_and_reject_later_versions():
    op = kda_recurrent.make_op(
        {"lengths": (4,), "heads": 4, "key_heads": 4,
         "state_contexts": (4096,), "state_acceptance": ((1, 2),)},
        requests=("r",))
    meta = dict(op.metadata)
    contract = meta["kda_state_contract"]
    assert contract["read_versions"] == (4096,)
    assert contract["checkpoint_versions"] == ((4097, 4098, 4099, 4100),)
    assert contract["accepted_offsets"] == (1,)
    assert contract["advances"] == (2,)
    assert contract["commit_versions"] == (4098,)
    assert contract["rejected_versions"] == ((4099, 4100),)
    assert contract["version_basis"] == "caller_context"
    actions = [(effect["action"], effect["version"])
               for effect in meta["kda_state_effects"]]
    assert actions == [
        ("read", 4096),
        ("write", 4097), ("write", 4098), ("write", 4099), ("write", 4100),
        ("commit", 4098), ("rollback", 4099), ("rollback", 4100),
    ]
    assert meta["state_checkpoint_versions"] == ((4097, 4098, 4099, 4100),)
    plan = kda_plan(op)
    assert plan.metadata["state_contract"] == contract
    assert plan.metadata["state_effects"] == meta["kda_state_effects"]


def test_undeclared_acceptance_is_pending_and_never_commits_every_row():
    op = kda_op((4,))
    contract = dict(op.metadata)["kda_state_contract"]
    assert contract["pending_acceptance"] is True
    assert contract["commit_policy"] == "pending_acceptance_event"
    assert contract["commit_versions"] == (None,)
    assert contract["checkpoint_versions"] == ((1, 2, 3, 4),)
    assert [effect["action"] for effect in dict(op.metadata)["kda_state_effects"]] == [
        "read", "write", "write", "write", "write"]
    with pytest.raises(ModelingError, match="advance_outside_query"):
        kda_op((4,), state_acceptance=((0, 5),))
    with pytest.raises(ModelingError, match="accepted_advance_mismatch"):
        kda_op((4,), state_acceptance=((2, 2),))


def test_accepted_versions_do_not_change_the_source_query_work():
    base = kda_plan(kda_op((4,), heads=4))
    accepted = kda_plan(kda_op(
        (4,), heads=4, state_contexts=(4096,),
        state_acceptance=((1, 2),)))
    assert dict(base.work.resources) == dict(accepted.work.resources)
    assert dict(base.work.matrix) == dict(accepted.work.matrix)


def test_state_dtype_and_uneven_sequences_are_priced_by_the_same_formula():
    lengths, heads, k, v = (1, 2, 3), 3, 128, 128
    parts = [kda_plan(kda_op((q,), heads=heads, key_heads=1,
                             state_contexts=(100 + q,),
                             state_acceptance=((0, 1),)))
             for q in lengths]
    total = Work()
    for plan in parts:
        total += plan.work
    multi = kda_plan(kda_op(lengths, heads=heads, key_heads=1,
                            state_contexts=(100, 101, 102),
                            state_acceptance=((0, 1),) * len(lengths)))
    assert dict(multi.work.resources) == dict(total.resources)
    assert dict(multi.work.matrix) == dict(total.matrix)

    fp32 = kda_plan(kda_op((4,), heads=heads, key_heads=1, state_dtype="fp32"))
    bf16 = kda_plan(kda_op((4,), heads=heads, key_heads=1, state_dtype="bf16"))
    state_fp32, state_bf16 = k * v * 4, k * v * 2
    initial = heads * (state_fp32 - state_bf16)
    checkpoints = 4 * heads * (state_fp32 - state_bf16)
    assert fp32.work.resources["mte2_aiv"] - bf16.work.resources["mte2_aiv"] == initial
    assert fp32.work.resources["mte3"] - bf16.work.resources["mte3"] == checkpoints
    assert fp32.work.resources["hbm"] - bf16.work.resources["hbm"] == (
        initial + checkpoints)


def test_exact_historical_slots_and_versions_share_one_contract():
    selection = {
        "kind": "exact", "index_layout": "speculative",
        "index_row_stride": 4, "prefix": [0, 2, 4],
        "state_indices": [[7, 8, -1, -1], [11, 12, -1, -1]],
        "accepted_tokens": [1, 2], "state_capacity": 16,
    }
    op = kda_recurrent.make_op(
        {"lengths": (2, 2), "heads": 2, "key_heads": 1,
         "state_selection": {**selection, "accepted_tokens": [1, 1],
                              "state_indices": [[7, 8, -1, -1], [11, 12, -1, -1]]},
         "state_contexts": (40, 80), "state_acceptance": ((0, 1), (1, 2))},
        requests=("a", "b"))
    contract = dict(op.metadata)["kda_state_contract"]
    assert contract["initial_slots"] == (7, 11)
    assert contract["checkpoint_slots"] == ((7, 8), (11, 12))
    assert contract["commit_versions"] == (41, 82)
    assert contract["rejected_versions"] == ((42,), ())
    assert dict(op.metadata)["state_selection"]["initial_slots"] == (7, 11)


def test_unknown_partition_is_not_averaged_and_duplicate_aliases_are_rejected():
    contract = version_transition((1, 7, 2), contexts=(0, 0, 0),
                                  acceptance=((0, 1), (0, 0), (0, 1)))
    assert contract["queries"] == (1, 7, 2)
    assert contract["advances"] == (1, 0, 1)
    assert contract["pending_acceptance"] is True
    assert contract["commit_versions"] == (1, None, 1)
    assert contract["checkpoint_versions"] == ((1,), tuple(range(1, 8)), (1, 2))

    with pytest.raises(ModelingError, match="cross_sequence_state_alias"):
        resolve({"kind": "exact", "index_layout": "packed",
                 "prefix": [0, 1, 2], "state_indices": [3, 3],
                 "accepted_tokens": [1, 1], "state_capacity": 8}, (1, 1))


def _conv_plan(conv, parsed):
    return conv_plan(conv, Geometry.from_devices(parsed.machine.dies))


def _conv_case(*, history=3, tp=1, q=4):
    model = load_model("kimi-k3")
    plan = next(p for p in model.layer_plans if p.attention == "kda")
    plan = replace(plan, conv_state_elements=plan.conv_width * history)
    model = replace(model, layers=1, hidden_size=512, layer_plans=(plan,),
                    quant=quant_preset("bf16"))
    runtime = Software.from_mapping(
        {"kernels": {"lowerings": {"causal_conv_update": CONV}}})
    return model, runtime, plan


def _conv_ops(views, tokens, *, history=3, tp=1, history_acceptance=None):
    model, runtime, plan = _conv_case(history=history, tp=tp)
    if history_acceptance is not None:
        model = replace(model, metadata=tuple(sorted({
            **dict(model.metadata),
            "causal_conv_history_acceptance": {
                str(plan.index): tuple(history_acceptance)}}.items())))
    ops = linear(prefix="kda", plan=plan, views=views, stage=0, deps="input",
                 model=model, parallel=ParallelSpec(tp=tp), quant=model.quant,
                 mode="decode", padded_tokens=tokens,
                 policy=AttentionPolicy(), runtime=runtime).ops
    return next(op for op in ops if op.recipe == CONV), ops


def test_speculative_conv_prices_all_candidates_and_rejects_later_versions():
    view = RequestView("r", 4, 4096, True, "decode",
                       draft_proposal_tokens=3, draft_accepted_tokens=1,
                       draft_advanced_tokens=2)
    conv, _ = _conv_ops((view,), 4, history=6, history_acceptance=(1,))
    meta = dict(conv.metadata)
    width = 2 * 96 * 128 + 96 * 128
    assert conv.local_shape == (4, width)
    assert meta["sequence_tokens"] == (4,)
    assert meta["speculative"] is True
    assert meta["state_history_slots"] == 6
    assert meta["state_read_bytes"] == (3 + 6) * width * 2
    assert meta["state_write_bytes"] == 6 * width * 2
    contract = meta["conv_state_contract"]
    assert contract["commit_versions"] == (4098,)
    assert contract["checkpoint_versions"] == ((4097, 4098, 4099, 4100),)
    assert contract["rejected_versions"] == ((4099, 4100),)
    effects = [(effect["action"], effect["version"])
               for effect in meta["conv_state_effects"]]
    assert effects == [
        ("read", 4096),
        ("write", 4097), ("write", 4098), ("write", 4099), ("write", 4100),
        ("commit", 4098), ("rollback", 4099), ("rollback", 4100),
    ]
    parsed = parse_case(hardware_case(units=4, split_copy=True))
    spec_work = _conv_plan(conv, parsed).work
    nonspec = replace(conv, metadata=tuple({
        **dict(conv.metadata), "speculative": False,
        "state_history_slots": 3}.items()))
    nonspec_work = _conv_plan(nonspec, parsed).work
    # All query rows execute in both plans; speculation adds two history
    # descriptors per candidate (2+q reads and 2+q writes in the source).
    assert (spec_work.resources["hbm"] - nonspec_work.resources["hbm"]
            == (2 + 4) * width * 2 + (2 + 4 - 3) * width * 2)
    assert spec_work.resources["vector"] == nonspec_work.resources["vector"]


def test_speculative_conv_without_extended_history_is_an_explicit_error():
    view = RequestView("r", 4, 4096, True, "decode",
                       draft_proposal_tokens=3, draft_accepted_tokens=1,
                       draft_advanced_tokens=2)
    with pytest.raises(ModelingError, match="speculative_requires_extended_history"):
        _conv_ops((view,), 4, history=3)


def test_padding_rows_execute_queries_but_own_no_state_history():
    active = RequestView("a", 1, 4096, True, "decode")
    padded = RequestView("p", 31, 4096, False, "decode", padding=True)
    conv, _ = _conv_ops((active, padded), 32, history=3)
    meta = dict(conv.metadata)
    width = 3 * 96 * 128
    assert conv.local_shape == (32, width)
    assert meta["sequence_tokens"] == (1, 31)
    assert meta["physical_sequence_tokens"] == (1, 31)
    assert meta["padding_slots"] == (1,)
    assert meta["state_read_bytes"] == 3 * width * 2
    assert meta["state_write_bytes"] == 3 * width * 2
    assert meta["conv_state_contract"]["checkpoint_versions"] == ((4097,), ())
    parsed = parse_case(hardware_case(units=4, split_copy=True))
    present = _conv_plan(conv, parsed)
    inactive = _conv_plan(replace(conv, metadata=tuple({
        **dict(conv.metadata), "sequence_tokens": (1, 0),
        "physical_sequence_tokens": (1, 0), "padding_slots": ()}.items())), parsed)
    # The padding slot retains the host grid and external tensors but emits no
    # convolution payload, vector work or state history.
    assert present.work.resources == inactive.work.resources


def test_prefill_versions_commit_each_chunk_and_keep_real_tail_rows():
    from mindie_sim.prediction.operators import kda_state

    params = {"lengths": [128, 96], "heads": 2, "key_heads": 2,
              "dk": 128, "dv": 128, "chunk": 64, "dtype": "bf16",
              "gate_dtype": "fp32", "state_dtype": "fp32"}
    op = kda_state.make_op(params, requests=("a", "b"),
                           state_contexts=(0, 512))
    meta = dict(op.metadata)
    contract = meta["kda_state_contract"]
    assert contract["read_versions"] == (0, 512)
    assert contract["checkpoint_versions"] == ((64, 128), (576, 608))
    assert contract["internal_commit_versions"] == (
        ((64, 128), (64, 128)), ((576, 608), (576, 608)))
    assert contract["commit_versions"] == (128, 608)
    assert contract["version_basis"] == "caller_context"
    assert contract["internal_record_count"] == 8
    assert [{
        key: record[key] for key in (
            "target", "version", "sequence", "storage", "dtype", "layout",
            "span", "bytes", "head_count", "action", "producer", "output",
            "generation", "ready")}
        for record in contract["publication_records"]] == [
        {"target": "kda.layer[0].recurrent_state[a]", "version": 128,
         "sequence": 0, "storage": "public_state:kda.layer[0].recurrent_state[a]", "dtype": "fp32",
         "layout": "V,K", "span": ((0, 2), (0, 128), (0, 128)),
         "bytes": 2 * 128 * 128 * 4, "head_count": 2, "action": "commit",
         "producer": "kda_fwd_h:layout_publication", "output": 0,
         "generation": 128, "ready": "kda_fwd_h:layout_publication"},
        {"target": "kda.layer[0].recurrent_state[b]", "version": 608,
         "sequence": 1, "storage": "public_state:kda.layer[0].recurrent_state[b]", "dtype": "fp32",
         "layout": "V,K", "span": ((0, 2), (0, 128), (0, 128)),
         "bytes": 2 * 128 * 128 * 4, "head_count": 2, "action": "commit",
         "producer": "kda_fwd_h:layout_publication", "output": 1,
         "generation": 608, "ready": "kda_fwd_h:layout_publication"}]
    actions = [(effect["action"], effect["version"])
               for effect in meta["kda_state_effects"]]
    assert actions == [
        ("write", 128), ("write", 608),
        ("commit", 128), ("commit", 608),
    ]
    # The 96-token sequence still clips its second chunk to 32 real rows.
    assert contract["queries"] == (128, 96)
    assert meta["chunks_per_sequence"] == (2, 2)


def test_decode_binder_consumes_exact_view_context_and_acceptance_not_average():
    from mindie_sim.prediction.family_kda_recurrent import bind_decode_scans
    from mindie_sim.prediction.kda_source import view_state_declaration

    views = (
        RequestView("a", 4, 40, False, "decode",
                    draft_proposal_tokens=3, draft_accepted_tokens=1,
                    draft_advanced_tokens=2),
        RequestView("b", 1, 80, False, "decode"),
    )
    contexts, acceptance = view_state_declaration(views)
    assert contexts == (40, 80)
    assert acceptance == ((1, 2), (0, 0))
    chosen = [version_transition((4,), contexts=(40,),
                                 acceptance=((1, 2),)),
              version_transition((1,), contexts=(80,),
                                 acceptance=((0, 0),))]
    assert chosen[0]["commit_versions"] == (42,)
    assert chosen[1]["commit_versions"] == (81,)
    assert chosen[0]["rejected_versions"] == ((43, 44),)

    scan = kda_op((1,), heads=2, key_heads=1)
    bound = bind_decode_scans(
        (scan,), lengths=(4, 1), heads=2, key_heads=1, dk=128, dv=128,
        dtype="bf16", state_dtype="fp32", requests=("a", "b"), deps=(),
        views=views)
    contract = dict(bound.metadata)["kda_state_contract"]
    assert contract["contexts"] == (40, 80)
    assert contract["checkpoint_versions"] == ((41, 42, 43, 44), (81,))
    assert contract["commit_versions"] == (42, 81)
    assert contract["rejected_versions"] == ((43, 44), ())


def test_non_spec_conv_commits_the_single_checkpoint_at_token_commit():
    view = RequestView("r", 1, 4096, True, "decode")
    conv, _ = _conv_ops((view,), 1, history=3)
    meta = dict(conv.metadata)
    contract = meta["conv_state_contract"]
    assert meta["speculative"] is False
    assert meta["state_history_slots"] == 3
    assert contract["read_versions"] == (4096,)
    assert contract["checkpoint_versions"] == ((4097,),)
    assert contract["commit_versions"] == (4097,)
    assert contract["rejected_versions"] == ((),)
    assert [(effect["action"], effect["version"])
            for effect in meta["conv_state_effects"]] == [
        ("read", 4096), ("write", 4097), ("commit", 4097)]


def test_pending_conv_verification_neither_commits_nor_rejects_early():
    view = RequestView("r", 4, 4096, False, "decode",
                       draft_proposal_tokens=3, draft_advanced_tokens=0)
    conv, _ = _conv_ops((view,), 4, history=6, history_acceptance=(1,))
    meta = dict(conv.metadata)
    contract = meta["conv_state_contract"]
    assert contract["pending_acceptance"] is True
    assert contract["commit_policy"] == "pending_acceptance_event"
    assert contract["commit_versions"] == (None,)
    assert contract["rejected_versions"] == ((),)
    assert [(effect["action"], effect["version"])
            for effect in meta["conv_state_effects"]] == [
        ("read", 4096), ("write", 4097), ("write", 4098),
        ("write", 4099), ("write", 4100)]


def test_conv_acceptance_outside_current_queries_is_explicit():
    view = RequestView("r", 4, 4096, False, "decode",
                       draft_proposal_tokens=4, draft_advanced_tokens=5)
    with pytest.raises(ModelingError, match="causal_conv_update_advance_outside_query"):
        _conv_ops((view,), 4, history=8)


def test_state_write_and_publication_fragments_are_executable():
    from mindie_sim.prediction.engine import Engine
    from mindie_sim.prediction.scheduling import FragmentConductor
    from mindie_sim.schema.operators import ProgramFragment, StateEffect

    op = kda_recurrent.make_op(
        {"lengths": (4,), "heads": 4, "key_heads": 4,
         "state_contexts": (4096,), "state_acceptance": ((1, 2),)},
        requests=("r",), state_decisions=("accept_event",))
    contract = dict(op.metadata)["kda_state_contract"]
    target = contract["targets"][0]
    fragments = kda_recurrent.state_fragments(op)
    engine = Engine(())
    conductor = FragmentConductor(engine, state={target: 4096})
    source = engine.submit("source_ready", {})
    read = conductor.submit(fragments["read"], inputs={target: source})
    write = conductor.submit(fragments["write"], after=(read.event,))
    conductor.advance(until=(write.event,))
    # Writes create the candidate versions but do not make them visible.
    assert conductor.state.committed_version(target) == 4096
    assert 4100 in conductor.state.live_versions(target)
    acceptance = engine.submit("accept_event", {})
    commit = conductor.submit(fragments["commits"][0], after=(write.event,),
                              inputs={"decision:" + target + ":4098": acceptance})
    conductor.advance(until=(commit.event,))
    assert conductor.state.committed_version(target) == 4098
    for fragment in fragments["rollbacks"]:
        port = fragment.ports.event_in[0]
        rollback = conductor.submit(
            fragment, after=(write.event,),
            inputs={port.name: acceptance})
        conductor.advance(until=(rollback.event,))
    assert 4099 not in conductor.state.live_versions(target)
    assert 4100 not in conductor.state.live_versions(target)
    # A next invocation may read the committed version; the rejected ones are
    # not selected.
    next_read = ProgramFragment(
        "next_read", state_effects=(StateEffect(target, "read", 4098),))
    handle = conductor.submit(next_read, inputs={target: acceptance})
    conductor.advance(until=(handle.event,))
    assert conductor.state.committed_version(target) == 4098
def test_pending_fragments_leave_committed_state_untouched():
    from mindie_sim.prediction.engine import Engine
    from mindie_sim.prediction.scheduling import FragmentConductor

    op = kda_op((4,))
    contract = dict(op.metadata)["kda_state_contract"]
    target = contract["targets"][0]
    fragments = kda_recurrent.state_fragments(op)
    conductor = FragmentConductor(Engine(()), state={target: 0})
    write = conductor.submit(fragments["write"])
    conductor.advance(until=(write.event,))
    assert conductor.state.committed_version(target) == 0
    assert conductor.state.live_versions(target) == (0, 1, 2, 3, 4)
    assert fragments["commits"] == () and fragments["rollbacks"] == ()


def test_prefill_fragments_publish_after_all_head_writes():
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.scheduling import FragmentConductor
    from mindie_sim.prediction.operators import kda_state
    from mindie_sim.schema.operators import ProgramFragment, WorkItem, WorkSet

    op = kda_state.make_op(
        {"lengths": [128, 96], "heads": 2, "key_heads": 2, "dk": 128,
         "dv": 128, "chunk": 64, "dtype": "bf16", "gate_dtype": "fp32",
         "state_dtype": "fp32"},
        requests=("a", "b"), state_contexts=(0, 512))
    contract = dict(op.metadata)["kda_state_contract"]
    fragments = kda_state.state_fragments(op)
    conductor = FragmentConductor(
        Engine([Resource("step", 1, unit="u")]),
        state=dict(zip(contract["targets"], contract["contexts"])))
    previous = None

    def step():
        nonlocal previous
        work = ProgramFragment(
            "step", work=WorkSet((WorkItem("step", 1, "u"),)))
        handle = conductor.submit(work, after=(() if previous is None else (previous,)))
        conductor.advance(until=(handle.event,))
        previous = handle.event
        return handle.event

    # The initialize stage materializes the internal heads from the public
    # subviews; no internal target is prefilled.
    boundary = step()
    initialize = fragments["initialize"]
    inputs = {port.name: boundary for port in initialize.ports.tensor_in}
    handle = conductor.submit(initialize, after=(boundary,), inputs=inputs)
    conductor.advance(until=(handle.event,))
    previous = handle.event
    for sequence, targets in enumerate(contract["internal_targets"]):
        for target in targets:
            assert conductor.state.committed_version(target) is None
    chunks = fragments["internal"]
    for key in sorted(chunks):
        boundary = step()
        fragment = chunks[key]
        inputs = {port.name: previous for port in fragment.ports.tensor_in}
        handle = conductor.submit(fragment, after=(boundary,), inputs=inputs)
        conductor.advance(until=(handle.event,))
        previous = handle.event
    # Both heads advanced their internal chunks; the public target still holds
    # the previous request version until the layout publication.
    for sequence, targets in enumerate(contract["internal_targets"]):
        for target in targets:
            assert conductor.state.committed_version(target) == (
                128 if sequence == 0 else 608)
    assert conductor.state.committed_version(contract["targets"][0]) == 0
    assert conductor.state.committed_version(contract["targets"][1]) == 512
    pub = kda_state.layout_fragment(op)
    boundary = step()
    inputs = {port.name: boundary
              for port in pub.ports.tensor_in + pub.ports.event_in}
    handle = conductor.submit(pub, after=(boundary,), inputs=inputs)
    conductor.advance(until=(handle.event,))
    assert conductor.state.committed_version(contract["targets"][0]) == 128
    assert conductor.state.committed_version(contract["targets"][1]) == 608
def test_duplicate_checkpoint_slot_keeps_only_the_last_covering_version():
    from mindie_sim.prediction.operators.kda_state_selection import state_records

    records = state_records(
        targets=("s",), lengths=(2,), contexts=(5,), acceptance=((0, 2),),
        checkpoint_slots=((7, 7),))
    assert records["write_records"] == (("s", 7, 7),)
    assert records["retained_versions"] == ((7,),)
    assert records["overwritten_versions"] == ((6,),)
    assert records["commit_records"] == (("s", 7),)
    with pytest.raises(
            ModelingError,
            match="accepted_checkpoint_overwritten_requires_materialization"):
        state_records(
            targets=("s",), lengths=(2,), contexts=(5,),
            acceptance=((0, 1),), checkpoint_slots=((7, 7),))


def test_conv_abi_uses_run_mode_1_and_typed_optional_acceptance():
    view = RequestView("r", 4, 4096, True, "decode",
                       draft_proposal_tokens=3, draft_accepted_tokens=1,
                       draft_advanced_tokens=2)
    conv, _ = _conv_ops((view,), 4, history=6, history_acceptance=(1,))
    meta = dict(conv.metadata)
    assert meta["run_mode"] == 1
    assert meta["num_accepted_tokens_present"] is True
    assert meta["num_accepted_tokens_dtype"] == "int32"
    assert meta["num_accepted_tokens_shape"] == (1,)
    assert meta["num_accepted_tokens_values"] == (1,)
    assert meta["state_token_offsets"] == (0,)
    assert meta["state_history_slots"] == 6
    assert meta["speculative_source_selector"] == \
        "num_accepted_tokens_presence_width4"

    accepted = RequestView("r", 4, 4096, True, "decode",
                           draft_proposal_tokens=3, draft_accepted_tokens=3,
                           draft_advanced_tokens=4)
    other, _ = _conv_ops((accepted,), 4, history=6,
                         history_acceptance=(3,))
    other_meta = dict(other.metadata)
    assert other_meta["state_token_offsets"] == (2,)
    parsed = parse_case(hardware_case(units=4, split_copy=True))
    assert _conv_plan(conv, parsed).work.resources == \
        _conv_plan(other, parsed).work.resources


def test_kda_program_binds_state_fragments_to_real_stages():
    from mindie_sim.prediction.program import RecipeContext

    op = kda_recurrent.make_op(
        {"lengths": (4,), "heads": 4, "key_heads": 4,
         "state_contexts": (4096,), "state_acceptance": ((1, 2),)},
        requests=("r",))
    plan = kda_plan(op)
    counts = {"hbm": 1, "mte2_aiv": 4, "mte2_aic": 4, "mte2": 4,
              "mte3": 4, "vector": 4}
    rates = {"hbm": 1e12, "mte2_aiv": 1e12, "mte2_aic": 1e12,
             "mte2": 1e12, "mte3": 1e12, "vector": 1e12}
    context = RecipeContext(
        plan.work.apply(op), {}, {key: key for key in rates}, rates, 2,
        setup_ms=0, minimum_ms=0, resource_units=counts,
        pool_names={"ub": "ub"}, pool_units={"ub": 4})
    program = plan.lower(context)
    outer = {stage.name: stage for stage in program.stages}
    inner = {stage.name: stage for stage in outer["body"].program.stages}
    assert inner["state_read"].fragment is not None
    assert inner["state_publish"].fragment is not None
    assert [(effect.action, effect.version)
            for effect in inner["state_publish"].fragment.state_effects] == [
        ("write", 4097), ("write", 4098),
        ("write", 4099), ("write", 4100)]
    assert [(effect.action, effect.version)
            for effect in inner["state_read"].fragment.state_effects] == [
        ("read", 4096)]


def test_prefill_plan_carries_chunk_fragments_and_publication_records():
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.operators import kda_state

    op = kda_state.make_op(
        {"lengths": [128, 96], "heads": 2, "key_heads": 2, "dk": 128,
         "dv": 128, "chunk": 64, "dtype": "bf16", "gate_dtype": "fp32",
         "state_dtype": "fp32"},
        requests=("a", "b"), state_contexts=(0, 512))
    geometry = Geometry(2, {"vector": 4, "hbm": 1, "mte2": 4, "mte3": 4},
                        {"ub": 1 << 20}, {"ub": 4})
    plan = kda_state.build(op, geometry)
    assert plan.metadata["state_fragment_counts"] == {
        "initialize": 1, "internal": 8, "publication": 2}
    contract = plan.metadata["state_contract"]
    assert [(record["target"], record["version"],
             record["producer"], record["output"])
            for record in contract["publication_records"]] == [
        (contract["targets"][0], 128, "kda_fwd_h:layout_publication", 0),
        (contract["targets"][1], 608, "kda_fwd_h:layout_publication", 1)]
    assert plan.metadata["layout_bytes"] == 2 * 128 * 128 * 4


def test_kda_exact_duplicate_slot_keeps_last_cover_and_rejects_unretained_accept():
    selection = {
        "kind": "exact", "index_layout": "packed", "prefix": [0, 2, 4],
        "state_indices": [7, 7, 11, 12], "accepted_tokens": [2, 1],
        "state_capacity": 16,
    }
    op = kda_recurrent.make_op(
        {"lengths": (2, 2), "heads": 2, "key_heads": 1,
         "state_selection": selection, "state_contexts": (9, 20),
         "state_acceptance": ((0, 2), (0, 1))},
        requests=("a", "b"))
    contract = dict(op.metadata)["kda_state_contract"]
    assert contract["retained_versions"][0] == (11,)
    assert contract["overwritten_versions"][0] == (10,)
    assert contract["write_records"][0] == (
        "kda.layer[0].recurrent_state[a]", 11, 7)
    with pytest.raises(
            ModelingError,
            match="accepted_checkpoint_overwritten_requires_materialization"):
        kda_recurrent.make_op(
            {"lengths": (2, 2), "heads": 2, "key_heads": 1,
             "state_selection": {**selection, "accepted_tokens": [1, 1]},
             "state_contexts": (9, 20),
             "state_acceptance": ((0, 1), (0, 1))},
            requests=("a", "b"))


def test_conv_recipe_forwards_the_same_contract_to_the_shared_plan():
    from mindie_sim.prediction.family_causal_conv import _planned

    view = RequestView("r", 4, 4096, True, "decode",
                       draft_proposal_tokens=3, draft_accepted_tokens=1,
                       draft_advanced_tokens=2)
    conv, _ = _conv_ops((view,), 4, history=6, history_acceptance=(1,))
    parsed = parse_case(hardware_case(units=4, split_copy=True))
    geometry = Geometry.from_devices(parsed.machine.dies)
    shared = conv_plan(conv, geometry)
    planned = _planned(conv, geometry)
    contract = dict(conv.metadata)["conv_state_contract"]
    assert planned.work.resources == shared.work.resources
    assert planned.metadata["state_contract"] == contract
    assert planned.metadata["conv_state_contract"] == contract
    assert planned.metadata["conv_state_effects"] == \
        dict(conv.metadata)["conv_state_effects"]


def test_conv_q4_accept_two_then_q1_rebases_committed_window():
    q4 = RequestView("r", 4, 4096, True, "decode",
                     draft_proposal_tokens=3, draft_accepted_tokens=2,
                     draft_advanced_tokens=3)
    _q4, _ = _conv_ops((q4,), 4, history=6, history_acceptance=(1,))
    q1 = RequestView("r", 1, 4099, True, "decode")
    conv, ops = _conv_ops((q1,), 1, history=6, history_acceptance=(2,))
    meta = dict(conv.metadata)
    assert meta["speculative"] is False
    assert meta["run_mode"] == 1
    assert meta["num_accepted_tokens_present"] is False
    rebase_record, = meta["history_rebase"]
    assert (rebase_record["sequence"], rebase_record["accepted_tokens"],
            rebase_record["source_offset"], rebase_record["dest_offset"],
            rebase_record["rows"], rebase_record["bytes"],
            rebase_record["order"], rebase_record["allocation"],
            rebase_record["last_use"]) == (
        0, 2, 1, 0, 3, 3 * 36864 * 2, "descending_overlap_in_place",
        "in_place_no_new_slot", "before_short_convolution")
    rebase = next(op for op in ops if op.recipe == "conv_history_rebase")
    width = 2 * 96 * 128 + 96 * 128
    assert rebase.local_shape == (3, width)
    assert rebase.mte2_bytes == rebase.mte3_bytes == 3 * width * 2
    assert rebase.op_id in conv.deps
    # The ordinary q1 UPDATE reads offset zero after the ordered copy; its
    # traffic remains the non-spec 3-row history, not a second q4 window.
    parsed = parse_case(hardware_case(units=4, split_copy=True))
    conv_work = _conv_plan(conv, parsed).work
    plain = RequestView("r", 1, 4099, True, "decode")
    plain_conv, _ = _conv_ops((plain,), 1, history=6)
    plain_work = _conv_plan(plain_conv, parsed).work
    assert conv_work.resources == plain_work.resources


def _run_fwd_h_with_stage_hooks(lengths, heads, contexts, *, state_values=None,
                                 state_ready=()):
    """Run the real FWD_H program and capture Stage.fragment at its tasks."""
    from unittest.mock import patch
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.mechanism import PlannedRecipe
    from mindie_sim.prediction.operators import kda_state
    from mindie_sim.prediction.program import RecipeBook, Stage
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import IterationWork
    from mindie_sim.schema.workload import IterationPlan, RequestView

    op = kda_state.make_op(
        {"lengths": list(lengths), "heads": heads, "key_heads": heads,
         "dk": 128, "dv": 128, "chunk": 64, "dtype": "bf16",
         "gate_dtype": "fp32", "state_dtype": "fp32"},
        requests=tuple(f"s{index}" for index in range(len(lengths))),
        state_contexts=tuple(contexts), state_values=state_values,
        state_ready=state_ready)
    parsed = parse_case(hardware_case(units=2))
    execution = Execution(
        parsed, CostModel(parsed),
        recipes=RecipeBook(recipes={op.recipe: PlannedRecipe(kda_state.build)}),
        trace_limit=0)
    captured = []
    original = Stage.submit

    def submit(self, engine, name, predecessors, scoped, owner, prepared=None):
        event = original(self, engine, name, predecessors, scoped, owner,
                        prepared)
        if self.fragment is not None and self.fragment.state_effects:
            captured.append((event, self.fragment, self.name))
        return event

    iteration = IterationWork(
        (op,), IterationPlan((RequestView("r", 1, 0, True, "prefill"),)),
        1, 1, 1, None, "prefill", "one")
    with patch.object(Stage, "submit", submit):
        result = execution.add(iteration, 0, 0)
        execution.engine.advance(until=(result.completion,))
    return op, captured, execution


def _apply_stage_hooks(op, captured, execution, *, publication=True):
    """Replay the real stage fragments into one running ledger.

    The ledger starts with only the public initial state.  Internals are
    materialized by the initialize stage, not prefilled, and the public version
    advances only when the layout stage completes.  ``publication=False`` stops
    before the public commit so a caller can inspect the held version.
    """
    from mindie_sim.prediction.scheduling import StateLedger

    contract = dict(op.metadata)["kda_state_contract"]
    public_targets = set(contract["targets"])
    ledger = StateLedger(dict(zip(contract["targets"], contract["contexts"])))
    max_live = 0
    snapshots = []
    for event, fragment, name in sorted(
            captured, key=lambda item: (item[0].time, item[0].sequence)):
        public_commit = any(
            effect.action == "commit" and effect.target in public_targets
            for effect in fragment.state_effects)
        if public_commit and not publication:
            snapshots.append((name, dict(ledger.snapshot()["committed"])))
            continue
        reads = [effect for effect in fragment.state_effects
                 if effect.action == "read"]
        for effect in reads:
            ledger.check_read(effect.target, effect.version, at=event.time)
        ledger.apply([effect for effect in fragment.state_effects
                      if effect.action != "read"],
                     at=event.time, sequence=event.sequence)
        snapshots.append((name, dict(ledger.snapshot()["committed"])))
        for targets in contract["internal_targets"]:
            for target in targets:
                max_live = max(max_live, len(ledger.live_versions(target)))
    if publication:
        from mindie_sim.prediction.engine import Engine, Resource
        from mindie_sim.prediction.scheduling import FragmentConductor
        from mindie_sim.prediction.operators import kda_state
        from mindie_sim.schema.operators import (
            ProgramFragment, WorkItem, WorkSet)

        engine = Engine([Resource("layout", 1e9, unit="byte")])
        conductor = FragmentConductor(engine, state=ledger)
        layout_bytes = float(contract["layout_bytes"])
        layout = ProgramFragment(
            "layout_output",
            work=WorkSet((WorkItem("layout", 2 * layout_bytes, "byte"),)))
        layout_handle = conductor.submit(layout)
        fragment = kda_state.layout_fragment(op)
        inputs = {port.name: layout_handle.event
                  for port in (fragment.ports.tensor_in
                               + fragment.ports.event_in)}
        handle = conductor.submit(fragment, after=(layout_handle.event,),
                                  inputs=inputs)
        conductor.advance(until=(handle.event,))
    return ledger, max_live, snapshots
def test_fwd_h_96_internal_store_and_public_512_hold_until_608():
    op, captured, execution = _run_fwd_h_with_stage_hooks((96,), 2, (512,))
    contract = dict(op.metadata)["kda_state_contract"]
    ledger, _, held = _apply_stage_hooks(op, captured, execution,
                                         publication=False)
    # Each head owns independent internal state; both chunk stores ran before
    # the public layout publication.
    for target in contract["internal_targets"][0]:
        assert ledger.committed_version(target) == 608
    assert ledger.committed_version(contract["targets"][0]) == 512
    for _name, committed in held:
        assert committed[contract["targets"][0]] == 512
    ledger, _, _ = _apply_stage_hooks(op, captured, execution,
                                      publication=True)
    assert ledger.committed_version(contract["targets"][0]) == 608


def test_fwd_h_4096_versions_are_bounded_and_not_registered_in_one_batch():
    op, captured, execution = _run_fwd_h_with_stage_hooks((4096,), 1, (0,))
    contract = dict(op.metadata)["kda_state_contract"]
    target = contract["internal_targets"][0][0]
    ledger, max_live, _ = _apply_stage_hooks(op, captured, execution,
                                             publication=False)
    assert max_live <= 2
    assert ledger.committed_version(target) == 4096
    assert ledger.committed_version(contract["targets"][0]) == 0
    # The 64 chunk stores are distinct real stage completions, not one global
    # write batch.  Applying them must not hit the 64-version ledger limit.
    times = {event.time for event, fragment, _name in captured
             if fragment.name.startswith("kda_fwd_h:state_chunk")
             and any(effect.target == target and effect.action == "write"
                     for effect in fragment.state_effects)}
    assert len(times) == 64


def test_fwd_h_heads_can_complete_at_different_store_boundaries():
    op, captured, execution = _run_fwd_h_with_stage_hooks((96, 128), 2,
                                                         (0, 0))
    contract = dict(op.metadata)["kda_state_contract"]
    times = {}
    for event, fragment, _name in captured:
        for effect in fragment.state_effects:
            if effect.action != "commit":
                continue
            if effect.target == contract["targets"][0]:
                continue
            if ".head[" in effect.target:
                times.setdefault(effect.target, []).append(event.time)
    assert len({round(value[-1], 12) for value in times.values()}) > 1


def test_conv_full_q4_then_q1_reads_x1_x2_x3_and_commits_4101():
    q4 = RequestView("r", 4, 4096, True, "decode",
                     draft_proposal_tokens=3, draft_accepted_tokens=3,
                     draft_advanced_tokens=4)
    _q4, _ = _conv_ops((q4,), 4, history=6, history_acceptance=(1,))
    q1 = RequestView("r", 1, 4100, True, "decode")
    conv, ops = _conv_ops((q1,), 1, history=6, history_acceptance=(4,))
    meta = dict(conv.metadata)
    rebase, = meta["history_rebase"]
    assert (rebase["source_offset"], rebase["rows"], rebase["bytes"]) == (
        3, 3, 3 * 36864 * 2)
    assert meta["num_accepted_tokens_present"] is False
    assert meta["conv_state_contract"]["commit_versions"] == (4101,)
    copy = next(op for op in ops if op.recipe == "conv_history_rebase")
    assert copy.op_id in conv.deps


def test_conv_full_q4_then_q2_uses_one_native_history_offset():
    q2 = RequestView("r", 2, 4100, True, "decode",
                     draft_proposal_tokens=1, draft_accepted_tokens=0,
                     draft_advanced_tokens=1)
    conv, ops = _conv_ops((q2,), 2, history=6, history_acceptance=(4,))
    meta = dict(conv.metadata)
    assert meta["speculative"] is True
    assert meta["history_rebase"] == ()
    assert meta["num_accepted_tokens_present"] is True
    assert meta["num_accepted_tokens_values"] == (4,)
    assert meta["state_token_offsets"] == (3,)
    assert not any(op.recipe == "conv_history_rebase" for op in ops)
    # Current q2 acceptance independently decides the next visible version.
    assert meta["conv_state_contract"]["commit_versions"] == (4101,)


def test_conv_mixed_q1_q4_applies_native_history_once_without_rebase():
    q1 = RequestView("a", 1, 4096, True, "decode")
    q4 = RequestView("b", 4, 4096, True, "decode",
                     draft_proposal_tokens=3, draft_accepted_tokens=1,
                     draft_advanced_tokens=2)
    conv, ops = _conv_ops((q1, q4), 5, history=6,
                          history_acceptance=(2, 1))
    meta = dict(conv.metadata)
    assert meta["speculative"] is True
    assert meta["history_rebase"] == ()
    assert meta["num_accepted_tokens_values"] == (2, 1)
    assert meta["state_token_offsets"] == (1, 0)
    assert not any(op.recipe == "conv_history_rebase" for op in ops)
    contract = meta["conv_state_contract"]
    assert contract["read_versions"] == (4096, 4096)
    assert contract["commit_versions"] == (4097, 4098)


def test_conv_active_q4_with_padding_uses_only_active_history_capacity():
    q4 = RequestView("a", 4, 4096, True, "decode",
                     draft_proposal_tokens=3, draft_accepted_tokens=1,
                     draft_advanced_tokens=2)
    pad = RequestView("p", 32, 4096, False, "decode", padding=True)
    conv, _ = _conv_ops((q4, pad), 36, history=6,
                        history_acceptance=(1, 0))
    meta = dict(conv.metadata)
    assert meta["padding_slots"] == (1,)
    assert meta["state_history_slots"] == 6
    assert meta["num_accepted_tokens_values"] == (1, 0)
    assert meta["conv_state_contract"]["read_versions"][1] is None


def test_kda_state_accesses_bind_targets_versions_and_public_span():
    from mindie_sim.prediction.operators import kda_recurrent, kda_state

    op = kda_recurrent.make_op(
        {"lengths": (4,), "heads": 4, "key_heads": 4,
         "state_contexts": (4096,), "state_acceptance": ((1, 2),)},
        requests=("r",))
    accesses = kda_recurrent.state_accesses(op)
    reads = [item for item in accesses if item["direction"] == "read"]
    writes = [item for item in accesses if item["direction"] == "write"]
    assert reads[0]["producer"] == "external"
    assert reads[0]["state_version"] == 4096
    assert reads[0]["bytes"] == 4 * 128 * 128 * 4
    assert {item["state_version"] for item in writes} == {
        4097, 4098, 4099, 4100}
    assert all(item["owner"] == "kda.layer[0]" for item in accesses)
    assert all(item["request"] == "r" for item in accesses)
    # Every checkpoint covers all four value heads in its own slot.
    by_version = {}
    for item in writes:
        by_version.setdefault(item["state_version"], []).append(item)
    assert {version: len(items) for version, items in by_version.items()} == {
        4097: 1, 4098: 1, 4099: 1, 4100: 1}
    assert all(item["bytes"] == 4 * 128 * 128 * 4 for item in writes)
    assert all(item["span"] == ((0, 4), (0, 128), (0, 128)) for item in writes)
    assert by_version[4098][0]["decision"] == "token_commit"
    assert by_version[4099][0]["last_use"] == "last_reader"

    prefill = kda_state.make_op(
        {"lengths": [96], "heads": 2, "key_heads": 2, "dk": 128, "dv": 128,
         "chunk": 64, "dtype": "bf16", "gate_dtype": "fp32",
         "state_dtype": "fp32"},
        requests=("r",), state_contexts=(512,))
    prefill_accesses = kda_state.state_accesses(prefill)
    assert any(item["producer"] == "external"
               and item["storage"] == "public_state:kda.layer[0].recurrent_state[r]"
               and item["bytes"] == 128 * 128 * 4
               for item in prefill_accesses)
    public = [item for item in prefill_accesses
              if item["direction"] == "write" and item["position"] == "layout"]
    assert [{
        key: public[0][key] for key in (
            "occurrence", "producer", "request", "direction", "position",
            "output", "storage", "owner", "representation", "persistent",
            "temporary", "state_version", "span", "bytes", "source",
            "decision", "last_use")}] == [{
        "occurrence": "kda_fwd_h:public:0",
        "producer": "kda_fwd_h:layout_publication", "request": "r",
        "direction": "write", "position": "layout",
        "output": "kda.layer[0].recurrent_state[r]",
        "storage": "public_state:kda.layer[0].recurrent_state[r]", "owner": "kda.layer[0]",
        "representation": "fp32", "persistent": True, "temporary": False,
        "state_version": 608, "span": ((0, 2), (0, 128), (0, 128)),
        "bytes": 2 * 128 * 128 * 4,
        "source": "kda_fwd_h:layout_publication",
        "decision": "layout_complete",
        "last_use": "after_layout_publication"}]
def test_conv_accesses_mark_rebase_source_dest_and_q2_native_source():
    from mindie_sim.prediction.family_causal_conv import state_accesses

    q1 = RequestView("r", 1, 4099, True, "decode")
    conv, ops = _conv_ops((q1,), 1, history=6, history_acceptance=(2,))
    rebase = next(op for op in ops if op.recipe == "conv_history_rebase")
    reads = dict(rebase.metadata)["state_accesses"]
    writes = dict(rebase.metadata)["state_accesses_write"]
    assert reads[0]["span"] == ((1, 4), (0, 36864))
    assert writes[0]["span"] == ((0, 3), (0, 36864))
    assert reads[0]["source"] == "external"
    assert reads[0]["state_version"] == 4099
    assert writes[0]["producer"] == rebase.op_id
    conv_access = state_accesses(conv)
    assert conv_access[0]["source"] == rebase.op_id
    assert conv_access[0]["state_version"] == 4099
    assert conv_access[0]["span"] == ((0, 3), (0, 36864))

    q2 = RequestView("r", 2, 4100, True, "decode",
                     draft_proposal_tokens=1, draft_accepted_tokens=0,
                     draft_advanced_tokens=1)
    q2_conv, _ = _conv_ops((q2,), 2, history=6, history_acceptance=(4,))
    q2_access = state_accesses(q2_conv)
    # The selected history window and the speculative reload are separate
    # ranges; no continuous seven-row span exceeds the six-row storage.
    assert q2_access[0]["span"] == ((3, 6), (0, 36864))
    assert q2_access[0]["bytes"] == 3 * 36864 * 2
    assert q2_access[1]["span"] == ((0, 4), (0, 36864))
    assert q2_access[1]["bytes"] == 4 * 36864 * 2
    assert q2_access[0]["source"] == "external"
    assert q2_access[1]["state_version"] == 4100
    assert q2_access[2]["decision"] == "token_commit"
    assert q2_access[2]["state_version"] == 4101
def test_conv_q4_partial_one_then_q1_reads_1_to_4_and_commits_4099():
    q4 = RequestView("r", 4, 4096, True, "decode",
                     draft_proposal_tokens=3, draft_accepted_tokens=1,
                     draft_advanced_tokens=2)
    _q4, _ = _conv_ops((q4,), 4, history=6, history_acceptance=(1,))
    q1 = RequestView("r", 1, 4098, True, "decode")
    conv, ops = _conv_ops((q1,), 1, history=6, history_acceptance=(2,))
    meta = dict(conv.metadata)
    rebase, = meta["history_rebase"]
    assert (rebase["source_offset"], rebase["dest_offset"],
            rebase["rows"], rebase["bytes"]) == (
        1, 0, 3, 3 * 36864 * 2)
    assert rebase["source_version"] == 4098
    assert rebase["source_window"] == (1, 4)
    assert rebase["dest_window"] == (0, 3)
    assert rebase["protected_rows"] == ()
    assert rebase["reader_hold"] is None
    assert meta["conv_state_contract"]["commit_versions"] == (4099,)
    # The copy is the UPDATE's state producer; no second native offset.
    assert meta["num_accepted_tokens_present"] is False
    copy = next(op for op in ops if op.recipe == "conv_history_rebase")
    assert copy.op_id in conv.deps
    assert dict(copy.metadata)["state_accesses"][0]["source"] == "external"


def _conv_lower(views, *, history=6, history_acceptance=None, state_values=None,
                state_slots=None, state_ready=(), state_reader_holds=None,
                state_staging=False):
    from mindie_sim.prediction.family_causal_conv import lower
    from mindie_sim.prediction.lowering import Builder
    from mindie_sim.prediction.tensors import TensorLowering

    model, runtime, plan = _conv_case(history=history)
    plan = replace(plan, conv_state_elements=plan.conv_width * history)
    model = replace(model, layers=1, layer_plans=(plan,))
    tokens = sum(view.q_len for view in views)
    b = Builder("kda", plan, 0, views, model, "decode")
    tensors = TensorLowering(b)
    key_heads = plan.linear_key_heads or plan.linear_heads
    heads = plan.linear_value_heads or plan.linear_heads
    dk = plan.linear_key_head_dim or plan.linear_head_dim
    dv = plan.linear_head_dim
    key_width, value_width = key_heads * dk, heads * dv
    width = 2 * key_width + value_width
    slices = ({"role": "q", "offset_elements": 0},
              {"role": "k", "offset_elements": key_width},
              {"role": "v", "offset_elements": 2 * key_width})
    producer = b.work(
        "qkv_pack", ("input",), (tokens, 3 * key_width),
        family="tensor_copy", dtype=model.activation_dtype,
        metadata=(("output_format", model.activation_dtype),
                  ("packed_projection_slices", slices)))
    lower(
        b, tensors, plan=plan, views=views, tokens=tokens, sources=(producer,),
        key_width=key_width, value_width=value_width,
        history_acceptance=history_acceptance, state_values=state_values,
        state_slots=state_slots, state_ready=state_ready,
        state_reader_holds=state_reader_holds, state_staging=state_staging)
    return b.ops[-1], b.ops, width


def test_decode_actual_pool_slots_all_heads_and_same_chain():
    from mindie_sim.prediction.operators import kda_recurrent
    from mindie_sim.schema.tensors import TensorLayout, TensorValue

    pool = TensorValue(
        "prev_kda", "fp32", TensorLayout.contiguous((16, 4, 128, 128)),
        output=0, storage="kda.recurrent_pool", binding="external",
        binding_name="fetch_event")
    selection = {
        "kind": "exact", "index_layout": "speculative",
        "index_row_stride": 4, "prefix": [0, 4],
        "state_indices": [[7, 9, 11, 13]], "accepted_tokens": [1],
        "state_capacity": 16,
    }
    op = kda_recurrent.make_op(
        {"lengths": (4,), "heads": 4, "key_heads": 4,
         "state_dtype": "fp32", "state_selection": selection,
         "state_contexts": (4096,), "state_acceptance": ((1, 2),)},
        requests=("r",), state_values=pool,
        state_ready=("fetch_event",), state_decisions=("accept_event",))
    contract = dict(op.metadata)["kda_state_contract"]
    read = contract["read_bindings"][0]
    assert (read["producer"], read["slot"], read["storage"],
            read["offset"], read["state_version"]) == (
        "prev_kda", 7, "kda.recurrent_pool", 458752, 4096)
    accesses = kda_recurrent.state_accesses(op)
    read_access = accesses[0]
    assert read_access["bytes"] == 4 * 128 * 128 * 4
    assert read_access["span"] == ((7, 8), (0, 4), (0, 128), (0, 128))
    slots = {item["state_version"]: item["slot"]
             for item in accesses if item["direction"] == "write"}
    assert slots == {4097: 7, 4098: 9, 4099: 11, 4100: 13}
    writes_4098 = [item for item in accesses
                   if item["direction"] == "write"
                   and item["state_version"] == 4098]
    assert len(writes_4098) == 1
    assert writes_4098[0]["bytes"] == 4 * 128 * 128 * 4
    assert writes_4098[0]["span"] == ((9, 10), (0, 4), (0, 128), (0, 128))
    assert all(item["decision"] == "token_commit" for item in writes_4098)
    rejected = [item for item in accesses
                if item["direction"] == "write" and item["state_version"] > 4098]
    assert rejected and all(item["last_use"] == "last_reader"
                            for item in rejected)
    assert all(item["decision_event"] == "accept_event" for item in rejected)

    # The next invocation consumes the previous candidate's actual output.
    committed = next(
        item for item in contract["write_bindings"]
        if item["state_version"] == 4098)
    next_value = kda_recurrent.state_value(committed)
    nxt = kda_recurrent.make_op(
        {"lengths": (1,), "heads": 4, "key_heads": 4,
         "state_dtype": "fp32", "state_contexts": (4098,)},
        requests=("r",), state_values=(next_value,), state_slots=(9,),
        state_ready=("prev_accept",), state_decisions=("next_accept",))
    next_read = dict(nxt.metadata)["kda_state_contract"]["read_bindings"][0]
    assert (next_read["producer"], next_read["slot"],
            next_read["state_version"]) == (op.op_id, 9, 4098)
    next_access = kda_recurrent.state_accesses(nxt)[0]
    assert next_access["source"] == "prev_accept"
    assert next_access["state_version"] == 4098
    assert dict(nxt.metadata)["kda_state_contract"]["commit_versions"] == (4099,)


def test_conv_rebase_reads_actual_subview_and_protects_reader_hold():
    from mindie_sim.prediction.family_causal_conv import state_accesses
    from mindie_sim.schema.tensors import TensorLayout, TensorValue

    model, _, plan = _conv_case(history=6)
    width = plan.conv_width
    value = TensorValue(
        "prev_conv", plan.conv_state_dtype,
        TensorLayout.contiguous((6, width)), output=0,
        storage="kda.layer[0].conv_state.pool", binding="external",
        binding_name="fetch_event", state_version=4098)
    q1 = RequestView("r", 1, 4098, True, "decode")
    hold = ({"window": (2, 5), "event": "reader_last_use", "grant": "read_grant"},)
    conv, ops, _ = _conv_lower(
        (q1,), history=6, history_acceptance=(2,),
        state_values=(value,), state_slots=(7,), state_ready=("fetch_event",),
        state_reader_holds=hold)
    meta = dict(conv.metadata)
    rebase, = meta["history_rebase"]
    assert (rebase["source_offset"], rebase["dest_offset"],
            rebase["source_window"], rebase["dest_window"],
            rebase["source_version"], rebase["protected_rows"],
            rebase["reader_hold"], rebase["grant"]) == (
        1, 0, (1, 4), (0, 3), 4098, (2, 3), "reader_last_use", "read_grant")
    copy = next(op for op in ops if op.recipe == "conv_history_rebase")
    assert "reader_last_use" in copy.deps
    assert "fetch_event" in copy.deps
    reads = dict(copy.metadata)["state_accesses"]
    writes = dict(copy.metadata)["state_accesses_write"]
    assert reads[0]["span"] == ((1, 4), (0, width))
    assert reads[0]["state_version"] == 4098
    assert reads[0]["source"] == "fetch_event"
    assert reads[0]["slot"] == 7
    assert writes[0]["span"] == ((0, 3), (0, width))
    assert writes[0]["state_version"] == 4098
    assert writes[0]["producer"] == copy.op_id
    update_read = state_accesses(conv)[0]
    assert update_read["source"] == copy.op_id
    assert update_read["state_version"] == 4098
    assert update_read["span"] == ((0, 3), (0, width))


def test_conv_overlap_without_reader_event_uses_paid_staging():
    from mindie_sim.prediction.family_causal_conv import state_accesses
    from mindie_sim.schema.tensors import TensorLayout, TensorValue

    model, _, plan = _conv_case(history=6)
    width = plan.conv_width
    value = TensorValue(
        "prev_conv", plan.conv_state_dtype,
        TensorLayout.contiguous((6, width)), output=0,
        storage="kda.layer[0].conv_state.pool", binding="external",
        binding_name="fetch_event", state_version=4098)
    q1 = RequestView("r", 1, 4098, True, "decode")
    conv, ops, _ = _conv_lower(
        (q1,), history=6, history_acceptance=(2,), state_values=(value,),
        state_slots=(7,), state_ready=("fetch_event",),
        state_reader_holds=({"window": (2, 5)},))
    meta = dict(conv.metadata)
    rebase, = meta["history_rebase"]
    assert rebase["order"] == "staged_then_write"
    assert rebase["allocation"] == "independent_staging"
    assert rebase["staging_bytes"] == 3 * width * dtype_bytes("bf16")
    staging = next(op for op in ops if op.recipe == "conv_history_staging")
    assert staging.hbm_bytes == 2 * rebase["staging_bytes"]
    assert staging.metadata and dict(staging.metadata)["grant"]
    copy = next(op for op in ops if op.recipe == "conv_history_rebase")
    assert staging.op_id in copy.deps
    staging_reads = [
        item for item in state_accesses(conv)
        if item["temporary"]]
    # The conv UPDATE itself has no temporary access; the staging producer
    # owns the paid temporary read/write records.
    assert staging_reads == []
    staging_accesses = dict(staging.metadata)["state_accesses"]
    assert any(item["temporary"] and item["grant"] for item in staging_accesses)
    assert sum(item["bytes"] for item in staging_accesses) == 2 * rebase["staging_bytes"]
    # The UPDATE consumes the rebase output, which consumes the staged value.
    update_read = state_accesses(conv)[0]
    assert update_read["source"] == copy.op_id
    assert update_read["state_version"] == 4098


def test_fwd_h_reads_real_public_t_subviews_and_publishes_layout_output():
    from mindie_sim.schema.tensors import TensorLayout, TensorValue

    public = TensorValue(
        "previous_model", "fp32", TensorLayout.contiguous((2, 128, 128)),
        output="state", storage="model.public_state", binding="external",
        binding_name="model_state_ready", state_version=512)
    op, captured, execution = _run_fwd_h_with_stage_hooks(
        (96,), 2, (512,), state_values=(public,),
        state_ready=("model_state_ready",))
    contract = dict(op.metadata)["kda_state_contract"]
    read = contract["read_bindings"][0]
    assert (read["producer"], read["storage"], read["ready"],
            read["state_version"]) == (
        "previous_model", "model.public_state", "model_state_ready", 512)
    by_head = {binding["target"]: binding
               for binding in contract["initialize_bindings"]}
    head0 = by_head[contract["internal_targets"][0][0]]
    head1 = by_head[contract["internal_targets"][0][1]]
    # The public [V,K] subview is permuted to internal [K,V]; head 1 starts at
    # its real public storage offset instead of a fabricated contiguous name.
    assert head0["layout"]["offset"] == 0
    assert head1["layout"]["offset"] == 128 * 128
    assert head1["storage"] == "state:0:head:1"
    assert head1["producer"] == "kda_fwd_h:initialize"
    publication = contract["publication_bindings"][0]
    assert publication["producer"] == "kda_fwd_h:layout_publication"
    assert publication["state_version"] == 608
    assert publication["storage"] == "public_state:kda.layer[0].recurrent_state[s0]"
    accesses = __import__(
        "mindie_sim.prediction.operators.kda_state",
        fromlist=["state_accesses"]).state_accesses(op)
    reads = [item for item in accesses if item["direction"] == "read"
             and item["position"] == "state"]
    assert reads[0]["storage"] == "model.public_state"
    assert reads[0]["source"] == "model_state_ready"
    assert reads[0]["span"] == ((0, 1), (0, 128), (0, 128))
    assert reads[1]["span"] == ((1, 2), (0, 128), (0, 128))
    public_write = [item for item in accesses if item["position"] == "layout"]
    assert public_write[0]["producer"] == "kda_fwd_h:layout_publication"
    assert public_write[0]["source_ready"] == "kda_fwd_h:layout_publication"
    assert public_write[0]["bytes"] == 2 * 128 * 128 * 4

    ledger, _, held = _apply_stage_hooks(op, captured, execution,
                                         publication=False)
    for _name, committed in held:
        assert committed[contract["targets"][0]] == 512
    ledger, _, _ = _apply_stage_hooks(op, captured, execution,
                                      publication=True)
    assert ledger.committed_version(contract["targets"][0]) == 608


def test_real_execution_binds_same_decode_access_set_for_memory_and_offload():
    from mindie_sim.prediction.execution_state import bind_op_accesses
    from mindie_sim.prediction.operators import kda_recurrent
    from mindie_sim.schema.operators import IterationWork
    from mindie_sim.schema.tensors import TensorLayout, TensorValue
    from mindie_sim.schema.workload import IterationPlan

    pool = TensorValue(
        "previous_kda", "fp32", TensorLayout.contiguous((16, 4, 128, 128)),
        output=0, storage="kda.recurrent_pool", binding="external",
        binding_name="fetch_event")
    selection = {
        "kind": "exact", "index_layout": "speculative",
        "index_row_stride": 4, "prefix": [0, 4],
        "state_indices": [[7, 9, 11, 13]], "accepted_tokens": [1],
        "state_capacity": 16,
    }
    op = kda_recurrent.make_op(
        {"lengths": (4,), "heads": 4, "key_heads": 4,
         "state_dtype": "fp32", "state_selection": selection,
         "state_contexts": (4096,), "state_acceptance": ((1, 2),)},
        requests=("r",), state_values=pool, state_ready=("fetch_event",),
        state_grants=("read_grant",), state_decisions=("accept_event",))
    view = RequestView("r", 4, 4096, True, "decode",
                       draft_proposal_tokens=3, draft_accepted_tokens=1,
                       draft_advanced_tokens=2)
    work = IterationWork(
        (op,), IterationPlan((view,)), 1, 4, 4, None, "decode", "one")
    bound = bind_op_accesses(work, op)
    expected = kda_recurrent.state_accesses(op)
    assert len(bound) == len(expected)
    access = bound[0].access
    memory = bound[0].memory_dict()
    assert access.direction == "read"
    assert access.producer == "previous_kda"
    assert access.state_version == 4096
    assert access.bytes == 4 * 128 * 128 * 4
    assert memory["owner"] == "kda.layer[0]"
    assert memory["representation"] == "state"
    assert memory["persistent"] is True
    assert memory["temporary"] is False
    assert memory["generation"] == 4096
    assert memory["tier"] == "hbm"
    assert memory["source_ready"] == "fetch_event"
    assert memory["grant_ready"] == "read_grant"
    write = next(item for item in bound if item.access.direction == "write"
                 and item.access.state_version == 4098)
    assert write.access.storage.endswith(":slot:9")
    assert write.memory_dict()["request_slot"] == 9
    assert write.memory_dict()["decision_event"] == "accept_event"
    # Compatibility sample for the shared service carrier: its public fields
    # are the existing StateAccess projection, while owner/representation/
    # persistent/temporary/generation/tier remain producer-owned memory fields
    # for the unified carrier to normalise.
    from mindie_sim.prediction.offload import _as_access
    sample = {
        key: value for key, value in {
            "occurrence": access.occurrence, "producer": access.producer,
            "request": access.request, "stage": access.stage,
            "layer": access.layer, "direction": access.direction,
            "position": access.position, "output": access.output,
            "storage": access.storage, "state_version": access.state_version,
            "span": access.span, "rank": access.rank, "scope": access.scope,
            "bytes": access.bytes, "source": access.source,
            "reader": access.reader, "grant": access.grant,
            "decision": access.decision, "last_use": access.last_use,
        }.items()}
    assert _as_access(sample).state_version == 4096


def _run_conv_execution(conv, ops, view, *, no_rebase, background=False):
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.family_causal_conv import _planned
    from mindie_sim.prediction.gdn_decode_programs import CONV
    from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
    from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
    from mindie_sim.prediction.structure import parse_case
    from test_kda_state_pipeline import hardware_case
    from mindie_sim.schema.operators import IterationWork, OpSpec
    from mindie_sim.schema.workload import IterationPlan

    def fetch_build(op, geometry):
        return OperatorPlan(
            Work(resources={"hbm": 1}),
            lambda context: KernelProgram((Stage("fetch", latency=5.0),)))

    fetch = OpSpec(
        "fetch", "state_fetch", "decode", 0, 0, ("r",), (1,), (1,), (1,),
        (1,), hbm_bytes=1, mte2_bytes=1, recipe="test_fetch",
        source_boundary="kernel", coverage="test")
    extra = ()
    if background:
        extra = (OpSpec(
            "bg", "hbm_load", "decode", 0, 0, ("r",), (1,), (1,), (1,), (1,),
            hbm_bytes=2_000_000_000, mte2_bytes=2_000_000_000,
            recipe="test_bg", coverage="test"),)
    parsed = parse_case(hardware_case(units=4, split_copy=True))
    execution = Execution(
        parsed, CostModel(parsed),
        recipes=RecipeBook(recipes={
            "test_fetch": PlannedRecipe(fetch_build),
            CONV: PlannedRecipe(_planned)}), trace_limit=5000)
    external = execution.engine.submit("input", {})
    iteration = IterationWork(
        tuple(ops) + (fetch,) + extra, IterationPlan((view,)), 1, 1, 1,
        None, "decode", "one")
    result = execution.add(iteration, 0, 0,
                           external_inputs={"input": external})
    execution.engine.advance(until=(result.completion,))
    reads = [item for item in execution.engine.trace
             if "state_read" in item["task"]
             or "conv_history_rebase:begin" in item["task"]]
    return result.completion.time, reads


def test_conv_state_load_and_rebase_wait_for_real_five_ms_fetch():
    from mindie_sim.prediction.family_causal_conv import state_accesses
    from mindie_sim.schema.tensors import TensorLayout, TensorValue

    model, _, plan = _conv_case(history=6)
    width = plan.conv_width
    value = TensorValue(
        "prev_conv", plan.conv_state_dtype,
        TensorLayout.contiguous((6, width)), output=0,
        storage="kda.layer[0].conv_state.pool", binding="external",
        binding_name="fetch_event", state_version=4098)
    q1 = RequestView("r", 1, 4098, True, "decode")

    # No rebase: the UPDATE's state_read stage starts after the actual fetch.
    plain, plain_ops, _ = _conv_lower(
        (q1,), history=6, state_values=(value,), state_slots=(7,),
        state_ready=("fetch",))
    plain_completion, plain_reads = _run_conv_execution(
        plain, plain_ops, q1, no_rebase=True)
    assert plain_completion >= 5.0
    assert plain_reads and all(item["start_ms"] >= 5.0
                               for item in plain_reads)

    # Rebase: the copy waits for the same fetch and an actual old-reader
    # last-use event; the UPDATE then consumes the copy output.
    held, held_ops, _ = _conv_lower(
        (q1,), history=6, history_acceptance=(2,),
        state_values=(value,), state_slots=(7,), state_ready=("fetch",),
        state_reader_holds=({"window": (2, 5), "event": "fetch"},))
    rebase = next(op for op in held_ops if op.recipe == "conv_history_rebase")
    assert "fetch" in rebase.deps
    held_completion, held_reads = _run_conv_execution(
        held, held_ops, q1, no_rebase=False)
    assert held_completion >= 5.0
    assert held_reads and all(item["start_ms"] >= 5.0
                              for item in held_reads)
    assert state_accesses(held)[0]["source"] == rebase.op_id

    # HBM contention changes the completion time, not this plan's issued work.
    contended, contended_ops, _ = _conv_lower(
        (q1,), history=6, state_values=(value,), state_slots=(7,),
        state_ready=("fetch",))
    baseline = plain.hbm_bytes
    assert contended.hbm_bytes == baseline
    contended_completion, _ = _run_conv_execution(
        contended, contended_ops, q1, no_rebase=True, background=True)
    assert contended_completion > plain_completion
    assert contended.hbm_bytes == plain.hbm_bytes


def test_conv_partial_two_three_chain_reads_2_to_5_and_commits_4100():
    q4 = RequestView("r", 4, 4096, True, "decode",
                     draft_proposal_tokens=3, draft_accepted_tokens=2,
                     draft_advanced_tokens=3)
    first, _ = _conv_ops((q4,), 4, history=6, history_acceptance=(0,))
    assert dict(first.metadata)["conv_state_contract"]["commit_versions"] == (
        4099,)
    q1 = RequestView("r", 1, 4099, True, "decode")
    conv, ops = _conv_ops((q1,), 1, history=6, history_acceptance=(3,))
    meta = dict(conv.metadata)
    rebase, = meta["history_rebase"]
    assert (rebase["source_offset"], rebase["dest_offset"],
            rebase["source_window"], rebase["dest_window"],
            rebase["source_version"], rebase["accepted_tokens"]) == (
        2, 0, (2, 5), (0, 3), 4099, 3)
    assert meta["conv_state_contract"]["commit_versions"] == (4100,)
    copy = next(op for op in ops if op.recipe == "conv_history_rebase")
    assert dict(copy.metadata)["state_accesses"][0]["span"] == (
        (2, 5), (0, 36864))
    assert dict(copy.metadata)["state_accesses"][0]["state_version"] == 4099


def test_conv_full_q4_then_q2_consumes_one_committed_history_window():
    q4 = RequestView("r", 4, 4096, True, "decode",
                     draft_proposal_tokens=3, draft_accepted_tokens=3,
                     draft_advanced_tokens=4)
    first, _ = _conv_ops((q4,), 4, history=6, history_acceptance=(1,))
    assert dict(first.metadata)["conv_state_contract"]["commit_versions"] == (
        4100,)
    q2 = RequestView("r", 2, 4100, True, "decode",
                     draft_proposal_tokens=1, draft_accepted_tokens=0,
                     draft_advanced_tokens=1)
    conv, ops = _conv_ops((q2,), 2, history=6, history_acceptance=(4,))
    meta = dict(conv.metadata)
    assert meta["history_rebase"] == ()
    assert meta["num_accepted_tokens_present"] is True
    assert meta["conv_state_contract"]["commit_versions"] == (4101,)
    selection = meta["conv_state_contract"]["history_selection"][0]
    assert selection["source_offset"] == 3
    assert selection["history_window"] == (3, 6)
    assert selection["current_window"] == (0, 4)
    assert not any(op.recipe == "conv_history_rebase" for op in ops)
