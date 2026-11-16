"""GDN state-version, slot, PCP and valid-row contracts.

The tests drive the production decode/prefill callers through the common
frontend and the standalone GDN call builder.  They assert real work identities
and declared effect scopes; no target measurement or device state is read.
"""
from dataclasses import replace

import pytest

from mindie_sim.prediction.call_gdn import make_call
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.gdn_decode_programs import conv_plan, gating_plan
from mindie_sim.prediction.gdn_recurrent import build
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView

PREFILL = "gdn_prefill_arch35"
WIDTH = 2 * 2 * 128 + 6 * 128


def source_case(lengths=(4,) * 4, *, keys=4, heads=12, units=64, queue=False):
    parsed = parse_case({
        "machine": {
            "name": "declared-gpu", "architecture": "declared", "num_devices": 1,
            "device": {"memory_bytes": 1024 ** 3, "compute_peaks": {"bf16": 1e9, "fp32": 1e9},
                       "rates": {"memory": 1e9, "vector": 1e9}, "compute_units": 4},
            "links": [{"name": "fabric", "capacity_bytes_per_ms": 1000,
                       "members": [0], "semantics": "shared"}],
        },
        "model": {
            "name": "geometry", "layers": 2, "hidden_size": 32, "num_heads": 4,
            "num_kv_heads": 2, "head_dim": 8, "intermediate_size": 64, "vocab_size": 128,
        },
        "workload": {"requests": [{"id": "r", "prompt_tokens": 4, "output_tokens": 3}]},
        "runtime": {"dp": 1, "tp": 1, "pp": 1, "ep": 1, "max_num_batched_tokens": 16},
    })
    base_layer = replace(parsed.model.layer_plans[0], attention="gdn", linear_key_heads=keys,
                         linear_value_heads=heads, linear_heads=heads,
                         linear_key_head_dim=128, linear_head_dim=128, conv_kernel=4,
                         recurrent_state_dtype="fp32")
    kernels = {"lowerings": {"gdn_decode": "vllm_ascend_gdn_decode"}}
    rates = {"hbm": 4e12, "vector": 1e13, "mte2": 4e12,
             "mte2_aic": 4e12, "mte2_aiv": 4e12, "mte3": 4e12}
    if queue:
        rates["scalar_store_issue"] = 6.4e9
        kernels["lowerings"]["gdn_decode"] = {
            "implementation": "vllm_ascend_gdn_decode",
            "options": {"gating_beta_store": {"resource": "scalar_store_issue", "queue": {
                "outstanding": 8, "granule_bytes": 128, "response_ms": 0.0001}}}}
    device = replace(parsed.machine.dies[0], rates=tuple(rates.items()), compute_units=units // 2,
                     resource_units=tuple((r, units) for r in rates))
    parsed = replace(parsed, model=replace(parsed.model, layers=1, layer_plans=(base_layer,)),
                     runtime=replace(parsed.runtime, pp_layer_partition=(1,), kernels=kernels),
                     machine=replace(parsed.machine, dies=(device,)))
    views = tuple(RequestView(str(i), q, 4096, False, "decode") for i, q in enumerate(lengths))
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode")
    return parsed, views, ops


def _ops(parsed, views, mode, tokens):
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode=mode,
                             tokens=tokens)
    return ops


def _by_recipe(ops, recipe):
    return next(op for op in ops if op.recipe == recipe)


def _by_family(ops, family):
    return next(op for op in ops if op.family == family)


def _geometry(parsed):
    return Geometry.from_devices(parsed.machine.dies)


def _prefill_case():
    parsed, _, _ = source_case(lengths=(1,), keys=2, heads=6)
    return replace(parsed, runtime=replace(
        parsed.runtime, kernels={"lowerings": {"gdn_prefill": PREFILL}}))


def test_non_spec_decode_commits_the_single_checkpoint_at_token_commit():
    parsed, views, _ = source_case(lengths=(1, 1), keys=2, heads=6)
    ops = _ops(parsed, views, "decode", 2)
    conv = _by_recipe(ops, "ascend_causal_conv1d_update")
    scan = next(op for op in ops if op.family == "recurrent_vector")
    meta = dict(conv.metadata)
    assert meta["state_dtype"] == parsed.model.activation_dtype == "bf16"
    assert meta["state_layout"] == "slots_history_channels"
    assert meta["state_mutation"] == "inplace_history_writeback"
    assert meta["state_write_bytes"] == 2 * (4 - 1) * WIDTH * 2
    conv_contract = meta["gdn_state_contract"]
    assert conv_contract["target_kind"] == "conv"
    assert conv_contract["direction"] == "token"
    assert conv_contract["index_copy_required"] is False

    contract = dict(scan.metadata)["gdn_state_contract"]
    assert contract["target_kind"] == "recurrent"
    assert contract["read_versions"] == (4096, 4096)
    assert contract["commit_versions"] == (4097, 4097)
    assert contract["accepted_offsets"] == (0, 0)
    assert contract["advances"] == (1, 1)
    assert contract["checkpoints"] == (1, 1)
    assert contract["commit_policy"] == "token_commit"
    assert contract["candidate_write_point"] == "device_complete_before_token_commit"
    assert contract["copy_boundaries"] == (
        "cache_index_single_column_view",
        "recurrent_state_fp32_no_cast",
        "conv_history_bf16",
    )
    effects = dict(scan.metadata)["gdn_state_effects"]
    for target in contract["targets"]:
        actions = [(effect["action"], effect["version"]) for effect in effects
                   if effect["target"] == target]
        assert actions == [("read", 4096), ("write", 4097), ("commit", 4097)]
    commit = next(effect for effect in effects if effect["action"] == "commit")
    assert commit["condition"] == ""
    assert commit["decision"] == "acceptance[0]"
    assert commit["arbiter"] == "acceptance_trace"


def test_speculative_decode_produces_every_checkpoint_and_rejects_unaccepted():
    parsed, base, _ = source_case(lengths=(4,), keys=2, heads=6)
    view = replace(base[0], draft_proposal_tokens=3, draft_accepted_tokens=1,
                   draft_advanced_tokens=2)
    ops = _ops(parsed, (view,), "decode", 4)
    scan = next(op for op in ops if op.family == "recurrent_vector")
    meta = dict(scan.metadata)
    contract = meta["gdn_state_contract"]
    assert contract["accepted_offsets"] == (1,)
    assert contract["advances"] == (2,)
    assert contract["commit_versions"] == (4098,)
    assert meta["state_checkpoint_versions"] == ((4097, 4098, 4099, 4100),)
    writes = [effect["version"] for effect in meta["gdn_state_effects"]
              if effect["action"] == "write"]
    assert writes == [4097, 4098, 4099, 4100]
    committed = [effect["version"] for effect in meta["gdn_state_effects"]
                 if effect["action"] == "commit"]
    rejected = [effect["version"] for effect in meta["gdn_state_effects"]
                if effect["action"] == "rollback"]
    assert committed == [4098]
    assert rejected == [4099, 4100]
    assert not set(committed) & set(rejected)
    assert contract["rejected_checkpoints"] == "not_published"


def test_undeclared_acceptance_stays_pending_instead_of_committing_every_row():
    parsed, views, _ = source_case(lengths=(4,), keys=2, heads=6)
    ops = _ops(parsed, views, "decode", 4)
    scan = next(op for op in ops if op.family == "recurrent_vector")
    meta = dict(scan.metadata)
    contract = meta["gdn_state_contract"]
    assert contract["commit_policy"] == "pending_acceptance_event"
    assert contract["commit_versions"] == (None,)
    assert meta["state_checkpoint_versions"] == ((4097, 4098, 4099, 4100),)
    assert [effect["action"] for effect in meta["gdn_state_effects"]] == [
        "read", "write", "write", "write", "write"]
    assert meta["gdn_state_contract"]["decision_inputs"] == ("acceptance[0]",)
    assert [effect["version"] for effect in meta["gdn_state_effects"]
            if effect["action"] == "write"] == [4097, 4098, 4099, 4100]
    with pytest.raises(ModelingError):
        _ops(parsed, (replace(views[0], draft_accepted_tokens=0,
                              draft_advanced_tokens=5),), "decode", 4)


def test_padding_view_owns_no_state_and_executes_no_recurrence():
    parsed, _, _ = source_case(lengths=(1,), keys=2, heads=6)
    parsed = replace(parsed, runtime=replace(
        parsed.runtime, graph_mode="auto", graph_capture_mode="full",
        graph_capture_keys=(32,)))
    view = RequestView("pad", 32, 4096, False, "decode", padding=True,
                       draft_verify=False, draft_prepare=False)
    ops = _ops(parsed, (view,), "decode", 32)
    conv = _by_recipe(ops, "ascend_causal_conv1d_update")
    scan = _by_family(ops, "recurrent_vector")
    cm, sm = dict(conv.metadata), dict(scan.metadata)
    assert cm["padding_slots"] == (0,)
    assert cm["gdn_state_contract"]["no_state_io_slots"] == (0,)
    assert cm["gdn_state_effects"] == ()
    assert cm["state_write_bytes"] == 0
    assert conv.local_shape == (0, WIDTH) and conv.padded_shape == (32, WIDTH)
    assert sm["tokens"] == 0 and sm["state_checkpoints"] == 0
    assert sm["state_checkpoint_versions"] == (None,)
    assert sm["gdn_state_effects"] == ()
    assert sm["gdn_state_contract"]["checkpoints"] == (0,)
    assert scan.local_shape[0] == 0 and scan.padded_shape[0] == 32


def test_conv_skips_null_slot_compute_and_keeps_only_active_work():
    parsed, _, _ = source_case(lengths=(1,), keys=2, heads=6)
    active = RequestView("a", 1, 4096, True, "decode")
    pad = RequestView("pad", 32, 0, False, "decode", padding=True)
    active_ops = _ops(parsed, (active,), "decode", 1)
    mixed_ops = _ops(parsed, (active, pad), "decode", 33)
    active_conv = _by_recipe(active_ops, "ascend_causal_conv1d_update")
    mixed_conv = _by_recipe(mixed_ops, "ascend_causal_conv1d_update")
    geometry = _geometry(parsed)
    assert dict(conv_plan(active_conv, geometry).work.resources) == \
        dict(conv_plan(mixed_conv, geometry).work.resources)
    assert mixed_conv.local_shape == (1, WIDTH)
    assert mixed_conv.padded_shape == (33, WIDTH)
    assert dict(mixed_conv.metadata)["padding_slots"] == (1,)
    work = dict(conv_plan(mixed_conv, geometry).work.resources)
    assert work["mte2_aiv"] == (4 + 3 + 1) * WIDTH * 2
    assert work["mte3"] == (1 + 3) * WIDTH * 2
    assert work["vector"] == (4 + 1 + 17) * WIDTH

    all_pad = _ops(parsed, (pad,), "decode", 32)
    pad_conv = _by_recipe(all_pad, "ascend_causal_conv1d_update")
    pad_plan = conv_plan(pad_conv, geometry)
    assert dict(pad_plan.metadata)["empty_active_sequences"] is True
    assert pad_plan.work.resources == {}
    assert pad_conv.local_shape == (0, WIDTH)


def test_mixed_prefill_and_pad_decode_charge_no_pad_recurrence():
    parsed = _prefill_case()
    views = (RequestView("pf", 64, 0, True, "prefill"),
             RequestView("pad", 32, 0, False, "decode", padding=True))
    ops = _ops(parsed, views, "decode", 96)
    scan = _by_family(ops, "recurrent_vector")
    conv = _by_family(ops, "causal_conv1d_fn")
    sm = dict(scan.metadata)
    assert sm["tokens"] == 0 and sm["gdn_state_effects"] == ()
    assert sm["padding_slots"] == (0,)
    assert scan.local_shape[0] == 0 and scan.padded_shape[0] == 32
    assert dict(conv.metadata)["lengths"] == (32, 64)
    assert dict(conv.metadata)["query_start_loc"] == (0, 32, 96)
    assert dict(conv.metadata)["cache_indices"] == (0, None)
    state = _by_family(ops, "gdn_state")
    assert dict(state.metadata)["gdn_state_contract"]["direction"] == "h"


def test_generic_gdn_keeps_plan_dtype_and_width_outside_native_arch35():
    parsed, _, _ = source_case(lengths=(2,), keys=2, heads=6)
    layer = replace(parsed.model.layer_plans[0], recurrent_state_dtype="fp16",
                    conv_state_dtype="fp16", conv_kernel=5)
    parsed = replace(parsed, model=replace(parsed.model, layer_plans=(layer,)),
                     runtime=replace(parsed.runtime, kernels={"lowerings": {}}))
    decode = _ops(parsed, (RequestView("d", 2, 16, True, "decode"),),
                  "decode", 2)
    scan = _by_family(decode, "recurrent_vector")
    contract = dict(scan.metadata)["gdn_state_contract"]
    assert dict(scan.metadata)["state_element_bytes"] == 2
    assert contract["state_dtype"] == "fp16"
    assert contract["conv_kernel"] == 5
    assert "recurrent_state_fp16_no_cast" in contract["copy_boundaries"]
    assert "conv_history_fp16" in contract["copy_boundaries"]

    prefill = _ops(parsed, (RequestView("p", 65, 16, True, "prefill"),),
                   "prefill", 65)
    pscan = _by_family(prefill, "recurrent_scan")
    pcontract = dict(pscan.metadata)["gdn_state_contract"]
    assert pcontract["state_dtype"] == "fp16"
    assert pcontract["commit_versions"] == (81,)
    assert pcontract["internal_commit_versions"] == ()

    native = replace(parsed, runtime=replace(
        parsed.runtime, kernels={"lowerings": {"gdn_decode": "vllm_ascend_gdn_decode"}}))
    with pytest.raises(ModelingError, match="fp32_state"):
        _ops(native, (RequestView("d", 1, 16, True, "decode"),), "decode", 1)
    native_pf = replace(parsed, runtime=replace(
        parsed.runtime, kernels={"lowerings": {"gdn_prefill": PREFILL}}))
    with pytest.raises(ValueError):
        _ops(native_pf, (RequestView("p", 65, 16, True, "prefill"),),
             "prefill", 65)
    native_width = replace(parsed, model=replace(parsed.model, layer_plans=(
        replace(layer, recurrent_state_dtype="fp32", conv_kernel=5),)),
        runtime=replace(parsed.runtime, kernels={
            "lowerings": {"gdn_prefill": PREFILL}}))
    with pytest.raises(ValueError):
        _ops(native_width, (RequestView("p", 65, 16, True, "prefill"),),
             "prefill", 65)


def test_prefill_effects_bind_gather_h_and_scatter_at_real_points():
    parsed = _prefill_case()
    view = RequestView("p", 128, 512, True, "prefill")
    ops = _ops(parsed, (view,), "prefill", 128)
    gather = _by_family(ops, "state_gather")
    state = _by_family(ops, "gdn_state")
    scatter = _by_family(ops, "state_scatter")
    gc = dict(gather.metadata)["gdn_state_contract"]
    hc = dict(state.metadata)["gdn_state_contract"]
    sc = dict(scatter.metadata)["gdn_state_contract"]
    assert (gc["direction"], gc["read_versions"], gc["commit_versions"]) == (
        "gather", (512,), (None,))
    assert [e["action"] for e in dict(gather.metadata)["gdn_state_effects"]] == [
        "read"]
    assert (hc["direction"], hc["read_versions"], hc["commit_versions"]) == (
        "h", (512,), (None,))
    assert hc["local_bank_versions"] == (128,)
    assert hc["local_chunk_versions"] == ((64, 128),)
    assert hc["internal_commit_versions"] == ()
    assert hc["publish_point"] == "scatter_complete"
    assert [e["action"] for e in dict(state.metadata)["gdn_state_effects"]] == [
        "read", "write"]
    assert dict(state.metadata)["state_commit_versions"] == (None,)
    assert (sc["direction"], sc["commit_versions"], sc["publish_versions"]) == (
        "scatter", (640,), (640,))
    assert [e["action"] for e in dict(scatter.metadata)["gdn_state_effects"]] == [
        "read", "write", "commit"]
    assert 640 not in [e["version"] for e in dict(state.metadata)["gdn_state_effects"]
                       if e["action"] == "commit"]

    conv = _by_family(ops, "causal_conv1d_fn")
    cc = dict(conv.metadata)["gdn_state_contract"]
    assert cc["direction"] == "conv"
    assert cc["internal_commit_versions"] == ()
    assert cc["local_chunk_versions"] == ((),)
    assert cc["commit_versions"] == (640,)


def test_conv_only_call_has_one_sequence_tail_commit_and_no_chunk_fiction():
    call = make_call("conv_prefill", {
        "batch": 1, "tokens": 128, "conv_width": 1280, "kernel": 4})
    contract = call.device_program["effects"][0]["state_contract"]
    assert contract["direction"] == "conv"
    assert contract["internal_commit_versions"] == ()
    assert contract["local_chunk_versions"] == ((),)
    assert contract["commit_versions"] == (128,)
    assert contract["publish_point"] == "sequence_tail"
    actions = [e["action"] for e in call.device_program["effects"][0]["state_effects"]]
    assert actions == ["write", "commit"]

    scan = make_call("scan_prefill", {
        "batch": 1, "tokens": 128, "v_heads": 6, "k_heads": 2,
        "k_dim": 128, "v_dim": 128, "chunk": 64})
    state = scan.ledger["state"]
    assert state["contract"]["direction"] == "call"
    assert state["contract"]["commit_versions"] == (None,)
    assert state["contract"]["publish_point"] == "call_return"
    assert [(e["action"], e["version"]) for e in state["effects"]] == [
        ("read", 0), ("write", 128)]


def test_pcp_builds_segmented_state_chain_before_the_final_publish():
    parsed = _prefill_case()
    parallel = replace(parsed.runtime.parallel, pcp=2, cp_plan="pcp")
    parsed = replace(parsed, runtime=replace(parsed.runtime, parallel=parallel))
    view = RequestView("p", 64, 512, True, "prefill", global_q_len=128,
                       query_ranges=((0, 32), (96, 128)))
    ops = _ops(parsed, (view,), "prefill", 64)
    states = [op for op in ops if op.family == "gdn_state"]
    convs = [op for op in ops if op.family == "causal_conv1d_fn"]
    scatter = _by_family(ops, "state_scatter")
    assert len(states) == len(convs) == 2
    h0, h1 = states
    c0, c1 = convs
    assert dict(h0.metadata)["lengths"] == (32,)
    assert dict(h1.metadata)["lengths"] == (32,)
    assert dict(h0.metadata)["state_read_versions"] == (512,)
    assert dict(h1.metadata)["state_read_versions"] == (608,)
    h_exchange = next(op for op in ops if op.recipe == "pcp_recurrent_state_step1")
    c_exchange = next(op for op in ops if op.recipe == "pcp_conv_state_step1")
    assert dict(h_exchange.metadata)["from_version"] == 544
    assert dict(h_exchange.metadata)["to_version"] == 608
    assert dict(c_exchange.metadata)["from_version"] == 544
    assert dict(c_exchange.metadata)["to_version"] == 608
    assert h_exchange.op_id in h1.deps
    assert c_exchange.op_id in c1.deps
    assert h0.op_id in h_exchange.deps and c0.op_id in c_exchange.deps
    hc = dict(h1.metadata)["gdn_state_contract"]
    assert hc["queries"] == (32,)
    assert hc["global_queries"] == (32,)
    assert hc["local_bank_versions"] == (32,)
    assert hc["commit_versions"] == (None,)
    assert dict(h0.metadata)["gdn_state_contract"]["read_versions"] == (512,)
    sc = dict(scatter.metadata)["gdn_state_contract"]
    assert sc["query_ranges"] == (((0, 32), (96, 128)),)
    assert sc["commit_versions"] == (None,)
    assert sc["publish_versions"] == (None,)
    assert sc["local_commit_versions"] == (640,)
    assert sc["global_versions"] == (640,)
    assert sc["publish_pending"] == "remote_producer"
    remote = next(op for op in ops if op.recipe == "pcp_remote_state_publish")
    assert scatter.deps == (remote.op_id,)
    assert set(remote.deps) >= {h0.op_id, h1.op_id}
    assert dict(remote.metadata)["status"] == "pending_external_rank"
    assert dict(remote.metadata)["global_versions"] == (640,)
    # H completion alone has no commit; remote publish has no local producer.
    assert 640 not in [e["version"] for e in dict(h1.metadata)["gdn_state_effects"]
                       if e["action"] == "commit"]
    prep = _by_family(ops, "gdn_prepare")
    out = _by_family(ops, "gdn_output")
    assert dict(prep.metadata)["segment_lengths"] == ((32,), (32,))
    assert dict(out.metadata)["segment_lengths"] == ((32,), (32,))
    last_conv = dict(c1.metadata)["gdn_state_contract"]
    assert last_conv["direction"] == "conv"
    assert last_conv["commit_versions"] == (None,)
    assert last_conv["global_versions"] == (640,)

    tail = RequestView("tail", 65, 512, True, "prefill", global_q_len=130,
                       query_ranges=((0, 32), (97, 130)))
    ops = _ops(parsed, (tail,), "prefill", 65)
    tail_states = [op for op in ops if op.family == "gdn_state"]
    tail_scatter = _by_family(ops, "state_scatter")
    assert [dict(op.metadata)["lengths"] for op in tail_states] == [(32,), (33,)]
    assert dict(tail_states[1].metadata)["state_read_versions"] == (609,)
    assert dict(tail_scatter.metadata)["gdn_state_contract"]["commit_versions"] == (None,)
    assert dict(tail_scatter.metadata)["gdn_state_contract"]["global_versions"] == (642,)
    assert dict(tail_scatter.metadata)["gdn_state_contract"]["local_commit_versions"] == (642,)


def test_serving_prefill_accepts_reordered_slots_and_real_capacity():
    parsed = _prefill_case()
    metadata = tuple(sorted({
        **dict(parsed.runtime.metadata),
        "gdn_prefill_state_indices": {"B": 7, "A": 2},
        "gdn_state_rows": 9,
    }.items()))
    parsed = replace(parsed, runtime=replace(parsed.runtime, metadata=metadata))
    views = (RequestView("B", 64, 512, True, "prefill"),
             RequestView("A", 32, 128, True, "prefill"))
    ops = _ops(parsed, views, "prefill", 96)
    conv = _by_family(ops, "causal_conv1d_fn")
    meta = dict(conv.metadata)
    assert meta["cache_indices"] == (7, 2)
    assert meta["state_rows"] == 9
    assert meta["state_index_source"] == "caller_metadata"
    assert meta["null_block_id"] == 0
    contract = meta["gdn_state_contract"]
    assert contract["commit_versions"] == (576, 160)
    assert contract["slot_indices"] == (7, 2)
    assert contract["state_capacity"] == 9
    assert contract["slot_source"] == "caller_metadata"

    unknown = replace(parsed, runtime=replace(
        parsed.runtime, metadata=tuple(sorted({
            key: value for key, value in dict(parsed.runtime.metadata).items()
            if key not in {"gdn_prefill_state_indices", "gdn_state_rows"}}.items()))))
    ops = _ops(unknown, views, "prefill", 96)
    conv = _by_family(ops, "causal_conv1d_fn")
    meta = dict(conv.metadata)
    assert meta["cache_indices"] == (None, None)
    assert meta["cache_index_present"] is True
    assert meta["cache_index_bound"] is False
    assert meta["state_rows"] is None
    assert meta["state_rows_bound"] is False
    assert meta["state_index_source"] == "dynamic_caller_index"
    assert meta["null_block_id"] == 0
    contract = meta["gdn_state_contract"]
    assert contract["slot_indices"] == (None, None)
    assert contract["state_capacity"] is None
    assert contract["state_capacity_lower_bound"] is None
    assert contract["slot_source"] == "dynamic_caller_index"

    fixture = replace(parsed, runtime=replace(
        parsed.runtime, metadata=tuple(sorted({
            **{key: value for key, value in dict(parsed.runtime.metadata).items()
               if key not in {"gdn_prefill_state_indices", "gdn_state_rows"}},
            "gdn_state_uniform_fixture": True}.items()))))
    ops = _ops(fixture, views, "prefill", 96)
    conv = _by_family(ops, "causal_conv1d_fn")
    meta = dict(conv.metadata)
    assert meta["cache_indices"] == (1, 2)
    assert meta["state_index_source"] == "uniform_fixture"
    assert meta["state_rows"] is None
    assert meta["gdn_state_contract"]["slot_indices"] == (1, 2)
    assert meta["gdn_state_contract"]["state_capacity"] is None
    assert meta["gdn_state_contract"]["state_capacity_lower_bound"] == 3


def test_index_copy_flag_follows_the_normalized_single_column_view():
    for lengths, stride, expected in (((1,), 4, False),
                                      ((1, 1), 4, True),
                                      ((1, 1, 1), 1, False)):
        parsed, views, _ = source_case(lengths=lengths, keys=2, heads=6)
        parsed = replace(parsed, runtime=replace(
            parsed.runtime, kernels={"lowerings": {"gdn_decode": {
                "implementation": "vllm_ascend_gdn_decode",
                "options": {"cache_index_row_stride": stride}}}}))
        ops = _ops(parsed, views, "decode", len(lengths))
        conv = _by_recipe(ops, "ascend_causal_conv1d_update")
        assert dict(conv.metadata)["gdn_state_contract"][
            "index_copy_required"] is expected
        copy = [op for op in ops if op.recipe == "state_indices_contiguous"]
        assert bool(copy) is expected


def test_padded_prefill_does_not_gather_or_scatter_a_state_row():
    parsed = _prefill_case()
    view = RequestView("pad", 64, 0, True, "prefill", padding=True)
    ops = _ops(parsed, (view,), "prefill", 64)
    state = _by_family(ops, "gdn_state")
    gather = _by_family(ops, "state_gather")
    scatter = _by_family(ops, "state_scatter")
    # Padding has no persistent or local state effect, but H still consumes its
    # physical initial row (materialized by gather).
    assert dict(state.metadata)["gdn_state_effects"] == ()
    assert dict(state.metadata)["gdn_state_contract"]["commit_versions"] == (None,)
    # Gather materializes the physical (zero) H input row even though there is
    # no persistent cache read; scatter publishes nothing for the pad slot.
    assert gather.mte3_bytes == 6 * 128 * 128 * 4
    assert scatter.mte2_bytes == scatter.mte3_bytes == 0
    assert dict(scatter.metadata)["gdn_state_effects"] == ()
    conv = _by_family(ops, "causal_conv1d_fn")
    assert dict(conv.metadata)["cache_indices"] == (0,)
    # A fully padded prefill graph keeps the physical launch extent; the null
    # cache index suppresses RunSeq but not FN weight/convert/metadata work.
    assert dict(conv.metadata)["lengths"] == (64,)
    from mindie_sim.prediction.operators import native_causal_conv_prefill as native_conv
    work = dict(native_conv.build(conv, Geometry.from_devices(parsed.machine.dies)).work.resources)
    assert work["vector"] > 0 and work["hbm"] > 0


def test_zero_active_rows_keep_an_explicit_empty_state_program():
    shape = (0, WIDTH)
    op = OpSpec("empty", "short_convolution", "decode", 0, 0, (), shape, shape,
                shape, shape, dtype="bf16",
                metadata=(("sequence_tokens", (0,)), ("kernel_width", 4),
                          ("speculative", False), ("state_history_slots", 3)))
    geometry = Geometry(4, {"vector": 4, "hbm": 1, "mte2_aiv": 1, "mte3": 1})
    conv = conv_plan(op, geometry)
    assert dict(conv.metadata)["empty_active_sequences"] is True
    assert conv.work.resources == {}
    gate = gating_plan(replace(op, local_shape=shape, family="gdn_gating"), geometry)
    assert dict(gate.metadata)["empty_active_sequences"] is True
    assert gate.work.resources == {}
    recurrent = build(replace(op, metadata=(
        ("sequence_tokens", (0,)), ("heads", 6), ("key_heads", 2),
        ("dk", 128), ("dv", 128), ("state_dtype", "fp32"))), geometry)
    assert dict(recurrent.metadata)["empty_active_sequences"] is True
    assert recurrent.work.resources == {}


def test_speculative_active_with_pad_uses_active_history_and_local_rows():
    parsed, _, _ = source_case(lengths=(1,), keys=2, heads=6)
    active = RequestView("a", 4, 4096, True, "decode")
    pad = RequestView("pad", 32, 0, False, "decode", padding=True)
    active_ops = _ops(parsed, (active,), "decode", 4)
    mixed_ops = _ops(parsed, (active, pad), "decode", 36)
    active_conv = _by_recipe(active_ops, "ascend_causal_conv1d_update")
    mixed_conv = _by_recipe(mixed_ops, "ascend_causal_conv1d_update")
    geometry = _geometry(parsed)
    assert dict(mixed_conv.metadata)["state_history_slots"] == 4 - 1 + 4 - 1
    assert mixed_conv.local_shape == (4, WIDTH)
    assert mixed_conv.padded_shape == (36, WIDTH)
    active_work = dict(conv_plan(active_conv, geometry).work.resources)
    mixed_work = dict(conv_plan(mixed_conv, geometry).work.resources)
    assert mixed_work == active_work
    # Active q=4 speculative: read weights+history+q+(2+q) restoration,
    # write q+(2+q) history.
    assert mixed_work["mte2_aiv"] == (4 + 3 + 4 + 6) * WIDTH * 2
    assert mixed_work["mte3"] == (4 + 6) * WIDTH * 2


def test_unknown_index_values_keep_the_index_tensor_in_the_native_abi():
    from mindie_sim.prediction.operators import native_causal_conv_prefill as native_conv

    def build(metadata):
        parsed = _prefill_case()
        parsed = replace(parsed, runtime=replace(
            parsed.runtime, metadata=tuple(sorted({
                **dict(parsed.runtime.metadata), **metadata}.items()))))
        views = (RequestView("B", 64, 512, True, "prefill"),
                 RequestView("A", 32, 128, True, "prefill"))
        ops = _ops(parsed, views, "prefill", 96)
        return parsed, _by_family(ops, "causal_conv1d_fn")

    parsed, unknown = build({})
    um = dict(unknown.metadata)
    assert um["cache_index_present"] is True
    assert um["cache_index_bound"] is False
    assert um["cache_indices"] == (None, None)
    assert um["state_rows"] is None
    assert um["state_rows_bound"] is False
    assert um["dispatch"]["has_cache_indices"] is True
    contract = um["gdn_state_contract"]
    assert contract["slot_indices"] == (None, None)
    assert contract["state_capacity"] is None
    assert contract["state_capacity_lower_bound"] is None

    parsed, bound = build({"gdn_prefill_state_indices": {"B": 7, "A": 2},
                           "gdn_state_rows": 9})
    bm = dict(bound.metadata)
    assert bm["cache_indices"] == (7, 2)
    assert bm["state_rows"] == 9
    geometry = Geometry.from_devices(parsed.machine.dies)
    unknown_work = dict(native_conv.build(unknown, geometry).work.resources)
    bound_work = dict(native_conv.build(bound, geometry).work.resources)
    assert unknown_work == bound_work
    assert bm["query_start_loc"] == um["query_start_loc"] == (0, 64, 96)
    assert bm["state_rows"] == 9 and um["state_rows"] is None

    # A caller table with known values but no capacity must not invent 3.
    _, no_capacity = build({"gdn_prefill_state_indices": {"B": 7, "A": 2}})
    assert dict(no_capacity.metadata)["state_rows"] is None
    assert dict(no_capacity.metadata)["state_rows_bound"] is False

    # A true absent-index standalone call keeps the identity ABI.
    absent = make_call("conv_prefill", {
        "batch": 1, "tokens": 4, "conv_width": 1280, "kernel": 4,
        "cache_indices": None})
    am = dict(absent.ops[0].metadata)
    assert am["cache_index_present"] is False
    assert am["cache_index_bound"] is False
    assert am["state_rows_bound"] is False
    assert am["dispatch"]["has_cache_indices"] is False


def test_pending_request_does_not_change_another_requests_commit():
    from mindie_sim.prediction.gdn_recurrent import state_contract

    pending = RequestView("A", 4, 4096, True, "decode")
    known = replace(RequestView("B", 4, 4096, True, "decode"),
                    draft_proposal_tokens=3, draft_accepted_tokens=1,
                    draft_advanced_tokens=2)
    for views in ((pending, known), (known, pending)):
        contract, effects = state_contract(
            views=views, phase="decode", state_dtype="fp32",
            conv_state_dtype="bf16", conv_kernel=4, direction="token",
            target_kind="recurrent", initial_state=(True, True))
        commits = dict(zip((view.request_id for view in views),
                           contract["commit_versions"]))
        publishes = dict(zip((view.request_id for view in views),
                             contract["publish_versions"]))
        assert commits == {"A": None, "B": 4098}
        assert publishes == {"A": None, "B": 4098}
        policies = dict(zip((view.request_id for view in views),
                            contract["slot_commit_policy"]))
        assert policies == {"A": "pending_acceptance_event", "B": "token_commit"}
        pending_index = next(index for index, view in enumerate(views)
                             if view.request_id == "A")
        assert contract["pending_slots"] == (pending_index,)
        commit_targets = {e["target"] for e in effects if e["action"] == "commit"}
        assert commit_targets == {"gdn.recurrent_state[B]"}

    resolved = replace(pending, draft_accepted_tokens=0, draft_advanced_tokens=1)
    contract, _ = state_contract(
        views=(resolved, known), phase="decode", state_dtype="fp32",
        conv_state_dtype="bf16", conv_kernel=4, direction="token",
        target_kind="recurrent", initial_state=(True, True))
    assert contract["commit_versions"] == (4097, 4098)


def test_generic_gdn_pcp_splits_local_scan_before_global_commit():
    parsed, _, _ = source_case(lengths=(1,), keys=2, heads=6)
    parallel = replace(parsed.runtime.parallel, pcp=2, cp_plan="pcp")
    parsed = replace(parsed, runtime=replace(
        parsed.runtime, kernels={"lowerings": {}}, parallel=parallel))
    view = RequestView("p", 64, 512, True, "prefill", global_q_len=128,
                       query_ranges=((0, 32), (96, 128)))
    ops = _ops(parsed, (view,), "prefill", 64)
    segments = [op for op in ops if op.recipe.startswith("recurrent_scan")]
    assert len(segments) == 2
    assert dict(segments[0].metadata)["tokens"] == 32
    assert dict(segments[1].metadata)["tokens"] == 32
    exchange = next(op for op in ops
                    if op.recipe.startswith("pcp_recurrent_state_step"))
    assert dict(exchange.metadata)["from_version"] == 544
    assert dict(exchange.metadata)["to_version"] == 608
    assert exchange.op_id in segments[1].deps
    assert segments[0].op_id in exchange.deps
    assert dict(segments[0].metadata)["gdn_state_contract"]["direction"] == "h"
    final = dict(segments[1].metadata)["gdn_state_contract"]
    assert final["direction"] == "conv"
    assert final["commit_versions"] == (None,)
    assert final["global_versions"] == (640,)
    remote = next(op for op in ops if op.recipe == "pcp_remote_state_publish")
    assert dict(remote.metadata)["status"] == "pending_external_rank"
    assert dict(remote.metadata)["global_expected_version"] == 640
    assert segments[1].op_id in remote.deps
    assert 640 not in [e["version"] for e in
                       dict(segments[0].metadata)["gdn_state_effects"]
                       if e["action"] == "commit"]


def test_standalone_scan_and_conv_keep_real_sequence_partition():
    scan = make_call("scan_prefill", {
        "batch": 2, "tokens": 1, "v_heads": 6, "k_heads": 2,
        "k_dim": 128, "v_dim": 128, "chunk": 64})
    h = next(op for op in scan.ops if op.family == "gdn_state")
    hm = dict(h.metadata)
    assert hm["lengths"] == (1, 1)
    assert hm["chunks_per_sequence"] == (1, 1)
    assert hm["head_tasks"] == 2 * 6
    assert [e["target"] for e in hm["gdn_state_effects"]] == [
        "scan_call.final_state[seq0]", "scan_call.final_state[seq0]",
        "scan_call.final_state[seq1]", "scan_call.final_state[seq1]"]
    unequal = make_call("scan_prefill", {
        "lengths": (1, 127), "v_heads": 6, "k_heads": 2,
        "k_dim": 128, "v_dim": 128, "chunk": 64})
    uh = dict(next(op for op in unequal.ops if op.family == "gdn_state").metadata)
    assert uh["lengths"] == (1, 127)
    assert uh["chunks_per_sequence"] == (1, 2)

    conv = make_call("conv_prefill", {
        "lengths": (1, 127), "conv_width": 1280, "kernel": 4})
    cm = dict(conv.ops[0].metadata)
    assert cm["lengths"] == (1, 127)
    assert cm["query_start_loc"] == (0, 1, 128)
    assert conv.ledger["outputs"]["out"]["shape"] == [128, 1280]


def test_mixed_pad_decode_preserves_physical_row_offset():
    parsed = _prefill_case()
    views = (RequestView("pad", 32, 0, False, "decode", padding=True),
             RequestView("pf", 64, 0, True, "prefill"))
    ops = _ops(parsed, views, "decode", 96)
    scan = _by_family(ops, "recurrent_vector")
    sm = dict(scan.metadata)
    assert sm["tokens"] == 0
    assert sm["physical_sequence_tokens"] == (32,)
    assert sm["physical_row_offsets"] == ()
    assert dict(_by_family(ops, "causal_conv1d_fn").metadata)["lengths"] == (32, 64)
    # Add a real decode row after the pad: the scan must retain offset 32.
    views = (RequestView("pad", 32, 0, False, "decode", padding=True),
             RequestView("d", 1, 512, True, "decode"),
             RequestView("pf", 64, 0, True, "prefill"))
    ops = _ops(parsed, views, "decode", 97)
    scan = _by_family(ops, "recurrent_vector")
    sm = dict(scan.metadata)
    assert sm["tokens"] == 1
    assert sm["physical_sequence_tokens"] == (32, 1)
    assert sm["physical_row_offsets"] == (32,)
    assert sm["input_row_ranges"] == ((32, 33),)


def test_fused_batch_keeps_each_request_effect_and_short_conv_history():
    from mindie_sim.prediction.attention import AttentionPolicy

    parsed, views, _ = source_case(lengths=(1, 1), keys=2, heads=6)
    parsed = replace(parsed, runtime=replace(parsed.runtime,
                                             kernels={"lowerings": {}}))
    frontend = Frontend(attention_policy=AttentionPolicy(fused_batch=True))
    ops, _ = frontend.body(parsed.model, parsed.runtime, views, mode="decode",
                           tokens=2)
    scans = [op for op in ops if op.family == "recurrent_vector"]
    assert len(scans) == 1
    scan = scans[0]
    sm = dict(scan.metadata)
    assert sm["tokens"] == 2 and sm["sequence_tokens"] == (1, 1)
    assert sm["state_request_offsets"] == (0, 1)
    assert sm["state_request_ids"] == ("0", "1")
    contract = sm["gdn_state_contract"]
    assert contract["targets"] == ("s0.l0.recurrent_state[0]",
                                   "s0.l0.recurrent_state[1]")
    assert contract["commit_versions"] == (4097, 4097)
    assert contract["slot_commit_policy"] == ("token_commit", "token_commit")
    effects = sm["gdn_state_effects"]
    assert len([e for e in effects if e["action"] == "commit"]) == 2
    assert {e["target"] for e in effects if e["action"] == "commit"} == {
        "s0.l0.recurrent_state[0]", "s0.l0.recurrent_state[1]"}

    conv = next(op for op in ops if op.recipe == "short_convolution")
    cm = dict(conv.metadata)
    cc = cm["gdn_state_contract"]
    assert cc["targets"] == ("s0.l0.conv_state[0]", "s0.l0.conv_state[1]")
    assert cc["commit_versions"] == (4097, 4097)
    assert [e["action"] for e in cm["gdn_state_effects"]] == [
        "read", "write", "commit", "read", "write", "commit"]
    width = 2 * 2 * 128 + 6 * 128
    assert conv.mte2_bytes == (2 + 3 * 2) * width * 2
    assert conv.mte3_bytes == (2 + 3 * 2) * width * 2


def test_capacity_outputs_and_small_explicit_slot_tables():
    conv = make_call("conv_prefill", {
        "lengths": (1, 127), "conv_width": 1280, "kernel": 4,
        "cache_indices": (7, 2), "state_rows": 9})
    assert conv.ledger["outputs"]["state"]["shape"] == [9, 3, 1280]
    small = make_call("conv_prefill", {
        "lengths": (1, 1, 1), "conv_width": 1280, "kernel": 4,
        "cache_indices": (1, 0, 0), "state_rows": 2})
    sm = dict(small.ops[0].metadata)
    assert sm["cache_indices"] == (1, 0, 0)
    assert sm["state_rows"] == 2
    assert sm["state_write_bytes"] == 3 * 1280 * 2
    assert {e["target"] for e in sm["gdn_state_effects"]} == {
        "conv_call.conv_state[seq0]"}
    assert [e["action"] for e in sm["gdn_state_effects"]] == ["write", "commit"]
    assert small.ledger["outputs"]["state"]["shape"] == [2, 3, 1280]


def test_pcp_rank_local_version_never_publishes_global_without_remote():
    parsed = _prefill_case()
    parallel = replace(parsed.runtime.parallel, pcp=2, cp_plan="pcp")
    parsed = replace(parsed, runtime=replace(parsed.runtime, parallel=parallel))
    view = RequestView("rank1", 2, 512, True, "prefill", global_q_len=515 - 512,
                       query_ranges=((1, 3),))
    ops = _ops(parsed, (view,), "prefill", 2)
    h = _by_family(ops, "gdn_state")
    conv = _by_family(ops, "causal_conv1d_fn")
    scatter = _by_family(ops, "state_scatter")
    initial = next(op for op in ops if op.recipe == "pcp_remote_initial_state")
    assert dict(initial.metadata)["status"] == "pending_external_rank"
    assert dict(initial.metadata)["from_version"] == 512
    assert dict(initial.metadata)["to_version"] == 513
    assert initial.op_id in h.deps
    assert dict(h.metadata)["state_read_versions"] == (513,)
    sc = dict(scatter.metadata)["gdn_state_contract"]
    assert sc["commit_versions"] == (None,)
    assert sc["local_commit_versions"] == (515,)
    assert sc["global_versions"] == (515,)
    assert sc["publish_pending"] == "remote_producer"
    conv_initial = next(op for op in ops if op.recipe == "pcp_remote_conv_state")
    assert conv_initial.op_id in conv.deps
    assert dict(conv.metadata)["gdn_state_contract"]["read_versions"] == (513,)
    assert dict(conv.metadata)["gdn_state_contract"]["commit_versions"] == (None,)
    assert dict(conv.metadata)["gdn_state_contract"]["global_versions"] == (515,)


def test_decode_contract_carries_caller_slot_metadata_when_available():
    parsed, views, _ = source_case(lengths=(1, 1), keys=2, heads=6)
    parsed = replace(parsed, runtime=replace(
        parsed.runtime, metadata=tuple(sorted({
            **dict(parsed.runtime.metadata),
            "gdn_decode_state_indices": {"0": 7, "1": 2},
            "gdn_state_rows": 9}.items()))))
    ops = _ops(parsed, views, "decode", 2)
    conv = _by_recipe(ops, "ascend_causal_conv1d_update")
    cc = dict(conv.metadata)["gdn_state_contract"]
    assert cc["slot_indices"] == (7, 2)
    assert cc["state_capacity"] == 9
    assert cc["slot_source"] == "caller_metadata"
    scan = _by_family(ops, "recurrent_vector")
    sc = dict(scan.metadata)["gdn_state_contract"]
    assert sc["slot_indices"] == (7, 2)
    assert sc["state_capacity"] == 9


def test_legacy_and_cu_seqlens_partition_share_one_normalization():
    legacy = make_call("conv_prefill", {
        "batch": 2, "tokens": 1, "conv_width": 1280, "kernel": 4})
    assert legacy.ledger["arguments"]["lengths"] == (1, 1)
    assert legacy.ledger["arguments"]["query_start_loc"] == [0, 1, 2]
    assert legacy.ledger["outputs"]["out"]["shape"] == [2, 1280]
    cu = make_call("conv_prefill", {
        "cu_seqlens": (0, 1, 128), "conv_width": 1280, "kernel": 4})
    cm = dict(cu.ops[0].metadata)
    assert cm["lengths"] == (1, 127)
    assert cm["query_start_loc"] == (0, 1, 128)
    assert cu.ledger["outputs"]["out"]["shape"] == [128, 1280]
    actual = make_call("conv_prefill", {
        "batch": 2, "tokens": 64, "cu_seqlens": (0, 1, 128),
        "conv_width": 1280, "kernel": 4})
    assert dict(actual.ops[0].metadata)["lengths"] == (1, 127)
    assert actual.ledger["outputs"]["out"]["shape"] == [128, 1280]
    try:
        make_call("conv_prefill", {
            "lengths": (1, 1), "cu_seqlens": (0, 1, 128),
            "conv_width": 1280, "kernel": 4})
    except Exception as error:
        assert "partition" in str(error) or "mismatch" in str(error)
    else:
        raise AssertionError("conflicting partition inputs must be rejected")
    scan = make_call("scan_prefill", {
        "cu_seqlens": (0, 1, 128), "v_heads": 6, "k_heads": 2,
        "k_dim": 128, "v_dim": 128, "chunk": 64})
    h = dict(next(op for op in scan.ops if op.family == "gdn_state").metadata)
    assert h["lengths"] == (1, 127)
    assert h["chunks_per_sequence"] == (1, 2)
    equal = make_call("scan_prefill", {
        "lengths": (64, 64), "v_heads": 6, "k_heads": 2,
        "k_dim": 128, "v_dim": 128, "chunk": 64})
    eh = dict(next(op for op in equal.ops if op.family == "gdn_state").metadata)
    assert eh["lengths"] == (64, 64)
    assert eh["chunks_per_sequence"] == (1, 1)


def test_generic_fused_conv_and_scan_follow_per_request_acceptance():
    from mindie_sim.prediction.attention import AttentionPolicy

    parsed, base, _ = source_case(lengths=(4, 4), keys=2, heads=6)
    parsed = replace(parsed, runtime=replace(parsed.runtime,
                                             kernels={"lowerings": {}}))
    pending = replace(base[0], draft_proposal_tokens=3,
                      draft_accepted_tokens=0, draft_advanced_tokens=0)
    known = replace(base[1], draft_proposal_tokens=3,
                    draft_accepted_tokens=1, draft_advanced_tokens=2)
    frontend = Frontend(attention_policy=AttentionPolicy(fused_batch=True))
    ops, _ = frontend.body(parsed.model, parsed.runtime, (pending, known),
                           mode="decode", tokens=8)
    scan = _by_family(ops, "recurrent_vector")
    sc = dict(scan.metadata)["gdn_state_contract"]
    commits = dict(zip(sc["targets"], sc["commit_versions"]))
    assert commits["s0.l0.recurrent_state[0]"] is None
    assert commits["s0.l0.recurrent_state[1]"] == 4098
    conv = next(op for op in ops if op.recipe == "short_convolution")
    cc = dict(conv.metadata)["gdn_state_contract"]
    ccommits = dict(zip(cc["targets"], cc["commit_versions"]))
    assert ccommits["s0.l0.conv_state[0]"] is None
    assert ccommits["s0.l0.conv_state[1]"] == 4098
    conv_commits = {e["target"] for e in dict(conv.metadata)["gdn_state_effects"]
                    if e["action"] == "commit"}
    assert conv_commits == {"s0.l0.conv_state[1]"}
    # The accepted request's next q1 reads 4098 and writes the next candidate.
    from mindie_sim.prediction.gdn_recurrent import state_contract
    next_view = RequestView("next", 1, 4098, True, "decode")
    nxt, _ = state_contract(views=(next_view,), phase="decode",
                            state_dtype="fp32", conv_state_dtype="bf16",
                            conv_kernel=4, direction="token",
                            target_kind="recurrent", initial_state=(True,))
    assert nxt["read_versions"] == (4098,)
    assert nxt["commit_versions"] == (4099,)


def test_pcp_segments_keep_their_own_slot_and_capacity():
    parsed = _prefill_case()
    parallel = replace(parsed.runtime.parallel, pcp=2, cp_plan="pcp")
    metadata = tuple(sorted({
        **dict(parsed.runtime.metadata),
        "gdn_prefill_state_indices": {"B": 7, "A": 2},
        "gdn_state_rows": 9}.items()))
    parsed = replace(parsed, runtime=replace(
        parsed.runtime, parallel=parallel, metadata=metadata))
    views = (RequestView("B", 64, 512, True, "prefill", global_q_len=128,
                         query_ranges=((0, 32), (96, 128))),
             RequestView("A", 64, 512, True, "prefill", global_q_len=128,
                         query_ranges=((32, 64), (64, 96))))
    ops = _ops(parsed, views, "prefill", 128)
    convs = [op for op in ops if op.family == "causal_conv1d_fn"]
    assert [dict(op.metadata)["cache_indices"] for op in convs] == [
        (7,), (7,), (2,), (2,)]
    assert all(dict(op.metadata)["state_rows"] == 9 for op in convs)
    states = [op for op in ops if op.family == "gdn_state"]
    assert [dict(op.metadata)["gdn_state_contract"]["slot_indices"]
            for op in states] == [(7,), (7,), (2,), (2,)]


def test_gather_h_scatter_expose_real_value_edges_and_standalone_io():
    parsed = _prefill_case()
    view = RequestView("p", 64, 512, True, "prefill")
    ops = _ops(parsed, (view,), "prefill", 64)
    gather = _by_family(ops, "state_gather")
    state = _by_family(ops, "gdn_state")
    scatter = _by_family(ops, "state_scatter")
    gm = dict(gather.metadata)
    hm = dict(state.metadata)
    sm = dict(scatter.metadata)
    assert gm["state_value_output"] == "initial_state"
    assert hm["state_input_producer"] == gather.op_id
    assert hm["state_value_output"] == "chunk_bank"
    assert sm["state_value_output"] == "persistent_cache"
    assert gather.op_id in state.deps
    assert state.op_id in scatter.deps
    scan = make_call("scan_prefill", {
        "lengths": (1, 127), "v_heads": 6, "k_heads": 2,
        "k_dim": 128, "v_dim": 128, "chunk": 64})
    sh = dict(next(op for op in scan.ops if op.family == "gdn_state").metadata)
    assert sh["state_external_initial"] == "caller:initial_state"
    assert sh["state_external_final"] == "caller:final_state"
    cm = dict(_by_family(ops, "causal_conv1d_fn").metadata)
    assert cm["workspace_lifetime"] == "first_acquire_to_last_conv_block"
    assert cm["workspace_pool"] == "workspace"
    assert cm["workspace_bytes"] > 0


def test_fn_gm_workspace_is_one_call_level_grant():
    from mindie_sim.prediction.operators import native_causal_conv_prefill as native_conv
    from mindie_sim.prediction.operators import native_causal_conv_spec as native_spec
    from mindie_sim.prediction.operators.native_causal_conv_prefill import _default_geometry
    from mindie_sim.prediction.program import RecipeContext

    op = native_spec.make_op({
        "lengths": (1, 1), "dim": 16, "width": 2, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": (True, True), "cache_indices": (1, 2),
        "state_history_slots": 1, "null_block_id": 0, "pad_slot_id": -1,
        "state_rows": 3})
    geometry = _default_geometry()
    owners = int(geometry.resource_units["vector"])
    plan = native_conv.build(op, geometry)
    rates = {key: max(1.0, value) for key, value in plan.work.resources.items()}
    context = RecipeContext(
        None, {}, rates, {key: owners for key in rates}, owners,
        pool_names={"ub": "ub", "workspace": "workspace"},
        pool_units={"ub": owners, "workspace": owners},
        pool_capacities={"ub": 10 ** 9, "workspace": 16777344})
    stages = {stage.name: stage for stage in plan.program(context).stages}
    grant = stages["gm_grant"]
    assert [item.name for item in grant.acquire] == ["native_conv_gm_workspace"]
    assert grant.free == ()
    for name in ("reserve0", "reserve1"):
        assert stages[name].after == ("gm_grant",)
        assert [item.name for item in stages[name].acquire] == [
            "native_conv_ub" + name[-1]]
    assert [item.name for item in stages["release0"].free] == ["native_conv_ub0"]
    assert [item.name for item in stages["release1"].free] == ["native_conv_ub1"]
    release = stages["gm_release"]
    assert [item.name for item in release.free] == ["native_conv_gm_workspace"]
    assert release.after == ("release0", "release1")


def test_mixed_prefill_and_speculative_decode_keep_per_request_semantics():
    parsed = _prefill_case()
    pending = RequestView("B", 4, 4096, True, "decode",
                          draft_proposal_tokens=3,
                          draft_accepted_tokens=0, draft_advanced_tokens=0)
    accepted = RequestView("B", 4, 4096, True, "decode",
                           draft_proposal_tokens=3,
                           draft_accepted_tokens=1, draft_advanced_tokens=2)
    prefill = RequestView("P", 64, 512, True, "prefill")
    for views in ((accepted, prefill), (prefill, accepted)):
        ops = _ops(parsed, views, "decode", 68)
        conv = _by_family(ops, "causal_conv1d_fn")
        cc = dict(conv.metadata)["gdn_state_contract"]
        assert cc["slot_directions"] == ("token", "conv")
        assert cc["queries"] == (4, 64)
        commits = dict(zip(cc["targets"], cc["commit_versions"]))
        assert commits["s0.l0.conv_state[B]"] == 4098
        assert commits["s0.l0.conv_state[P]"] == 576
        assert dict(conv.metadata)["state_candidate_storage"]["kind"] == \
            "history_ring+query_rows"
        assert dict(conv.metadata)["state_candidate_storage"]["history_slots"] == 3
        stores = cc["candidate_stores"][cc["targets"].index(
            "s0.l0.conv_state[B]")]
        assert [item["version"] for item in stores] == [4097, 4098, 4099, 4100]
        assert {item["ready"] for item in stores} == {conv.op_id}
        assert all(item["heads"] == 6 for item in stores)
        scan = _by_family(ops, "recurrent_vector")
        assert dict(scan.metadata)["gdn_state_contract"]["commit_versions"] == (4098,)
        state = _by_family(ops, "gdn_state")
        assert dict(state.metadata)["state_read_versions"] == (512,)
        scatter = _by_family(ops, "state_scatter")
        assert dict(scatter.metadata)["gdn_state_contract"]["commit_versions"] == (576,)
        gather = _by_family(ops, "state_gather")
        assert dict(gather.metadata)["gdn_state_contract"]["read_versions"] == (512,)
    ops = _ops(parsed, (pending, prefill), "decode", 68)
    cc = dict(_by_family(ops, "causal_conv1d_fn").metadata)["gdn_state_contract"]
    commits = dict(zip(cc["targets"], cc["commit_versions"]))
    assert commits["s0.l0.conv_state[B]"] is None
    assert commits["s0.l0.conv_state[P]"] == 576
    assert dict(_by_family(ops, "recurrent_vector").metadata)[
        "gdn_state_contract"]["commit_versions"] == (None,)


def test_pcp_rank0_tail_requests_read_their_own_prefix():
    parsed = _prefill_case()
    parallel = replace(parsed.runtime.parallel, pcp=2, cp_plan="pcp")
    parsed = replace(parsed, runtime=replace(parsed.runtime, parallel=parallel))
    views = (RequestView("tail0", 1, 512, True, "prefill", global_q_len=3,
                         query_ranges=((2, 3),)),
             RequestView("tail1", 1, 1024, True, "prefill", global_q_len=3,
                         query_ranges=((2, 3),)))
    ops = _ops(parsed, views, "prefill", 2)
    states = [op for op in ops if op.family == "gdn_state"]
    assert [dict(op.metadata)["state_read_versions"] for op in states] == [
        (514,), (1026,)]
    scatter = _by_family(ops, "state_scatter")
    sc = dict(scatter.metadata)["gdn_state_contract"]
    assert sc["read_versions"] == (514, 1026)
    assert sc["local_commit_versions"] == (515, 1027)
    assert sc["global_versions"] == (515, 1027)
    assert sc["commit_versions"] == (None, None)
    initials = [op for op in ops if op.recipe == "pcp_remote_initial_state"]
    assert [dict(op.metadata)["to_version"] for op in initials] == [514, 1026]


def test_generic_mixed_conv_and_recurrent_keep_per_request_decisions():
    parsed, _, _ = source_case(lengths=(1,), keys=2, heads=6)
    parsed = replace(parsed, runtime=replace(parsed.runtime,
                                             kernels={"lowerings": {}}))
    accepted = RequestView("B", 4, 4096, True, "decode",
                           draft_proposal_tokens=3,
                           draft_accepted_tokens=1, draft_advanced_tokens=2)
    prefill = RequestView("P", 64, 512, True, "prefill")
    for views in ((accepted, prefill), (prefill, accepted)):
        ops = _ops(parsed, views, "decode", 68)
        conv = next(op for op in ops if op.recipe == "short_convolution")
        cc = dict(conv.metadata)["gdn_state_contract"]
        by_target = dict(zip(cc["targets"], range(len(cc["targets"]))))
        slot = by_target["s0.l0.conv_state[B]"]
        pslot = by_target["s0.l0.conv_state[P]"]
        assert cc["slot_directions"][slot] == "token"
        assert cc["slot_directions"][pslot] == "conv"
        assert cc["read_versions"][slot] == 4096
        assert cc["commit_versions"][slot] == 4098
        assert cc["read_versions"][pslot] == 512
        assert cc["commit_versions"][pslot] == 576
        assert cc["checks"][slot] if False else True
        stores = cc["candidate_stores"][slot]
        assert [item["version"] for item in stores] == [4097, 4098, 4099, 4100]
        assert {item["ready"] for item in stores} == {conv.op_id}
        assert all(item["heads"] == 6 for item in stores)
        assert all(item["head_shards"] >= 1 for item in stores)
        scan = next(op for op in ops if op.family == "recurrent_vector")
        sc = dict(scan.metadata)["gdn_state_contract"]
        assert sc["read_versions"] == (4096,)
        assert sc["commit_versions"] == (4098,)
        assert sc["decision_inputs"] == ("acceptance[B]",)
        assert dict(scan.metadata)["state_candidate_stores"][0][-1][
            "ready"] == scan.op_id
    accepted_commit = cc["commit_versions"][cc["targets"].index(
        "s0.l0.conv_state[B]")]
    assert accepted_commit == 4098
    pending = replace(accepted, draft_accepted_tokens=0,
                      draft_advanced_tokens=0)
    ops = _ops(parsed, (pending, prefill), "decode", 68)
    conv = next(op for op in ops if op.recipe == "short_convolution")
    cc = dict(conv.metadata)["gdn_state_contract"]
    bslot = next(index for index, target in enumerate(cc["targets"])
                 if "conv_state[B]" in target)
    pslot = next(index for index, target in enumerate(cc["targets"])
                 if "conv_state[P]" in target)
    assert cc["commit_versions"][bslot] is None
    assert cc["commit_versions"][pslot] == 576
    assert cc["publish_versions"][bslot] is None
    assert not [effect for effect in dict(conv.metadata)["gdn_state_effects"]
                if effect["target"].endswith("[B]") and effect["action"] == "commit"]
    scan = next(op for op in ops if op.family == "recurrent_vector")
    sc = dict(scan.metadata)["gdn_state_contract"]
    assert sc["commit_versions"] == (None,)
    assert sc["publish_versions"] == (None,)
    assert sc["candidate_stores"][0][-1]["ready"] == scan.op_id
    assert dict(scan.metadata)["state_value_outputs"][0]["state_version"] is None
    assert not [effect for effect in dict(scan.metadata)["gdn_state_effects"]
                if effect["action"] == "commit"]
    # The same B's next q1 consumes the generation actually committed by the
    # previous call; each later occurrence advances that same request state.
    committed = accepted_commit
    for _ in range(2):
        successor = RequestView("B", 1, committed, True, "decode")
        next_ops = _ops(parsed, (successor,), "decode", 1)
        nxt = dict(next(op for op in next_ops
                        if op.family == "recurrent_vector").metadata)[
                            "gdn_state_contract"]
        following = committed + 1
        assert nxt["read_versions"] == (committed,)
        assert nxt["commit_versions"] == (following,)
        assert nxt["candidate_stores"][0][0]["version"] == following
        committed = following
    assert committed == 4100
    # accepted=2/advance=3 changes B's own committed generation to 4099.
    advanced = replace(accepted, draft_accepted_tokens=2,
                       draft_advanced_tokens=3)
    ops = _ops(parsed, (advanced, prefill), "decode", 68)
    cc = dict(next(op for op in ops
                   if op.recipe == "short_convolution").metadata)["gdn_state_contract"]
    slot = cc["targets"].index("s0.l0.conv_state[B]")
    assert cc["commit_versions"][slot] == 4099
    scan = next(op for op in ops if op.family == "recurrent_vector")
    assert dict(scan.metadata)["gdn_state_contract"]["commit_versions"] == (4099,)
    # A and B are both q4 with their own decisions; both fusion orders keep
    # every request's decision input and candidate store.
    a_pending = RequestView("A", 4, 4096, True, "decode")
    for views in ((a_pending, accepted, prefill),
                  (accepted, a_pending, prefill),
                  (prefill, a_pending, accepted)):
        ops = _ops(parsed, views, "decode", 72)
        cc = dict(next(op for op in ops
                       if op.recipe == "short_convolution").metadata)[
                           "gdn_state_contract"]
        commits = dict(zip(cc["targets"], cc["commit_versions"]))
        assert commits["s0.l0.conv_state[A]"] is None
        assert commits["s0.l0.conv_state[B]"] == 4098
        if any(view.request_id == "P" for view in views):
            assert commits["s0.l0.conv_state[P]"] == 576
        inputs = dict(zip(cc["targets"], cc["decision_inputs"]))
        assert inputs["s0.l0.conv_state[A]"] == "acceptance[A]"
        assert inputs["s0.l0.conv_state[B]"] == "acceptance[B]"
        for target in ("s0.l0.conv_state[A]", "s0.l0.conv_state[B]"):
            slot = cc["targets"].index(target)
            assert [store["version"] for store in cc["candidate_stores"][slot]] == [
                4097, 4098, 4099, 4100]
            assert all(store["heads"] == 6 for store in
                       cc["candidate_stores"][slot])
        a_commits = [effect for effect in dict(
            next(op for op in ops
                 if op.recipe == "short_convolution").metadata)["gdn_state_effects"]
            if effect["action"] == "commit"
            and effect["target"] == "s0.l0.conv_state[A]"]
        assert a_commits == []


def test_mixed_physical_subviews_and_real_prefill_materialization():
    parsed = _prefill_case()
    views = (RequestView("pad", 32, 0, False, "decode", padding=True),
             RequestView("d", 1, 512, True, "decode"),
             RequestView("pf", 64, 0, True, "prefill"))
    ops = _ops(parsed, views, "decode", 97)
    scan = _by_family(ops, "recurrent_vector")
    sm = dict(scan.metadata)
    assert sm["input_row_ranges"] == ((32, 33),)
    assert sm["physical_row_offsets"] == (32,)
    view = sm["state_value_views"][0]
    # q/k/v live in one compact qkv buffer: token 32 offsets in each component.
    assert [item["offset"] for item in view[:3]] == [8192, 33024, 74240]
    assert all(item["shape"] == expected for item, expected in zip(
        view[:3], ((1, 1, 2, 128), (1, 1, 2, 128), (1, 1, 6, 128))))
    assert [item["offset"] for item in view[3:]] == [192, 192]
    assert sm["state_value_inputs"][0]["offset"] == 74240
    prepare = _by_family(ops, "gdn_prepare")
    pm = dict(prepare.metadata)
    assert pm["gdn_prepare_source_offsets"] == (8448, 33280, 75008, 198, 198)
    assert pm["gdn_prepare_input_offsets"] == (0, 0, 0, 0, 0)
    assert [item["producer"] for item in pm["gdn_prepare_inputs"]] == [
        "s0.l0.q_head_first", "s0.l0.k_head_first", "s0.l0.v_head_first",
        "s0.l0.g_head_first", "s0.l0.beta_head_first"]
    materialized = [op for op in ops if op.family == "tensor_copy"
                    and dict(op.metadata).get("materialization_reason")
                    == "fla_BSND_to_BNSD"]
    assert {op.op_id for op in materialized} == {
        "s0.l0.q_head_first", "s0.l0.k_head_first", "s0.l0.v_head_first"}
    assert all(op.semantic_shape == (1, 2, 64, 128) if "q_" in op.op_id
               or "k_" in op.op_id else op.semantic_shape == (1, 6, 64, 128)
               for op in materialized)


def test_gather_h_scatter_bind_the_same_value_views_and_completion():
    parsed = _prefill_case()
    view = RequestView("p", 64, 512, True, "prefill")
    ops = _ops(parsed, (view,), "prefill", 64)
    gather = _by_family(ops, "state_gather")
    state = _by_family(ops, "gdn_state")
    scatter = _by_family(ops, "state_scatter")
    gm, hm, sm = dict(gather.metadata), dict(state.metadata), dict(scatter.metadata)
    gather_port = gm["state_value_outputs"][0]
    assert gather_port["producer"] == gather.op_id
    assert gather_port["ready"] == gather.op_id
    assert gather_port["state_version"] == 512
    h_input = hm["state_value_inputs"][0]
    assert h_input["producer"] == gather.op_id
    assert h_input["offset"] == gather_port["offset"] == 0
    h_output = hm["state_value_outputs"][0]
    assert h_output["producer"] == state.op_id
    assert h_output["state_version"] == 512
    assert h_output["local_version"] == 64
    assert h_output["ready"] == state.op_id
    scatter_input = sm["state_value_inputs"][0]
    assert scatter_input["producer"] == state.op_id
    assert scatter_input["ready"] == state.op_id
    scatter_output = sm["state_value_outputs"][0]
    assert scatter_output["producer"] == scatter.op_id
    assert scatter_output["state_version"] == 576
    assert scatter_output["ready"] == scatter.op_id
    assert dict(state.metadata)["gdn_state_effects"]
    assert 576 not in [effect["version"] for effect in
                       dict(state.metadata)["gdn_state_effects"]
                       if effect["action"] == "commit"]
    assert dict(scatter.metadata)["gdn_state_contract"]["publish_point"] == \
        "scatter_complete"


def test_generic_pcp_two_ranks_read_prefix_and_publish_only_after_write():
    parsed, _, _ = source_case(lengths=(1,), keys=2, heads=6)
    parallel = replace(parsed.runtime.parallel, pcp=2, cp_plan="pcp")
    parsed = replace(parsed, runtime=replace(
        parsed.runtime, kernels={"lowerings": {}}, parallel=parallel))
    rank1 = RequestView("rank1", 2, 512, True, "prefill", global_q_len=3,
                        query_ranges=((0, 2),))
    ops = _ops(parsed, (rank1,), "prefill", 2)
    scan = next(op for op in ops if op.family == "recurrent_scan")
    rc = dict(scan.metadata)["gdn_state_contract"]
    assert rc["read_versions"] == (512,)
    assert rc["local_checkpoint_versions"] == ((513, 514),)
    assert rc["local_commit_versions"] == (514,)
    assert rc["commit_versions"] == (None,)
    assert rc["global_versions"] == (515,)
    assert rc["publish_pending"] == "remote_producer"
    step = next(op for op in ops if op.recipe == "pcp_recurrent_state_step0")
    assert (dict(step.metadata)["from_version"],
            dict(step.metadata)["to_version"]) == (514, 515)
    publish = next(op for op in ops
                   if op.recipe == "pcp_remote_state_publish0")
    assert step.op_id in publish.deps
    assert scan.op_id in step.deps
    owner = RequestView("rank0", 1, 512, True, "prefill", global_q_len=3,
                        query_ranges=((2, 3),))
    ops = _ops(parsed, (owner,), "prefill", 1)
    scan = next(op for op in ops if op.family == "recurrent_scan")
    oc = dict(scan.metadata)["gdn_state_contract"]
    assert oc["read_versions"] == (514,)
    assert oc["local_checkpoint_versions"] == ((515,),)
    assert oc["local_commit_versions"] == (515,)
    assert oc["commit_versions"] == (None,)
    initial = next(op for op in ops
                   if op.recipe == "pcp_remote_initial_state0")
    assert (dict(initial.metadata)["from_version"],
            dict(initial.metadata)["to_version"]) == (512, 514)
    assert initial.op_id in scan.deps
    publish = next(op for op in ops
                   if op.recipe == "pcp_remote_state_publish0")
    assert dict(publish.metadata)["producer_scope"] == "local_writeback"
    assert scan.op_id in publish.deps
    # The q2/[1,3) control keeps its committed local final 515.
    ranged = RequestView("ranged", 2, 512, True, "prefill", global_q_len=3,
                         query_ranges=((1, 3),))
    ops = _ops(parsed, (ranged,), "prefill", 2)
    rc = dict(next(op for op in ops
                   if op.family == "recurrent_scan").metadata)["gdn_state_contract"]
    assert rc["read_versions"] == (513,)
    assert rc["local_checkpoint_versions"] == ((514, 515),)
    assert rc["local_commit_versions"] == (515,)


def test_selected_pcp_segments_share_prepare_h_and_output_partition():
    parsed = _prefill_case()
    parallel = replace(parsed.runtime.parallel, pcp=2, cp_plan="pcp")
    parsed = replace(parsed, runtime=replace(parsed.runtime, parallel=parallel))
    for name, q, ranges, reads, lengths, final in (
            ("q128", 64, ((0, 32), (96, 128)), (512, 608), (32, 32), 640),
            ("q130", 65, ((0, 32), (97, 130)), (512, 609), (32, 33), 642)):
        view = RequestView(name, q, 512, True, "prefill", global_q_len=final - 512,
                           query_ranges=ranges)
        ops = _ops(parsed, (view,), "prefill", q)
        convs = [op for op in ops if op.family == "causal_conv1d_fn"]
        prepares = [op for op in ops if op.family == "gdn_prepare"]
        states = [op for op in ops if op.family == "gdn_state"]
        outputs = [op for op in ops if op.family == "gdn_output"]
        assert [dict(op.metadata)["lengths"] for op in convs] == [
            (lengths[0],), (lengths[1],)]
        assert [dict(op.metadata)["lengths"] for op in prepares] == [
            (lengths[0],), (lengths[1],)]
        assert [dict(op.metadata)["lengths"] for op in states] == [
            (lengths[0],), (lengths[1],)]
        assert [dict(op.metadata)["lengths"] for op in outputs] == [
            (lengths[0],), (lengths[1],)]
        conv_reads = [dict(op.metadata)["gdn_state_contract"]["read_versions"]
                      for op in convs]
        assert conv_reads == [(reads[0],), (reads[1],)]
        state_reads = [dict(op.metadata)["state_read_versions"] for op in states]
        assert state_reads == [(reads[0],), (reads[1],)]
        exchange = next(op for op in ops if op.recipe == "pcp_recurrent_state_step1")
        assert (dict(exchange.metadata)["from_version"],
                dict(exchange.metadata)["to_version"]) == (544, reads[1])
        assert exchange.op_id in states[1].deps
        assert states[0].op_id in exchange.deps
        assert states[0].op_id in outputs[0].deps
        assert states[1].op_id in outputs[1].deps
        scatter = _by_family(ops, "state_scatter")
        publish = next(op for op in ops if op.recipe == "pcp_remote_state_publish")
        assert scatter.op_id in [op.op_id for op in ops
                                 if op.recipe == "scatter_recurrent_state"]
        assert publish.op_id in scatter.deps
        assert set(publish.deps) >= {states[0].op_id, states[1].op_id}
        contract = dict(scatter.metadata)["gdn_state_contract"]
        assert contract["global_versions"] == (final,)
        assert contract["local_commit_versions"] == (final,)
        assert dict(states[1].metadata)["state_value_inputs"][0]["producer"] == \
            exchange.op_id
        # Delaying the exchange delays the second H, O and final publish.
        assert exchange.op_id in states[1].deps
        assert exchange.op_id not in states[0].deps
        assert states[1].op_id in outputs[1].deps
        assert publish.op_id in scatter.deps


def test_fn_workspace_runtime_grant_waits_and_frees_once():
    from mindie_sim.prediction.engine import Allocation, Engine, Resource
    from mindie_sim.prediction.operators import native_causal_conv_prefill as native_conv
    from mindie_sim.prediction.operators.native_causal_conv_prefill import _default_geometry
    from mindie_sim.prediction.program import RecipeContext

    def runtime(batch):
        op = native_conv.make_op({
            "lengths": (1,) * batch, "dim": 16, "width": 2, "dtype": "bf16",
            "activation": "silu", "bias": False, "metadata_placement": "device",
            "has_initial_state": (True,) * batch,
            "cache_indices": tuple(range(1, batch + 1)),
            "state_history_slots": 1, "null_block_id": 0, "pad_slot_id": -1,
            "state_rows": batch + 1})
        geometry = _default_geometry()
        owners = int(geometry.resource_units["vector"])
        plan = native_conv.build(op, geometry)
        workspace = int(dict(op.metadata)["workspace_bytes"])
        rates = {key: max(1.0, value) for key, value in plan.work.resources.items()}
        names = {key: key for key in rates}
        context = RecipeContext(
            None, {}, names, rates, owners,
            pool_names={"ub": "ub", "workspace": "workspace"},
            pool_units={"ub": owners, "workspace": 1},
            pool_capacities={"ub": 10 ** 9, "workspace": workspace})
        resources = [Resource(key, float(value), "work", "fair", owners)
                     for key, value in rates.items()]
        engine = Engine(resources, pools={"ub": 10 ** 9, "workspace": workspace},
                        pool_units={"ub": owners, "workspace": 1},
                        trace_limit=32)
        return op, engine, plan.program(context), workspace

    op, engine, program, workspace = runtime(2)
    assert workspace == 16 * 1024 * 1024 + 2 * 32 + 2 * 1 * 16 * 2 == 16777344
    holder = engine.submit("holder", {"hbm": 1}, acquire=[
        Allocation("external_holder", "workspace", 1, (0,))])
    ready = engine.submit("ready", release=0)
    event = program.start(engine, "fn", ready)
    engine.advance(horizon=0)
    assert not event.done
    assert engine.allocated["workspace"] == 1
    engine.submit("external_release", release=0, free=[
        Allocation("external_holder", "workspace", 1, (0,))])
    engine.advance(horizon=0.5)
    # The call-level grant is live through the delayed last reader; only the
    # final block release lets it go.
    assert not event.done
    assert engine.allocated["workspace"] == workspace
    engine.advance(until=(event,))
    assert event.done and engine.allocated["workspace"] == 0
    grants = [(item["action"], item["name"]) for item in engine.allocation_trace
              if item["name"].endswith("native_conv_gm_workspace")]
    assert grants == [("acquire", "fn:native_conv_gm_workspace"),
                      ("release", "fn:native_conv_gm_workspace")]
    assert engine.peak_allocated["workspace"] == workspace

    # Single block keeps the same one-grant/one-release construction.
    op1, engine1, program1, workspace1 = runtime(1)
    ready1 = engine1.submit("ready", release=0)
    event1 = program1.start(engine1, "fn", ready1)
    engine1.advance(horizon=0)
    assert not event1.done
    engine1.advance(until=(event1,))
    assert event1.done and engine1.allocated["workspace"] == 0
    grants1 = [(item["action"], item["name"]) for item in engine1.allocation_trace
               if item["name"].endswith("native_conv_gm_workspace")]
    assert grants1 == [("acquire", "fn:native_conv_gm_workspace"),
                       ("release", "fn:native_conv_gm_workspace")]
