"""Draft preparation, integer acceptance and media execution dependencies."""
from dataclasses import replace
import json
from pathlib import Path

from mindie_sim import Case, predict
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
from mindie_sim.catalog.draft_weights import draft_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.scheduler import DraftPolicy, Progress
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.dspark import DsparkSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def draft_case():
    parsed = Case.from_mapping(case(prompt=4, output=6))
    draft = DraftSpec(kind="mtp", layers=1, hidden_size=32, intermediate_size=64,
                      num_heads=4, num_kv_heads=2, head_dim=8, q_width=32, kv_width=16,
                      vocab_size=128, target_shared_embedding=True, target_shared_lm_head=True)
    return replace(parsed, model=replace(parsed.model, auxiliary=AuxiliarySpec(mtp=draft)),
                   runtime=replace(parsed.runtime, metadata=(
                       ("draft_mode", "mtp"), ("draft_proposal_tokens", 2), ("draft_accept_tokens", 1))))


def test_draft_prefill_populates_cache_after_target_first_token_then_verifies_once():
    parsed = draft_case()
    plain = replace(parsed, runtime=replace(parsed.runtime, metadata=()))
    baseline = predict(plain)
    result = predict(parsed, event_limit=2000)
    assert result["requests"][0]["generated_tokens"] == 6
    assert result["requests"][0]["speculative"]["proposed_tokens"] > 0
    assert result["requests"][0]["ttft_ms"] == baseline["requests"][0]["ttft_ms"]
    commits = [event for event in result["events"] if event["task"].endswith("token_commit:complete")]
    prepared = [event for event in result["events"] if "draft_prefill." in event["task"]
                and not event["task"].endswith(":submit")]
    assert prepared and min(event["start_ms"] for event in prepared) >= commits[0]["end_ms"]
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    view = RequestView("r", 1, 4, True, "decode", draft_proposal_tokens=2)
    work = Frontend()(IterationPlan((view,)), parsed.model, parsed.runtime, parsed.machine)
    target = [op for op in work.ops if op.recipe == "qkv_projection" and not op.op_id.startswith("draft")]
    assert len(target) == 2 and all(op.local_shape[0] == 3 for op in target)
    assert len([op for op in work.ops if op.recipe == "draft_features"]) == 2


def test_prefix_marginals_match_categorical_counts_and_samples_are_reproducible():
    spec = SpeculativeSpec(3, (.8, .4, .1))
    assert spec.expected_committed_tokens() == 2.3
    conditional = SpeculativeSpec(3, (.8, .5, .25), acceptance_mode="conditional")
    assert spec.acceptance_distribution() == conditional.acceptance_distribution()
    parsed = draft_case()
    raw = {"draft_len": 2, "accepted_distribution": [.5, 0, .5]}
    runtime = replace(parsed.runtime, metadata=(("draft_mode", "mtp"), ("speculative", raw)))
    parsed = replace(parsed, runtime=runtime)
    request = parsed.workload.requests[0]
    policy = DraftPolicy.for_request(parsed, request)
    a, b = Progress(request), Progress(request)
    sequence = []
    for step in range(100):
        a.decode_steps = b.decode_steps = step
        value = policy.committed(a, 2)
        assert value == policy.committed(b, 2)
        sequence.append(value)
    assert set(sequence) == {1, 3}
    assert DsparkSpec(2, (.5, .2), confidence=(.1, .8),
                      confidence_threshold=.5).step_spec().draft_len == 0


def test_encoded_media_keeps_sequence_positions_and_skips_only_the_tower():
    path = Path(__file__).parents[1] / "examples/cases/qwen3.8-27b-media.json"
    raw = json.loads(path.read_text())
    raw["workload"] = {"requests": [{"id": "media", "text_tokens": 4, "media_tokens": 4, "output_tokens": 1}]}
    parsed = Case.from_mapping(raw)
    fresh = RequestView("media", 8, 0, True, "prefill")
    ready = replace(fresh, media_encoded=True)
    a = Frontend()(IterationPlan((fresh,)), parsed.model, parsed.runtime, parsed.machine)
    b = Frontend()(IterationPlan((ready,)), parsed.model, parsed.runtime, parsed.machine)
    assert a.valid_tokens == b.valid_tokens == 8
    assert any(op.op_id.startswith("media0:") for op in a.ops)
    assert not any(op.op_id.startswith("media0:") for op in b.ops)
    assert sum(op.flops for op in a.ops) > sum(op.flops for op in b.ops)
    assert parsed.workload.requests[0].prompt_tokens == 8
    patches = parsed.model.vision.patch_tokens_for_media(4)
    qk = next(op for op in a.ops if op.op_id == "media0:vision.l0.qk")
    pv = next(op for op in a.ops if op.op_id == "media0:vision.l0.pv")
    assert qk.flops == pv.flops == 2 * patches**2 * parsed.model.vision.query_width
    assert pv.hbm_bytes == pv.mte2_bytes + pv.mte3_bytes


def test_draft_quantization_and_projection_shards_match_resident_weights():
    from mindie_sim.catalog.quant import quant_preset
    parsed = draft_case()
    quant = quant_preset("w8a8_dynamic")
    draft = replace(parsed.model.auxiliary.mtp, weight_category="mtp")
    model = replace(parsed.model, quant=quant, auxiliary=AuxiliarySpec(mtp=draft))
    views = (RequestView("r", 1, 4, True, "decode", draft_proposal_tokens=2),)
    work = Frontend()(IterationPlan(views), model, parsed.runtime, parsed.machine)
    draft_matrices = [op for op in work.ops if op.op_id.startswith("draft") and op.flops]
    assert all(op.dtype == "bf16" for op in draft_matrices)
    body = [op for op in work.ops if not op.op_id.startswith("draft") and op.recipe == "qkv_projection"]
    assert all(op.dtype == "int8" for op in body)
    term = next(term for term in draft_terms(draft, model) if term.name == "draft_q")
    assert term_bytes(term, quant, parsed.runtime.parallel) == draft.q_width * draft.hidden_size * 2


def test_one_token_response_does_not_prepare_an_unused_draft():
    parsed = draft_case()
    workload = replace(parsed.workload, requests=(replace(parsed.workload.requests[0], output_tokens=1),))
    parsed = replace(parsed, workload=workload)
    result = predict(parsed, event_limit=2000)
    assert not any("draft_prefill." in event["task"] for event in result["events"])
    assert result["metrics"]["makespan_ms"] == result["requests"][0]["ttft_ms"]
