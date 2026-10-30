"""Independent serving oracles and cross-device mechanism controls."""
from dataclasses import replace

import pytest

from mindie_sim.prediction import predict
from mindie_sim.prediction.capacity import MemoryPolicy
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.primitives import PrimitiveStore
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView


def case(*, dp=1, tp=1, pp=1, ep=1, prompt=4, output=3, requests=None):
    ranks = dp * tp * pp
    return {
        "machine": {
            "name": "declared-gpu", "architecture": "cuda", "num_devices": ranks,
            "device": {"memory_bytes": 1024 ** 3, "compute_peaks": {"bf16": 1e9, "fp32": 1e9},
                       "rates": {"memory": 1e9, "vector": 1e9}, "compute_units": 4},
            "links": [{"name": "fabric", "capacity_bytes_per_ms": 1000,
                       "members": list(range(ranks)), "semantics": "shared"}],
        },
        "model": {
            "name": "geometry", "layers": 2, "hidden_size": 32, "num_heads": 4,
            "num_kv_heads": 2, "head_dim": 8, "intermediate_size": 64, "vocab_size": 128,
        },
        "workload": {"requests": requests or [
            {"id": "r", "prompt_tokens": prompt, "output_tokens": output}]},
        "runtime": {"dp": dp, "tp": tp, "pp": pp, "ep": ep, "max_num_batched_tokens": 16},
    }


def linear_work(plan, model, runtime, machine):
    tokens = sum(view.q_len + view.draft_proposal_tokens for view in plan.requests)
    shape = (tokens,)
    op = OpSpec("work", "gemm", "body", 0, 0, tuple(v.request_id for v in plan.requests),
                shape, shape, shape, shape, flops=tokens * 1e6, dtype="bf16")
    return IterationWork((op,), plan, 1, tokens, tokens, None, "body", "test_linear")


def test_metrics_follow_commits_not_step_count_or_padding():
    result = predict(case(), lower=linear_work)
    row = result["requests"][0]
    assert row["ttft_ms"] == 4
    assert row["tpot_ms"] == 1
    assert row["finish_ms"] == 6
    assert result["metrics"]["output_tps"] == 500
    assert row["itl"] == {"count": 2, "mean_ms": 1, "max_ms": 1}


def test_arrival_during_work_waits_for_the_next_batch_and_keeps_queue_time():
    value = case(requests=[
        {"id": "a", "prompt_tokens": 4, "output_tokens": 2, "arrival_ms": 0},
        {"id": "b", "prompt_tokens": 2, "output_tokens": 1, "arrival_ms": 2},
    ])
    value["runtime"]["max_num_batched_tokens"] = 4
    rows = predict(value, lower=linear_work)["requests"]
    assert [(r["first_token_abs_ms"], r["finish_ms"]) for r in rows] == [(4, 7), (7, 7)]
    assert [r["ttft_ms"] for r in rows] == [4, 5]


def test_independent_data_replicas_overlap_without_double_counting_elapsed_time():
    value = case(dp=2, requests=[
        {"id": "a", "prompt_tokens": 4, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 4, "output_tokens": 1},
    ])
    result = predict(value, lower=linear_work)
    assert result["metrics"]["makespan_ms"] == 4
    assert result["metrics"]["output_tps"] == 500
    assert result["execution"]["work"]["flops"] == 8e6


def test_closed_loop_arrivals_are_released_by_predicted_completion():
    value = case(dp=2, requests=[
        {"id": "a", "prompt_tokens": 2, "output_tokens": 1, "chain_id": 1},
        {"id": "b", "prompt_tokens": 3, "output_tokens": 1, "chain_id": 1, "gap_ms": 5},
    ])
    rows = predict(value, lower=linear_work)["requests"]
    assert rows[0]["finish_ms"] == 2
    assert rows[1]["arrival_ms"] == 7
    assert rows[1]["first_token_abs_ms"] == 10
    assert rows[1]["ttft_ms"] == 3


def test_speculative_rejections_preserve_integer_commits_and_real_gaps():
    value = case(output=6)
    value["runtime"]["metadata"] = {
        "draft_mode": "mtp", "draft_proposal_tokens": 2,
        "draft_accept_trace": [0, 2], "draft_accept_tokens": 1,
    }
    result = predict(value, lower=linear_work)
    row = result["requests"][0]
    assert row["generated_tokens"] == 6
    assert row["iterations"] == 4
    assert row["finish_ms"] == 11
    assert row["itl"] == {"count": 5, "mean_ms": 7 / 5, "max_ms": 3}
    assert row["speculative"] == {"proposed_tokens": 4, "accepted_tokens": 2, "rolled_back_tokens": 2}


def test_capacity_failure_stays_in_slo_denominator():
    value = case()
    value["slo"] = {"ttft_ms": 100}
    result = predict(value, memory_policy=MemoryPolicy(workspace_bytes=2 * 1024 ** 3))
    assert result["status"] == "infeasible"
    assert result["requests"][0]["generated_tokens"] == 0
    assert result["metrics"]["slo"]["passing_fraction"] == 0
    assert result["memory"]["status"] == "oom"


def test_shared_interconnect_charges_every_sender_on_the_same_wire():
    parsed = parse_case(case(tp=2))
    topology = Topology(parsed.machine, parsed.runtime.parallel)
    shape = (1,)
    op = OpSpec("reduce", "allreduce", "decode", 0, 0, (), shape, shape, shape, shape, link_bytes=100)
    demand, latency = topology.demand(op, 0)
    engine = Engine(topology.resources())
    event = engine.submit("reduce", demand, latency=latency)
    engine.advance(until=(event,))
    assert demand == {"link:fabric": 200}
    assert event.time == .2


def test_model_label_and_measurement_availability_do_not_change_the_program():
    parsed = parse_case(case())
    plan = IterationPlan((RequestView("r", 4, 0, True, "prefill"),))
    lower = Frontend()
    first = lower(plan, parsed.model, parsed.runtime, parsed.machine)
    renamed = replace(parsed.model, name="unseen-architecture-label")
    second = lower(plan, renamed, parsed.runtime, parsed.machine)
    assert first.ops == second.ops
    without = predict(parsed)
    with PrimitiveStore(":memory:") as parameters:
        with_store = predict(parsed, primitives=parameters)
    assert without["metrics"] == with_store["metrics"]
    assert without["execution"]["serviced"] == with_store["execution"]["serviced"]


def test_ep_cohort_includes_idle_replicas_and_finishes_all_collectives():
    value = case(dp=2, ep=2, output=2)
    value["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    result = predict(value, event_limit=500)
    assert result["requests"][0]["status"] == "served"
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    collectives = [event for event in result["events"] if event["owner"] == "expert_group"]
    assert collectives
    assert all(event["end_ms"] >= event["start_ms"] for event in collectives)


def test_generic_gpu_and_split_copy_engine_use_the_same_serving_program():
    gpu = case(output=2)
    accelerator = case(output=2)
    accelerator["machine"]["architecture"] = "new-accelerator"
    # Even a one-of-four tail must supply the full shared memory rate.
    accelerator["machine"]["device"]["rates"].update(read=4e9, write=4e9)
    a, b = predict(gpu), predict(accelerator)
    assert a["metrics"]["output_tps"] == pytest.approx(b["metrics"]["output_tps"])
    assert a["metrics"]["ttft_ms"]["mean"] == pytest.approx(b["metrics"]["ttft_ms"]["mean"])
    assert a["execution"]["work"] == b["execution"]["work"]
    assert not any(key.startswith("mte") for key in a["execution"]["serviced"])
    assert any(key.startswith("mte") for key in b["execution"]["serviced"])
    accelerator["machine"]["device"]["rates"].update(read=2e9, write=2e9)
    limited = predict(accelerator)
    assert limited["metrics"]["output_tps"] < b["metrics"]["output_tps"]
    assert limited["execution"]["work"] == b["execution"]["work"]


def test_pipeline_microbatches_produce_finite_fill_and_drain():
    value = case(pp=2, requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1},
    ])

    def staged(plan, model, runtime, machine):
        tokens = plan.valid_tokens
        shape = (tokens,)
        ops = tuple(OpSpec(
            f"stage{stage}", "gemm", "prefill", stage, stage, tuple(v.request_id for v in plan.requests),
            shape, shape, shape, shape, flops=tokens * 1e6,
            deps=("stage0",) if stage else (), dtype="bf16",
        ) for stage in range(2))
        return IterationWork(ops, plan, 2, tokens, tokens, None, "prefill", "two_stage")

    serial = predict(value, lower=staged)
    value["runtime"]["pipeline_microbatches"] = 2
    pipeline = predict(value, lower=staged)
    assert serial["metrics"]["makespan_ms"] == 4
    assert pipeline["metrics"]["makespan_ms"] == 3
    assert [row["first_token_abs_ms"] for row in pipeline["requests"]] == [2, 3]
    assert pipeline["execution"]["work"] == serial["execution"]["work"]
