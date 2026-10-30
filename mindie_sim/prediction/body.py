"""An explicit target-body forward probe through the ordinary graph engine.

[Role] Expose device-only forward spans in public prediction entry points.
[Invariants] Query tokens are actual verification rows, never accepted tokens.
  A probe retains declared padding, routing and complete EP cohorts. Device
  communication remains causal. Host, sampling and inter-step waits are outside.
[Timeouts & Locks] One finite graph; no request history or external work.
[Side-Effect Policy] Pure local simulation; no hardware access or calibration.
"""
from .graph_analysis import graph_costs
from .operator_analysis import target_graphs, cohort_sequence_counts
from .structure import parse_case
from mindie_sim.schema.common import ModelingError


def body_forward(case, *, phase="decode", tokens=1, context=0, sequences=1,
                 recipes=None, primitives=None, event_limit=0):
    case = parse_case(case)
    if case.model.is_standalone_draft:
        raise ModelingError("draft_requires_target_model:attach_auxiliary_to_target;catalog_inspection_available")
    graphs = target_graphs(case, phase=phase, tokens=tokens, context=context,
                           sequences=sequences, scope="body",
                           apply_body_defaults=recipes is None)
    default_selection = (graphs.default_selection if recipes is None else
                         {"profile": "explicit_recipe_book_preserved"})
    result = graph_costs(case, graphs, recipes=recipes, primitives=primitives,
                         external_inputs={"analysis_input": 0},
                         include_submission=False, isolated=False,
                         trace_limit=event_limit)
    counts = cohort_sequence_counts(sequences, tuple(graphs))
    from .balanced_routing import cohort_padded_tokens
    from mindie_sim.schema.workload import RequestView
    padded = {r: cohort_padded_tokens((tuple(
        RequestView(f"geometry{r}:{i}", tokens, context, False, phase)
        for i in range(count)),), case.runtime) for r, count in counts.items()}
    from .decode_probe import probe_capacity
    capacity = probe_capacity(case, context=context, tokens=tokens,
                              sequences=sequences, phase=phase)
    return {
        **result, "schema": "mindie-sim.body-forward/1", "status": "modeled",
        "scope": "body", "body_forward_ms": result["duration_ms"],
        "metric_boundary": "target_body_device_graph",
        "geometry": {"phase": phase, "tokens_per_sequence": tokens,
                     "context_tokens": context, "sequences_per_replica": sequences,
                     "participating_replicas": list(graphs),
                     "sequences_by_replica": counts,
                     "valid_query_tokens_by_replica": {r: tokens * count for r, count in counts.items()},
                     "padded_query_tokens_by_replica": padded,
                     "valid_query_tokens": tokens * sum(counts.values()),
                     "padded_query_tokens": sum(padded.values())},
        "routing": case.runtime.routing or {"policy": "probability_moments"},
        "default_mechanisms": default_selection,
        "capacity": capacity,
        "boundary_details": {
            "start": "input hidden states and KV cache ready",
            "end": "target transformer body output ready on all cohort participants",
            "excluded": ["embedding", "final_norm", "lm_head", "draft_generation",
                         "sampling", "host_submission", "scheduler", "pd_handoff",
                         "inter_step_wait"],
            "speculation": "tokens includes every target verification row; acceptance does not scale work",
            "prefix_cache": "context KV is ready; query rows still execute",
            "deployment": "first independent DP cohort; no request admission or capacity guarantee",
        },
    }


def pd_body_forward(raw, **options):
    common = {k: v for k, v in raw.items() if k not in {"pools", "connector"}}
    pools = {name: body_forward(
        {**common, **raw["pools"][name], "model": raw["model"], "workload": raw["workload"]},
        **options) for name in ("prefill", "decode")}
    from mindie_sim.coverage import combine_coverage
    models = [parse_case({**common, **raw["pools"][name], "model": raw["model"],
                          "workload": raw["workload"]}).model for name in pools]
    return {
        "schema": "mindie-sim.body-forward/1", "status": "modeled", "scope": "body",
        "deployment": "disaggregated", **combine_coverage(models),
        "metric_boundary": "independent_target_body_device_graph_per_pool",
        "body_forward_ms": {name: value["body_forward_ms"] for name, value in pools.items()},
        "pools": pools,
    }
