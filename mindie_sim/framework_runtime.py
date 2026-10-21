"""Framework lifecycle adapters for the existing predictive resource engine.

Host CPU costs are explicit profile inputs. Engine commit gates subsequent
scheduling; ordered per-request stream tasks use DMA and never gate it.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .a3_types import A3InputError
from .framework_tasks import FrameworkProfile, add_framework_task


def generated_counts(step: Mapping[str, Any]) -> dict[str, int]:
    """Count committed tokens, including clipped speculative output bursts."""
    states = {state.request.request_id: state for state in step['selected']}
    result: dict[str, int] = {}
    for view in step['views']:
        state = states[view.request_id]
        remaining = max(0, state.output - state.generated)
        count = 0
        if view.samples:
            count = 1
            if view.draft_verify:
                proposal = max(0, int(view.draft_proposal_tokens))
                accepted = max(0, min(proposal, int(view.draft_accepted_tokens)))
                count = accepted + 1
        result[view.request_id] = min(remaining, count)
    return result


def submission_ids(step: Mapping[str, Any], replacements: Mapping[str, Any]) -> list[str]:
    if step['costed'] is None:
        return ['backend']
    return [cost.op.op_id for cost in step['costed']
            if cost.op.op_id not in replacements
            and not dict(cost.op.metadata).get('witness_only')
            and not dict(cost.op.metadata).get('not_execution')
            and any(amount > 0 for _, amount in cost.demands)]


def separate_host_intercepts(
    step: dict[str, Any], case: Any, profile: FrameworkProfile, *, cost_model: Any = None,
) -> None:
    """Replace eager host intercepts, retaining declared graph device dispatch.

    Whole-span and MC2 envelopes cannot be separated into host/device terms
    reliably; refuse them on opt-in instead of silently charging host twice.
    """
    if step['costed'] is None:
        return
    from .operator_plan import EstimateContext, evaluate_kernel_plan
    from .a3_cost import PredictiveCostModel, grouped_matmul_launch_count
    from .a3_physical_calibration import ACCEPTED_TIMING_SCOPES

    replayed = step['work'].graph_bucket is not None
    changed = []
    for cost in step['costed']:
        if cost.mc2 or dict(cost.demands).get('completion_envelope_ms', 0):
            raise A3InputError(f'framework_opaque_span_not_separable:{cost.op.op_id}')
        if grouped_matmul_launch_count(cost.op) > 1:
            raise A3InputError(f'framework_multiple_launches_not_supported:{cost.op.op_id}')
        physical = dict(cost.physical or {})
        actual_scope = str(physical.get('fitted_timing_scope') or physical.get('timing_scope') or '')
        if physical.get('applied') and actual_scope not in {
            scope for scope in ACCEPTED_TIMING_SCOPES if scope.startswith('device_event_')
        }:
            raise A3InputError(f'framework_physical_scope_not_separable:{cost.op.op_id}:{actual_scope}')
        fabric_ms = 0.0
        if cost.op.link_bytes > 0:
            if (cost.calibration.observations_used > 0 or physical.get('applied')
                    or dict(cost.op.metadata).get('comm_stages')):
                raise A3InputError(f'framework_communication_intercept_not_separable:{cost.op.op_id}')
            # Match the same physical fabric resolver as the cost model. The
            # fixed wire setup is not host submission and must survive it.
            resolver = cost_model or PredictiveCostModel(
                case.machine, compute_dtype=case.model.compute_dtype)
            wire = resolver._link(cost.op)
            if wire is not None:
                fabric_ms = float(wire.latency_us) / 1000.0
        old_launch = float(cost.launch_ms)
        executable = not (dict(cost.op.metadata).get('witness_only')
                          or dict(cost.op.metadata).get('not_execution'))
        launch = ((old_launch if replayed else profile.device_launch_us / 1000.0)
                  + fabric_ms) if executable else 0.0
        detail = {**dict(cost.launch_detail or {}),
                  'framework_host_intercept_removed_ms': 0.0 if replayed else max(0.0, old_launch-fabric_ms),
                  'framework_device_dispatch_ms': launch,
                  'framework_fabric_setup_ms': fabric_ms,
                  'framework_host_submission': 'explicit_cpu_task'}
        if cost.execution_plan is not None:
            if cost.execution_plan.includes_dispatch:
                raise A3InputError(f'framework_plan_dispatch_not_separable:{cost.op.op_id}')
            context = EstimateContext(
                compute_dtype=case.model.compute_dtype,
                weight_dtype=case.model.weight_dtype,
                activation_dtype=case.model.activation_dtype,
                timing_scope=getattr(case, 'timing_scope', 'device_event_warm'),
                quantization=case.model.quantization,
            )
            elapsed = evaluate_kernel_plan(cost.execution_plan, case.machine, context,
                                           launch, stage=cost.op.stage)['elapsed_ms']
            estimation = {**dict(cost.operator_estimation or {}),
                          'isolated_plan_ms': elapsed,
                          'framework_host_intercept_removed_ms': detail['framework_host_intercept_removed_ms']}
            changed.append(replace(cost, launch_ms=launch, estimated_ms=elapsed,
                                   service_ms=max(0.0, elapsed-launch),
                                   operator_estimation=estimation, launch_detail=detail))
        else:
            # service_demands precede the old serial/folded launch composition.
            demands = tuple(cost.service_demands) or tuple(
                (key, value) for key, value in cost.demands if key != 'launch')
            changed.append(replace(cost, demands=demands, launch_ms=launch,
                                   estimated_ms=cost.service_ms+launch, launch_detail=detail))
    step['costed'] = changed


def before_device(
    engine: Any, profile: FrameworkProfile, step: dict[str, Any], replica: int,
    kernels: Sequence[str], previous_output: str | None,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    counts = {'requests': len(step['selected']), 'tokens': step['work'].valid_tokens,
              'kernels': len(kernels)}
    common = dict(prefix=step['prefix'], replica=replica, release_ms=step['start_ms'])
    schedule = add_framework_task(engine, profile, 'schedule', **counts, **common,
                                  dependencies=(previous_output,) if previous_output else ())
    inputs = add_framework_task(engine, profile, 'input', **counts, **common,
                                dependencies=(schedule,))
    replayed = step['work'].graph_bucket is not None
    tasks: list[str] = []
    deps: dict[str, tuple[str, ...]] = {}
    previous = inputs
    groups = [list(kernels)] if replayed and kernels else [[key] for key in kernels]
    for index, group in enumerate(groups):
        submit = add_framework_task(
            engine, profile, 'submit', requests=counts['requests'], tokens=counts['tokens'],
            kernels=1, **common, dependencies=(previous,), name=f'submit{index}')
        tasks.append(submit)
        previous = submit
        for key in group:
            deps[key] = (submit,)
    info = {'source': profile.source, 'schedule_task': schedule, 'input_task': inputs,
            'submit_tasks': tasks, 'submission_mode': 'graph_replay' if replayed else 'eager',
            'device_op_count': len(kernels), 'output_token_counts': generated_counts(step)}
    return info, deps


def after_device(
    engine: Any, profile: FrameworkProfile, step: dict[str, Any], replica: int,
    device_gate: str, previous_streams: dict[str, str],
    streams: dict[str, list[tuple[str, int]]],
) -> str:
    info = step['framework']
    output_counts = info['output_token_counts']
    common = dict(prefix=step['prefix'], replica=replica, release_ms=step['start_ms'])
    output = add_framework_task(
        engine, profile, 'output', requests=len(step['selected']),
        tokens=sum(output_counts.values()), kernels=info['device_op_count'],
        **common, dependencies=(device_gate,))
    tasks = {}
    for index, (rid, count) in enumerate(output_counts.items()):
        if count <= 0:
            continue
        deps = (output, previous_streams[rid]) if rid in previous_streams else (output,)
        task = add_framework_task(engine, profile, 'stream', requests=1, tokens=count,
                                  kernels=0, **common, dependencies=deps, name=f'stream{index}')
        previous_streams[rid] = task
        streams.setdefault(rid, []).append((task, count))
        tasks[rid] = task
    info.update(device_gate=device_gate, output_task=output, stream_tasks=tasks)
    return output


def request_with_client_times(state: Any, streams: Sequence[tuple[str, int]], engine: Any) -> dict[str, Any]:
    result = state.to_dict()
    bursts = [float(engine.completed_at[task]) for task, _ in streams]
    tokens = [end for end, (_, count) in zip(bursts, streams) for _ in range(count)]
    if len(tokens) != state.generated:
        raise A3InputError(f'framework_client_token_count_mismatch:{state.request.request_id}')
    first = tokens[0] if tokens else None
    finish = tokens[-1] if tokens and state.finished else None
    result.update(
        engine_token_event_times_ms=list(state.token_event_times_ms),
        client_token_times_ms=bursts, client_token_event_times_ms=tokens,
        client_first_token_ms=first, client_finish_ms=finish,
        client_ttft_ms=first-state.request.arrival_ms if first is not None else None,
        client_tpot_ms=(finish-first)/(state.generated-1)
        if finish is not None and state.generated > 1 else None,
        client_e2e_ms=finish-state.request.arrival_ms if finish is not None else None,
    )
    return result


def client_metrics(replica_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from .a3_metrics import _merge_replicas
    replicas = []
    for replica in replica_results:
        requests = []
        for item in replica['requests']:
            bursts = item['client_token_times_ms']
            requests.append({**item, 'first_token_ms': item['client_first_token_ms'],
                             'finish_ms': item['client_finish_ms'],
                             'ttft_ms': item['client_ttft_ms'], 'tpot_ms': item['client_tpot_ms'],
                             'e2e_ms': item['client_e2e_ms'],
                             'itl_ms': [b-a for a, b in zip(bursts, bursts[1:])]})
        replicas.append({'requests': requests, 'traces': []})
    return _merge_replicas(replicas)['metrics']
