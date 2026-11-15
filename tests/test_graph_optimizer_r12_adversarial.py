"""NR-S2-FIX12/13/14: complete control quotient and raw port preservation.

[Role] Prove that contraction ordering reads the complete known dependency
  relation, including control declarations on surviving nodes the rewrite did
  not touch, for all five supported control keys, all four declaration forms and
  both scheduled/unscheduled data variants.  The same witnesses cover duplicate
  same-key metadata entries and a genuine multi-hop control cycle.  Boundary
  duplicates cover incoming/outgoing edges, both declaration orders and both
  by-name modes; touched survivors retain every target and positional operand.
  Split port declarations survive projection, contraction, events and prefetch.
[Invariants] An acyclic contraction is accepted and placed so that every
  multi-hop control edge points forward (``gate < fused < early < tail``), its
  evaluation is positive, and inserting then stripping generated stream events
  preserves that order.  A control cycle formed through an untouched survivor
  raises ``fusion_boundary_order_invalid`` rather than being silently accepted.
  Untouched surviving ``OpSpec`` objects are never replaced, and control names
  unavailable in the known graph are never invented as prerequisites.
[Timeouts & Locks] Pure in-process simulation with bounded per-case graphs; no
  clocks, sockets or locks.
[Side-Effect Policy] No filesystem, environment or device mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from mindie_sim.prediction.fusion_bridge import (
    Rewrite, _strip_events, add_prefetch, apply_prefetch_rewrite, apply_rewrite,
    canonical_plan, fused_op_spec, insert_stream_events, rewrite_fusion,
    settle_prefetch_rewrite, timeline_by_op,
)
from mindie_sim.prediction.fusion_control_rewrite import (
    apply_incoming_controls, ordering_prerequisites, region_boundary_ports,
    rewrite_consumers,
)
from mindie_sim.prediction.fusion_ports import (
    CONTROL_METADATA_KEYS, control_dependency_names, declared_control_order,
    full_operand_ports, input_port_entries, projected_ports, retained_ports,
)
from mindie_sim.prediction.fusion_recipe import (
    RejectedFusion, synthesize_vector,
)
from mindie_sim.prediction.weight_prefetch import PrefetchDecision

from graph_opt_fixtures import hw
from test_graph_optimizer_r8_adversarial import _add, case

_HARDWARE = hw(max_streams=2, stream_switch_ms=0.01)
_FORMS = ("tuple", "list", "string", "mapping")
_MATRIX = tuple(
    (key, form, scheduled, duplicate)
    for key in CONTROL_METADATA_KEYS
    for form in _FORMS
    for scheduled in (False, True)
    for duplicate in (False, True)
)


def _declare(form, name):
    """One control name in each supported declaration form."""
    return {
        "tuple": (name,),
        "list": [name],
        "string": name,
        "mapping": {name: "ticket"},
    }[form]


def _start_ms(row):
    """Scheduled start from a canonical timeline row."""
    start = row.get("start_ms")
    return float(row["ready_ms"] if start is None else start)


def _candidate(plan):
    candidate = synthesize_vector(
        plan.graph, ("a", "b"), tile_rows=1,
        capacity_per_owner=plan.hardware.ub_bytes_per_aiv)
    assert not isinstance(candidate, RejectedFusion), candidate.reasons
    return candidate


def _acyclic_ops(key, form, scheduled, duplicate):
    """``early -> tail`` on untouched nodes; the duplicate variant adds a
    second same-key ``gate`` declaration after ``early``.  Both entries agree
    with the required order, so acceptance proves duplication is tolerated."""
    tail_metadata = [(key, _declare(form, "early"))]
    if duplicate:
        tail_metadata.append((key, _declare(form, "gate")))
    tail_data = (("analysis_input", "early") if scheduled
                 else ("analysis_input",))
    return (
        _add("a", ("analysis_input",)),
        _add("early", ("analysis_input",),
             metadata=((key, _declare(form, "a")),)),
        _add("tail", tail_data, metadata=tuple(tail_metadata)),
        _add("gate", ("analysis_input",)),
        _add("b", ("a", "gate"),
             metadata=((key, _declare(form, "gate")),)),
        _add("z", ("b",)),
    )


def _cycle_ops(key, form, scheduled, duplicate):
    """``fused:a+b -> x -> gate -> fused:a+b`` through untouched ``gate``.
    The duplicate entry adds an unavailable name so a metadata dict would lose
    the essential ``x`` edge; an unknown control name must never be invented."""
    gate_metadata = [(key, _declare(form, "x"))]
    if duplicate:
        gate_metadata.append((key, _declare(form, "external_ticket")))
    gate_data = (("analysis_input", "x") if scheduled
                 else ("analysis_input",))
    return (
        _add("a", ("analysis_input",)),
        _add("x", ("analysis_input",),
             metadata=((key, _declare(form, "a")),)),
        _add("gate", gate_data, metadata=tuple(gate_metadata)),
        _add("b", ("a", "gate"),
             metadata=((key, _declare(form, "gate")),)),
        _add("z", ("b",)),
    )


def test_r12_multi_hop_acyclic_control_order():
    for key, form, scheduled, duplicate in _MATRIX:
        plan = canonical_plan(case(), _acyclic_ops(
            key, form, scheduled, duplicate))
        assert plan.graph.validate()
        assert plan.evaluate()["duration_ms"] > 0
        originals = {op.op_id: op for op in plan.ops}
        rewrite = rewrite_fusion(plan, _candidate(plan))
        after = apply_rewrite(plan, rewrite)
        after.graph.validate()
        fused_id = rewrite.fused.op_id
        order = {op.op_id: index for index, op in enumerate(after.ops)}
        assert order["gate"] < order[fused_id] < order["early"] < order["tail"]
        for survivor in ("gate", "tail"):
            assert next(op for op in after.ops
                        if op.op_id == survivor) is originals[survivor]
        measurement = after.evaluate()
        assert measurement["duration_ms"] > 0
        rows = timeline_by_op(measurement)
        assert rows["early"]["complete_ms"] <= _start_ms(rows["tail"])
        assert rows["early"]["stream"] == rows["tail"]["stream"]
        moved = tuple(
            replace(op, stream="aux") if op.op_id in {"early", "tail"} else op
            for op in after.ops)
        inserted = insert_stream_events(moved, _HARDWARE)
        assert inserted != moved
        for ops in (inserted, _strip_events(inserted)):
            fresh = after.rewrite(ops)
            fresh.graph.validate()
            fresh_order = {op.op_id: index for index, op in enumerate(fresh.ops)}
            assert (fresh_order["gate"] < fresh_order[fused_id]
                    < fresh_order["early"] < fresh_order["tail"])
            assert fresh.evaluate()["duration_ms"] > 0


def test_r12_multi_hop_control_cycle_is_rejected():
    for key, form, scheduled, duplicate in _MATRIX:
        plan = canonical_plan(case(), _cycle_ops(
            key, form, scheduled, duplicate))
        assert plan.graph.validate()
        assert plan.evaluate()["duration_ms"] > 0
        with pytest.raises(ValueError, match="fusion_boundary_order_invalid"):
            rewrite_fusion(plan, _candidate(plan))


def _raw_controls(metadata, selected=None):
    """Independent raw reader; retain repeats so dedup assertions are meaningful."""
    pairs = metadata.items() if isinstance(metadata, Mapping) else metadata
    return tuple(str(name) for key, value in pairs
                 if key in CONTROL_METADATA_KEYS
                 if selected is None or key == selected
                 for name in ((value,) if isinstance(value, str) else value))


def _quotient(ops, members=()):
    """Form expected edges from original declarations before contracting ids."""
    def contracted(name):
        return "fused:a+b" if name in members else name
    known = {contracted(op.op_id) for op in ops}
    edges = {name: set() for name in known}
    for op in ops:
        names = (*op.deps, *_raw_controls(op.metadata))
        pairs = op.metadata.items() if isinstance(op.metadata, Mapping) else op.metadata
        for key, value in pairs:
            if key == "fusion_input_ports":
                names += tuple(dep for dep, _port in value)
        target = contracted(op.op_id)
        edges[target].update(contracted(name) for name in names
                             if contracted(name) in known and contracted(name) != target)
    return edges


def _boundary_rewrite(plan, with_by_name):
    candidate = _candidate(plan)
    if with_by_name:
        return rewrite_fusion(plan, candidate)
    members = set(candidate.member_ops)
    specs = tuple(op for op in plan.ops if op.op_id in members)
    layout = {op.op_id: full_operand_ports(op, None) for op in plan.ops}
    boundary = region_boundary_ports(plan.ops, members, layout)
    fused = fused_op_spec(candidate, declared_hbm=int(sum(op.hbm_bytes for op in specs)),
                         member_specs=specs, boundary_ports=boundary)
    _, ports, _ = retained_ports(candidate.analysis, specs, fused.op_id)
    fused = apply_incoming_controls(fused, specs, members, boundary, None)
    rewritten, anchor = rewrite_consumers(plan.ops, members, layout, fused, ports, None)
    return Rewrite((*rewritten[:anchor], fused, *rewritten[anchor:]), fused,
                   candidate.recipe, candidate.candidate_id, candidate.member_ops)


def test_r13_duplicate_boundary_quotient():
    for key in CONTROL_METADATA_KEYS:
        for form in _FORMS:
            for with_by_name in (False, True):
                for side in ("outgoing", "incoming"):
                    for reverse in (False, True):
                        for cycle in (False, True):
                            ops = list(_acyclic_ops(key, form, False, False))
                            ops[4] = replace(ops[4], deps=("a",))
                            if cycle:
                                ops[3] = replace(ops[3], metadata=((key, _declare(form, "tail")),))
                            index, name = (1, "a") if side == "outgoing" else (4, "gate")
                            entries = ((key, _declare(form, name)), (key, _declare(form, "external_ticket")))
                            ops[index] = replace(ops[index], metadata=entries[::-1] if reverse else entries)
                            ops = tuple(ops)
                            before = {op.op_id: i for i, op in enumerate(ops)}
                            assert all(before[p] < before[c] for c, deps in _quotient(ops).items() for p in deps)
                            expected = _quotient(ops, {"a", "b"})
                            plan = canonical_plan(case(), ops)
                            assert plan.graph.validate() and plan.evaluate()["duration_ms"] > 0
                            if cycle:
                                with pytest.raises(ValueError, match="^fusion_boundary_order_invalid$"):
                                    _boundary_rewrite(plan, with_by_name)
                                continue
                            rewrite = _boundary_rewrite(plan, with_by_name)
                            after = apply_rewrite(plan, rewrite)
                            assert after.graph.validate() and after.evaluate()["duration_ms"] > 0
                            assert _quotient(after.ops) == ordering_prerequisites(after.ops) == expected
                            position = {op.op_id: i for i, op in enumerate(after.ops)}
                            assert all(position[p] < position[c] for c, deps in expected.items() for p in deps)
                            surviving = {op.op_id: op for op in after.ops}
                            assert surviving["gate"] is ops[3] and surviving["tail"] is ops[2]
                            target = surviving["early"] if side == "outgoing" else rewrite.fused
                            names = ("fused:a+b" if side == "outgoing" else "gate", "external_ticket")
                            assert _raw_controls(target.metadata) == (names[::-1] if reverse else names)
                            for op in after.ops:
                                assert not {"a", "b"} & set(_raw_controls(op.metadata))
                                assert "external_ticket" not in op.deps
                            assert not {"a", "b", "external_ticket"} & set(after.graph.external_inputs)
                            assert _boundary_rewrite(plan, with_by_name).ops == rewrite.ops


def test_r13_touched_duplicate_controls_keep_targets_and_data():
    for key in CONTROL_METADATA_KEYS:
        for form in _FORMS:
            for with_by_name in (False, True):
                for other in ("x", "external_ticket"):
                    for explicit in (False, True):
                        metadata = tuple((key, {name: i} if form == "mapping" else _declare(form, name))
                                         for i, name in enumerate(("a", other, "b", "a", other)))
                        positions = (("a", "a:out"), ("x", "x:out"), ("a", "a:out"))
                        if explicit:
                            metadata += (("fusion_input_ports", positions),)
                        ops = (_add("a", ("analysis_input",)), _add("b", ("a",)),
                               _add("x", ("analysis_input",)),
                               _add("c", ("a", "x", "a") if explicit else ("analysis_input",), metadata=metadata))
                        by_name = {op.op_id: op for op in ops} if with_by_name else None
                        layout = {op.op_id: full_operand_ports(op, by_name) for op in ops}
                        if explicit:
                            assert layout["c"] == positions
                        fused = _add("fused:a+b", ("analysis_input",))
                        rewritten, anchor = rewrite_consumers(
                            ops, {"a", "b"}, layout, fused, {"a:out": "fused:a+b:a_out"}, by_name)
                        consumer = next(op for op in rewritten if op.op_id == "c")
                        assert _raw_controls(consumer.metadata, key) == (fused.op_id, other)
                        if form == "mapping":
                            assert [(name, payload) for k, v in consumer.metadata if k == key
                                    for name, payload in v.items()] == [(fused.op_id, 0), (other, 1)]
                        expected = ((fused.op_id, "fused:a+b:a_out"), ("x", "x:out"),
                                    (fused.op_id, "fused:a+b:a_out")) if explicit else layout["c"]
                        assert input_port_entries(consumer.metadata) == full_operand_ports(consumer) == expected
                        assert consumer.deps == ((fused.op_id, "x") if explicit else ("analysis_input", fused.op_id))
                        assert next(op for op in rewritten if op.op_id == "x") is ops[2]
                        assert ops[3].metadata == metadata and "external_ticket" not in consumer.deps
                        after = (*rewritten[:anchor], fused, *rewritten[anchor:])
                        assert _quotient(after) == _quotient(ops, {"a", "b"})
                        position = {op.op_id: i for i, op in enumerate(after)}
                        assert all(position[p] < position[c] for c, deps in _quotient(after).items() for p in deps)


def test_r13_raw_metadata_order_and_mapping_input():
    keys = tuple(reversed(CONTROL_METADATA_KEYS))
    for container in (tuple, list, dict):
        for form in _FORMS:
            pairs = [(key, _declare(form, f"gate{i}")) for i, key in enumerate(keys)]
            if container is not dict:
                pairs += [(keys[0], _declare(form, "extra")), (keys[0], _declare(form, "gate0"))]
            pairs += [("fusion_input_ports", (("gate0", "gate0:out"),))]
            metadata = container(pairs)
            expected = tuple(f"gate{i}" for i in range(5)) + (() if container is dict else ("extra",))
            assert declared_control_order(metadata) == expected
            op = replace(_add("member", ("gate0",)), metadata=metadata)
            assert control_dependency_names(op) == frozenset(expected) - {"gate0"}
            fused = apply_incoming_controls(
                replace(_add("fused", ()), metadata={"unchanged": 1}), (op,), {"member"},
                (("gate0", "gate0:out"),), None)
            assert _raw_controls(fused.metadata) == expected[1:]
            assert fused.deps == ("gate0",) and full_operand_ports(fused) == (("gate0", "gate0:out"),)
            gate = _add("gate0", ("analysis_input",))
            ops = (gate, _add("gate1", ("analysis_input",)), op)
            rewritten, _ = rewrite_consumers(
                ops, {"gate1"}, {o.op_id: full_operand_ports(o) for o in ops},
                _add("fused:gate1", ("analysis_input",)), {}, None)
            assert rewritten[0] is gate
            assert _raw_controls(rewritten[1].metadata) == ("gate0", "fused:gate1", *expected[2:])
            assert rewritten[1].deps == ("gate0", "fused:gate1")
            assert full_operand_ports(rewritten[1]) == (("gate0", "gate0:out"),)


def test_r14_raw_ports_survive_projection_contraction_events_and_prefetch():
    positions = (("event:g", "event:g:out"), ("x", "x:out"),
                 ("y", "y:out"), ("x", "x:out"))
    records = (("p:early", "bf16", (1, 4), "strided", 16, True),
               ("p:out", "bf16", (1, 4), "contiguous", 8, False))
    expected = tuple(port for _, port in positions)
    decision = PrefetchDecision("p", (), 256, 256, 0, 256, 0,
                                1.0, 1e6, 0.0, True, "regression", ())
    for key in CONTROL_METADATA_KEYS:
        for form in _FORMS:
            for container in (tuple, list):
                for split in (False, True):
                    metadata = [(key, _declare(form, n)) for n in ("a", "event:g", "external_ticket")]
                    for name, rows in (("fusion_input_ports", positions), ("retained_ports", records)):
                        metadata.extend(((name, rows[:1]), (name, rows[1:])) if split else ((name, rows),))
                    metadata += [("resource_work", {"mte1": 3.0}), ("prefetch_hit_bytes", -1.0)]
                    producer = replace(_add("p", tuple(dep for dep, _ in positions)),
                                       stream="aux", metadata=container(metadata))
                    ops = (_add("a", ("analysis_input",)), _add("b", ("a",)),
                           _add("event:g", ("analysis_input",), family="event"),
                           _add("x", ("analysis_input",)), _add("y", ("analysis_input",)), producer,
                           _add("c", ("p",), metadata=(("fusion_input_ports", (("p", "p:early"),)),)))
                    plan = canonical_plan(case(), ops)
                    after = apply_rewrite(plan, rewrite_fusion(plan, _candidate(plan)))
                    for base in (plan, after):
                        eventful = insert_stream_events(base.ops, _HARDWARE)
                        assert eventful != base.ops
                        assert insert_stream_events(eventful, _HARDWARE) == eventful
                        for variant in (base.ops, eventful, _strip_events(eventful)):
                            stage = base.rewrite(variant)
                            assert stage.graph.validate() and stage.evaluate()["duration_ms"] > 0
                            assert stage.graph.ops["p"].inputs == expected
                            assert stage.graph.ops["c"].inputs == ("p:early",)
                            prior = next(op for op in variant if op.op_id == "p")
                            pending, prefetch_id = add_prefetch(stage, decision, _HARDWARE)
                            rewrite = settle_prefetch_rewrite(pending, decision, prefetch_id, 256, _HARDWARE)
                            settled = apply_prefetch_rewrite(pending, rewrite, prefetch_id)
                            assert _raw_controls(rewrite.consumer.metadata) == _raw_controls(prior.metadata)
                            changed_keys = {"resource_work", "prefetch_hit_bytes", "fusion_input_ports"}
                            assert tuple((k, v) for k, v in rewrite.consumer.metadata if k not in changed_keys) == (
                                tuple((k, v) for k, v in prior.metadata if k not in changed_keys))
                            assert tuple(r.metadata_record() for r in projected_ports(
                                rewrite.consumer.metadata)) == records
                            scalars = dict(rewrite.consumer.metadata)
                            assert scalars["resource_work"] == {"mte1": 259.0}
                            assert scalars["prefetch_hit_bytes"] == 256.0
                            assert rewrite.consumer.hbm_bytes == prior.hbm_bytes - 256
                            inserted = insert_stream_events(settled.ops, _HARDWARE)
                            for final_ops in (settled.ops, inserted, _strip_events(inserted)):
                                final = settled.rewrite(final_ops)
                                assert final.graph.validate() and final.evaluate()["duration_ms"] > 0
                                assert final.graph.ops["p"].inputs == (*expected, f"{prefetch_id}:out")
                                assert final.graph.ops["c"].inputs == ("p:early",)
                                final_p = next(op for op in final.ops if op.op_id == "p")
                                assert _raw_controls(final_p.metadata) == _raw_controls(prior.metadata)
                                for record in records:
                                    value = final.graph.values[record[0]]
                                    assert (value.dtype, value.shape, value.layout,
                                            value.storage_bytes, value.public) == record[1:]
                    assert producer.metadata == container(metadata)
