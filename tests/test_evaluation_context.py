"""One canonical evaluation context per call; every isolated basis stays local."""
from dataclasses import replace

from mindie_sim.compose import compose_case
from mindie_sim.prediction.cost import CostLedger
from mindie_sim.prediction.efficiency import analyze_network_efficiency
from mindie_sim.prediction.evaluation import (
    BoundedCache, EvaluationContext, ExecutionBindings, FrozenHardwareView,
)
from mindie_sim.prediction.structure import parse_case


def _case():
    return parse_case(compose_case(
        "qwen3-32b", "a3", tp=2, layers=4, prompt_tokens=16,
        output_tokens=2, concurrency=1))


def test_one_cost_model_and_one_binding_set_per_evaluation():
    from mindie_sim.prediction import cost as cost_module
    from mindie_sim.prediction import evaluation as evaluation_module
    counts = {"cost": 0, "bindings": 0, "device_dict": 0}
    original_cost = cost_module.CostModel.__init__
    original_bindings = evaluation_module.ExecutionBindings.create.__func__
    original_to_dict = evaluation_module.FrozenHardwareView.from_machine.__func__
    from mindie_sim.schema import hardware

    original_device = hardware.Device.to_dict

    def counted_cost(self, *args, **kwargs):
        counts["cost"] += 1
        return original_cost(self, *args, **kwargs)

    def counted_bindings(cls, *args, **kwargs):
        counts["bindings"] += 1
        return original_bindings(cls, *args, **kwargs)

    def counted_device(self):
        counts["device_dict"] += 1
        return original_device(self)

    cost_module.CostModel.__init__ = counted_cost
    evaluation_module.ExecutionBindings.create = classmethod(counted_bindings)
    hardware.Device.to_dict = counted_device
    try:
        case = _case()
        report = analyze_network_efficiency(
            case, phase="decode", tokens=1, context=16, sequences=1, scope="body")
    finally:
        cost_module.CostModel.__init__ = original_cost
        evaluation_module.ExecutionBindings.create = classmethod(original_bindings)
        hardware.Device.to_dict = original_device
    assert report["operators"]
    assert counts["cost"] == 1
    assert counts["bindings"] == 1
    assert counts["device_dict"] <= len(case.machine.dies)


def test_evaluation_local_caches_are_bounded():
    cache = BoundedCache(limit=3)
    for index in range(6):
        cache.put(index, index)
    assert len(cache) == 3
    assert cache.evictions == 3
    assert cache.get(5) == 5
    assert cache.hits == 1
    assert cache.get(99) is None
    assert cache.misses == 1


def test_frozen_hardware_counterfactual_changes_signature():
    case = _case()
    first = FrozenHardwareView.from_machine(case.machine)
    rates = tuple((name, float(value) * 0.5 if name == "hbm" else value)
                  for name, value in case.machine.dies[0].rates)
    changed_die = replace(case.machine.dies[0], rates=rates)
    changed_machine = replace(
        case.machine, dies=(changed_die, *case.machine.dies[1:]))
    second = FrozenHardwareView.from_machine(changed_machine)
    assert first.digest() != second.digest()


def test_isolated_rows_do_not_leak_into_main_basis():
    from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs
    case = _case()
    context = EvaluationContext.create(case)
    ops = target_graphs(case, phase="decode", tokens=1, context=16,
                        sequences=1, scope="body")[0][:8]
    report = operator_costs(case, tuple(ops), evaluation=context)
    counts = context.cost.summary()["operator_occurrences"]
    assert counts.get("analytic_work", 0) == 0
    assert all(count == 0 for count in counts.values())
    assert report["operator_occurrences"] > 0
    # Every isolated row owns a ledger with only its own records.
    for row in report["operators"]:
        assert row["basis"]["operator_occurrences"].get("analytic_work", 0) >= 1
    assert CostLedger().counts["analytic_work"] == 0
