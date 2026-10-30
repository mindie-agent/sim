"""Check frozen response selection without replaying already verified Bodies.

[Role] Verify hardware-domain guards on actual frontend attention contracts.
[Invariants] This is binding validation, not a new timing-accuracy measurement.
  Changed physical supplies must retain the analytical source schedule.
[Timeouts & Locks] Finite metadata graphs only; no processes, device or waits.
[Side-Effect Policy] Write a compact local binding receipt.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from mindie_sim import target_graphs
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.structure import parse_case

HERE = Path(__file__).resolve().parent


def bindings(raw, geometry):
    case = parse_case(raw)
    graphs = target_graphs(case, scope="body", phase="decode", tokens=4,
                          context=geometry["context_tokens"],
                          sequences=geometry["sequences_per_replica"])
    ops = [op for op in graphs[0] if op.recipe == "flash_attention_decode"]
    cost = CostModel(case)
    execution = Execution(case, cost, include_submission=False)
    contexts = [execution._context(op, 0, "full") for op in ops]
    selected = [c for c in contexts if c.joint_native_service]
    return len(selected), [dict(c.work) for c in selected]


def main():
    old = json.loads((HERE / "body-validation.json").read_text())
    records = []
    for row in old["cases"]:
        path = ROOT / row["case_path"]
        raw = json.loads(path.read_text())
        count, work = bindings(raw, row["geometry"])
        expected = sum(rule["queries"].get("joint_native_service", 0)
                       for rule in row["service_regimes"])
        assert count == expected
        perturbations = []
        for section, role in (("rates", "hbm"), ("rates", "mte2_aic"),
                              ("rates", "vector"), ("compute_peaks", "bf16")):
            changed = deepcopy(raw)
            die = parse_case(raw).machine.dies[0]
            rate = dict(getattr(die, section))[role]
            changed["machine"]["device"].setdefault(section, {})[role] = rate/2
            altered, _ = bindings(changed, row["geometry"])
            assert altered == 0
            perturbations.append(dict(section=section, role=role, joint_calls=altered))
        records.append(dict(case_id=row["case_id"], unchanged_joint_calls=count,
                            unchanged_joint_work=work, perturbations=perturbations,
                            guarded_case_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (HERE / "domain-validation.json").write_text(json.dumps({
        "boundary": "frontend/context selection only, no timed Body rerun",
        "body_reference_commit": "1c5939cd", "cases": records}, indent=2)+"\n")
    print(len(records), "unchanged selections and", 4*len(records), "rate-domain misses verified")


if __name__ == "__main__":
    main()
