"""Isolated complete SituMxQuant call on the shared engine.

The example is deliberately standalone: it builds one BF16-to-FP8/E8M0 source
call, prices it on an existing hardware case, and prints only the device work
and bounded buffer state. It does not read measurements or run hardware.

Run:

    PYTHONPATH=. .venv/bin/python examples/situ_mx_quant_complete_call.py
"""
import json
from pathlib import Path

from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.family_situ_mx import Config, make_op, recipe_book
from mindie_sim.prediction.structure import parse_case


ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = json.loads((ROOT / "examples/cases/kimi-k3-a5-current-checkpoint-conditional.json").read_text())
    # Keep the existing A5 hardware and model declaration; the example supplies
    # its own small tensor and explicit source recipe, not a model-wide profile.
    case = parse_case(raw)
    config = Config(rows=1, width=768, output_dtype="fp8_e4m3",
                    beta=4.0, linear_beta=25.0)
    result = call_costs(case, (make_op(config),), recipes=recipe_book(),
                        invocations=1, mode="eager")
    services = {key: value for key, value in result["execution"]["serviced"].items()
                if value}
    print(json.dumps({
        "boundary": result["boundary"],
        "single_call": result["single_call"],
        "duration_ms": result["duration_ms"],
        "serviced": services,
        "peak_allocated_per_unit": result["execution"]["peak_allocated_per_unit"],
        "live_allocated": {key: value for key, value in
                           result["execution"]["live_allocated"].items() if value},
        "live_tasks": result["execution"]["live_tasks"],
        "note": ("source lane work includes both tanh paths, exponent reduction, "
                 "E8M0 scales and 256-wide FP8 conversion; the retained ~65 us "
                 "observation remains unexplained by this source/resource program"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
