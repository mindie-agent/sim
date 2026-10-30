"""Import retained call protocols and compare explicitly supplied predictions.

[Role] Stream original captures into the common observation/comparison contract.
[Invariants] No split promotion, timing fit or synthesized prediction. Exported
  samples keep source identities; empty prediction coverage is a normal result.
[Timeouts & Locks] Streaming sources and JSONL output; finite comparison points.
[Side-Effect Policy] Writes only requested local reports and optional samples.
"""
import argparse
import json
from pathlib import Path

from mindie_sim.observations.captures import capture_inventory, iter_observations
from mindie_sim.observations.comparison import comparison_report
from mindie_sim.observations.store import json_items


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--predictions", type=Path,
                        help="accuracy-input/1 packet with explicit prediction identities")
    parser.add_argument("--samples-output", type=Path, help="Optional streamed normalized JSONL")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = next(json_items(args.protocol))
    campaign = next(json_items(args.predictions)) if args.predictions else {"predictions": []}

    def samples():
        if args.samples_output:
            with args.samples_output.open("w") as stream:
                for sample in iter_observations(protocol, root=args.root):
                    stream.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")
                    yield sample
        else:
            yield from iter_observations(protocol, root=args.root)

    report = {
        "schema": "mindie-sim.capture-import/1",
        "protocol": str(args.protocol),
        "inventory": capture_inventory(protocol, root=args.root),
        "comparison": comparison_report({**campaign, "observations": samples()}),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "timing_conditions": report["inventory"]["timing_conditions"],
        "samples": report["inventory"]["event_or_kernel_sum_samples"],
        "descriptive_matches": report["comparison"]["descriptive"]["matched_points"],
        "heldout_matches": report["comparison"]["validation"]["coverage"]["paired_heldout_points"],
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
