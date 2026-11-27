"""Report independent timing accuracy and declared scaling/configuration checks.

[Role] Score explicit prediction and observation campaigns.
[Invariants] Recompute comparisons from timings and exclude training samples.
[Timeouts & Locks] Bounded metadata reads and CPU-only comparisons.
[Side-Effect Policy] Reads sources and writes only the requested report.
"""
import argparse

from mindie_sim.__main__ import emit, read_json
from mindie_sim.observations.comparison import accuracy_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign")
    parser.add_argument("--absolute-tolerance-ms", type=float)
    parser.add_argument("--relative-tolerance", type=float)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    override = {name: value for name, value in (
        ("absolute_ms", args.absolute_tolerance_ms), ("relative", args.relative_tolerance)
    ) if value is not None}
    campaign = read_json(args.campaign)
    campaign["tolerances"] = {**campaign.get("tolerances", {}), **override}
    result = accuracy_report(campaign)
    emit(result, args.output)
    print(result["coverage"])


if __name__ == "__main__":
    main()
