# Explicit collective rendezvous

Run from the repository root:

```bash
python -m mindie_sim collective-plan --input examples/collectives/shared-link.json --output collective-result.json
```

The synthetic fixture uses 100 physical wire bytes/ms. Group `a` waits until
5 ms for a zero-payload member; group `b` starts at 1 ms. Wire competition gives
group completion times **12 ms** and **9 ms**, a **13 ms** total with local
consumer work, and exactly **1100 serviced wire bytes**.

These are normalized dependency-engine checks, not NPU measurements or an
AllToAll implementation's traffic formula. The original JSON includes input
provenance, member readiness, gate IDs and actual resource service intervals.
See [the input/API contract](../../docs/collective-plans.md).
