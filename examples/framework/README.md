# Synthetic framework lifecycle example

Run from the repository root:

```bash
python -m mindie_sim predict \
  --input examples/framework/case.json \
  --operator-config examples/framework/operator-config.json \
  --framework-config examples/framework/profile.json \
  --output /tmp/sim-framework.json
```

The files separate three explicit inputs:

| File | Purpose |
| --- | --- |
| `case.json` | One-layer Dense/GQA network, hidden size 16; one request with 16 input and 3 output tokens |
| `operator-config.json` | The synthetic Vector/Cube profiles from `../operator-plans/mixed.json`, with explicit legacy fallback allowed |
| `profile.json` | Synthetic host/stream coefficients and sharing scopes; eager device dispatch is 1 us |

The machine and operator rates are arithmetic fixtures, not an Ascend SKU.
The framework coefficients are declared examples, not measured CPU costs or
vLLM defaults. The command uses only Python's standard library and performs no
NPU work.

For this eager case, the three engine commits are 63.0436, 109.4932 and
155.9588 us; ordered client delivery is 68.1436, 114.5932 and 161.0588 us.
The second engine iteration starts before the first client stream completes.
Mean TPOT is 46.4576 us. These are reproducible analytic outputs, not hardware
latency measurements.

Inspect the original `metrics` for engine-scoped results and
`framework.client_metrics` for client-scoped results. `traces[].framework`
names the schedule, input, submit, output and stream tasks;
`traces[].events` contains the actual resource timeline. This example has 11
eager submissions per iteration, including one for each fine-plan operator
rather than one for every internal stage.

Omit `--framework-config` to keep legacy framework accounting. Omit
`--operator-config` to keep the ordinary operator cost path while still
modeling framework tasks. Supported graph replay uses one host submission;
its device dispatch remains owned by the existing graph cost model. The
bundled case explicitly selects eager mode.

See [the lifecycle guide](../../docs/framework-lifecycle.md) for coefficient
counts, CPU/DMA sharing, dispatch ownership, refusal conditions, speculative
output accounting and the distinction between engine and client metrics.
