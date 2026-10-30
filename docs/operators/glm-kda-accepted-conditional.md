# GLM recurrent KDA accepted-state conditional bodies

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The [two explicit cases](../../examples/cases/high-concurrency-20261030/glm-kda-accepted-conditional/)
add native-compatible KDA source selection to the earlier attention variants.
Original inputs remain unchanged. Matrix mappings, attention variants,
quantization, routing and the declared current query geometry are retained.
The only new physical program is KDA: arch22 for the A3 declaration, arch35
for A5, with kernel-owned Q/K normalization, gate and beta sigmoid. It replaces
the corresponding existing analytic members rather than adding another kernel.

Both cases explicitly assume valid speculative state metadata. Unknown
accepted counts and slot values are not synthesized. This is the
[symbolic state-selection domain](kda-accepted-state.md), not an exact capture
address match. The old structural missing-state label is removed because the
selected mechanism now exists; unknown capture identity stays in runtime
evidence assumptions. Other unmodeled structures remain marked partial.

The local GLM upper caller requests `sigmoid_beta=True` and `compute_gate=True`
for decode. Its imported adapter at the same retained revision actually
computes gate and FP32 sigmoid outside a Triton kernel. This differs from the
native RecurrentKda capture with BF16 gate/beta inputs. Consequently, the new
cases select an explicit compatible native program; they do not claim that
the local mirror's complete call chain is the deployed one. Both file hashes,
their clean Git state and line ranges are preserved in
caller-chain evidence.

Runtime `091dc59c`, package SHA
`5fdd5143223ea50ed2e527fdd5f6ee02c845fce4795df9638763f6a9be41749d`
completed all eight four-layer/full-body controls under a 90-second per-process
budget. Every complete source body contains 34 native KDA invocations across
45 target layers; no live tasks/programs or buffers remain.

| Declared body geometry | Prior analytic KDA body ms | Explicit native-compatible KDA body ms | Source execution wall seconds |
| --- | ---: | ---: | ---: |
| A3 TP16, B4/q6, conditional context4096 | 36.54043 | 36.25136 | 3.162 |
| A5 TP8, B32/q4, conditional context4104 | 28.49220 | 27.57639 | 5.748 |

These are simulator comparisons with preserved inputs, not measured accuracy.
Native KDA invocation activity sums are 0.64593ms and 0.93487ms respectively;
overlapping invocation spans are not additive body critical-path components.
The B/q values come from retained profiler geometry receipts, while context
remains conditional. Neither context nor active rows are inferred from backing
tensor capacity. Scalar GetValue/flag issue latency remains unpriced.

The report index
links each input hash and individual execution result. Reproduce locally with:

```sh
python data/verification/glm-kda-accepted-body-20261030.producer.py \
  -o data/verification/glm-kda-accepted-body-20261030.json
```

The producer uses serial child processes with explicit timeouts. A failed or
timed-out child remains visible in the report and produces a nonzero exit.
