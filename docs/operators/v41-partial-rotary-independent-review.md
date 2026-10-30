# V4.1 partial rotary caller independent review

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

Reviewed runtime: `1f46176f`. This review found no blocking caller-graph issue.
The retained JSON
contains the exact runtime file hashes, cases, invocation counts and durations.
The producer
keeps the original controls; its portability changes do not change the stored results.

Six undeclared-selector controls cover V4.1, GLM78 W4A8 and Kimi BF16 at batch
one and two. They execute the old `mla` callable loaded directly from Git object
`1f46176f^`, then the reviewed callable through the ordinary attention builder.
Operator graphs and effective kernel declarations are equal in all six controls.
No old implementation is copied into production code.

Four actual three-layer graphs cover symmetric DP batch one/two and a single
active first/last DP replica. They verify Q rotation after Q projection, KV
rotation after KV normalization and before cache quantization/storage, both
inputs before attention, inverse rotation after every attention output, and
output activation quantization after inverse rotation. Only the rotary slices
are transferred; the old combined rotation is absent. Invocation counts are
72, 72, 9 and 9. All runs release live tasks/programs and buffers (floating point
allocation residue below 1e-7 bytes). Four existing focused tests also pass.

Reproduce from a checkout containing the reviewed commit and inputs:

```sh
python data/verification/v41-partial-rotary-independent-review-20261030.producer.py \
  --root /path/to/reviewed/sim --output /tmp/v41-partial-rotary-review.json
```

The producer records the executed HEAD and runtime hashes. Running it on later
code is a new check, not evidence that the original results used that later code.
The target Git parent must remain available because it supplies the old callable.

This verifies caller dependencies and analytic traffic, not captured binary
identity, native core dispatch or measured performance. The declared coefficient
sharing and aggregate resource timing remain analytic assumptions. The reviewer
checked retained slice-kernel semantics and the caller assembly; the private
caller image was not independently reacquired. See the source-domain discussion
in [shared KV partial rotary](shared-kv-partial-rotary.md).
