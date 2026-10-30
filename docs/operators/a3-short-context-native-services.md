# A3 short-context native operator responses

`data/observations/parameters/a3-native-decode-20261030/produce.py` streams the
retained 2026-10-23 A3 captures. Only 4K native operator durations fit the
nonnegative fixed-plus-row responses in `parameters.json`; 16K operators and
the ten 64K/128K Body comparisons remain separate. The historical experiment
roles were unassigned. These explicitly selected analysis roles are not a
claim of pristine heldout experiments.

The physical conditions are hardware 9362, native tensor geometry and precision,
caller/recipe, recurrence checkpoints, convolution history, and row count.
No rule selects a model name, context length, Body duration, or desired error.
The row domain is the observed training range; larger inputs retain analytical
simulation. These are inclusive kernel completion bounds. They retain the
analytical work and do not supply a measured HBM or vector bandwidth under
hardware perturbations.

Records preserve complete input/output ABI, core kind and block count separately.
An explicit Conv output different from its flat input marks a dynamic caller
outside this calibration. Gating and norm descriptors, when present, must agree
with that caller's rows; absent descriptors retain the source association as a
condition. Routing uses the INT8 packed output path, excluding the BF16 path.
The native Cumsum has missing tensor fields in decode. Its immediate next task
on the same stream must be the fused INT8 gate/up consumer with 256 descriptors;
other streams may interleave. This pairing supplies a declared ABI, not captured
values of the count list. The source conversion is pinned in
`prediction/moe_group_list.py`.

The separate 16K set has 201 in-domain physical groups. Under the existing
15% / 15-us small-call criterion, 199 pass and two fused gate/up groups fail.
Groups with different native ABI are retained separately; this is not a count
of independent hardware runs.

All source files are hashed in 1 MiB chunks. Each physical group retains at most
the first 4096 durations, with its full event count and retention policy. Operator
evidence includes failed comparisons. This calibration is not appropriate for
Kimi KDA, a different expert width, a different recurrence checkpoint count, or
unseen routing distributions merely because a Body has a similar gap.

The ten separate Body controls select the source FAI host mapping, the explicit
count-to-prefix producer and fused INT8 gate/up boundary where applicable, and
the corrected BF16 shared expert operands with one TP sum after their join.
The resulting `body-validation.json` retains **6/10** within 15%. Four lower-row
Qwen3.5 controls remain about 16–19% low. No total-time offset or parameter
refit uses these residuals. The four Qwen3.6 controls are about 7–14% low.
The remaining Cumsum/control and small-helper differences are not hidden as
zero-duration observations.

Reproduce the operator packet from the repository root:

```sh
python data/observations/parameters/a3-native-decode-20261030/produce.py
python data/observations/parameters/a3-native-decode-20261030/validate_body.py \
  --profiler-root /path/to/profiler
```

The returned `service_regimes` are an explicit case declaration; they are not
installed globally or automatically selected by model name. The native binary
identity and actual expert count values were not recovered from these captures.
