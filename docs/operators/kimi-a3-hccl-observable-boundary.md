# Kimi A3 TP communication: source caller and observed boundary

The four-layer TP8/TP16 Kimi captures contain nine AllGather and eight
ReduceScatter calls per complete Body. The source-compatible SP caller has
four attention AG/RS pairs, one dense FFN pair, three TP-sharded shared-expert
FFN pairs, and a final AG. Its BF16 hidden width is 7168. With native graph
storage capacity `T`, one rank owns `ceil(T/TP)` rows. AG input bytes are
`ceil(T/TP) * 7168 * 2`; RS input bytes are `TP` times that amount. These
payloads are source-conditional: the HCOM capture arguments themselves are N/A.

The exact source blobs, canonical cases, capture SHA256s, extraction identities,
and train/validation assignments are in
`data/observations/a3-kimi-hccl-joint-20261030/inputs.json`. The Ascend source
revision is `8c30e4f491f73cba737a1dacc522894e437cdf3d`, and the SP helper is from
vLLM `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`. Captured binary identity is
not established. Single-node rank counts 8 and 16 are separate physical
domains; unseen ranks or topologies are not interpolated.

## Training boundary and counterexamples

The retrospective training set uses 4K TP8 B2/B24/B48 and TP16 B2/B12/B48.
Originally TP16 B24 was selected. Its extraction included an overlong RS
envelope that crossed the next Body and pulled an LMHead AG into 78 of 177
Body windows. CSV line 28155 ends after both its following AttnRes and the
next forward start. B12 replaced this source before its HCOM durations were
read. The excluded source remains validation evidence. The original sources
were unassigned; this is not a prospective blind split.

The retained observation is one native HCOM `COMMUNICATION` envelope.
Overlapping `allgatherAicpuKernel` / `reduce_scatterAicpuKernel` controllers are
excluded. No Body duration, Body error, context value, batch identity, or call
ordinal is a response coordinate. Adjacent captured tensor shapes and every
call-ordinal distribution remain in `train-native.json` for inspection.

The resulting payload response is not a bandwidth curve. TP8 AG mean duration
falls from 70.633 to 54.458 to 51.682 microseconds as declared input bytes rise
from 14336 to 43008 to 86016. TP16 B2 and B12 have the same 14336-byte input,
but their means differ: 166.510 and 147.115 microseconds. Within one source,
early AG roles can cost several times later roles with the same payload.
Peer readiness and control effects are therefore still mixed into this
observed boundary. A negative segment slope is not a negative physical
bandwidth estimate.

`rules.json` retains bounded continuous interpolation over distinct payload
means, weighting each source equally at a repeated payload. TP16 has only two
distinct payloads. All remaining sources are validation; an input beyond the
training payload interval stays unsupported. No target timing is used to
extend the interval or refit a failed point.

## Untuned source control

`data/verification/a3-kimi128k-mechanism-audit-20261030.json` retains two complete
128K B1 controls with an explicit source-compatible `AICPU_TS/mesh/separate`
collective. TP8 Body changed from 2.746722 to 2.493107 ms against observed
3.396537 ms; TP16 changed from 2.958313 to 2.366992 ms against 4.092640 ms.
These are larger underestimates, not improvements. Copy, reduction and control
work remain unpriced in that untuned source program. No HCOM envelope was
converted into an independent per-pipe rate.

The existing Sim native-group response boundary can constrain a single
collective invocation once while retaining its local phases and physical
transport. It must not be installed once per internal phase or as a Body
floor. The local-only `joint_native` replacement guard remains intact.
