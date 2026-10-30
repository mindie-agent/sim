# Verification scope

Current evidence is limited to the release acceptance receipt, the selected
Decode Body comparisons and three independently executable physical controls.
Development-round reports are not retained in this directory. Maintained tests
read their independent inputs from `tests/fixtures/verification/`; their
`sources.json` records the original revision and SHA-256 for each input.

## Decode Body comparisons

The [current scoreboard](../data/verification/model-scoreboard-20261029-final.md)
and [compact receipt](../data/verification/decode-body-current-20261030.json)
retain **65/66** selected completed Body comparisons within 15%, with 66/66
latest attempts completed. Qwen3.5-397B c48 remains outside the threshold.

These are selected comparisons from multiple recorded runtime revisions, not a
uniform replay at the release tag. Each row retains source identity, geometry,
topology and rank assumptions, observed mean/P50, prediction, and calibration
relationship. Some operator services use the same observed population as the
comparison; an in-sample pass does not establish heldout or unseen-shape
accuracy. Source-compatible programs are also conditional on their declared
caller and implementation identity.

Body completion, isolated operator duration, accumulated family activity and
Serving/client TTFT or TPOT have distinct boundaries. They must not be added or
substituted for each other. Prefill and full host/client serving accuracy are
separate acceptance work; passing Decode Body rows does not establish them.

## Executable physical controls

| Control | Retained evidence | Reproduction |
| --- | --- | --- |
| A3 TP8/TP16 SIO and HCCS hierarchy | [Results](../data/verification/a3-physical-hierarchy-20261030.json) | [Producer](../data/verification/a3-physical-hierarchy-20261030.producer.py) |
| PD connector capacity, framing, switch and endpoint memory | [Results](../data/verification/pd-physical-connector-20261030.json) | [Producer](../data/verification/pd-physical-connector-20261030.producer.py) |
| Explicit two-Die CCU ownership and finite buffer service | [Results](../data/verification/hccl-ccu-two-die-20261030.json) | [Producer](../data/verification/hccl-ccu-two-die-20261030.producer.py), [sources](../data/verification/hccl-ccu-two-die-20261030.sources.json) |

These controls exercise the common Engine with declared resources. They verify
work conservation, input readiness, capacity response and cleanup; they are not
new device measurements or end-to-end accuracy claims. The producers print
JSON and run without model weights, device contexts or remote jobs.

## Package and maintained-suite acceptance

The release gate runs the maintained suite with the companion profiler import
path supplied, builds the wheel, verifies the installed package outside the
checkout, and records the exact source revision and package identity. Test
results from a different revision remain distinct from release acceptance.

The [0.8.0 local release receipt](../data/verification/release-0.8.0.json)
records **3301 maintained tests**, **144 isolated installed-wheel checks** and
all three release CPU benchmarks passing on runtime `439779a7`.
The [wheel receipt](../data/verification/installed-wheel-0.8.0.json) preserves
its package hash and individual checks. All 293 package Python/JSON files were
byte-equal to that runtime. Final release-tag CI and artifact hashes accompany
the formal GitHub Release.

Complete V4.1 DP8 Body controls at 2, 8 and 16 sequences per replica peak at
354.5, 357.5 and 359.7 MiB. Sharing immutable hardware contexts and MC2 source
plans preserves the measured-before Body durations; the 8-sequence control
also preserves its work, resource-service and timeline hashes. All runs release
tasks and programs. The largest geometry has no independent before-run; its
49–88 second wall times include concurrent local acceptance work. These are
memory/correctness controls, not new device-accuracy evidence.
