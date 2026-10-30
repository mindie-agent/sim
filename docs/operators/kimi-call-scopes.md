# Kimi call scopes

Standalone analysis only: `tools/kimi_call_scopes.py` builds a declared
GroupedMatmulSituQuant semantic reference and a source-conditioned
ChunkKdaFwd stage ledger. It does not edit the runtime, schema, scoreboard,
catalog, existing tools or profiler data.

## Situ semantic reference

```python
from kimi_call_scopes import situ_reference

packet = situ_reference(case, descriptors, quant)
packet["ops"]       # tuple[OpSpec]: gate_up, SiTU, down.quantize
packet["recipes"]   # RecipeBook: explicit matrix_envelope
packet["ledger"]    # JSON-compatible geometry/work/traffic/inventory
```

The DAG is:

```text
situ.input (FP8/E8M0 descriptor)
  -> situ.gate_up (aggregate grouped GEMM, BF16 output)
  -> situ.activation (SiTU, BF16)
  -> situ.down.quantize (FP8 payload + E8M0 scales)
```

The input is already FP8. `activation_quantization` is run only on the SiTU
consumer closure, not on the gate-up input, so no input quantizer is inserted.
A dummy down consumer requests the output quantizer and is then discarded. The
generated `situ.down.quantize` op remains; the down projection itself is never
returned.

The matrix recipe is an explicit aggregate `ResourceEnvelope`, not a native
grouped schedule. Descriptor `group_list` values are absent. All declared weight
banks are charged, and no equal-row split or tiled expert distribution is
invented. The source plan's `ffn_activation` must be `situ`; beta and linear beta
come from that case plan.

The effective routed-expert weight format and activation scale representation
must also match the descriptors. Category exclusions cannot silently replace FP4
with BF16, and FP32 or differently packed activation scales cannot be reported as
the descriptor's E8M0 layout. The helper checks the generated operand metadata
and quantized output bytes before returning the ledger.

The ledger marks this as a declared semantic reference, not a fused native
kernel, ABI identity or measured traffic reconstruction. BF16 gate and SiTU
outputs stay materialized under `preserve_workspaces`.

### Retained fixture examples

| M | FLOPs | FP4 weight payload | weight E8M0 scales | FP8 input | input scales | FP8 output | output scales | SiTU scalar ops | output-quant ops | BF16 intermediate R+W |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2875 | 126,615,552,000 | 308,281,344 B | 19,267,584 B | 10,304,000 B | 322,000 B | 8,832,000 B | 276,000 B | 105,984,000 | 35,880,000 | 105,984,000 B |
| 3286 | 144,716,070,912 | 308,281,344 B | 19,267,584 B | 11,777,024 B | 368,032 B | 10,094,592 B | 315,456 B | 121,135,104 | 41,009,280 | 121,135,104 B |

The retained descriptors have K=3584, gate-up N=6144, 28 FP4 weight banks, an
output width of 3072 and E8M0 group size 32. The intermediate R+W count is
`2*gate_bf16 + 2*situ_bf16`: gate-up write plus SiTU read, then SiTU write plus
output-quantizer read. Storage is reported separately from traffic.

## KDA source stage ledger

```python
from kimi_call_scopes import kda_stage_ledger

ledger = kda_stage_ledger(descriptors, chunk_size=64)
```

This API returns stage membership, recognized product work, source formula
evidence, unresolved work, source output/workspace footprint and arithmetic
alignment. It does not return a duration, a stage timing split or a
reconstructed historical call count.

Pinned source revision: `ae923905948f360f4b7ab06df52d0789a0c9316e`. The wrapper
selection is:

```text
useDenseA5FastPath =
  cu_seqlens is absent && q is BF16 && chunk == 64 &&
  dk == 128 && dv == 128 && tokens % chunk == 0
splitStages = IsAscend950 && totalChunks > 1 && !useDenseA5FastPath
```

Descriptors do not carry SoC identity; the ledger states the Ascend950 source
assumption explicitly. With the retained `cu_seqlens[2]` descriptor present,
the dense path is disabled. For T=8192, chunk=64 there are 128 chunks, so the
source-conditioned ledger has four sequential launches:
`GATE_PREPARE`, `POST_WU`, `FWD_H`, `FINALIZE`. Each launch exposes the same 13
tensor output descriptors. The stage scalar is absent from captured events.
The pinned host implementation accepts only chunk sizes 64 and 128
(`aclnn_chunk_kda_fwd.cpp:489-490`); the ledger uses the same domain. A single
chunk example must use matching token and output geometry, such as T=64/C=64.

For T=8192, H=12, dk=dv=128, C=64 (`pairs=524288`):

| Stage | Recognized source product work | FLOPs |
| --- | --- | ---: |
| GATE_PREPARE | QK `2*pairs*H*dk` + KK `2*pairs*H*dk` | 3,221,225,472 |
| POST_WU | w/u/kg formula not established in this ledger | unresolved |
| FWD_H | recurrence formula/traffic not established in this ledger | unresolved |
| FINALIZE | `qgScaled@h` `2*T*H*dk*dv` + `aqk@vNew` `2*pairs*H*dv` | 4,831,838,208 |

Before `10c2e531`, the semantic member program in `recurrent.py` placed only one
QK/KK-class product (1,610,612,736 FLOPs) in `chunk_products` and an extra score
product in `recurrent_output`. That commit moves the product to its causal
prepare stage without changing combined work:

```text
current chunk_products     3,221,225,472
source prepare             3,221,225,472
current recurrent_output   4,831,838,208
source finalize            4,831,838,208
current phase shift                    0
historical phase shift     1,610,612,736
combined FLOPs             equal
```

Equality of arithmetic does not make generic member execution a source phase
timing. The ledger preserves the historical correction, plus the unresolved
POST_WU/FWD_H source work and generic `wy_transform` work. It never
relabels existing semantic member timings as source stage timings.

`tools/model_scoreboard_kimi.py` also executes four disjoint semantic projections
on the shared graph engine. Each carries actual FLOPs, vector work, read/write
traffic, resource service, predicted duration and remaining live state. Their
operator union must equal the complete semantic core exactly once. The current
core is a serial DAG, so the four projection times sum to its complete duration;
that property is checked independently. This explains the mechanism's phase
costs without manufacturing observed phase durations or complete-call counts.

`tools/kimi_stage_identity.py` streams the original 110 MB capture through the
shared profiler decoder. The retained
`data/verification/model-scoreboard-20261029-kimi-kda-stage-identity.json`
finds 1104 events and 881.997838 ms of native activity. All share the same
name, tensor signature and stream 61. The 48 CSV columns contain no stage,
attributes or tiling key, and task IDs lack a source-stage mapping. Thus
direct per-stage observed durations and counts remain unknown. Neither event
count nor duration clustering supplies the missing identity.

A separate source-compatible block comparison is now available in
`data/verification/model-scoreboard-20261029-kimi-kda-call-alignment.json`.
`tools/kimi_kda_blocks.py` independently identifies 276 maximal four-event runs
with matching external convolution/normalization/transpose/output boundaries.
`tools/kimi_kda_alignment.py` compares those blocks against retained core and
stage predictions under the pinned source's stage-order assumption. It keeps
the original 1104-launch statistics separate. Complete core activity is
881.997838 ms observed versus 60.706774 ms predicted, with the largest gaps in
prepare and state update. This additional conditional domain does not claim
that stage attributes or capture-installed binary identity were recovered.
The exact cu-seqlens value is also absent: one sequence with 128 chunks permits
8129-8192 valid tokens; the scenario explicitly uses 8192.

Storage is retained separately:

- 13 output descriptors: 327,942,144 B unique footprint.
- Known internal FP32 score workspace: 50,331,648 B.
- Total source-stage read/write traffic: unresolved; descriptor storage is not
  traffic and must not be multiplied by four launches.
- No aggregate measured duration, no `1104/4` historical-call reconstruction,
  no per-event stage allocation.

## Evidence

The ledger embeds pinned paths, SHA-256 hashes and line ranges from the local
audit. Key paths:

- `/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp`
- `/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_prepare.h`
- `/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_post_wu.h`
- `/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_finalize.h`
- `/Users/maoxx241/code/sim-rewrite-20261027/mindie_sim/prediction/recurrent.py`
- `/Users/maoxx241/code/profiler/tests/fixtures/fusion_contracts/a5_kimi_k3.csv`

The fixture rows are provenance examples. The ledger identifies the pinned
source condition, not an installed capture binary.

## Limitations

- This helper is not integrated into `tools/model_scoreboard_kimi.py`; the root
  owner performs that integration.
- The Situ matrix is an aggregate envelope for unknown group-list rows.
- The KDA ledger is not native timing, not a historical launch reconstruction
  and not a complete traffic model.
- POST_WU and FWD_H formulas and traffic remain unresolved.
- Source-conditioned stage membership can be compared conditionally with
  evidence, but not presented as binary identity or measured accuracy.
