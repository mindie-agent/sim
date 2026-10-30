# Explicit causal-convolution update

Select `runtime.kernels.lowerings.causal_conv_update` as
`ascend_causal_conv1d_update`. The ordinary KDA decode caller then replaces its
semantic convolution with the existing `gdn_decode_programs.conv_plan` program.
This is an independent selector; neither selecting native KDA recurrence nor
the model name enables it. Prefill keeps its previous semantic program.

The first caller domain is q1, non-speculative, valid state slots, BF16/FP16
activation and matching convolution-state dtype, width 2–4, SiLU, no bias.
The retained KDA producer requires key/value heads divisible by TP; the
prepared weight inventory preserves that source domain instead of treating
a fraction of a head as a valid local weight shard.
The recurrent state remains independently FP32. The source update reads W−1
history rows and writes W−1 new history rows; it does not only emit an output
activation. The existing finite AIV ownership, UB ring, input/output pipeline
and serial task compaction are reused without rate or floor changes.

Retained source anchors:

- `vllm-ascend-v41-private/vllm_ascend/ops/kimi_kda.py` SHA256
  `35365b3b353fc492b4dc5d75b88916431f6802c21bf51b61b4893d39d54f11f3`:
  lines 233–255 register prepared convolution weights; 386–414 call the custom
  operator with activation1/no bias; 544–545 slice active rows; 635–648 select
  non-speculative update mode and retain Q/K/V output views.
- `vllm-ascend-v41-private/csrc/moe/causal_conv1d/op_host/causal_conv1d_def.cpp`
  lines 24–28 declare `x.AutoContiguous()`; `csrc/torch_binding.cpp` lines
  641–674 forward the custom call to `aclnnCausalConv1d`. The Python file
  `ops/causal_conv1d.py` is a reference implementation, not this native route.
- `vllm-v0.27.1-k3-ops-validation/vllm/models/kimi_k3/nvidia/kda.py` SHA256
  `3663c3b273abf41e9727e2b1724e87628b64b6a9dc048a2c48d002b1f9085420`,
  lines 470–499: a packed projection exposes its leading QKV as a view.
- The reusable program pins the retained native kernel at
  `data/observations/sources/native-causal-conv-a4a7958`. Its update branch
  performs weight/history load, token convolution and state writeback.
  This physical program is a declared source-compatible implementation, not
  identification of the capture's loaded binary or exact source revision.

A packed QKV view is materialized only when its strides require it. In
particular, one leading row remains contiguous; multiple rows with a larger
packed-projection stride require a copy of selected QKV bytes. When the model
declares three separate Q/K/V projections, the adapter performs one explicit
row-wise pack to satisfy the one-input kernel. That adaptation is not a claim
that the retained packed source caller contains a `cat` operation. The
projection declarations themselves remain unchanged.

Weight packing is not performed on every invocation. `_pack_conv_weights`
updates the separately registered `ascend_conv1d_weight`; it does not delete
the original FP32 `conv1d.weight`. The selected model therefore adds one
activation-dtype runtime `WeightTerm` per KDA layer, with `source_owned=False`.
Capacity remains in the existing stage weight inventory and is removed when
the selector is removed from a parsed case. The 69-layer KDA portion of the
new Kimi input adds 2543616 resident bytes per TP8 rank.

Scalar metadata issue/cache latency and invalid-slot skipping are not modeled
by this program; all declared requests have valid state slots. The existing
serial-run fluid compaction is retained and is not an instruction-accurate
schedule. No observed convolution duration is a parameter of this binding.

The focused tests cover full 93-layer B1/B4/B32 graphs, packed B1 alias versus
B4 copy, state traffic and readiness through the actual Engine, independent
FP32 recurrent state, prepared weight capacity, selector removal, and unchanged
unselected/prefill behavior. The single source-case variant and bounded report
are produced by `kimi-causal-conv-update-20261030.producer.py`.

The retained controls at runtime `30423d1f` show the actual result, including
the existing device minimum. Source kernels remain near that minimum:

| Batch | Previous envelope us | Native update us | Pack + update us | New resident bytes/rank |
|---|---:|---:|---:|---:|
| 1 | 1.312 | 1.312 | 2.624 | 2543616 |
| 4 | 1.312 | 1.312 | 2.624 | 2543616 |
| 32 | 1.312 | 1.31778 | 2.62978 | 2543616 |

The separate producer adaptation adds its own real copy invocation. These
numbers do not establish agreement with a captured convolution wrapper or
explain its remaining scalar/dispatch latency. No measured duration was used
to raise the floor.

The explicit B32/context4096 variant also completed all 93 layers in public
`predict(scope="body")`: 27.75109 ms, 14.03 s local wall / 13.91 s CPU,
153.7 MiB peak, 1115824 tasks, and zero live tasks/programs/allocations. Its
90-second worker budget and start/end package SHA256
`e38cd3e9602531b1688db678297ef37325c9a3b91c58c0bf17026f4c96cae574` were retained.
Relative to the preceding identical source-host control, body cost increases
by 0.09093 ms, HBM work by 183140352 bytes and vector work by 142442496 units;
matrix FLOPs and communication bytes remain unchanged. The later integral
head and non-speculative-state guards do not change this supported input's
physical program. The report is conditional prediction evidence, not an
accuracy pass.
