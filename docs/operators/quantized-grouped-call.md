# INT8 FRACTAL_NZ grouped-matmul complete call

This report covers the retained standalone `torch_npu.npu_grouped_matmul`
callback for the INT8 dynamic W8A8, FRACTAL_NZ-weight branch. It is an
analytical tensor/dataflow and shared-matrix-program binding, not an
identification of the historical loaded CANN kernel.

The implementation owns:

- `mindie_sim/prediction/call_grouped.py` -- public descriptor/builder and the
  narrow `mapped_matmul` binding;
- `mindie_sim/observations/grouped_binding.py` -- audited producer/capture
  binder;
- the `quantized_grouped` entry in `prediction.call_builders.CALL_BUILDERS`;
- the standalone `grouped_calls` binder in
  `observations.call_alignment._grouped`;
- `tests/test_quantized_grouped_call.py` and
  `examples/calls/quantized-grouped-small.json`.

No Engine, matrix program, BF16/ND source planner, routing realization or
timing fit is copied.

## Retained producer and boundary

Producer:

```text
/Users/maoxx241/code/mindie-agent/.workstreams/hp-serving/qwen35-grouped/scripts/probe_grouped_graph.py
sha256 e04e38ccf8a6f27be886ad7db9856fa97491c1cf9d41bd11cf3efa170edb8e5a
```

Protocol and retained fixtures:

```text
data/observations/protocols/grouped-call-protocol.json
tests/fixtures/observations/capture_adapters/grouped_producer.py.txt
tests/fixtures/observations/capture_adapters/grouped_probe.json
```

The timed callback is:

```python
torch_npu.npu_grouped_matmul(
    x=[qx], weight=[wq_nz], scale=[ws], bias=None, per_token_scale=[ps],
    split_item=2, group_type=0, group_list=counts, group_list_type=1,
    output_dtype=torch.bfloat16,
)[0]
```

Activation quantization, weight conversion to FRACTAL_NZ, allocation and
initialization of the input tensors and counts are outside the timed region.
Output allocation and API host dispatch are inside the Python closure, but
their native cost is unresolved. The binder does not infer either cost from
event residuals.

The eager one-call event, graph replay of one whole call, and graph replay of
eight whole serial calls are separate identities. The graph-eight condition
retains `invocations=8`, one replay boundary and the original 40 whole-replay
samples. The binder never divides those samples by eight. The producer's
per-call display field is intentionally not imported as a new sample identity.

## API and geometry contract

The binder verifies the complete audited producer SHA and every retained API
argument, including the tensor-list strings, `scale`/`per_token_scale` dtypes,
`split_item=2`, `group_type=0`, `group_list_type=1`, direct-count group-list
semantics and BF16 output. It also verifies ND input/output, FRACTAL_NZ weight,
phase and the eager/graph-one/graph-eight execution declaration. A mismatch is
an error; there is no nearest-branch fallback.

`expert_rows` is an explicit sequence of length `E`. Every nonzero count must
be the same positive value. Zero counts retain their original expert ids and do
not contribute weight payload. Logical `M=sum(expert_rows)`, active groups
`G=count(nonzero)`, and each active expert owns the same row domain. The bind
rejects an all-zero list rather than inventing a matrix launch.

For the current descriptor:

| Quantity | Declared bytes/work |
| --- | ---: |
| INT8 activation payload | `M*K` |
| FP32 per-token scales | `4*M` |
| INT8 active weight payload | `G*K*N` |
| FP32 active per-channel weight scales | `4*G*N` |
| packed active weight | `G*K*N + 4*G*N` |
| BF16 output | `2*M*N` |
| INT64 count tensor scan | `8*E` |
| full weight-bank residency | `E*K*N` |
| full channel-scale-bank residency | `4*E*N` |
| matrix FLOPs | `2*M*N*K` |
| matrix domains | `G`, each with `M/G` rows |

The full bank residency is kept separate from the per-invocation active
payload. Only the active payload and scales are matrix input traffic; the bank
is a geometry/residency declaration, not `E` launched invocations.

The one grouped operator declares `matrix_groups=G`, so the common
`MappedMatmul`/`HierarchicalMatmul` core keeps separate equal-row domains. It
does not collapse the call into one global `[M,K] x [K,N]` dense matrix or
launch one kernel per expert. INT8, INT32 accumulation and BF16 output remain
explicit in metadata, precision work and output bytes.

The count bytes are a separate, once-per-call HBM argument fetch before the
grouped matrix program. This stage shares the same operator/launch boundary and
does not become an activation operand, enter L0, repeat per K panel, or consume
matrix FLOPs. It is an analytical shared-fetch assumption; the native per-core
scalar scan and cache behavior remain unresolved. Counts are included in
`hbm_bytes` but excluded from `mte2_bytes`.

## FRACTAL_NZ evidence and domain

The retained layout domain is checked against the INT8 layout validation in
the local open-source GMM implementation:

```text
/Users/maoxx241/code/vllm-ascend/csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/gmm_dsq_base.h
sha256 08e07b078ca73626df3a135bff230c9b7b61108b235e0f98f22420f4c94f9733
```

`CheckSingleTensorListTypeA8W8` (lines 142-157) maps logical `[E,K,N]`
to `[E,N/32,K/16,16,32]`. The dtype-specific domain is `N % 32 == 0`
and `K % 16 == 0`. The same INT8 block extents occur in the plain grouped
A8W8O16 reference executor:

```text
/Users/maoxx241/code/refs/ops-transformer/gmm/grouped_matmul/tests/st/aclnnGroupedMatmulV5_A8W8O16/executor_aclnnGroupedMatmulV5_A8W8O16.py
sha256 aabac587d789d6cce4a3284f7b6e59867357269027d3138771465fd89e321896
```

These references establish storage geometry and numerical semantics; the
SwiGLU implementation is not asserted to be the historical plain grouped
dispatch. The initially proposed `16x64` NZ extent came from a different
pseudo-quantization example and was rejected during independent review.
The historical probe shapes satisfy the INT8 whole-tile conditions:
gate/up `K=2048,N=256` and down `K=128,N=2048`.

This adapter therefore accepts only those historical whole-tile K/N multiples,
where logical and storage byte counts agree exactly. It records the storage
shape and declares internal tiled-load order, cache/alignment behavior and
padded traffic unresolved. It does not invent padded traffic or select the
BF16 ND grouped planner. Non-multiple K/N branches are rejected.

All ten producer shape plans for the two projections are bound by the focused
test as one operator each, without running a whole-model simulation. The
retained fixture itself contains the two c1 served rows (gate/up and down).

## Common matrix-core integration

The descriptor uses `RecipeBook` with a single explicit recipe name,
`quantized_grouped_int8_nz`, whose implementation subclasses the existing
`MappedMatmul`. `MappedMatmul` selects a `HierarchicalMatmul` and executes it on
the ordinary Engine. No second engine or local matrix pipeline exists.

The shared matrix core executes a final-K epilogue. The call's numerical
contract is:

```text
int32_scale_per_token_per_channel_to_bf16/1
```

The descriptor supplies `OpSpec.vector_ops=4*M*N`, corresponding to INT32
to FP32 conversion, two FP32 multiplies and one BF16 conversion per output
element. Its `matrix_epilogue` metadata declares:

```json
{
  "kind": "row_column_scale",
  "operations_per_output": 4,
  "row_scale_bytes": "4*M",
  "column_scale_bytes": "4*G*N"
}
```

The scale byte entries are numeric values in actual descriptors. The common
core removes these scales from K-panel matrix inputs and loads them before
the final epilogue. Vector service occurs after complete accumulation and
before the existing BF16 output store. It does not repeat per K panel, invent
a global INT32 intermediate, or add another operator launch.

The operation count is semantic vector work. It is not an assertion that an
Ascend `AscendDequant` intrinsic becomes one instruction per arithmetic
operation. The plain A8W8O16 CPU reference converts the INT32 accumulation to
FP32 before applying the channel and token scales, then converts the output.

The original DSH commit refused execution while this core mechanism was absent.
The acceptance revision binds the shared mechanism directly; no source-text
inspection or optional compatibility path guesses whether metadata was
consumed. Eager one-call, graph one-call and graph eight-call protocols all
compose through `call_costs`.

## Builder and binder

Public builder:

```python
build_quantized_grouped_call({
    "expert_rows": [1, 0, 1, 0],
    "k": 16,
    "n": 64,
    "phase": "primitive",
})
```

The public CLI example separates the hardware case from the call declaration:

```sh
python -m mindie_sim call \
  examples/calls/quantized-grouped-small.json \
  examples/calls/quantized-grouped-small-call.json
```

That declaration repeats eight complete calls in a single graph replay.

It returns a `CallDefinition`-compatible result through
`CALL_BUILDERS["quantized_grouped"]`: `(ops, recipes, ledger, limitations)`.
Only the narrow API values are accepted; unknown arguments are errors.

`bind_grouped_condition` reads only geometry, dtype, layout, API contract,
execution protocol and provenance from the retained condition. It never reads
sample durations, canary values, sample splits or observation statistics. It
deep-copies the original target identity into the ledger and preserves the
condition id, canary, samples and `unassigned` role unchanged in
`compare_calls`.

## Validation

Focused command (provided interpreter, repository cwd):

```sh
/Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python -m pytest -q \
  tests/test_quantized_grouped_call.py tests/test_call_alignment.py \
  tests/test_capture_adapters.py tests/test_matrix_epilogue.py
```

The tests establish exact independent byte/FLOP arithmetic, empty-expert id
preservation, separate full-bank and active payloads, one operator boundary,
recipe reuse, all historical shapes, NZ whole-tile rejection, audited binder
checking, duration/canary/split invariance, original eager/graph-one/graph-eight
sample spans and eight whole invocations. Complete-call execution is required
for acceptance; a pending-epilogue refusal does not count as a passing
execution test.

Independent acceptance on 2026-10-29 passed **45 tests** with common epilogue
revision `604c4491`. A separate explicit-tile Engine test checks three
`[2,32] @ [32,64]` expert domains across two N tiles and two K panels:
all six output epilogues follow their complete matrix reductions; counts
never enter matrix operands; adding two empty expert slots adds exactly
16 bytes of count traffic and changes no other resource work. The delayed
input control and final task/allocation checks pass.

Both actual public CLI commands completed with exit 0 and empty stdout/stderr
when writing output files. The small eight-call control reports eight
operators, 63,488 HBM bytes, 12,288 vector operations, and zero live tasks
or programs. Its duration is a declared-resource prediction, not a measured
kernel result.

The retained full protocol also completed:

```sh
python -m mindie_sim compare-calls \
  examples/calls/quantized-grouped-small.json \
  data/observations/protocols/grouped-call-protocol.json --include-submission
```

It preserves **30 conditions, 1,200 samples, all `unassigned`**, with 60
separately declared device/submitted predictions and no failed condition
scenario. The final comparison ran from a clean export of `475a0b71`;
`data/verification/quantized-grouped-calls-20261029.json` retains the source
hashes, complete condition identities, canaries, statistics, resource work
and conditional comparisons. Raw timings remain at the original capture
path. `validation.strict_pairs` remains zero.
Original canaries sample at most the first three active experts; they are
preserved as producer checks, not widened into whole-tensor validation.

The graph-one-call comparisons expose substantial remaining differences:

| Projection | M | Active experts | Device prediction ms | Event median ms |
| --- | ---: | ---: | ---: | ---: |
| gate/up | 32 | 8 | 0.002111153 | 0.040576499 |
| gate/up | 128 | 8 | 0.002357260 | 0.040498501 |
| gate/up | 32 | 32 | 0.004266247 | 0.043451499 |
| gate/up | 128 | 32 | 0.004339601 | 0.043814000 |
| gate/up | 256 | 256 | 0.033827303 | 0.092419002 |
| down | 32 | 8 | 0.000629553 | 0.039828500 |
| down | 128 | 8 | 0.000779456 | 0.039610501 |
| down | 32 | 32 | 0.002239751 | 0.045317000 |
| down | 128 | 32 | 0.002353041 | 0.045569500 |
| down | 256 | 256 | 0.017615335 | 0.103879999 |

These conditional differences range from about -98.4% to -63.4%. The
submitted scenario adds the existing nominal 0.020 ms launch service; this
parameter was not fitted to the capture and does not establish event-boundary
equivalence. The results establish dataflow integration, not timing
convergence, heldout accuracy, or complete-model accuracy.

The example selects the existing `Ascend950DT_9582` profile. The shared
matrix core now binds matrix-operand reads to its declared `mte2_aic` role
and output-scale reads to `mte2_aiv`, with the corresponding owner counts.
Every final mapping reports these bindings and
`unmodeled_transfer_roles=[]`; no new rate was inserted to close the
residual. The earlier pre-binding results are superseded by the table above.
The profile's nominal memory-path supply rates are not measured instruction
rates. Identifying the historical dispatch/control schedule remains separate
work.

## Remaining limits

- No historical loaded kernel binary, exact internal dispatch or native kernel
  count is identified.
- Output allocation, API host dispatch, graph replay launch gaps and runtime
  scheduling remain unresolved and are not inferred from timings.
- Internal FRACTAL_NZ tiled access, L1/L0 ordering and padding beyond the
  declared whole-tile byte identity are not modeled.
- Count-scan scalar execution and epilogue instruction scheduling are not
  reconstructed. Only their declared bytes and vector work are exposed.
- Native L0C-to-vector transfer ports and storage traffic are unresolved.
  The logical epilogue does not invent a chip-specific port rate or a global
  accumulation spill.
- No hardware run, heldout accuracy, calibration or end-to-end serving claim is
  made.
