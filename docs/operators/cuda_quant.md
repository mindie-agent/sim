# CUDA per-token INT8 quantization

`cuda_quant.py` selects `dynamic_scaled_int8_quant_kernel`, symmetric and without
AZP, from the declared vLLM checkout. The input is contiguous BF16, FP16 or FP32;
the output is INT8 plus one FP32 scale per row. The block has `min(width,256)`
threads. CUDA and pinned CCCL source identities are retained in the operator
metadata. They do not establish the toolchain of a historical measurement.

## Source work and branches

The two input passes both include low-precision-to-FP32 conversion. The first
forms thread maxima and calls CUB's float maximum; the second recomputes the
inverse from absmax and uses `cvt.rni.sat.s8.f32`. Reading the rounded scale back
and taking its reciprocal is numerically different and is not the source path.

Every live warp invokes five shuffles. A maximum runs only when
`lane + offset <= last_lane`. For a full warp this is 5 warp calls, 160 lane
participants and 129 maxima. These are separate quantities. Cross-warp
aggregation runs in thread0 with sequential shared reads and maxima.

`input_byte_offset` and `output_byte_offset` preserve the pointer alignment of
contiguous views. BF16 width32 with input offset2 and output offset0 takes the
all-scalar output path. Address summaries enumerate the repeating row-alignment
period without a per-row history. They do not invent a vectorization speedup:
memory sectors and compiled load/store issue are not yet modeled.

`zero_absmax` declares the uniform data branch for the analyzed rows. The zero
branch omits reciprocal division but retains the scale computation and output
work. The default declares nonzero rows. Mixed zero/nonzero row schedules are
not inferred from shape.

## Service and dependency boundary

Source lane work has typed roles such as `fp32_max`, `fp32_div` and
`cvt_bf16_fp32`. Shuffle demand uses `warp_shuffle_calls`; shared accesses retain
bytes. Rates must be declared explicitly, either as resources or as service
modes sharing a physical pipe. There is no automatic equal-cost mapping to a
generic vector operation, no measured-evidence admission check, and no guessed
instruction latency. Synthetic campaign rates are only structural controls.

Producer reads, conversions, arithmetic, reduction and stores are separate
phases. Scale service starts after reduction and division; output service starts
after multiplication and conversion. Source barriers preserve readiness without
an invented barrier duration. One resident row per compute owner is a declared
schedule; different owners may run concurrently and contend for global memory.
The row phases are service envelopes, not a reconstruction of warp scheduling.

When a shared-memory pool is declared, each active owner retains 48 bytes for
the whole CTA and releases it before reusing the pipeline. This is the pinned
source layout estimate, not a compiler resource report. Without a pool the
capacity is explicitly unconstrained. Shared lvalue traffic assumes no compiler
register reuse; instruction hoisting and eliminated accesses require compiled
evidence before refining those counts.

## Verification

The original returned candidate passed its own tests but failed independent
predicated-max, scale-readiness and storage-offset controls. Its failure report
is retained under `data/verification/cuda-quant-review/`. The revised tests
execute all256 CUB block sizes, enumerate real alignment paths, perturb one typed
rate, inspect producer/store traces, and check owner isolation, shared memory,
zero-row work, delayed input and bounded row issue.

This module is an explicitly selectable isolated extension. It is not yet a
model frontend default, a GPU latency validation or a claim that the complete
compiled instruction stream has been modeled.
