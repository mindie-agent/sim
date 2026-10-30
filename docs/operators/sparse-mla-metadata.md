# Sparse MLA metadata is an AI_CPU dependency

`vllm_ascend_sparse_mla_metadata` binds the NoPE, BF16, dense paged sparse MLA
caller. It models the metadata kernel and the caller's separate copy into a
persistent 1024-word INT32 buffer. Attention waits for that publication. It
does not turn the AI_CPU call into Vector work or host submission.

Select it under `runtime.kernels.lowerings.sparse_mla_metadata` with an
explicit `options.layer_groups` list. For example, `[[0, 4, 8]]` shares one
metadata invocation among those layers, while `[[0], [4], [8]]` declares three
distinct invocations. Members must have the same attention geometry and stage.
The first member names the result and later members consume it. All selected
metadata builds and publications run before layer execution, dependent on Body
input readiness rather than a layer's RMSNorm. Layers not
listed receive no new metadata call.

`options.schedule="layer_ordered"` instead retains each group at its first
layer, after index selection and before sparse attention. This is an explicit
capture caller contract: the independent profiler's
`acceptance-20261030-service-glm/metadata-placement-evidence.json` retains six
A5 examples inside MLA layers 3/7, after scoring and directly before attention,
not together before the forward. The source-builder placement remains the
default `pre_forward`; these two execution boundaries are not conflated.

This distinction matters: retained `model_runner_v1.py:5590–5650` groups layers
by backend, cache specification and rotary configuration. Its metadata builder
keeps state per draft index (`sfa_v1.py:405,594–597`), so the source does not prove
one metadata kernel per layer. The observed A5 NoPE captures contain 11 calls
per Body, but applying that count requires an explicit capture caller grouping.

The source contract is pinned at revision
`8c30e4f491f73cba737a1dacc522894e437cdf3d`. `sfa_v1.py:72–104,175–228`
declares INT32 query starts, sequence lengths and per-query top-k lengths;
positions include graph padding. The planner therefore retains both physical
metadata rows and valid query rows. The persistent buffer copy remains separate
from the native kernel response.

The physical lower bound reads the declared input arrays and writes the output.
An explicitly declared `aicpu_metadata_call` resource adds one controller call,
owned by controller 0. Alternatively an observed `service_regimes` response
may match `kernel: SparseFlashMlaMetadata` and its geometry. That response is
inclusive at this native boundary, not an addition to a Body residual. Absent
either declaration the unknown controller work is visible in
`unpriced_service_work`; no undocumented AICPU instruction rate is invented.

Tests cover one and eight requests, delayed input, shared versus distinct
metadata groups across three layers, graph padding, one response per native
call and 0.5/1/2 controller supply with unchanged payload bytes.
