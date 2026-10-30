# Packed INT8 sparse MLA with a BF16 rotary suffix

`indexed_attention_packed_int8` now accepts latent512 with rope0 or rope64.
The rope64 cache row contains 512 INT8 latent bytes, 128 BF16 rotary bytes and
16 FP32 scale bytes: **656 bytes**, padded to 672 only in local UB. It does not
read 656 INT8 elements or dequantize the BF16 rotary suffix. Publishing the
expanded row uses width576; QK includes the rotary dimensions, PV does not.

The retained source is vllm-ascend revision
`8c30e4f491f73cba737a1dacc522894e437cdf3d`,
`kv_quant_sparse_flash_attention_service_vector_mla.h:652–851`, SHA256
`3eed5ab746e056319bf873a488a3306cffbabe8dd19f70294da85685f4e3cfd3`.
`CopyInSingleKv` and its paired path transfer the valid mixed row;
`DequantAndND2NZ` converts the latent tile with four scales and copies the
BF16 rope separately. The rope0 path remains an analytic adaptation.

The whole batch owns one attention call. Body defaults select it only inside
the declared tensor, paired-pipe and scratch domain. The selected prolog and
Body cache declaration use the same mixed row for persistent capacity, not
576 INT8 bytes; only freshly produced rows count as per-call store traffic.
The cache format supports a declared unquantized suffix independently of model
names. Native query outputs retain separate latent and rotary tensor views.

This change establishes representation, work and invocation boundaries. It
keeps the declared key tile/ownership and offchip scratch policy; it does not
assert that this is the captured dispatch or that its remaining timing error
has converged. A source-domain isolated smoke at q6/B8 produced about356us,
so matching the ABI alone must not be reported as <=15% accuracy.

Validation: 50 focused tests passed, covering the packed prolog, batched
attention, public Body defaults, mixed row capacity, finite buffers, readiness,
work conservation and a single inclusive native-call service envelope.
