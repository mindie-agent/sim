# Shared-KV partial rotary caller

Select `runtime.kernels.lowerings.v41_rotary = "v41_partial_rotary"` for the
unfused latent-MQA, window/compressed-cache caller. The declaration replaces
the former combined Q/KV rotary node with independent Q and KV invocations,
then inserts output inverse rotation between attention and the output
projection's activation quantization. Other callers and undeclared inputs
retain their existing graph.

Only the BF16 rotary slice is read and written. The NoPE suffix/prefix is not
copied. FP32 sine/cosine traffic is declared once per token across heads;
coefficient sharing is an analytic assumption. Source caller/kernel hashes and
full tensor views are retained on each node. The three invocations keep their
boundaries through generic graph transforms and use the shared device's AIV,
HBM, MTE2_AIV and MTE3 supplies and invocation policy.

This is a source-derived traffic and invocation decomposition. Native BlockDim,
per-owner reload, and instruction scheduling are not reconstructed, so timing
uses the explicit resource envelope. It does not establish captured binary
identity or introduce a duration fitted from the 136-call observation count.
Indexer and compressed-cache rotations are unaffected; no extra Q/KV nodes
are added alongside the replaced aggregate.

Four focused tests check partial-slice arithmetic, independent dependencies,
NoPE-width invariance, and complete three-layer/EP-cohort execution with
quantization after inverse rotation and all resources released.
