# MXFP8 single-group projection wrapper

`prediction.matrix_wrappers.apply_matrix_wrapper(ops, selection, runtime)`
declares the tensor view of an explicitly selected eager projection. It returns
the same operators with metadata only. FLOPs, bytes, dependencies, operator IDs,
source boundaries and selected physical recipes remain unchanged.

The implementation name is `vllm_ascend_mxfp8_single_group_eager`. Its selection
option `projection_recipes` selects the supported `latent_projection` caller
recipe; other caller recipes are rejected. The independent current-checkpoint
Kimi declaration has two `latent_projection` operators per MoE FFN sublayer.
Those correspond to the wrapper's routed input/output transforms. Attention
and other matrix recipes are not selected by this declaration.

The supported domain requires explicit `runtime.graph_mode="eager"` (or the
existing none/off/disabled aliases), a two-dimensional input, one dense weight
matrix, no bias, FP8 E4M3 operands, BF16/FP16 output, and per-output/group-32
E8M0 weight scales. Activation scales retain `(M, ceil(K/64), 2)`. Existing
source-bound operators are retained. An explicit selection outside the domain
raises instead of silently producing a different native contract.

The wrapper creates `group_list=[M]`, one bank and `group_type=0`. Structure
export therefore emits the existing strict `grouped_matrix` contract with
geometry `(1,M,N,K)`. The observed grouped contract remains grouped. Multi-bank
MoE routing, allocation padding, active expert histograms and MXFP4 execution
are outside this wrapper. In particular, the existing partial native MXFP4
prediction scope is not removed.

Source evidence is the retained campaign image
`sha256:883670fcc91cbd75a7070b00501be0ecd81a266b81366a21bf02139d8e6a33e1`.
`vllm_ascend/models/kimi_k3.py:120` selects `apply_single_group_gmm` only for an
eligible two-dimensional, bias-free eager invocation.
`vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py:167` implements the wrapper;
line 209 constructs the one-element row-count tensor. Exact file hashes are
retained in the declaration metadata. The full source texts and image-layer
provenance are retained by the profiler's
`tests/fixtures/kimi_grouped_attnres/source_image_883670fc/`.

Source selection establishes an independently declared candidate tensor domain.
It does not prove historical graph/eager dispatch, actual loaded binary identity
or an event's position in a Transformer block. The current-checkpoint case is
also explicitly conditional on its independently inspected checkpoint metadata.
KDA's separated Q/K/V operators do not become the source's fused QKV projection
merely because they could be concatenated to an observed width.
