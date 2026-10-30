# Native fusion resource projection

Declared semantic subgraphs and selected executable kernels are separate views
of the same frontend DAG. A subgraph establishes an independently declared
operation boundary. It does not by itself select a kernel, reduce memory
traffic, or establish that a captured event executed that implementation.

`prediction.native_fusion.lower_native_fusions(ops, runtime)` is the optional
last resource pass. It reads only `runtime.kernels`. The regular
`target_graphs`/`target_operators` and serving frontend include the pass;
`resource_fusion=False` skips just this final pass and retains the other
implementation and quantization lowerings. Structure export constructs that
semantic graph once and projects it into the selected resource graph.

## Selection

Selection names roles already declared by the attention, FFN or residual
builder. It does not infer a group from an operator name or capture:

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "mla_prolog": "vllm_ascend_mla_prolog_v3",
        "native_fusion": {
          "implementation": "declared_native_fusion",
          "options": {
            "roles": ["mla_prolog", "lightning_indexer", "hc_pre"],
            "traffic_profile": {
              "mla_prolog": "mla_prolog_arch35_split_m_n_ae923905",
              "hc_pre": "hc_pre_m_k_split_175ef4d4"
            }
          }
        }
      }
    }
  }
}
```

Every selected role must occur in the supplied graph. Choose only roles present
in that model's declaration; the example shows configuration syntax, not a
claim that every model contains those three roles. A single profile string
applies to every selected group; a role-to-profile mapping defaults omitted
roles to `preserve_workspaces`.

The pass requires disjoint, complete groups on one stage, phase and stream.
It runs after capture regions are declared and cannot merge across a capture
segment or turn an eager member into captured work.
An unknown multi-launch stage, such as the retained ChunkKdaFwd capture
boundary, cannot become a complete one-launch kernel. A source-call boundary
or collective requires its own executable planner and is not silently folded
into this resource projection. Selection errors remain visible.

## Resource execution

Each selected group becomes one `OpSpec` with an exact `PlannedRecipe`.
Its finite `KernelProgram` contains the original members' resource programs
and internal dependency edges. Matrix precision, vector work, extended
resource demands and existing recipe choices remain explicit. The enclosing
kernel performs one submission and setup; internal phases use
`RecipeContext.for_phase` and do not repeat the enclosing startup or inherit
an inclusive measured kernel duration.

This is a resource DAG calculation, not a sum of standalone kernel durations.
It retains the existing analytical member recipes, including their matrix
mapping and workspace approximations. Its internal overlap follows declared
dependencies; it is not a reconstruction of native instruction scheduling.
All external dependencies and consumers are rewired to the kernel boundary.
Outputs remain available only at that boundary's completion.

An explicit source plan may replace the members' analytical resource
representation when the implementation changes it. The HC M/K-split plan,
for example, computes RMS square sums with a shared-input Gram product instead
of a separate full-X vector reduction. Its work is prepared from the same
source profile and hardware geometry before export and execution.

The structure manifest retains semantic members for attribution and adds the
selected executable operator under `resource_operators`. Its subgraph records
`resource_boundary.operator_ids`, physical work, the selected profile, source
evidence and fidelity limits. Internal semantic members are not independent
matchable kernels under that selection. Physical phase identities use local
ordinals so repeated Transformer blocks with equal work and implementation
have equal physical signatures.

## Memory profiles and their limits

| Profile | Selected source domain | Eliminated model transfers | Transfers retained |
| --- | --- | --- | --- |
| `preserve_workspaces` | No local storage selection | None | All member bytes; only kernel submission is combined |
| `mla_prolog_arch35_split_m_n_ae923905` | vllm-ascend revision `ae923905`, arch35 split-M/split-N MlaPrologV3 | Normalized latent K write and cache-store read; K-rope cache-store read; K-rope write only when its producer has no external consumer | Q projection, norm and quantization workspaces; external q-norm/Q-rope output; final cache writes |
| `hc_pre_m_k_split_175ef4d4` | vllm-ascend revision `175ef4d4`, HcPre M/K split | Shared-X projection/Gram source plan replaces separate vector RMS; coefficient intermediates stay local | Cast ring, split-K projection/Gram partials, original-X reread and y/post/comb writes; all GM bytes retain HBM service |
| `hc_post_d_split_175ef4d4` | Same revision, HcPost D-split | No external tensor byte removal | Source 2048-column dependencies, FP32 conversions and bounded queues |
| `grouped_swiglu_quant_int4_post_ae923905` | vllm-ascend revision `ae923905`, A8W4 MSD postprocess | SwiGLU-to-quantization temporary write/read | Existing GEMM workspace and quantized output/scales |
| `grouped_swiglu_quant_int8_split_workspace_ae923905` | vllm-ascend revision `ae923905`, A8W8 key-1 split-workspace path | SwiGLU-to-quantization temporary write/read | Existing GEMM workspace and quantized output/scales |

Every eliminated transfer is bounded by the actual producer write and consumer
read in the declared graph. A producer with a consumer outside the group keeps
its write. This is conservative for a multi-output producer: the MLA rotary
node exposes Q-rope as well as K-rope, so its full write remains when Q-rope is
externally live. Internal cache-store reads can still be removed.
An explicitly selected shape-generated memory recipe cannot accept these
envelope-only deductions: the pass rejects that combination instead of reporting
a saving that its executed program would reconstruct. Unmodified matrix and
other phases retain their existing selected recipes.

The grouped profiles establish local reuse of the activation result. They do
not reconstruct the native A8W4 dual-FP16 workspace, A8W8 INT32 accumulator
workspace, packing, scale arithmetic or instruction tiling. Those remain
explicit fidelity limits of the existing member recipes. Selecting the profile
is not a claim of complete native-kernel fidelity.
The separate `program_profiles` selection can replace that member program with
`ascend_w4a8_msd`; its source ledger includes the MSD phases and epilogue.
Structure comparison preserves partial-member limitations for ordinary traffic
profiles and removes only work explicitly implemented by the selected source
program. A single launch alone never establishes a complete cost model.

LightningIndexer and QuantLightningIndexer source uses global score and
partial-top-k workspaces. Their default combined resource program retains
those bytes. HcPost is already one semantic resource operation, so selecting
its boundary does not invent an additional saving. Its D-split source profile
adds conversion work and dependent stages; see
[HC source resource plans](operators/hyper-connection-source.md).
AttnResFwd has a semantic
normalization/projection/softmax/mixture group, but the independently inspected
Triton alternative does not prove that native kernel's storage behavior; its
default profile retains member transfers.

Source path, revision and available hashes are embedded in each profile's
accounting ledger. Source selection does not identify a historical capture's
binary, tiling key or dispatch. In particular, the inspected MLA implementation
is arch35 source; a same-named A3 event is not proof that it follows that path.
Historical conditional timing comparisons must retain this distinction.

## Verification

`tests/test_native_fusion_resources.py` uses existing V4 and GLM model cases
and actual frontend builders. It checks resource conservation, a single host
submission through the common execution engine, bounded source-specific
transfer reductions, externally live outputs, MLA/indexer dependency rewiring,
extended resource additions, repeated-group signatures, and rejection of a
complete-kernel interpretation for unknown KDA stage launches.

Profiler capture matching and conditional cost comparison remain independent
consumers. No observed duration or captured tensor shape is used to construct
these semantic groups or resource programs.
