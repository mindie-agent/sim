# Composable model inputs

`ModelSpec.from_mapping` and the ordinary `predict`, `target_graphs`,
`graph_costs`, and `operator_costs` paths accept Hugging Face configuration
mappings. No weights, CUDA, CANN, or framework model objects are loaded.

```python
import json
from mindie_sim import predict, target_graphs, graph_costs
from mindie_sim.model_catalog import load_model

config = json.loads(open("config.json").read())
case["model"] = config
# Equivalently: {"hf_config": config, "name": "my-checkpoint"}
result = predict(case)
graph = graph_costs(case, target_graphs(case, phase="prefill", tokens=32))
model = load_model("/models/my-checkpoint/config.json")
```

Local paths and architecture preset keys use the same construction path.
`minimax-m3`, `longcat-flash-lite`, `bailing-v3`, `qwen3.5-moe`, and
`hunyuan-video-dit` now produce executable structural records. The DiT preset
is only a partial transformer-component study: it does not model video
patchification, double streams, diffusion iterations, or cache algorithms,
and its request metrics are not video generation latency.

## Structural adapters

| Input | Preserved structure | Remaining declared limitations |
| --- | --- | --- |
| Qwen2 / Qwen2.5 / Qwen3 dense and MoE | Native head width, GQA, dense/MoE schedule, QK norm where declared, shared expert | QKV bias, unsupported activation/gating and optional draft variants are explicit gaps |
| DeepSeek V2 / V2-Lite / V3 / R1 | MLA, direct Q when q-rank is absent, first dense layers and modulo MoE, shared experts | Router correction and score scaling; optional MTP when its complete schema is absent |
| GLM4 / GLM4.5 / Air | GQA/MQA, dense/MoE, distinct QK norm settings; unindexed MLA geometry can compose MLA | Extra GLM4 norms, projection bias, half RoPE and custom routing are explicit gaps |
| Kimi K2 / VL text | Standard MLA+MoE; no inherited K3 KDA/latent/DSpark assumptions | Vision tower/projector and custom router semantics remain explicit |
| MiniMax M1 / Text01 | Native attention schedule, known Lightning QKV/gate/output projections, recurrent state size, FFN | Lightning core and gated norm, postnorm and custom scaling |
| MiniMax M3 preset | Standard projections, dense/MoE, shared KV producers and cache allocation | Sparse attention/index selection; unspecified FP8 scale geometry uses a declared native-precision proxy |
| Step3 | Nested text geometry, explicit MoE layer list, shared/routed FFNs, embeddings and norms | Shared-query attention geometry and vision are omitted explicitly |
| Other HF types | Recognized GQA/MLA/recurrent declarations and FFNs | Unknown fields and submodules are retained in coverage metadata |

Official config fixtures retain repository, revision and hash under
`tests/fixtures/hf-standard-families`, `tests/fixtures/standard-qwen-kimi`, and
`tests/fixtures/composable-hf`. Step2 is tested with a declared structural
example, not misrepresented as an official fetched checkpoint: the attempted
official config endpoint was not publicly readable.

MiniMax M1's independently read source is
`MiniMaxAI/MiniMax-M1-80k@8d1494b1a260e22040d5b9b2eb332eb44500b34d`,
`modeling_minimax_m1.py`: lines 210–223 declare QKV, output gate and output
matrices; lines 1203–1217 distinguish numeric Lightning/full-attention layer
codes. Those facts select structural records in the adapter. The Engine
contains no MiniMax-specific performance branch.

## Coverage and timing boundaries

Every public prediction report carries:

```json
{
  "model_coverage": "partial",
  "unmodeled_features": ["..."],
  "degraded_submodules": ["..."],
  "prediction_scope": "known_components_with_declared_degradations"
}
```

The same fields are retained in `ModelSpec.metadata`, along with original
unrecognized field values and the source config hash. Coverage survives JSON
roundtrips. Unknown model types do not disable known work. Malformed mandatory
dimensions, invalid ownership, and inconsistent schedules still produce errors.
Explicit simulator-schema inputs keep strict field validation; HF config
interpretation does not make simulator typos silently acceptable.

An unsupported attention core creates a visible zero-work dependency marker
between known projections. It does not become GQA or GDN. Its missing work,
memory and scheduling behavior are not inferred from the observed residual.
For an unknown cache contract no dense KV cache is invented. Known Lightning
state storage is counted independently from its unmodeled update program.

`complete` means that the declared structural schema has no known omissions.
It is not measured accuracy, full compiler fidelity, or a heldout validation
result. A partial model's graph/request times describe only the retained work
under the declared proxy assumptions. They are not a complete-model performance
claim and are not guaranteed lower bounds when a proxy changes representation.

## Ownership and precision

Shared GQA consumers omit their K/V projection weights and cache writes. They
depend on the actual owner store, retaining a real PP transfer when the owner
is on another stage. Query projection, Q normalization/rotary, attention reads
and output remain per consumer. Capacity counts the owner's persistent KV once.
Cross-PP shared KV currently models the transport dependency and payload but
does not fully account for the receiving mirror/cache and endpoint copy service.
Such a placement gains an explicit partial marker during case normalization.

Native activation/weight categories and KV precision are independent. Explicit
FP8 block dimensions and scale formats determine packed payload plus scales;
standard V3/R1/K2 do not inherit V4's routed FP4 default. Unresolved quantization
variants or module-path exclusions produce an explicit native-precision proxy
with the original declaration retained. They never fabricate block dimensions,
device utilization, a timing multiplier, or a measured dispatch floor.

## Acceptance scope

Tests execute the ordinary frontend and shared Engine, checking independent
projection FLOPs, native precision, shared-cache producer dependencies and
storage, graph liveness, and equal report coverage at operator/graph/serving
boundaries. Small dimensional controls preserve the source's structure and
are labeled as such. They establish software and geometry behavior, not
full-checkpoint accuracy. The measured six-model scoreboard remains a separate
acceptance artifact; its unresolved whole-step residuals are not erased by
catalog generalization.
