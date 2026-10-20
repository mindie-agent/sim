# Model reference packages

Each package preserves the actual archived model dimensions and a declared,
adequate A3 topology.  The JSON structure is derived from
`mindie_sim/model_configs/` and recomputes for new shapes; it is not a latency
replay table.

- `structure/<model>.json`: normalized body schedule, attention/FFN/cache/state
  ownership, quantization, optional draft/vision/Engram records, per-rank PP
  stage weights, weight category breakdown and cache at `max_model_len=256`.
- `cases/<model>.json`: machine/model/workload/runtime case using the full model
  and the declared topology.
- `cases/<model>.should.json`: finite should-cost constraints (exact placement,
  unfused/fused variants, scalar `graph_padding: false`, three-plan simulation
  budget).
- `cases/<model>.search.json`: finite two-candidate search domain with unfused
  and fused candidates.
- `results/`: current executed public predict/should/search summaries for all
  nine models plus the Qwen media predict case. The workload is a
  prompt8/output1 structural smoke case, not a serving-viability claim. These
  are token-ready analytic-reference summaries with separate preparation/drain
  accounting; not measured A3 rows. A single declared
  intra-node link spans 8–128 dies; that is synthetic for >8-die topologies, not
  a validated multi-node fabric.
- `pending-measurement-inventory.json`: current real-lowering inventory
  (11,386 conditions, 318 selected requests, 4,902 pending conditions, 82
  pending family/recipe groups, bounded concrete shape examples) and executable
  campaign commands; no measured samples or heldout outcomes.
- `generate_reference.py`: deterministic regeneration (no device/network).

## Commands

```bash
python -m mindie_sim predict --input examples/model-reference/cases/kimi-k3.json
python -m mindie_sim should  --input examples/model-reference/cases/kimi-k3.json \
  --constraints examples/model-reference/cases/kimi-k3.should.json \
  --search-budget 3
python -m mindie_sim search  --input examples/model-reference/cases/kimi-k3.json \
  --search-spec examples/model-reference/cases/kimi-k3.search.json
```

Replace `kimi-k3` with any key in `index.json` (all nine are present).  The
per-request media path is additionally exercised by
`cases/qwen3.8-27b-media.json` with the `media_tokens_by_request` metadata
mapping; its tower executes once on the first prefill chunk and is skipped on
later chunks/decode.

## Scope

The model configs are the real archived configs; request lengths may be changed
without changing the architecture. Topologies are chosen so the full model fits
per-rank HBM under the declared A3 profile and are not a production
recommendation. Cases include the accepted public matrix's Ascend910_9391
24-AIC/48-AIV, 512KiB L1, 64/64/128KiB L0A/L0B/L0C, 192KiB UB and 192MiB L2
declarations so finite fusion candidate evaluation is physical; compute and HBM
peaks are unchanged. No NPU execution, measured latency claim or heldout data is
included. Stored TTFT/U are token-ready analytic estimates. Model-derived
preparation is enabled for several cases; each result records prep-task counts
and engine-drain timestamps separately, so background preparation/drain is not
silently folded into or subtracted from the token-ready number. Missing
calibration yields analytic estimates plus explicit measurement gaps.

Regenerate structure/cases/index:

```bash
python examples/model-reference/generate_reference.py
```

Regenerate the pending inventory from current lowering without changing measured
campaign artifacts:

```bash
python -m mindie_sim.a3_operator_campaign plan \
  --config examples/operator-campaign/campaign-config.json \
  --output-dir /tmp/campaign-current --budget 320
```

The nine-model results are validated by the public command forms above; no
reduced-model substitution is used.
