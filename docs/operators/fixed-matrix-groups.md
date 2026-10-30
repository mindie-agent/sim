# Fixed equal-matrix groups in `TiledMatmul`

`TiledMatmul` remains the fixed-mapping control. When an operator declares
`metadata["matrix_groups"] = G`, the leading M axis is G equal independent
matrices, not one dense axis:

- `G` must be a positive integer and `totalM % G == 0`. The group-local M size
  is `totalM / G`. A missing key means `G = 1`; an explicit invalid grouped
  domain raises instead of falling back to dense geometry.
- One kernel and one setup materialize the finite program. Flat M/N tiles are
  ordered group-major and assigned once across owners, with K panels innermost.
  Each tile reads only its own group's B matrix. A is still copied privately
  for each N tile.
- With ideal reuse, the shared HBM totals remain the original input, weight and
  output bytes. Disabling `reuse_inputs` or `reuse_weights` charges the actual
  per-tile copies instead. MTE2 private copy service is independent of those
  flags.
- For explicit multiple groups, the L1 panel allocation is the declared A+B copy
  payload, including packed weights and activation scales. It is not a
  compute-width approximation.
- Each owner tile acquires its accumulator on the first K panel and releases it
  after the final store. Different groups never share that accumulator identity.
- Only the finite slot window is live: active state scales with
  `cores * slots`, not with the total group/tile count.

The focused oracle lives in `tests/test_tiled_matmul_groups.py`; it independently
enumerates scalar operand addresses over ragged geometry and all reuse
flag combinations, and includes an ordinary frontend MLA decode op.

The single-group path retains its previous buffer approximation. In particular,
`matrix_mapping_domain="aggregate_experts"` still describes expected rows and
active banks without individual expert geometry. Its aggregate weight traffic
must not become a fictitious single physical L1 allocation. Resolving that
approximation requires a separate declaration of expert geometry; this change
does not silently replace it or alter its timing.

```sh
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_tiled_matmul_groups.py
```

## Independent acceptance on 2026-10-29

The focused fixed-group, mapping, phase, MLA and balanced-expert checks passed
50 tests. The fixed-group oracle counts explicit BF16 operand addresses and
private copies, including all four reuse settings. Separate packed-INT8 checks
cover scale-containing panel residency. Source-selected MLA attention remains
an `OperatorPlan`; the surrounding ordinary head matrices use the corrected
fixed program.

Ten actual Kimi, GLM and V4.1 frontend matrix controls preserved logical FLOPs
and HBM traffic. Representative isolated predictions with the same declared
hardware and 128/128/128 fixed tiles were:

| Ordinary frontend operation | Old fixed (us) | Corrected fixed (us) | Private MTE2 bytes before / after |
| --- | ---: | ---: | ---: |
| Kimi prefill256 `q_absorb`, G=12 | 10.796 | 2.193 | 40,894,464 / 6,291,456 |
| Kimi prefill256 `v_expand`, G=12 | 14.138 | 2.604 | 40,894,464 / 6,291,456 |
| Kimi decode1 `v_expand`, G=12 | 12.718 | 1.060 | 1,585,152 / 1,585,152 |
| V4.1 prefill256 `grouped_output_a`, G=8 | 152.068 | 34.965 | 606,601,216 / 136,380,416 |
| V4.1 decode1 `grouped_output_a`, G=8 | 33.872 | 8.469 | 33,857,536 / 33,857,536 |

The decode cases change owner distribution even when total private bytes stay
constant. Twelve ordinary aggregate-expert controls across Qwen3.5, Kimi and
V4.1, prefill and decode, retained exactly the old duration, serviced work and
buffer results. These are mechanism comparisons on identical parameters, not
measured latency improvements. They do not include later changes to hardware
materialization or other frontends.
