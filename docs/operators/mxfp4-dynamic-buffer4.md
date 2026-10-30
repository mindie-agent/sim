# MXA8W4 DYNAMIC_BUFFER4 producer pipeline

`producer_mode="dynamic_buffer4"` explicitly selects the retained CANN source dependency program. The existing `serial` mode remains the default and an inspectable control. Both modes use the same integer expert histogram, matrix geometry, bytes, register inventory, hardware rates, finite buffer capacities and outer invocation.

## Source boundary

The selected source is `MXA8W4_NZNK / VECTOR_ANTIQUANT / N_FIRST_TAIL_RESPLIT / DYNAMIC_BUFFER4` in retained `ops-transformer` revision `a3925fd49cd256ad478d2faf983bf2961b8ae2c8`. Exact local file hashes are in `data/verification/mxfp4-dynamic-producer-20261030.json`. This source branch selection does not prove the captured binary identity or peer `group_list` contents.

| Source location under `gmm/grouped_matmul/op_kernel` | Dependency represented |
|---|---|
| `grouped_matmul_apt.cpp:75,425–432` | Dynamic configuration selects four input buffers. |
| `arch35/weight_quant_basic_block/basic_block_config.h:208–225` | Packed input: 80 KiB / four slots; widened output: 128 KiB / four slots. |
| `weight_quant_vec_compute.h:387–450,735–791` in that directory | Scale and packed-weight reads advance the same input-ring counter; a scale read occurs at each 4096-K boundary. Vector completion releases the input slot. |
| `weight_quant_vec_compute.h:755–791,793–835` | Two scale-transpose output slots and four widened output slots retire after MTE3 publication. |
| `weight_quant_basic_block.h:267–338,417–452` | AIV-to-AIC publication waits on MTE1; the reverse L1-reuse wait is on MTE3. It does not block independent MTE2 reads or Vector operations. The last MTE1 read releases the B-L1 slot while Cube may still consume L0. |
| `weight_quant_cube_compute.h:135–233,273–335` | Activation and activation-scale refills have independent two-slot release periods. |
| `basic_api/weight_quant_basic_api_v1.h:101–130,202–220` in that directory | Two L0 slots, serial operand/Cube streams, and Cube-to-MTE1 slot retirement. |

The old detailed program also serialized the entire producer at every K panel. Therefore it remains a conservative implementation control; it is not the detailed oracle for the new overlap mechanism.

## Execution and compact boundary

A shared dependency emitter records each instruction stream and the input, widened-output, transpose, L1 and L0 rings. The detailed program submits those dependencies to the existing Engine. Its bounded generation window includes two activation/scale refill periods; it is not an additional physical buffer. Actual physical buffers continue to be reserved by the ordinary grouped kernel.

The nominal solver uses exact binary ticks and only jumps after the full normalized dependency state, ring cursors and periodic phase repeat. Unused zero-time input anchors remain absolute across a jump. Finite tails continue from the live state. It is a max-plus dependency calculation, not a second contention engine.

The compact program retains a real first producer and final Cube operation, with all intermediate work in the existing coupled fluid phase. Both producer mode and slot contract enter the owner-template and bound-IR cache keys. The generic `compact_refill` implementation is unchanged.

Resource arbitration under unrelated, asymmetric contention remains an approximation. The shared dependency emitter does not establish arbitrary-contention equality between detailed and compact programs.

## Independent controls

Hand-computed three-pipe tests use read 3, Vector 4 and publish 2. One, two and three panels finish at 9, 13 and 17 respectively, rather than 9, 18 and 27. Additional controls cover one/two/four slots, earliest legal slot reuse, slow Cube, L1 reuse before Cube completion, the extra scale-read ring increment, a one-sided AIV tail, zero experts, late input, periodic jumps, nonuniform tails and zero live state. A 64-panel source program agrees with the eagerly submitted full dependency graph, checking the bounded generation window.

The final affected run has 94 passing checks. Zero-expert initialization and source work use the established numerical tolerance.

The retained ordinary-path scan contains 32 conditions / 72 actual Engine runs. All work differences are below `8.6e-14` relative; live tasks and remaining buffers are zero. At the declared A5 rates, the c64 symmetric rank0 projection results are:

| Projection | Shape `(M,N,K)` | Prior serial compact µs | Source detailed µs | Source compact µs |
|---|---|---:|---:|---:|
| Expert gate/up | `(48,4608,5120)` | 447.025 | 216.484 | 215.839 |
| Expert down | `(48,5120,2304)` | 209.013 | 108.862 | 107.298 |

These are isolated kernel predictions. They are not measured error improvements, and multiplying them by layer count is not a body latency calculation.

Across the declared profile and individual 0.5x/2x HBM, AIC/AIV copy, MTE1, MTE3 and Vector changes, the largest compact-versus-detailed difference is −5.21%. This limited scan does not establish a universal 15% bound: the retained synthetic single-owner control with an overlapping AIV0 Vector burst is **+43.13%** in compact execution. Its work remains conserved. Simultaneous and delayed HBM bursts and a delayed AIV read burst in the same component control agree to numerical tolerance. No rate, floor or residual correction hides the asymmetric-Vector counterexample.

Independent review adds 160 separately written FIFO/ring numerical oracles, 24 actual Engine controls and 18 source-shape/profile comparisons. Its limited source scan reaches −5.32%. More demanding one-owner contention, beginning 1 µs after the input with 20 µs of competitor service, gives the following counterexamples:

| Matrix component `(M,N,K)` | Competing owner resource | Compact vs detailed span |
|---|---|---:|
| `(1,160,5120)` | AIV0 MTE2 | +72.45% |
| `(17,160,4160)` | AIV0 register UB stores | +84.66% |
| `(1,128,5120)` | MTE1 | +22.08% |

HBM contention in the same independent control agrees to numerical tolerance. Both programs conserve resource work and release all state. The discrepancy is the compact interior's reordering of heterogeneous contention, not a newly inferred source-ring defect. The retained detailed program is required when resolving this boundary; the compact mode is not accepted as universally equivalent. Evidence: `data/verification/mxfp4-dynamic-independent-review-20261030.json`, with independent numerical/source and contention producers alongside it. These are synthetic representation comparisons, not hardware observations.

## Explicit full-model inputs

Seven source variants are retained in `examples/cases/high-concurrency-20261030/v41-dynamic-buffer4-conditional/`. They add only the producer mode, explicit compact-panel choice and source provenance to the existing cases. The single-active-rank0 and symmetric-DP8 conditions remain separate. Client concurrency and graph capacity do not prove which peer workload occurred.

`data/verification/v41-dynamic-buffer4-body-20261030.producer.py` runs all seven labels under both conditions, retaining 40 layers, fresh process-local caches, a 60-second simulation budget and 10-second receipt/cleanup budget per row. Timeouts remain reported; they are not retried with a longer budget. Body results and CPU cost belong to that separate report, not to the isolated projection table above.


All 14 full-model rows completed at the stable package hash `90f5ed66354d75a33c0940a8e62124912ee3f5db95196f9e866df3428b7369e7`, whose runtime files equal integrated revision `12e2073b`. The original 12-row campaign was followed by the two c4 rows omitted from the initial request; no completed row was rerun. Each worker retained all 40 layers, no input patch, zero live tasks/programs/buffers and the original 60-second budget. Maximum wall time was 44.44 seconds; peak RSS was 380.1 MiB.

| Client label | Rank0 B | Single active body ms | Symmetric DP8 body ms | Single / symmetric wall seconds | Single / symmetric task count |
|---|---:|---:|---:|---:|---:|
| c1 | 1 | 6.706860 | 8.057932 | 14.10 / 36.83 | 634,042 / 1,468,488 |
| c4 | 1 | 6.706860 | 8.057932 | 13.74 / 32.38 | 634,042 / 1,468,488 |
| c8 | 1 | 6.706860 | 8.057932 | 14.45 / 36.58 | 634,042 / 1,468,488 |
| c16 | 2 | 6.998600 | 9.700744 | 44.44 / 39.72 | 819,194 / 1,530,184 |
| c32 | 4 | 7.312806 | 12.986367 | 22.18 / 37.42 | 825,978 / 1,584,456 |
| c64 | 8 | 8.223169 | 19.570291 | 20.60 / 39.95 | 839,585 / 1,693,312 |
| c128 | 16 | 10.079983 | 19.884427 | 20.52 / 41.77 | 866,799 / 1,911,024 |

The q1/context2048 geometry and the two peer conditions are explicit scenario inputs. The c1/c4/c8 labels share the same observed rank0 B1 input in these retained cases, hence their identical predictions within each condition. Graph capacity, client concurrency and exchange symmetry do not establish the capture's valid peer rows or expert histogram. The table is therefore a full-model execution and CPU-budget result, not a whole-body accuracy scoreboard. Its separate family activity sums include overlapping intervals and cannot be substituted for the graph span. The 0.63–1.91 million tasks also remain above the desired seconds-scale exploration cost; no timing shortcut hides that limit.
