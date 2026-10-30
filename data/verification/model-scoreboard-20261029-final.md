# Model timing scoreboard

<!-- current-body-evidence:start -->
## October 30 current Decode-body evidence

**65/66** selected completed comparisons satisfy
the strict 15% body criterion. These are the latest completed arms for
each retained source/geometry, **not a replay of every case at current HEAD**.
Runtime hashes below keep that distinction visible. Selection remains
conditional on the declared topology, context and rank coverage.

Latest attempts: **66/66 completed**, **0 failed**, **65 passed**.
An older completed prediction retained below does not turn a newer failed attempt into success.

The HF-wrapper precision bug invalidated the earlier Kimi predictions.
Superseded predictions are excluded; their measured populations remain.
The calibration column preserves the source's evaluation role;
`in_sample` is a calibrated comparison, not independent validation.
Micro-kernel tolerance does not relax body acceptance.

| Case / variant | Observed mean ms | Predicted ms | Signed gap ms | Absolute gap ms | Relative | Pass | Calibration | Runtime package |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| qwen35-397b-a3-tp16-c48:q4_b64_m256 / q397-matrix-gdn-21c30821 | 76.089 | 61.285 | -14.804 | 14.804 | -19.46% | no | observed | `9f88dac92812` |
| qwen35-397b-a3-tp16-c136:q4_b160_m640 / q397-matrix-gdn-21c30821 | 119.054 | 104.482 | -14.573 | 14.573 | -12.24% | yes | observed | `9f88dac92812` |
| a3_glm53_w4a8_tp16_tpot50_c2:q6_b2_m12 / glm78-packed-prolog-joint | 57.817 | 49.554 | -8.263 | 8.263 | -14.29% | yes | observed | `9f88dac92812` |
| a3_glm53_w4a8_tp16_tpot30_c2:q6_b2_m12 / glm78-packed-prolog-joint | 57.639 | 49.582 | -8.058 | 8.058 | -13.98% | yes | observed | `9f88dac92812` |
| a3_glm53_dc4k_b4:q1_b4_m4 / body-runtime40e4300e-glm-controller-corrected | 43.438 | 38.631 | -4.807 | 4.807 | -11.07% | yes | not declared | `ab51c56356b3` |
| a3_glm53flash_tp16_aclgraph_c8:q6_b8_m48 / body-runtime40e4300e-glm-controller-corrected | 46.859 | 42.056 | -4.804 | 4.804 | -10.25% | yes | not declared | `ab51c56356b3` |
| a3_glm53flash_tp16_aclgraph_c16:q6_b16_m96 / body-runtime40e4300e-glm-controller-corrected | 53.893 | 49.637 | -4.256 | 4.256 | -7.90% | yes | not declared | `ab51c56356b3` |
| a5_kimi_k3_p32768_c1:q1_b1_m1 / body-runtime40e4300e-remaining-controller-corrected | 32.555 | 28.561 | -3.995 | 3.995 | -12.27% | yes | not declared | `ab51c56356b3` |
| a3_glm53flash_tp16_aclgraph_c32:q6_b32_m192 / body-runtime40e4300e-glm-controller-corrected | 69.361 | 65.400 | -3.961 | 3.961 | -5.71% | yes | not declared | `ab51c56356b3` |
| glm53flash-a3-tp16-c4:q6_b4_m24 / body-runtime40e4300e-glm-controller-corrected | 39.841 | 36.251 | -3.590 | 3.590 | -9.01% | yes | not declared | `ab51c56356b3` |
| v4flash-a3-tp8-c4:q2_b4_m8 / a3-retained-cohort-service-21c30821 | 32.486 | 29.272 | -3.214 | 3.214 | -9.89% | yes | observed | `9f88dac92812` |
| glm53flash-a3-tp16-c2:q6_b2_m12 / a3-pooled-active-contract-service | 34.118 | 31.018 | -3.101 | 3.101 | -9.09% | yes | observed | `9f88dac92812` |
| a3_v41_w8a8_tp4_dp4_ep16_dspark7_c8:q8_b4_m32 / body-runtime40e4300e-remaining-controller-corrected | 48.582 | 51.673 | +3.091 | 3.091 | +6.36% | yes | not declared | `ab51c56356b3` |
| v4flash-a3-tp8-c8:q2_b8_m16 / body-a3-v4-caller-bf16-cc245da6 | 37.763 | 34.707 | -3.056 | 3.056 | -8.09% | yes | not declared | `1c7fc78cb8be` |
| a5_kimi_tp8_e16_bf16_c32_p4k:q1_b32_m32 / composed-source-service-3b72ea2d | 40.019 | 36.969 | -3.050 | 3.050 | -7.62% | yes | observed | `57520ed725bf` |
| a3_glm53_dc4k_b8:q1_b8_m8 / body-runtime40e4300e-glm-controller-corrected | 47.113 | 44.125 | -2.989 | 2.989 | -6.34% | yes | not declared | `ab51c56356b3` |
| a5_glm53flash_tp8_c8_p4k:q4_b8_m32 / observed-metadata-layer-13862d5e | 28.833 | 25.915 | -2.919 | 2.919 | -10.12% | yes | observed | `13100107089f` |
| a5_glm53flash_tp8_c16_p8k:q4_b16_m64 / observed-metadata-layer-13862d5e | 33.616 | 30.768 | -2.848 | 2.848 | -8.47% | yes | observed | `13100107089f` |
| a5_glm53flash_tp8_c16_p4k:q4_b16_m64 / observed-metadata-layer-13862d5e | 33.031 | 30.187 | -2.844 | 2.844 | -8.61% | yes | observed | `13100107089f` |
| a5_glm53flash_tp8_c4_p8k:q4_b4_m16 / observed-metadata-layer-13862d5e | 21.418 | 18.623 | -2.795 | 2.795 | -13.05% | yes | observed | `13100107089f` |
| a3_glm53_w4a8_tp16_tpot88_c8:q6_b8_m48 / glm78-packed-prolog-joint | 69.556 | 66.764 | -2.792 | 2.792 | -4.01% | yes | observed | `9f88dac92812` |
| qwen36-a5-tp2-c12:q4_b12_m48 / observable-complete-cohort-service-3b72ea2d | 24.226 | 21.480 | -2.746 | 2.746 | -11.33% | yes | observed | `57520ed725bf` |
| a3_v41_w8a8_tp4_dp4_ep16_dspark7_c4:q8_b2_m16 / body-runtime40e4300e-remaining-controller-corrected | 47.654 | 50.270 | +2.616 | 2.616 | +5.49% | yes | not declared | `ab51c56356b3` |
| qwen35-35b-a5-tp4-c128-graph:q4_b128_m512 / q35-shared-native-21c30821 | 28.585 | 26.049 | -2.536 | 2.536 | -8.87% | yes | observed | `9f88dac92812` |
| a5_kimi_tp8_e16_bf16_c32_p16k:q1_b32_m32 / composed-source-service-3b72ea2d | 44.666 | 42.156 | -2.511 | 2.511 | -5.62% | yes | observed | `57520ed725bf` |
| a3_v41_w8a8_tp4_dp4_ep16_dspark7_c16:q8_b5_m40 / body-runtime40e4300e-remaining-controller-corrected | 49.985 | 52.474 | +2.490 | 2.490 | +4.98% | yes | not declared | `ab51c56356b3` |
| glm53flash-a5-tp8-c3:q4_b4_m16 / observed-metadata-layer-13862d5e | 21.068 | 18.663 | -2.404 | 2.404 | -11.41% | yes | observed | `13100107089f` |
| glm53flash-a5-tp8-c1:q4_b1_m4 / observed-metadata-layer-13862d5e | 15.683 | 13.652 | -2.031 | 2.031 | -12.95% | yes | observed | `13100107089f` |
| qwen35-35b-a5-tp4-c64-graph:q4_b64_m256 / q35-shared-native-21c30821 | 21.957 | 19.975 | -1.982 | 1.982 | -9.03% | yes | observed | `9f88dac92812` |
| a5_glm53flash_tp8_c32_p8k:q4_b32_m128 / observed-metadata-layer-13862d5e | 38.713 | 36.852 | -1.862 | 1.862 | -4.81% | yes | observed | `13100107089f` |
| a5_glm53flash_tp8_c32_p4k:q4_b32_m128 / observed-metadata-layer-13862d5e | 37.669 | 35.809 | -1.860 | 1.860 | -4.94% | yes | observed | `13100107089f` |
| glm53flash-a5-tp8-c96:q4_b96_m384 / body-runtime40e4300e-glm-join-reduce | 57.590 | 59.368 | +1.778 | 1.778 | +3.09% | yes | not declared | `ab51c56356b3` |
| a5_v41_fp8_tp1_dp8_ep8_c128_p2k:q1_b16_m16 / observable-service-2f48346b | 27.286 | 25.540 | -1.746 | 1.746 | -6.40% | yes | retained_operator_service_in_sample | `e71d76a0e8fb` |
| a3_v41_w8a8_tp4_dp4_ep16_dspark7_c64:q8_b17_m136 / body-runtime40e4300e-remaining-controller-corrected | 61.811 | 63.541 | +1.729 | 1.729 | +2.80% | yes | not declared | `ab51c56356b3` |
| qwen35-35b-a5-tp4-c4-graph:q4_b4_m16 / q35-shared-native-21c30821 | 11.551 | 9.901 | -1.651 | 1.651 | -14.29% | yes | observed | `9f88dac92812` |
| a5_kimi_tp8_e16_bf16_c8_p16k:q1_b8_m8 / observable-service-source-force-eplb-84d80b40 | 35.964 | 37.585 | +1.621 | 1.621 | +4.51% | yes | retained_operator_service_in_sample | `0e5422506707` |
| a5_kimi_tp8_e16_bf16_c8_p4k:q1_b8_m8 / observable-service-source-force-eplb-84d80b40 | 34.819 | 36.421 | +1.602 | 1.602 | +4.60% | yes | retained_operator_service_in_sample | `0e5422506707` |
| qwen36-a5-tp2-c1:q4_b1_m4 / observable-complete-cohort-service-3b72ea2d | 12.318 | 10.719 | -1.599 | 1.599 | -12.98% | yes | observed | `57520ed725bf` |
| a5_v41_fp8_tp1_dp8_ep8_tpot30ms_c64:q1_b8_m8 / observable-service-2f48346b | 26.328 | 24.812 | -1.517 | 1.517 | -5.76% | yes | retained_operator_service_in_sample | `e71d76a0e8fb` |
| a5_kimi_tp8_e16_bf16_c16_p16k:q1_b16_m16 / observable-service-2f48346b | 39.247 | 40.764 | +1.516 | 1.516 | +3.86% | yes | retained_operator_service_in_sample | `e71d76a0e8fb` |
| a5_glm53flash_tp8_c64_p4k:q4_b64_m256 / observed-metadata-layer-13862d5e | 46.528 | 47.971 | +1.443 | 1.443 | +3.10% | yes | observed | `13100107089f` |
| a5_kimi_tp8_e16_bf16_c16_p4k:q1_b16_m16 / observable-service-2f48346b | 36.985 | 38.388 | +1.403 | 1.403 | +3.79% | yes | retained_operator_service_in_sample | `e71d76a0e8fb` |
| a5_v41_fp8_tp1_dp8_ep8_c32_p2k:q1_b4_m4 / observable-service-2f48346b | 18.625 | 17.361 | -1.263 | 1.263 | -6.78% | yes | retained_operator_service_in_sample | `e71d76a0e8fb` |
| a3_glm53_dc4k_b2:q1_b2_m2 / glm78-low-complete-native-service | 40.970 | 39.767 | -1.203 | 1.203 | -2.94% | yes | observed | `9f88dac92812` |
| a5_v41_fp8_tp1_dp8_ep8_c16_p2k:q1_b2_m2 / observable-service-2f48346b | 14.929 | 13.749 | -1.181 | 1.181 | -7.91% | yes | retained_operator_service_in_sample | `e71d76a0e8fb` |
| a5_v41_fp8_tp1_dp8_ep8_tpot10ms_c8:q1_b1_m1 / observable-service-2f48346b | 13.292 | 12.155 | -1.137 | 1.137 | -8.55% | yes | retained_operator_service_in_sample | `e71d76a0e8fb` |
| a5_kimi_tp8_e16_bf16_c1_p16k:q1_b1_m1 / observable-service-source-force-eplb-84d80b40 | 30.823 | 31.866 | +1.042 | 1.042 | +3.38% | yes | retained_operator_service_in_sample | `0e5422506707` |
| a5_kimi_tp8_e16_bf16_c1_p4k:q1_b1_m1 / observable-service-source-force-eplb-84d80b40 | 30.637 | 31.672 | +1.035 | 1.035 | +3.38% | yes | retained_operator_service_in_sample | `0e5422506707` |
| qwen36-a5-tp2-c96:q4_b96_m384 / observable-native-service-2f48346b | 63.853 | 62.838 | -1.015 | 1.015 | -1.59% | yes | in_sample | `e71d76a0e8fb` |
| a3_v41_w8a8_tp4_dp4_ep16_dspark7_c32:q8_b9_m72 / body-runtime40e4300e-remaining-controller-corrected | 54.967 | 55.972 | +1.005 | 1.005 | +1.83% | yes | not declared | `ab51c56356b3` |
| a3_kimi4_tp8_w4a8_c64:q1_b64_m64 / a3-pooled-active-contract-service | 9.615 | 8.699 | -0.916 | 0.916 | -9.53% | yes | observed | `9f88dac92812` |
| a3_glm53_dc4k_b1:q1_b1_m1 / glm78-low-complete-native-service | 38.955 | 38.046 | -0.909 | 0.909 | -2.33% | yes | observed | `9f88dac92812` |
| a3_glm53_dc16k_b1:q1_b1_m1 / glm78-low-complete-native-service | 39.112 | 38.334 | -0.778 | 0.778 | -1.99% | yes | observed | `9f88dac92812` |
| a5_v41_fp8_tp1_dp8_ep8_tpot5ms_c1:q1_b1_m1 / observable-service-2f48346b | 13.135 | 12.436 | -0.699 | 0.699 | -5.32% | yes | retained_operator_service_in_sample | `e71d76a0e8fb` |
| a5_v41_fp8_tp1_dp8_ep8_c4_p2k:q1_b1_m1 / observable-service-2f48346b | 13.239 | 12.550 | -0.689 | 0.689 | -5.21% | yes | retained_operator_service_in_sample | `e71d76a0e8fb` |
| a3_kimi4_tp8_w4a8_c16:q1_b16_m16 / a3-observable-service-2f48346b | 4.807 | 4.174 | -0.633 | 0.633 | -13.17% | yes | observed | `e71d76a0e8fb` |
| a3_kimi4_tp8_w4a8_c32:q1_b32_m32 / a3-pooled-active-contract-service | 6.714 | 6.097 | -0.617 | 0.617 | -9.19% | yes | observed | `9f88dac92812` |
| a3_glm53flash_tp16_aclgraph_c64:q6_b64_m384 / body-runtime40e4300e-glm-controller-corrected | 97.375 | 97.858 | +0.483 | 0.483 | +0.50% | yes | not declared | `ab51c56356b3` |
| a3_kimi4_tp8_w4a8_c4:q1_b4_m4 / a3-observable-service-2f48346b | 3.476 | 3.196 | -0.280 | 0.280 | -8.07% | yes | observed | `e71d76a0e8fb` |
| a5_kimi_tp8_e16_bf16_c4_p4k:q1_b4_m4 / observable-service-source-force-eplb-84d80b40 | 34.194 | 34.462 | +0.268 | 0.268 | +0.78% | yes | retained_operator_service_in_sample | `0e5422506707` |
| a5_kimi_tp8_e16_bf16_c4_p16k:q1_b4_m4 / observable-service-source-force-eplb-84d80b40 | 34.816 | 35.074 | +0.258 | 0.258 | +0.74% | yes | retained_operator_service_in_sample | `0e5422506707` |
| a5_kimi_tp8_e16_bf16_c2_p16k:q1_b2_m2 / observable-service-source-force-eplb-84d80b40 | 33.876 | 34.119 | +0.243 | 0.243 | +0.72% | yes | retained_operator_service_in_sample | `0e5422506707` |
| a5_kimi_tp8_e16_bf16_c2_p4k:q1_b2_m2 / observable-service-source-force-eplb-84d80b40 | 33.548 | 33.778 | +0.230 | 0.230 | +0.69% | yes | retained_operator_service_in_sample | `0e5422506707` |
| glm53flash-a3-tp16-c4-livepre:q6_b4_m24 / livepre-complete-native-joint-f6e68d5e | 29.762 | 29.545 | -0.216 | 0.216 | -0.73% | yes | observed | `9f88dac92812` |
| glm53flash-a3-tp16-c16-livepre:q6_b16_m96 / livepre-complete-native-joint-f6e68d5e | 41.069 | 40.882 | -0.187 | 0.187 | -0.46% | yes | observed | `9f88dac92812` |
| a3_kimi4_tp8_w4a8_c1:q1_b1_m1 / a3-observable-service-2f48346b | 3.180 | 3.238 | +0.058 | 0.058 | +1.82% | yes | observed | `e71d76a0e8fb` |

[Compact receipt](decode-body-current-20261030.json) retains source hashes, per-case
geometry, mean/P50, assumptions and original comparison receipts.
Measured family activity and isolated operator sums are separate
diagnostics: neither is added to body span. Communication wrapper
and HCOM stream activity must not be counted twice. A symmetric-DP
input declaration does not establish that every captured DP rank
had a request; inspect each row's retained assumptions.
<!-- current-body-evidence:end -->
