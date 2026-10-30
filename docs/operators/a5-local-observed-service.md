# A5 local service responses

`data/observations/a5-local-service-regimes-20261030/producer.py` assigns five
native-call responses from the retained B16 and B64 populations. B32 supplies
an unused-in-this-assignment check; these previously examined records are not
pristine heldout data. No Body duration enters parameter assignment.

| Native service | B32 response error |
| --- | ---: |
| Noncontiguous state gather | +0.97% |
| Noncontiguous state scatter | -2.80% |
| MXFP8 routing | +0.99% |
| AICPU mesh OneShot AllReduce | +2.44% |
| MXFP8 grouped SwiGLU | -0.009% |

The parameters retain direction, tensor dimensions, expert counts, collective
rank count and native-boundary provenance. The variable gather/scatter terms
scale with MTE2_AIV/MTE3 respectively; routing and fused SwiGLU variable terms
scale with vector rate. This is a declared effective-service scaling assumption,
not an isolated measurement of those pipes. The AllReduce response is a fixed
protocol response; the physical source program still prices link/HBM work.

The complete 45-layer control uses all five rules: 34 gather, 34 scatter,
90 AllReduce, 42 routing and 42 fused SwiGLU queries, with no unmatched rule.
Its c16/c32/c64 Body spans are 25.008/29.353/38.769 ms, errors
-24.29%/-22.08%/-16.68%. These controls remain below the measured Body times;
they do not calibrate convolution or recurrence. More complete source cases
and their results must be reported separately rather than replacing this
partial calibration result.

Run the producer, then:

```sh
python tools/replay_service_regime_cases.py \
  data/observations/a5-local-service-regimes-20261030/cases.json \
  -o data/verification/glm-observable-service-20261030.json
```

The replay uses the ordinary Body graph and causal Engine, retaining physical
work, query counts, resource cleanup and package/input fingerprints. Each
complete case has a 60-second process budget.
