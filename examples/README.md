# A3 dual-model examples

All numeric values here are **declared inputs**, not A3 measurements. The
predictive path uses them to produce a reviewable estimate; the normative path
uses raw declared peaks to produce a semantic lower bound L and a feasible
target U. No `measured` field is fabricated.

```bash
python -m mindie_sim predict --input examples/a3_dense_predict.json
python -m mindie_sim predict --input examples/a3_mla_moe_predict.json
python -m mindie_sim should  --input examples/a3_should_dense.json
python -m mindie_sim search  --input examples/a3_search_dense.json
python -m mindie_sim pipeline-microbatches --stages 3,5 --microbatches 4 --transfer-ms 1
```

Calibration database and the measurement loop:

```bash
python -m mindie_sim calibration --db /tmp/mindie-calibration.sqlite inventory
python -m mindie_sim predict --input examples/a3_dense_predict.json \
  --calibration-db /tmp/mindie-calibration.sqlite --record-misses \
  --requests-output /tmp/mindie-measurement-requests.json
python -m mindie_sim calibration --db /tmp/mindie-calibration.sqlite manifest --limit 32
# After an external runner (mindie-bench/coordinator) produces results:
python -m mindie_sim calibration --db /tmp/mindie-calibration.sqlite ingest --results results.json
```

The `run` subcommand can execute the CPU closed loop locally or the
`torch-npu` runner when the coordinator has already injected a visible device:

```bash
python -m mindie_sim calibration --db /tmp/mindie-calibration.sqlite run --runner cpu --limit 4
# Managed device only; sim never selects or leases a card:
python -m mindie_sim calibration --db /tmp/mindie-calibration.sqlite run --runner torch-npu --limit 1 --run-id external-run-42
```

`source_class=measured` is accepted only from a successful runner/ingest path;
CPU samples are `derived` and do not calibrate the predictive model.
