# Explicit observed native service

The default `service_regimes` response is an inclusive completion floor over
the analytical program. An explicitly selected `joint_native` response instead
runs one complete local native call on a declared service queue. This supports
an observed operator service that is faster or slower than an analytical
schedule, including a cold-scratch model that overestimates real traffic.

```json
{
  "id": "declared-sparse-native-domain",
  "match": {
    "hardware": "Ascend910_9362",
    "recipe": "latent_attention_batch",
    "metadata.indexed_attention_geometry.cache_dtype": "int8"
  },
  "response": {"fixed_us": 100},
  "execution": {
    "kind": "joint_native",
    "native_kernel": "KvQuantSparseFlashAttention",
    "resource": "joint_sparse_service",
    "reference_rate": 1000000000
  },
  "provenance": {"kind": "syntax_example_not_calibration"}
}
```

The example is only syntax; 100 µs is not a measured parameter. A real rule
must declare the actual native input, representation, implementation and shape
domain, with its observation source and calibration/evaluation role. The usual
physical `match`, coordinates, domain and response terms apply. Model names,
layer IDs and measured Body residuals are not matching or fitting coordinates.

The device declares `rates.joint_sparse_service = 1e9`,
`work_units.joint_sparse_service = "busy_ns"` and
`resource_units.joint_sparse_service = 1`. The reference supply converts the
response to queue work. A 100 µs response contributes 100,000 busy ns. Calls
sharing this resource on a device contend even when issued on separate streams;
different devices and explicitly separate queues are independent. Queue grouping
is an explicit scheduling assumption, not occupancy inferred from a duration.

The same Engine retains input readiness, stream order, graph dependencies and
normal host submission. The response owns device startup and tail; invocation
parameters are not stacked onto it. The replaced source schedule is not also
executed. Its FLOPs and traffic remain in `analytical_reference_work` and the
operator description. `declared_service` and `serviced` contain only the actual
joint-service demand. Execution `work_basis` marks reference work explicitly.
Internal resource and buffer occupancy remain unmodeled in this mode; a whole
native duration does not identify HBM traffic, cache hits or instruction rates.

The mode applies only to one local kernel. Multi-phase native groups and
collectives retain their source programs and transport dependencies; a joint
replacement on these boundaries raises a visible modeling error. Out-of-domain
rules leave the analytical path active and report the domain miss.

Changing the declared service capacity changes its queued duration. Response
terms may also declare `scale_with` for a justified hardware transfer assumption.
No HBM/AIC/AIV relationship is inferred automatically from a complete kernel
observation. Keep the analytical arm alongside the observed arm when comparing
hardware outside the measured domain.
