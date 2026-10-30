# Mixed source domains and device invocation classes

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

`program_execution.execution_class` is an explicit timing-class selector.
It is not a descriptive hardware label: `mixed` and `cube_vector` have no
ordinary A3/A5 timing declaration and therefore suppress the inferred
Cube/Vector invocation constraints. Four real programs were affected:
MXFP8 grouped SwiGLU quantization, PagedMLA, MQSMLA, and packed INT8 indexed
attention. Their descriptive label now uses `source_execution_domain`.

This leaves the existing policy to select the maximum declared Cube/Vector
startup once and the maximum completion minimum once from actual work.
The minimum starts at original input readiness and overlaps startup+body;
it is not added after them. Source work, stages, buffers and physical owners
are unchanged. Explicit user `invocation_class` and runtime invocation rules
retain precedence. No hardware parameter or fitted timing was changed.

`tests/test_mixed_source_invocation.py` reproduces all four cases through
actual frontend operators and the Engine. Each ignored a declared0.5ms
startup before the change. Trace controls now show one startup and one
inclusive minimum, no service before readiness, unchanged resource work and
zero live state. A supported tiny fused source inherits the existing A5
1.578µs minimum; it does not add that minimum onto useful service.

The GLM control receipt
prices the existing c16/c32/c64 q4,H4096 conditional source cases. Gate/up
spans remain163.5503/165.2034/168.4841µs respectively: all exceed the existing
minimum, so the correction does not move those large-operator predictions.
The receipt retains a4.34e−19 matrix-service summation roundoff in c16;
other serviced quantities match exactly. These are source operator controls,
not new capture accuracy or a whole-body result. Six new controls and69
related source/invocation tests passed.
