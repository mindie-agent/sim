# Strict execution replay

```bash
python -m mindie_sim execution-replay --input examples/execution-replay/mixed.json --output replay-result.json
```

The input wraps the existing synthetic `operator-plan` mixed example and declares
the intended single-call warm device-event protocol. `evidence` is empty: no NPU
measurements or accuracy claims are bundled. The original shared graph takes
**0.0550192 ms**; the three isolated fine durations are **0.011752**,
**0.0371472**, **0.006120 ms**. The serial predecessor waits add no extra
execution span to those kernels.

The native JSON contains full plan/config/hardware/protocol identities, hashes,
legacy and fine isolated times, actual graph spans and explicit fallback/error
coverage. Add only observations whose full executed identity and protocol have
been established; synthetic records must remain labeled synthetic. This command
does not import, fit or change runtime costs.

See [the identity and evidence contract](../../docs/execution-replay.md).
