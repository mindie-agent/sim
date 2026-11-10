# mindie-sim 0.9.10

[中文](sim-0.9.10.md)

This maintenance release repairs publication and automatic-update delivery. It retains the 0.9.9 simulation engine and makes no new model-accuracy or NPU-performance measurement claim. Historical experiments retain their original versions and provenance.

- Publication starts from protected main through `release.yml`, after all source CI succeeds.
- Packages, clean-source benchmarks, checksums and old-updater acceptance are prepared in a draft. Downloaded artifact bytes are verified before tag creation and immutable publication.
- The released updater performs discovery, actual dependency installation, generation construction and CLI probes in isolation. A second upgrade uses the public feed. Both reports remain with the workflow; `upgrade-acceptance.json` ships with the release.
- Automatic updates continue to target the newest formal components without lowering installed version floors. Failed publication cannot be reported as up to date.

The attached reports and corresponding CI record the checks actually executed for this version.
