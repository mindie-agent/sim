"""Independent dataset-assembly and frozen-holdout scoring for the A2/A3 accuracy round.

[Role] Offline evidence tooling that sits *outside* ``mindie_sim/``: it assembles one
  observation dataset from measured artifacts, freezes predictions against a declared
  geometry-only query manifest, and scores the frozen file against the frozen outcomes.
[Invariants] The scorer never imports model code, never fits, and never reselects a
  variant; every metric is recomputed from frozen artifacts whose hashes are recorded
  before any outcome is read. A heldout geometry is never admitted as train.
[Timeouts & Locks] No device, network or subprocess work in the library modules; each
  CLI is a bounded single pass over local files.
[Side-Effect Policy] Reads measurement/prediction artifacts; writes only the output
  directory handed to it by the caller (never a measurement dataset or a calibration DB).
"""

__all__ = ["metrics", "assemble_dataset", "freeze_predictions", "score_frozen"]

__version__ = "1.0.0"
