"""Independent conservation oracles for the model catalog.

[Role]
  Package root for reference implementations that are derived only from raw
  model JSON configs and the pinned constructor-shape declarations.  Nothing in
  this package imports the simulator's weight, cache or ownership helpers.

[Invariants]
  * Oracle modules import only the standard library, the sibling oracle module
    and (lazily) ``torch``.
  * Every parameter is a real ``torch.device("meta")`` ``nn.Parameter``; no
    device context, weight load or raw pointer access happens here.

[Timeouts & Locks]
  Pure process-local parsing and arithmetic; no locks.

[Side-Effect Policy]
  Reads repository config/manifest files; never writes.
"""
