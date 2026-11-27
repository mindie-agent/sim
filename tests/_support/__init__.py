"""Import-only support package shared by the product-orthogonal test domains.

[Role] Hold the single definition of cross-domain test helpers (parametrize
  axes, temp-dir ownership, accelerator/program builders) so no test module
  imports another test module.
[Invariants] Contains no tests; every module has at least one importer;
  SKU / ISA / quant literals live only in ``axes.py``.
[Timeouts & Locks] None.
[Side-Effect Policy] Import has no side effects.
"""
