"""Reviewed operator planners; callers bind implementations through RecipeBook.

[Role] Namespace for independent source-derived operator mechanisms.
[Invariants] No implicit discovery, model-name dispatch or import side effects.
[Timeouts & Locks] No I/O or locks.
[Side-Effect Policy] Importing the namespace does not activate a recipe.
"""
