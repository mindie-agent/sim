# One effective declaration for body prediction and comparison

`predict(scope="body")`, `target_graphs(scope="body")`,
`target_operators(scope="body")`, structure export, cohort comparison and body
studies now resolve the same supported attention/GDN defaults before lowering
the frontend. This fixes a semantic mismatch: a plain body prediction could
execute the source GDN caller while the exported structure still described the
generic recurrence and omitted caller materializations.

The shared selector uses declared geometry and physical resources. It does not
read observations or select a recipe by model name or latency residual. Existing
explicit `recipes`, `programs`, `families` and `lowerings` retain priority. No
new default is installed for `scope="model"`.

Body target collections retain the effective kernel declaration once, through
dict-compatible `TargetGraphSet` and tuple-compatible `TargetOperators`.
`graph_costs` and `operator_costs` consume that declaration, using the hardware,
service parameters and explicit kernel entries supplied to the cost call. A
hardware sweep therefore keeps its new resource rates; it does not resurrect
the hardware used when constructing the graph. Recipe overrides replace a
whole recipe entry, so an envelope override cannot inherit stale tiled-program
options. A graph remains a fixed declared implementation: regenerate it to
reselect resource-dependent tiles for a different hardware capacity.

An explicit `RecipeBook` passed to prediction or costing is retained. For a
custom book that owns frontend implementation selection, construct targets
with `apply_body_defaults=False`; `predict(..., recipes=book, scope="body")`
does this automatically. Plain manually supplied operators remain plain and
are never silently re-lowered. If reconstructing operators into a plain tuple/list, also retain their
effective declaration or
pass the corresponding book; collection conversion intentionally has normal
Python semantics and does not copy hidden state into every operator. Natural
single-rank extraction, rank-map filtering, operator slicing and shallow/deep
copy preserve the declaration; filtered rank maps require consistent declarations.

`mindie_sim.structure` and the `graph`/`operators` CLI commands accept
`--no-body-defaults` for a declared generic control. Studies accept the same
choice as `apply_body_defaults` in a probe. The default and its selection
record are included in body structure identity. Cohort comparison regenerates
that declaration and checks `graph_id`; old generic mappings without an
explicit opt-out fail with
`cohort_case_graph_id_mismatch:regenerate_mapping_for_this_case`. They must be
remapped from the retained observations. No old duration or match is silently
attached to a different source branch.

Validation executes the ordinary Engine rather than comparing only config
strings. It covers equal body/graph timelines and service for source GDN,
staged GQA in isolated and graph paths, independent hardware slowdown, explicit
books and overrides, unchanged model scope, real retained profiler norm rows
through mapping/comparison, rejection of legacy graph identity, and study/CLI
execution. The combined affected suite passes 84 tests. These checks establish
branch consistency; they do not establish measured body accuracy.
