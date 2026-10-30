# GDN decode state-index materialization

The selected `vllm_ascend_gdn_decode` caller now accepts the explicit option
`cache_index_row_stride` (default `1`). The retained `gdn.py.txt` source
(`e9ed210b4ca4431ee40916f06f0272e5e781eb04b61b084bacea216799fbca98`,
lines 138–143 and 435–445) selects `cache_indices[:, 0]` and makes that view
contiguous before passing it to the convolution update.

This is an INT32 input-layout contract. A row stride greater than one with
multiple requests creates one `state_indices_contiguous` vector invocation.
It reads and writes four bytes per request, retains the actual source strides,
and gates only the convolution. A contiguous column or singleton request needs
no materialization. Query rows, graph padding, and the allocation's unused rows
do not increase this payload. The existing tensor-copy execution recipe prices
the transfer; no measured duration or new launch floor is inserted.

The six high-concurrency A5 captures have an INT32 `[512, 4]` backing table and
`[batch, 1]` selected output. The six **native** scenarios under
`examples/cases/high-concurrency-20261030/source-gdn-native/` therefore explicitly
declare stride `4`. Their preparation plan records the changed input and fresh
scenario identities. The earlier analytic-core scenarios and their measured
comparison report remain unchanged. Native body runs are still owned by the
profiler session; this preparation does not claim a new body result.

Focused validation: 21 tests passed, including variable query lengths, q=1,
graph padding, singleton views, exact 16-byte read/write for four requests,
global input readiness, convolution-only dependency, and complete task/buffer
retirement. These validate the physical boundary, not latency accuracy for the
small copy. The unresolved high-batch ACL output-layout moves remain separate.
