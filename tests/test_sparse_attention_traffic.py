"""Independent integer and explicit-selection oracles for sparse reads."""
from mindie_sim.prediction.sparse_attention import (
    selected_query_owners, sparse_attention_traffic)


def test_different_queries_cannot_share_an_index_set_by_tile_membership():
    selections = ((0, 1), (2, 3))
    result = sparse_attention_traffic(2, 4, 2, query_tile=128)
    assert result.gather_rows == sum(len(selection) for selection in selections) == 4
    assert result.index_read_bytes == 16
    assert result.window_sweep_rows == 0
    # A dense tile sweep sees only K=2 rows and loses half of these real gathers.
    assert result.gather_rows > min(4 + 2, 2)


def test_compressed_selections_and_dense_window_follow_separate_read_paths():
    for query in (0, 1, 3, 7, 17):
        for context in (0, 1, 6, 31):
            for ratio in (1, 4, 8):
                for topk in (1, 3, 16):
                    for tile in (1, 4, 128):
                        result = sparse_attention_traffic(
                            query, context, topk, compression_ratio=ratio,
                            window=5, query_tile=tile)
                        selected = sum(min((context + index) // ratio, topk)
                                       for index in range(1, query + 1))
                        windows = sum(min(context + min(query, start + tile), 5)
                                      for start in range(0, query, tile))
                        assert result.gather_rows == selected
                        assert result.window_sweep_rows == windows
                        assert result.total_cache_rows == selected + windows
                        assert result.index_read_bytes == selected * 4


def test_large_geometry_stays_exact_without_materializing_query_history():
    query, context = 2**55 + 7, 2048
    result = sparse_attention_traffic(query, context, 2048)
    assert result.gather_rows == query * 2048
    assert result.index_read_bytes == query * 2048 * 4


def test_4k_prefill_selected_rows_preserve_causal_ramp_and_topk_saturation():
    result = sparse_attention_traffic(4096, 0, 2048)
    assert result.gather_rows == 2048 * 2049 // 2 + 2048 * 2048 == 6_292_480
    assert result.index_read_bytes == 25_169_920
    assert result.gather_rows * (512 + 64) * 2 == 7_248_936_960


def test_single_query_decode_owns_one_aic_task_not_the_whole_chip():
    # The source distributes query head-group tasks over AICs.  One decode row
    # is one task (one AIC plus its two AIV subblocks), while a 20-row decode
    # can occupy 20 AIC tasks.  Traffic remains per selected row either way.
    assert selected_query_owners(1, 20) == 1
    assert selected_query_owners(2, 20) == 2
    assert selected_query_owners(4096, 20) == 20
    assert selected_query_owners(0, 20) == 0
