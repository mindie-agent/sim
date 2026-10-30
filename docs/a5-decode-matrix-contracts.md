# A5 Decode 主体矩阵合同对照

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](verification.md).

2026-10-30，成本运行时4499695a，导出合同747c4f5f。读取 profiler 已确认的287个完整主体窗口，逐条筛选原生矩阵；不重新分段，不纳入 MTP、Embedding 或 LM head。六个 case × tiled/steady 两个明确候选完成，39组 M/N/K、两侧操作数 dtype 与每个 Forward 的调用数全部匹配，无矩阵跨出主体边界。所有case运行前后package hash一致。

表中是主体内矩阵 activity sum 与独立矩阵预测之和，不是主体关键路径，也不是其他算子误差的替代。NZ、scale ABI、cache/实际 dispatch 仍作为未完全匹配的合同保留；不以名字相同宣称二进制等价。steady 是明确资源程序候选，未按实测耗时为每个模型选择。

| 场景 | 实测矩阵 ms/body | Tiled ms | Steady ms | Tiled signed gap | Steady signed gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen36-a5-tp8-c4 | 2.661027 | 1.686996 | 1.656835 | -36.60% | -37.74% |
| qwen36-a5-tp8-c40 | 4.252140 | 4.173681 | 2.976114 | -1.85% | -30.01% |
| qwen36-a5-tp8-c128 | 6.498053 | 7.654338 | 6.751864 | +17.79% | +3.91% |
| qwen38-a5-tp4-c4 | 4.134083 | 2.899640 | 2.849597 | -29.86% | -31.07% |
| qwen38-a5-tp4-c32 | 5.716802 | 5.559978 | 4.768452 | -2.74% | -16.59% |
| qwen38-a5-tp4-c128 | 11.544615 | 15.675703 | 12.336099 | +35.78% | +6.86% |

两种程序的误差方向会随工作量变化，不能把矩阵统一增加6–7us来收敛整网。例如Qwen3.6 c40的Tiled矩阵总和接近实测，但个别矩阵仍存在正负抵消；Steady减少了部分开销，反而扩大该条件的总低估。

Qwen3.8 c128重型矩阵：QKVZ实测55.58us，Tiled80.24us、Steady51.85us；gate/up实测64.80us，Tiled93.74us、Steady77.18us。对应的程序选择必须随结果报告，不能把其他探针的预测数值混入当前默认case。普通单层body图另行核对确认Tiled的80.24/93.74us与独立算子一致。

每个角色的调用数、原始shape/dtype/layout、预测metadata、单次Mean/P50、绝对/相对差距和原始文件SHA保留在 JSON。这是抓取大项的条件对照，不改变算子资源速率或完整body验收门槛。

```sh
.venv/bin/python data/verification/a5-decode-matrix-contracts-20261030.producer.py \
  -o /tmp/a5-decode-matrix-contracts-current.json
```
