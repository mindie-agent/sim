# Qwen3.6 A3 TP4：同几何主体 Forward 对照

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](verification.md).

2026-10-30，完整 64 层 Qwen3.6-27B W8A8、A3 `Ascend910_9362` TP4。
公共 `predict(scope="body", phase="decode", tokens=4, context=4096, sequences=1)`
以输入 hidden/KV 已就绪为起点，排除 embedding、final norm/head、Host 提交和跨步等待。
两条件仅改变 GQA attention recipe；没有改时钟、带宽、量化、层数或调用几何。

| 条件 | 主体预测 ms | 实测均值 ms | Signed Gap ms | 相对差距 |
|---|---:|---:|---:|---:|
| 显式旧 `flash_attention_decode=envelope` | 10.035361 | 16.659023 | -6.623662 | -39.76% |
| 当前默认 `StagedGQADecode` | 11.238223 | 16.659023 | -5.420800 | -32.54% |

当前默认追回 1.202862 ms，尚未达到整步 15%。旧 envelope 的当前重跑结果与
profiler 历史预测完全一致。两条件逻辑工作量相等，均完成 1,408 个算子，Host
service 为零，结束时 live tasks/programs/buffers 均为零。默认实际执行 16 次
`StagedGQADecode`，query/key tile 为 64/64、两个窗口、默认一个 KV split；
声明的 MTE1/MTE2_AIC 独立管道确实产生服务量，未退回 envelope。

## 残差定位

下表按结构层归组，**包含对应层的投影、FFN 和通信**，不能解读为 GDN 核本身的误差。

| 结构层 | 层数 | 实测层跨度之和 ms | 默认预测 ms | Signed Gap ms | 相对差距 |
|---|---:|---:|---:|---:|---:|
| GDN attention 层 | 48 | 12.662196 | 7.696818 | -4.965378 | -39.21% |
| GQA attention 层 | 16 | 3.953385 | 3.541405 | -0.411980 | -10.42% |

剩余主体差距约 91.60% 落在 GDN 结构层。实测层跨度之和为 16.615581 ms，和主体
跨度相差 0.043442 ms；该层边界间隙保留，不塞入任何算子。GQA 层的改善完全来自
attention 调用跨度之和从 0.080000 增至 1.282862 ms，其余算子族不变。

当前默认预测中的主要算子族：

| 族 | 调用数 | Timeline elapsed sum ms |
|---|---:|---:|
| gate_up_projection | 64 | 2.387038 |
| allreduce | 128 | 1.960124 |
| gemm | 144 | 1.402725 |
| attention | 16 | 1.282862 |
| down_projection | 64 | 1.195236 |
| elementwise | 288 | 0.720000 |
| quantize | 256 | 0.640000 |
| recurrent_vector | 48 | 0.576311 |

这里的每行是实际图上 invocation-ready 到完成的跨度。该 TP4 单代表图恰好串行，
族跨度和等于主体耗时；这不是一般成立的可加算子成本定义，也不是实测 kernel
activity。报告同时保留 interval union、完整 timeline、每层观测与预测和每个 recipe
的调用均值，下一步可在不重复统计通信 wrapper 的前提下配对原始设备事件。

## 证据与复现

- 完整条件报告
- 可执行 producer
- [原始 case](../examples/cases/qwen36-a3-tp4.json)

每个条件使用独立本地进程，上限 59 秒；实际约 13 秒、最大 RSS 不到 60 MiB。
两个进程均校验运行前后 217 个包内 Python/JSON 文件，稳定 package hash 为
`3246745a5e426b986fce7ce4be7d0a190343ff1f183f17ef477acbf950c61e35`。
算例文件 SHA256 与 profiler 原始 receipt 相同；完整 normalized 资源已随结果保留。
执行任务数为默认 1,557,459、旧 envelope 1,548,899，说明该整图仍有默认矩阵程序的
任务展开成本，但本次没有超时或内存超限。

```sh
.venv/bin/python data/verification/decode-body-qwen36-defaults-20261030.producer.py \
  --receipt /tmp/decode-forward-q36-first.json
```

实测均值来自现有采集的 15 次单 rank、完整拓扑对齐主体跨度；活跃 KV 增长未被恢复，
4096 是客户端 prompt 提供的代表初始 context。强制均衡、高投机接收率和 100% prefix
hit 均未在该采集中证实。这是条件对账，不代表高并发生产条件已经达标。
