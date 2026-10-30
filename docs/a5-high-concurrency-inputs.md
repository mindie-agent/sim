# 新 A5 高并发采集：输入合同审查

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](verification.md).

2026-10-30 外部采集的六个目录已就绪。本次审查 metadata、源文件身份、每个 CSV
前 2000 行的矩阵 tensor 描述与完整模型输入。自动主体切片和最新 wheel 回放由
profiler 会话负责，避免重复执行。
本次没有新真机、容器或 DSH 操作，也没有把 Step/TPOT 当成 body Forward。

## 实际声明与未证实项

六个目录位于 `/Volumes/data/mindie-sim-data/a5/*high_conc_decode_tpot*20261030`，
共同镜像为 `quay.io/ascend/vllm-ascend:v0.30.0rc2-a5 (c3d00a196905)`。

| 模型 | TP | Client concurrency | prompt tokens | 总 prompt cache 命中率 | profile draft-token acceptance |
|---|---:|---:|---:|---:|---:|
| Qwen3.6 | 8 | 4 | 4104 | 99.8051% | 67.7928% |
| Qwen3.6 | 8 | 40 | 4104 | 99.8051% | 66.6667% |
| Qwen3.6 | 8 | 128 | 4104 | 99.8051% | 65.5496% |
| Qwen3.8 | 4 | 4 | 4104 | 87.3294% | 67.5725% |
| Qwen3.8 | 4 | 32 | 4104 | 87.3294% | 64.2432% |
| Qwen3.8 | 4 | 128 | 4104 | 87.3294% | 63.1045% |

4104 来自共享 prefix 4096 加独立 suffix 8，且 client 标记 `token_exact=true`。
Qwen3.6 平均 cached prompt tokens 为 4096/request；Qwen3.8 为 3584/request。
因此 Qwen3.6 的总 prompt 命中率略低于 100% 与 8-token suffix 一致；Qwen3.8
不能按 100% 共享 prefix 命中处理。profile 窗口开始/结束 inflight 数一致、零抢占，
但这些只是服务窗口证据，不能替代逐次 Forward 的实际 query/batch 提取。

profile 计数支持每个 draft event 提议 3 个 token。上表是**聚合 draft-token 接收率**，
不是每个投机位置的配置目标概率；例如配置目标 0.8 和聚合接收率约 0.65 并不天然矛盾。
完整 server 参数、配置目标是否为 0.8、实际 target query 行数和 graph bucket 均未在
这两份 metadata 中证实，保留待解项。

context 的代表初值可声明为 4104；profile 窗口内已生成 KV 令其增长，真实动态值仍
待独立证据。不能用缓存容量、client concurrency 或聚合 tokens-per-step 代替实际
context、sequence count 或 query rows。

## 旧例子不能直接套用

- `qwen38-a5.json` 仍为 TP1、单 die；本批是 TP4。旧例子还有专用
  `scalar_store_issue` 和通用 MTE 覆盖，不能无说明地传到本批次。
- `qwen36-a5.json` 的 TP8 正确，但 `mte1_bw_bytes_per_s=4e12` 和
  `mte_bw_bytes_per_s=4e12` 覆盖了已更新的 profile。新 SKU profile 的每 die
  MTE1/MTE3 为 13.5168 TB/s，MTE2_AIC/AIV 为 4.224 TB/s。
- 新目录的权重路径/镜像不同于旧 A3 对账；不能把旧 A3 的 GDN BF16 role 修补
  自动当作新 A5 checkpoint 的事实。本批逐目录检查前 2000 行的 shape/dtype：
  Qwen3.6 BA/O 为 BF16、QKVZ 为 INT8；Qwen3.8 三项均为 BF16，与其目录声明一致。
  因而仅本批 Qwen3.6 case 增加 `gdn.beta_alpha/gdn.o` BF16 role 覆盖。

有限 tensor 抽样还观察到所有六项 FIA 的 31-slot ABI，slot14 block-table batch 分别
为 4/40/128 和 4/32/128，query rows 为对应 batch 的 4 倍。这里只确认这些样本的
query=4；尚未声称整个窗口没有其他几何。Qwen3.6 GDN/GQA QKV 的物理 shape 相同，
其 INT8 精度可以核对，但不能由 shape 唯一分配模块调用数；GDN output 另以最近的
RecurrentGatedDeltaRule anchor 区分相同 shape 的 GQA output。全部配对保留 locator，
不使用耗时匹配。

新增输入位于 [examples/cases/high-concurrency-20261030](../examples/cases/high-concurrency-20261030)，
均保留完整 64 层、使用当前 `Ascend950DT_9582` profile、声明实际 TP 与设备数。
这是 device-body 分析输入，不是原服务器启动配置的完整复刻。
`graph_mode=none` 表示调用者须传入 profiler 确认的物理 query 行数；它不表示原服务
使用 eager 模式。`workload` 中的单个请求是解析占位，不用于模拟该并发窗口。

## 源选择与下一步交接

六个目录的顶层 `kernel_details.csv` 与 `prof/rank0/kernel_details.csv` SHA256
逐个相同，分析只能选择一个；它们不是两条不同采集。新输入审查统一选择后者。
各 rank 文件是否存在也已记录；不能把 rank0 当作已经验证了全 rank 同时完成。

审查 JSON包含源 SHA、
metadata SHA、量化/模型声明、normalized 资源、旧例子资源差异、独立服务指标与缺项。
producer
对大 CSV 使用 1 MiB 块的流式哈希，并有界检查前 2000 行；不加载整文件、不执行
六图预测。精度、fusion 和 query 几何的全窗口覆盖仍由 profiler 回放确认。

profiler 下一步可复用六份 case：先确认 target body 的完整 64 层拓扑和实际 q/batch，
再用显式 context 条件调用 `predict(scope="body")`；通信 wrapper 与设备通信须保留
不同边界。历史 Step、客户端 TPOT 和有/无 profiling 的差值继续留作服务诊断。
