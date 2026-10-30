# A5 高并发 GDN source caller 条件对照

2026-10-30，运行时 `d74cddc3`。六个新场景位于
`examples/cases/high-concurrency-20261030/source-gdn/`，保留完整模型和原始六个 case。
通过普通 `runtime.kernels.lowerings.gdn_decode` 选择
`vllm_ascend_gdn_decode`；实际执行第一层 GDN 调用的 15 个算子及其依赖图。
范围为投影结果就绪至 output norm/gate 完成，不含投影、通信、FFN、Host 提交或跨步等待。
本次 recurrence 仍是 analytic core，不能把结果表述为 native recurrence 已验收。

仅新增 `scalar_store_issue`：每 AIV 88,397,790.055 次/s，64 AIV 共
5,657,458,563.536 次/s；保留所有当前 A5 profile 的 HBM、MTE 和算力值。
参数来自已存在的独立 BF16 scalar-store primitive，8 个 outstanding slots、
128B address granule、91.25ns response、7ns row control，输入路径为
`data/observations/gating-probe-20261028/native-queue-holdout/parameters.json`。
未读取新采集时延来设置或修改任何模型参数，未增加 A5 通用 kernel floor。

校准使用 Ascend950DT_9582、镜像
`sha256:c3d00a196905159dbf0b97588d34f085086c2708712fa0d1c8fedd807f0e48a4`。
新 capture 的实际 gating binary 尚未逐一核验。保留编译产物支持该连续布局下
FP32 g 走 DMA、BF16 beta 走 scalar queue；当前结果是既有 primitive 参数向新
heads=6/12 地址布局的机制应用，不是新二进制或完整调用的独立精度认证。

| 场景 | shape 匹配完整调用数 | 实测 gating Mean us | 无队列预测 us | 有队列预测 us | 有队列相对差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 TP8 c4 | 3504 | 12.731 | 0.072 | 8.215 | -35.47% |
| Qwen3.6 TP8 c40 | 2592 | 41.796 | 0.098 | 32.009 | -23.42% |
| Qwen3.6 TP8 c128 | 1824 | 44.290 | 0.098 | 32.009 | -27.73% |
| Qwen3.8 TP4 c4 | 2160 | 22.255 | 0.145 | 15.176 | -31.81% |
| Qwen3.8 TP4 c32 | 2304 | 68.790 | 0.196 | 56.488 | -17.88% |
| Qwen3.8 TP4 c128 | 1392 | 75.749 | 0.196 | 56.488 | -25.43% |

前四种 gating 单次实测低于 50us，绝对差 4.52–12.28us，在 15us 容差内；
Qwen3.8 c32/c128 属于大调用，相对误差仍未达到 15%。不将微算子受控表述为整个
caller 或 body 达标。Qwen3.8 c128 的 convolution 预测 6.98us 对实测39.37us，
analytic recurrence 预测231.67us对实测183.80us，仍明确保留。

每组 CSV 均流式读取，仅选 `prof/rank0/kernel_details.csv`，不重复叠加顶层文件别名。
根据 recurrent 的 Q/K/V、batch 元数据 shape，关联同流的 convolution、gating、
两个 L2norm、输出拷贝与 norm。所有六组均匹配 16 个观察角色、每角色调用数完全相同，
没有不完整链或不同 geometry 被静默纳入。gating/L2norm 本身无 CSV shape，报告明确
其 geometry 是来自相邻有 shape 的 convolution/recurrent 和源码顺序，而非时延匹配。
观察覆盖所有匹配层；预测是普通前端首个 GDN 层，不声称观察定位仅第一层。

其中 15 个观察角色与预测显式配对，另一个是 `(512,4) → (batch,1)` 的状态索引
Slice，单独记录为 `unmodeled_state_index_copy`，没有冒充现有 payload copy。
不读取通信行参与这些调用耗时，也不计任何重复 HCOM wrapper。实测各族 activity
之和与预测图 span 分开保存；这里没有从调用统计推断完整 forward 或 TPOT。

12/12 条件完成，单 worker 0.12–0.70s，峰值 RSS 43.59MiB；零残留任务、程序和 buffer。
每次运行前后 package fingerprint 一致：
`6aa0161d6ce560db5522aa8c2458c2137ce398b2c7f47dc541dd919ae2e68d3f`（219 文件）。
完整原始模拟输出、各角色 Mean/P50、signed/absolute gap、shape/dtype/BlockNum 示例、
输入和参数 SHA 均在 `data/verification/a5-gdn-source-caller-20261030.json`。

```sh
.venv/bin/python data/verification/a5-gdn-source-caller-20261030.producer.py \
  -o /tmp/a5-gdn-source-caller-current.json
```

复跑使用当前 runtime；若 recurrence 后续更新，应写新报告并保留本次 analytic-core 对照。
