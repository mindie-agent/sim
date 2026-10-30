# V4.1 DP4 的当前 body 输入审计

旧 `v41-a5-dp4*.json` 六个输入保留不变。新 `examples/cases/v41-a5-dp4-body-current.json` 明确使用当前 A5 SKU 管道、264 B/row 的 host FP8+E8M0 Engram 表表示，保留原 TP1 / DP4 / EP4、96 GiB HBM、链路与主存/H2D 标称假设。

审计和可重复 producer 在 `data/verification/v41-current-body-input-audit-20261030{.json,.producer.py}`。此输入是完整 40 层模型；目标是调用边界和参数兼容，**没有沿用旧 Step -4% 等精度结论**，也没有把 measured pool / aggregate dequant 耗时导入新预测。

层数有部署 checkpoint 的直接证据：`/Volumes/data/mindie-sim-data/a5/2026-10-23/v41-flash-a5/env/checkpoint-structure.json`，SHA256 `c9328a7a7d03ece026939aab03f3bfe0ee5519e7dc926c695ff6d3b31c05da44`。其中 `model_dir=/mnt/weight/DeepSeek-V4.1-Flash-1017`、`text_config.num_hidden_layers=40`、384专家/top6、Engram层1/14和MTP3均与本地config一致；alt采集副本同SHA。其header-only probe源SHA为 `c0fd10627cdd1747077add299b98660721bc8b1868714e5f8ee19226b46938cb`，没有读取权重值。它未保留原始config字节SHA，因此本地config SHA `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`不冒充部署configSHA。**body分割应对齐40个Transformer层；不能按口述80层改结构。** 80是否指Attention+FFN两类子块尚未证实。

## 实际生效的变化

| 输入 | 旧声明 | 当前输入 | 含义 |
|---|---:|---:|---|
| `launch_us` | 20 µs | 当前 profile 0 | body API 排除 Host submission；服务提交应另行显式配置，不用旧20µs/op |
| `mte2_aic` | 4 TB/s | 4.224 TB/s | 删除旧通用 `mte_bw_bytes_per_s` 覆盖，恢复独立管道 |
| `mte2_aiv` | 4 TB/s | 4.224 TB/s | 同上 |
| `mte3` | 4 TB/s | 13.5168 TB/s | 恢复当前 profile 的写方向服务 |
| Engram host row | 512 B，device BF16 表声明 | 264 B，FP8+E8M0 | 存储表示修正；输出仍是512 B BF16 |

这里没有调整算力、HBM、链路或容量来匹配实测。主存64 GB/s、host vector50 Gops/s、H2D32 GB/s沿用旧输入的明确标称假设，并不冒充本机标定。普通 linears 的 FP8、group32 MXFP4 专家、BF16 KV、INT8 indexer都保持 catalog 声明。

旧输入的 `mxfp4_dequant_row`、`measured_rows`、`kv_pool_tokens`、`kv_pool_blocks`、`dp_step_admission` 等只存在于 metadata。当前源码不把这些注释作为参数读取，部分注释声称会拒绝不同dequant调用数/按测量池准入，已不符合当前行为。新输入移除这些注释，原值在旧文件和审计中保留；没有把“缺少测量”转成解析失败或禁止分析。

## Engram 不能只按 DP 数推断通信

保留的实际日志：`.workstreams/hp-serving/v41-flash-a5/async-off-16k/raw/remote/logs/serve.log`。

- line23/30：`cpu_offload=True, embedding_across_dp=False, dp_shared_memory=True`。
- line310–317：每个 Worker 都记录完整 384,006,168 / 384,016,682 rows × 256 的两张 registered-host 表，各约94.42 GiB。这符合264 B/row的源表示；日志中的“per rank”是各进程的可见映射，不足以证明物理主存为每个DP复制一份。
- checkpoint header另直接记录 `layers.1.engram.embed.weight` 为F8_E4M3、shape[384006168,256]，scale为F8_E8M0、shape[384006168,8]，因此264 B/row不是用日志耗时推算。
- `dp_shared_memory=True` 不等于每次查询需要跨DP交换行。当前输入不根据DP4凭空添加Engram跨DP通信。

两个已检查的源码分支有不同适用域，不能混成旧binary的已验证实现：

1. `vllm-ascend-v41-private` revision `ae923905948f360f4b7ab06df52d0789a0c9316e`，`engram_hbm.py` SHA256 `6db8f20f5d5608ca19707b4ebbdc4e84a45f43ab871c25034da40a0fd7afee5c`。line85–93明确只支持TP8；节点内ID all-to-all、owner lookup、reverse BF16 payload和TP broadcast分别在line336–402。这不是旧TP1 CPU-offload的合同。
2. `refs/inferact-sra-tracking` revision `0a5cb4e592a4a35a92f15b445cd3cc5340635f6e`，`common/engram.py` SHA256 `41c5bdf25cf8337088247be768e3d43fa41e6358397800f8b602ebf1ede69480`。line585–605聚合DP hash，line683–808采用按hash-head分片和UVA，line979–996 / 1093–1104用all-gather把token换回heads。该较新CUDA分支也不能替代旧A5的 `embedding_across_dp=False` 实现。

实际旧binary的 `embedding.py` 尚未在已留存源码中找到。因此CPU解码+H2D与UVA边界、共享host带宽归属、hash/history准备是否在实测body内仍显式未决。通用 `catalog/engram.py` 的TP-only ID比例和zero-byte reverse join不足以表达上面两个非本次合同的分布式分支；后续应按已证实选择的实现补机制，不能直接把DP当TP或为当前case加入虚构传输。

## 调用验证

短 `target_graphs(scope="body", phase="decode", tokens=1, context=2048, sequences=1)` 验证完整四副本EP cohort：8,256个节点、每副本40层、每副本两次Engram主存读取，每次24×264 B stored、24×512 B staged。没有裁剪模型层数。

`predict(scope="body")` 的度量边界是输入hidden/KV ready到整个参与cohort输出ready；其中真实依赖的Engram主存路径属于body工作，Host逐算子提交、采样、MTP生成、跨step等待不属于该边界。序列数是**每个参与副本**的数目：本例sequences=1代表四个DP副本各一个声明请求，不能冒充历史c1只激活单个DP的窗口。

初次旧control执行已在根线程要求避免与全量测试竞争时主动取消，未保留数值、不算成功或超时。全量测试结束后已执行当前完整body probe，达到60秒预算超时，子进程已终止；没有预测耗时数值，也没有裁剪层数重跑。短图生成通过不代表完整执行已完成。该次超时仅保留执行结束后的package hash，开始hash未采集，未重构或假称全程锁定；producer已补齐后续超时的开始/结束hash记录。尚无实测body配对，任何后续数值也只代表该声明几何的分析预测。

```sh
.venv/bin/python data/verification/v41-current-body-input-audit-20261030.producer.py
# Full body only after the shared-runtime verification window:
.venv/bin/python data/verification/v41-current-body-input-audit-20261030.producer.py --run-body
```
