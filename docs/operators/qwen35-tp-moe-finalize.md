# Qwen3.5 TP4 MoE prepare/finalize 的实测张量合同

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

完整采集中的 40 次 AllGather 位于 MC2 Combine 之后，恢复被 TP 切分的 routed 输出。它与使用 AllGather/ReduceScatter 的 EP 路由策略是不同机制。当前 Sim 对照图虽声明了 AllGather 源程序，但没有生成这 40 次调用；应修复 prepare/finalize 的结构边界，不能将已有 EP all-to-all 改名。

流式证据和可复跑 producer读取 profiler 已选完整 forward population，保留 CSV SHA、整数纳秒时间及行定位。三份 70–88 MiB CSV 均流式读取；只保留有限前后事件窗口和聚合计数。

| Client B（每序列 q=4） | 完整 forward | AG 设备调用 | Router 输入 M | TopK / Dispatch / Combine M | AG 推断本地 BF16 payload | AG activity / body | AG 后 TensorMove / body |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 84 | 3360 | 16 | 4 | 16 KiB | 0.422939 ms | 0 |
| 64 | 79 | 3160 | 256 | 64 | 256 KiB | 0.962846 ms | 0 |
| 128 | 76 | 3040 | 512 | 128 | 512 KiB | 1.206973 ms | 160 次，0.559880 ms |

每个 forward 均恰好 40 条链，全部张量合同检查通过：

1. Router MatMul 输入 `[4B,2048]`、输出 `[4B,256]`；其后 TopK 的输入已是 `[B,256]`，权重和 IDs 为 `[B,8]`。
2. `MoeDistributeDispatchV3` 接收 `[B,2048]` BF16；gate/down 是 INT8 权重与 INT8 激活的分组矩阵。记录中的 `[32B,2048]` 等输出仅证明分配容量，不证明有效专家行数或 group_list 数值。
3. `MoeDistributeCombineV3` 输出 `[B,2048]` BF16，然后 Block64 的 `hcom_allGather_` AivKernel。
4. Shared expert 使用完整 `[4B,2048]`，独立 TP AllReduce；最终 `ForeachAddList` 两个输入都是 `[4B,2048]` BF16。

c128 的每次 AG 后、shared expert 前，同一 AG stream 上还有恰好四个 `[128,2048]` BF16 `aclnnInplaceCopy` TensorMove。所有 3040 条链均如此；c4/c64 所选链均没有该调用。它们是独立的物理搬运事件，应保留为包装边界证据；还未证明触发该分支的 allocator/collective 条件，不能据此对任意 512 KiB 输入自动补四次复制，也不能把其时间加成网络启动底板。

## 固定版本源码对应

本地 `vllm-ascend` Git 对象 `8c30e4f491f73cba737a1dacc522894e437cdf3d`（未修改工作树）：

- `ops/fused_moe/moe_comm_method.py:215–232`：MC2 选择 `TokenDispatcherWithMC2` 和 `PrepareAndFinalizeWithMC2`。
- `ops/fused_moe/prepare_finalize.py:285–306`：`replace_allreduce=False` 时切分 MC2 mask、hidden states 和 router logits；padding 来自 forward context。
- 同文件 `316–325`：该分支继承 All2All finalize；`211–228` 新建完整输出、按 TP 切片，以 `dist.all_gather` 收集本地 hidden states，必要时裁掉 padding。
- `ops/register_custom_ops.py:127–155`：MC2 routed 输出不再次 TP reduce，shared expert 输出单独执行 TP AllReduce。

这给出当前采集形状链的明确源码解释，但不把仓库对象宣称为采集二进制身份。HCCL 行自身没有 tensor descriptor，也没有实际算法/协议配置；上表 payload 是邻接张量和源码共同支持的推断。设备 event 与 HCOM summary 的起止时间在全部 9560 次 AG 上一一相同，后者不计入 activity。activity sum、包装复制 activity 和完整 body critical path 分别报告，不线性相加声称整步误差改善。
