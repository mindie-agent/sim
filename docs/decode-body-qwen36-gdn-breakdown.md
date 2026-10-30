# Qwen3.6 Decode：GDN 结构层残差与通信去重

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](verification.md).

完整 64 层、A3 TP4、query=4、sequence=1、代表 context=4096 的
[同几何对照](decode-body-qwen36-defaults.md)表明：整步剩余差距 5.420800 ms 中，
48 个含 GDN 的结构层占 4.965378 ms。本报告继续拆解这 48 层中的投影、递归、
FFN、小算子和通信，不将结构层差距整体归因于 GDN 核。

对原始 12.24 MB `kernel_details.csv` 做两次有界流式扫描，使用 profiler 已有的
完整拓扑对齐逻辑恢复全部 15 次 Forward 的层边界。720 个 GDN 结构层的正耗时事件数
逐层均与 profiler 声明一致，主体/层统计与此前 receipt 完全一致。

## 通信的重复视图

这 720 层含 1,440 条 `hcom_allReduce_` 设备通信和 1,467 条
`allreduceAicpuKernel` AI_CPU 包装事件。后者记录跨越算子执行/等待的调度包装，
不能再加到 HCOM 耗时上：

| 边界 | 实测每 Forward ms | 预测每 Forward ms |
|---|---:|---:|
| HCOM 设备通信活动 | 1.337940 | 1.470093 |
| AI_CPU 包装活动之和，仅诊断 | 11.767938 | 不配对、不相加 |

真实 HCOM 通信预测约高 9.88%，它没有造成主体低估。包装事件保留原始计数和时长，
没有按流名称删除观测，也没有把其重叠等待视作额外网络工作。

## 显式算子/子图配对

以下均限定在 48 个 GDN 结构层。时间是每次完整 Forward 的活动之和，预测端取
已保存的真实 graph timeline；该代表图中这些调用串行。单次均值仅在一对一边界下
用于误差判断，多调用子图明确保留不同调用数。

| 配对角色 | 实测 ms | 预测 ms | Signed Gap ms | 实测单次 / 预测单次 us |
|---|---:|---:|---:|---:|
| materialization：Slice/Zero 与输出视图 | 0.722389 | 0.000983 | -0.721406 | 每层 8 次 vs 1 次，不作单核比值 |
| GDN output projection | 1.029297 | 0.319260 | -0.710037 | 21.444 / 6.651 |
| RecurrentGatedDeltaRule | 1.233098 | 0.576311 | -0.656787 | 25.690 / 12.006 |
| GDN gating | 0.521653 | 0.120000 | -0.401653 | 10.868 / 2.500 |
| FFN down projection | 1.296605 | 0.896427 | -0.400178 | 27.013 / 18.676 |
| GDN QKVZ projection | 1.175170 | 0.843466 | -0.331704 | 24.483 / 17.572 |
| GDN beta/alpha projection | 0.514912 | 0.240000 | -0.274912 | 10.727 / 5.000 |
| CausalConv1d | 0.392269 | 0.120000 | -0.272269 | 8.172 / 2.500 |
| FFN gate/up projection | 2.040481 | 1.790278 | -0.250203 | 42.510 / 37.297 |

其余 norm/quant/gate 子图和通信合并后，全部非包装活动为 **12.030165 ms**，预测
7.696818 ms，差 **4.333347 ms**。层跨度剩下 **0.632031 ms** 未被这些设备事件覆盖，
保留为内部同步/派发/等待尚未分解的区间，不拟合到 kernel 或带宽。

这里所有一对一普通算子的**单次均值都不超过 50 us**；最大的 gate/up 均值为
42.51 us。部分单次异常高值仍在 JSON 的 max/count 中保留。大部分一对一绝对差
落在 15 us 容差内，但很多小差距在 48 层重复后仍使主体误差超标。单核“受控”
不能替代主体 Forward 的 15% 验收，也不能把 12.66 ms 的结构层整体冠以“GDN 核”。

## 真实量化表示差异与第三条件

15 次 Forward 中每个矩阵角色有 720 次一致的 shape/dtype 证据：

| 角色 | 实测输入 → 输出 | 实测 dtype/layout | 原声明 |
|---|---|---|---|
| qkvz | `[4,5120] → [4,4096]` | INT8×INT8，NZ 权重 `[128,320,16,32]` | INT8，正确 |
| beta/alpha | `[4,5120] × [24,5120] → [4,24]` | BF16×BF16，ND | INT8 |
| output | `[4,1536] × [5120,1536] → [4,5120]` | BF16×BF16，ND | INT8 |

公共前端现已提供继承 `gdn` 的 `gdn.qkvz`、`gdn.beta_alpha`、`gdn.o` 精度角色。
第三条件通过独立 case 输入，仅把后两个角色的 weight/activation format 声明为
BF16；原 case 和全模型族默认保持不变。完整重新执行结果：

| 条件 | 主体 ms | 对实测 16.659023 ms 的差距 |
|---|---:|---:|
| 新默认、原精度声明 | 11.238223 | -32.54% |
| 新默认、观测精度声明 | 11.430704 | -31.38% |

增加 **0.192481 ms**，没有把差距强行补齐：output projection 从 6.651 增至
13.161 us/call，增加 0.312481 ms；48 次多余 output activation quantization 随 BF16
输入消失，减少 0.120000 ms。beta/alpha 仍受已有 5 us 调用下界限制。

第四条件在 `bc375e61` 的 per-owner recurrent AIV copy 修复后，使用同一输入重新
执行，得到 **11.460154 ms（-31.21%）**。相对第三条件仅增加 **0.029449 ms / 0.2576%**；
该物理修复影响小于 1%，不将其描述为主体精度收口。逻辑工作量不变，新的资源服务
差值、每个 recipe 的跨度变化和运行前后 package hash 单独保留。

该表示改变保持 FLOPs 50,312,708,096、link bytes 7,864,320 不变；HBM bytes 从
6,429,900,032 增至 6,811,701,248，向量工作随取消量化减少。当前 runtime 重建原精度
case 的逻辑工作量与此前保存的预测严格一致。第三条件在 commit `56079db0` 的
217 文件稳定 package hash `6d7a0b2c9144d1bbdb1203c53f0063f13fa13619fe076909a2ac9962cd1abeed`
上运行 11.55 秒，RSS 54.52 MiB，live tasks/programs/buffers 为零。

下一步应检查 materialization 的真实调用边界、递归核每个活跃 owner 的搬运管道、
小矩阵及调用粒度；内部未覆盖区间应继续作为独立证据问题。上述结果没有支持
增加全模型倍率、GDN 核倍率或额外通信底板。

## 文件与复现

- 720 层配对与原始 contract
- 流式分析 producer
- 观测精度输入
- 第三条件完整结果
- 第三条件 producer
- 第四条件结果
- 第四条件 producer

```sh
.venv/bin/python data/verification/decode-body-qwen36-gdn-breakdown-20261030.producer.py
.venv/bin/python data/verification/decode-body-qwen36-observed-contract-20261030.producer.py
```

这些证据沿用原采集的单 rank、代表初始 context、条件拓扑对齐限制，未证明高并发、
强制均衡、高接收率或 100% prefix cache 命中。没有新增远端或硬件采集。
