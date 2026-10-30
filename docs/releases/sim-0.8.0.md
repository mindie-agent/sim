# MindIE Sim 0.8.0

0.8.0 将旧预测器替换为可组合的硬件中立建模系统。模型结构、算子执行、
通信与请求调度使用同一个因果 Engine；解析仿真不要求实测数据。

## 本版交付

- Decode Body、算子族、完整图与 Serving 分别输出成本和时间边界。
  `predict(..., scope="body", phase="decode")` 输出主体 Forward；
  默认 Serving 路径处理实际请求、MTP 接受/提交与 PD 的 KV handoff。
- A3/A5 的分核计算与搬运、有限缓冲流水、量化与专家分组、Paged/Sparse
  Attention、GDN/KDA、mHC/AttnRes 和显式源程序绑定；紧凑周期避免逐 panel
  展开产生数百万任务。
- 按声明区分 AIV、AICPU、CCU 通信程序，支持 Tx/Rx 端口竞争、报文开销、
  Switch hop 与分层链路；硬件速率覆盖进入同一执行图。
- 可观测服务响应及可选指令能力保留适用域、来源和物理 owner。
  校准、解析模型、独立验证相互分离，不使用模型名倍率或整步残差拟合。
- 通用 HF 结构适配与 partial coverage；未知子模块明确列出，已知主干
  继续建模。CLI/API 的使用方式见[迁移说明](../migration.md)。

## Decode 验收与适用范围

[当前留存记分牌](../../data/verification/model-scoreboard-20261029-final.md)
按真实 caller 和精度角色更新后为 **65/66** 个来源的选中 Body 对比满足
15% 相对误差，中位绝对相对误差 **6.35%**、均值 **6.82%**。
Qwen3.5-397B c48 **−19.46%** 仍未通过。

这些结果保留各自的输入、runtime SHA、拓扑/并发假设和校准关系；它们包含
同源算子级校准，是不同已完成 runtime 的选中回放，不能当作本标签的全量
独立验证或未见并发泛化。已有未见并发试验仍存在失败，后续版本继续改进
连续物理坐标响应。Body 通过也不等于每个重型算子或客户端 TPOT 都通过。

Prefill 的因果/缓存 Attention、chunk 线性注意力及请求执行已有路径；
大 M 跨场景精度、完整 Host 输入/结果准备、图捕获回退和客户端 TTFT/TPOT
闭环继续独立推进。0.8.0 不将这些后续目标计为已完成精度验收。

## 安装与迁移

需要 Python 3.11+。发行包包含硬件 profile 和模型配置，依赖 `ijson`；
运行无需加载模型权重、初始化 GPU/NPU 或连接真机。

```sh
uv pip install mindie-sim-0.8.0-py3-none-any.whl
mindie-sim predict case.json --scope body --phase decode --tokens 4 --sequences 8 --context 4096
mindie-sim predict serving-case.json
```

旧 `a3-*`、floor fitting、gate 和 `migrate-case` CLI 已移除。仓库只保留
当前 Schema、当前运行时及必要 fixture；旧源码快照和过程性产物已清除。
外部历史输入须按当前声明重写，不提供运行时兼容层。
可选 profiler 适配与对应版本的能力分开验证；不修改已安装插件或活动会话。

## 发布验证

本标签的发布流水线执行非真机合同测试、pipeline/serving/pd-serving 三项
CPU 基准，并随 Release 交付 wheel、sdist、精确源码 benchmark 和 SHA256。
CPU 基准检验接口、执行完成和资源守恒，不替代真机精度测试。
本地源码/隔离 wheel 结果及历史精度边界见[验证记录](../verification.md)。

首次发布流水线在 Linux 上发现外部源码测试路径依赖及 Prefill 完整图的
30 秒主机速度门限。已将带许可证的精确源码保留为小型测试 fixture；
Prefill 重复 key-panel 描述复用保持工作量、事件及尾块一致。完整图测试
采用端到端 90 秒预算，同时保留 512 MiB、有限活动任务和资源释放门禁。
旧失败流水线记录保留；正式产物只由修正后标签的门禁生成。
