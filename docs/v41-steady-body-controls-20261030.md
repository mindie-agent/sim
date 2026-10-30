# V4.1 完整 body 的显式稳态流水对照

在相同 `v41-a5-dp4-body-current.json` 上，仅更改软件程序声明，两个完整40层、TP1/DP4/EP4的body均在60秒预算内完成。没有改主运行时默认策略，没有裁剪层数或参与副本。

| 对照 | 普通矩阵 | 专家计算 | body预测 | predict墙钟 | 已完成Engine任务 |
|---|---|---|---:|---:|---:|
| probability-steady-backbone | SteadyMatmul | 概率占用，原TiledMatmul | 8.309919 ms | 21.964 s | 1,734,132 |
| balanced-steady | SteadyMatmul | 显式整数均衡，SteadyMatmul | 7.818983 ms | 11.202 s | 372,212 |

两组都有8,256节点、四个参与副本、0 live tasks、0 live programs。几何是Decode q1、context2048、**每副本**一个请求，即全cohort四个请求；不是历史单客户端请求的body窗口。图生成分别0.378/0.274秒，其余主要是Engine执行。原默认程序的独立60秒超时见 `v41-current-body-input-audit-20261030.json`；那次没有开始package hash，因此这里只把它作为已有超时问题，不能做严格同版本的精确加速比。

这不是实测精度报告。当前160个attention节点仍选择 `ResourceEnvelope`；Engram CPU路径与旧binary边界仍有审计中列明的未知项。没有把旧Step误差或当前8ms数值当作body已满足15%目标。

## 输入与版本

- `examples/cases/v41-body-controls-20261030/probability-steady-backbone.json`：只设置 `runtime.kernels.programs.matmul.kind="steady_matmul"`，**没有**覆盖 `families.grouped_gemm`。
- `examples/cases/v41-body-controls-20261030/balanced-steady.json`：在上述设置上增加显式 `routing.policy="balanced"`；这是实验声明，旧V4.1采集没有forced-EPLB证据。
- Receipt：`data/verification/v41-steady-body-controls-20261030.json`，含输入SHA、完整domain工作、实际recipe选择、资源服务与完成计数。
- 两组开始/结束package SHA均为 `bf197a1179c2898ff5ba0f9575713570047186906e92e6a3d56effaba05ba601`。它包含当时main已存在的运行时代码内容，不能仅按HEAD名字代替实际源身份。

普通矩阵共有1,760个SteadyMatmul。概率路由的320个grouped调用实际仍是TiledMatmul，mapping domain为`aggregate_experts`。显式均衡的320个grouped调用实际为SteadyMatmul，每rank96个本地expert中6个有1行、90个0行，mapping domain为`equal_expert_groups`。没有把概率aggregate当成整数专家组，也没有用大M把概率模型强行套进steady。

## 分域工作检查

下表为四副本所有节点声明工作之和，GB为十进制。它不是关键路径时间，也不是单rank权重驻留容量。

| 域 | 节点 | FLOPs (G) | 概率HBM (GB) | 均衡HBM (GB) |
|---|---:|---:|---:|---:|
| 普通矩阵 | 1,760 | 55.8295 | 30.1267 | 30.1267 |
| Grouped GEMM | 320 | 67.9477 | 17.6560 | 18.0746 |
| Attention | 160 | 12.8849 | 0.12195 | 0.12195 |
| Vector及其他 | 5,648 | 0.4364 | 0.22431 | 0.22431 |
| Engram非矩阵部分 | 48 | 0 | 0.000919 | 0.000919 |
| Collective | 320 | 0 | 0 | 0 |

两组collective link工作均为14.7456 MB。除grouped权重读以外，全部域的FLOPs、向量、HBM和link工作在浮点累计误差内相等。均衡的grouped权重读比概率占用多418.625 MB，来自实际6个活跃bank与概率占用期望的差别；因此两组不是严格同一物理工作，而是明确的两种路由假设。任务数下降136万，不能解释成删掉了这部分权重工作。

Receipt 的 `family_span_sums` 是节点ready→complete时间在四副本的累计，含stream、资源竞争或rendezvous等待，**不能当kernel activity sum，也不能直接加成body或当残差贡献**。真实body来自共享Engine的cohort输出完成事件；实际Engine任务数只报告总量，未伪造逐family任务归因。

## 复现与限制

```sh
.venv/bin/python data/verification/v41-steady-body-controls-20261030.producer.py
```

两个worker串行、各60秒预算。producer先输出真实图/recipe/工作摘要，50秒时可记录只读Engine计数和Python栈，因此超时仍保留可定位证据。初次执行在predict返回后的receipt汇总因误用不存在的`start_ms`失败；修正为实际`elapsed_ms`并明确度量边界后重跑成功，前次失败在报告中保留，没有伪装成模型错误或静默通过。

结果支持优先压缩普通矩阵的规则K-panel展开，同时为整数专家复用同一机制。概率专家的期望占用不能通过换一个recipe名字等同于强制均衡；是否改公共默认策略由核心实现与跨模型验证决定。
