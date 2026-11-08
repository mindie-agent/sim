**拓展性、零 profiling 与唯一执行路径：主审查者 + Astra 联合审查**

2026-11-07。基线为 605d40f5 加当前工作区改动。Astra 使用 gpt-6-astra / max 独立只读审查；主审查者核查执行路径、接口和历史证据，再汇合结论。没有把主审查者冒称为 Antigravity。

本次交付是架构攻击、公式合同、删除清单和直接切换计划，**尚未实施 runtime 删除或切换**。原有未提交代码、未跟踪子图 schema、source contracts 和外部 raw captures 均保留。

**1. 决议：一条路径，但不能许诺不充分输入下的精确硬件时间**

只接受：
\[
Config\rightarrow\{G_s(\Theta_s)\}\xrightarrow{H,I,T}
\{P_m(X,Z;\Theta_{s,m})\}\rightarrow Composition
\rightarrow Execution\rightarrow Engine.
\]
H 为芯片能力，I 为实现/调度规则，T 为物理拓扑，X 为请求几何，Z 为路由、地址、驻留、就绪等充分状态。
隔离算子的分段函数是 \(f_m=MakeSpan_{Engine}(P_m)\)。通用组合必须消费资源程序 P；仅串行、无额外等待且无外部竞争时，才可对孤立时长 f 求和。

撤销原 systematic-three-level-piecewise-blueprint.md 第 7 节允许保留 completion_floor、joint_native、经验 service curve 和逐域渐退的决定。全部退出可执行预测接口；没有兼容 flag、旧 schema 翻译器、nearest-shape lookup 或 calibrated fallback。一个 Engine 内用观测时长替换资源程序，仍是不同计时路径。

| 过强命题 | 最小反例 | 必须替换成的合同 |
| --- | --- | --- |
| BW 只由 burst/stride/active cores 决定 | 相同三轴，地址对齐、cache warm/cold、核间地址冲突或请求依赖不同 | 地址交易 + 缓存/队列/依赖状态 |
| SRAM 唯一决定 native tile/stages | 同一容量容纳多个合法 tile，以及单/双 buffer，时间不同 | 容量约束 + ISA/layout + 明示实现选择 |
| collective 只需 topology/bytes/ranks/chunks | 等每 rank 总量可全部节点内交换，也可全部跨节点；同总量还可产生热点接收方 | directed traffic + 算法/协议 + readiness/credits |
| residual 唯一指认缺失机制 | 少算 load 与少算 sync 都能令串行总时间少 5 μs | 有区分证据才归因，否则保留不可辨识 |

这些反例不否定零目标 profiling；它们否定“不充分输入也能唯一精确预测”。

“零 profiling”必须区分两个范围：

- **零目标 profiling**：新增模型、shape、特性组合不提供延迟训练样本；H/I/T 冻结后，仅由结构编译。这是本方案必要交付条件。
- **绝对零硬件表征**：芯片参数也只来自规格/ISA/供应商事实。未公开的队列深度、延迟、地址映射或调度不能从峰值推出；应保留未知项、符号表达式或明确假设下的敏感性范围。一次性独立芯片表征可以跨模型复用，但不能标成绝对零 profiling。

任何参数都不得按 model、shape、batch、context、family 或 recipe 名选择。将这些条件隐藏到 HardwareSpec、access_class、implementation hash 或多个“同芯片 profile”，仍违反要求。参数数量不应随模型/shape 数增长。缺失参数不能置零后声称完整准确时间，也不能借“需要硬件定律”之名启动密集 profiling。

**2. 旧架构依赖 profiling 的根因：观测总时间吸收了缺失机制**

以下为当前代码事实，不是按文件名推测：

| 代码位置 | 实际行为 | 泛化问题 |
| --- | --- | --- |
| prediction/cost.py:93–113 | PrimitiveKey exact lookup，duration 进入 OpCost | 同 shape/revision 锚点不能识别资源占用 |
| prediction/execution.py:568–597 | primitive timing 变 minimum；regime 再提高 floor | max(physical, measured) 遮住欠账且不能修正 physical 过估 |
| prediction/execution.py:578–588、456–462 | joint_native 清除原 work/setup/invocation，执行 busy_ns queue | 真实 HBM/Cube/buffer 竞争消失，analytic work 仅留 reference |
| prediction/response_functions.py:61–99 | fixed_us、affine、knots、multilinear grid | “物理坐标”命名不使经验系数变成物理推导 |
| prediction/services.py:102–225 | response curve 转 per-invocation supply cap | 保留 bytes 仍不等于能推断新 stride/owner/cache 下供给 |
| prediction/efficiency.py:428–458 | 从整体时长反求各 pipe effective utilization | 一个时长无法识别多个资源；默认效率补了不可辨识部分 |
| prediction/efficiency.py:371–409 | per-family target efficiency；整图按孤立算子总和比例缩放 | 关键链/竞争会改变，该预测绕过 Engine |
| observations/a5_comm_calibration.py:448–495 | 0.55、0.45 等手写 contention 系数和 transfer/sync 常数 | “analytic”标签不是物理证据；此处是观察分析 helper，不能误报为所有 predict 均调用 |
| backends/profiles/ascend.json；invocation_binding.py:77–118 | execution-class measured minimum | 搬进芯片文件仍是校准；普通 Add/MatMul 最小时间不是纯 dispatch 延迟 |
| body_defaults.py:115–154；operator_analysis.py:75–126 | Body 专用默认选择及独立 lowering 编排 | Body/Model/Serving 可能选不同程序，观测补偿掩盖路径分歧 |
| family_quant_indexer.py:44–67、121–122 | complete boundary envelope，但限定 (64,128,512)；reload/top-k 跨 owner 未知 | 完整边界不等于完整性能机制或任意 indexer 几何支持 |

缺失的物理内容主要是访问放大与复用、owner occupancy、tile tails、buffer 限制、fill/drain、同步、真实融合物化，以及通信数据分布和协议控制。一个 duration 将这些因素混合，插值不能稳定外推；扩大 knots 只会把问题推到下一个组合。

两项纠正：当前 body_defaults.py **未发现 model latency multiplier**，其问题是旁路策略位置、固定候选和架构选择；应迁移有效机制后删除该模块。也只发现一个 Frontend 类；重复的是 Body/Serving 编排，不是第二个完整 Frontend。真实 callback/source builder 不能仅因数量多就当作重复代码删除。

**3. 参数所有权：硬件参数归芯片，软件规则不能伪装成硬件参数**

仓库目前类型名为 Hardware/Device，不是 HardwareSpec。直接重构当前类型，不新增兼容 schema。

| 对象 | 允许内容 | 禁止内容 |
| --- | --- | --- |
| H：具体芯片/SKU/频率 | pipe/端口、每 dtype ISA 吞吐、存储容量/归属、交易能力、物理队列、信号能力 | model/shape 条件、kernel elapsed minima、经验效率曲线 |
| I：实现合同 | 算法、layout/packing、loop order、dispatch guards、融合/物化、protocol、tile policy | 按 capture elapsed 选算法或拟合 coefficient |
| T：部署拓扑 | rank→die/package/node/NIC、定向边/交换机容量、路由、物理传播 | “同为 A5 所以相同带宽”、per-model link utilization |
| X/Z：动态状态 | ragged 长度、实际专家/peer 行数、地址/页表、cache/queue、各 rank ready | 用 measured total 反推状态后宣称零样本 |
| observations | 原始数据/边界/单位/source hash、独立残差/诊断证据 | 向 runtime 回注 floor/rate/guard |

Host 提交、HCCL 算法、kernel launch 编排不是芯片常数。若声称 Serving 精度，还必须有 host 平台和软件工作描述；加速器峰值不能推出完整 host 时间。

当前 profile 中三个代表 SKU 的声明如下，**仅核对代码，非本次硬件测量或独立规格认证**：

| 当前 SKU / family | AIC/AIV | L1/L0A/L0B/L0C/UB，KiB | 当前污染风险 |
| --- | --- | --- | --- |
| Ascend910B2 / A2 | 24/48 | 512/64/64/128/192 | vector 2.5 μs、cube 5 μs completion minima |
| Ascend910_9391 / A3 | 24/48 | 512/64/64/128/192 | 同上，不能当作芯片天然 dispatch 时延 |
| Ascend950DT_9582 / A5 | 32/64 | 512/64/64/256/248 | Add/MatMul/empty-AIC/W8A8 measured minima；独立 scalar-store 表征参数 |

供应商提供按平台读取 L1/L0/UB 容量的接口，因此应固定具体 SKU/revision、可用空间和来源。[GetCoreMemSize](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/Utils-API/platform_info/PlatformAscendC/GetCoreMemSize.md)。
A2/A3/A5 产品族不是充分的微架构身份，A5 未知参数不得由 A3 比例缩放填入。每项 H 参数要有单位、owner、频率、证据类别和来源。测得的物理参数与规格参数保持不同标签；绝对零表征声明不得使用前者。

**4. MTE2/MTE3：交易取整 + 有限队列，而非每算子的有效带宽**

正确输入为惰性访问段流：
(address, length, owner, R/W, source_space, dest_space, ready, cache_policy)。
DMA 对齐、cache line、HBM transaction、page、DRAM row、地址冲突区域是不同概念。不得把 32/64/256/512 B 当成通用效率拐点；例如 DataCopy 的 32 B 数据块单位不能单独证明 HBM wire transaction 大小。[DataCopy slice transfer](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0105.html)。

**精确交易离散式。** 已声明 granule g，段长 b>0、起点 a，且无跨段合并、无 cache reuse：
\[
n_g(a,b)=\left\lceil\frac{(a\bmod g)+b}{g}\right\rceil,\qquad
q_g(a,b)=g n_g(a,b).
\]
b=0 时二者为 0。a 对齐时：
\[
q_g(0,b)=kg\quad ((k-1)g<b\le kg),\qquad
\eta_{transaction}=b/(kg).
\]
这是地址覆盖推导，不是 profiling 拟合。示例 g=64、b=64：a=0 需一个 granule，a=1 需两个；相同 burst/stride/cores 已不足。示例不声称任何 SKU 的真实 g=64。

规则 stride s 定义为 start-to-start；若 API 给 tail-to-next-head gap，先换成 s=b+gap。
a_j=a_0+js，周期 \(L=g/\gcd(s,g)\)，\(n_j=n_g(a_j,b)\)：
\[
W_g=g\left(\left\lfloor N/L\right\rfloor\sum_{j=0}^{L-1}n_j+
\sum_{j=0}^{N\bmod L-1}n_j\right).
\]
这是无合并合同内精确的 stride/对齐放大。多维 tensor 递归生成访问段；paged gather 需要实际页映射，不能用平均 stride 代替。允许缓存/合并时按实际生命周期计算命中，不能全程永久去重。MTE payload 与 HBM wire bytes 分账，不能都乘同一放大率。

**有限 FIFO primitive 的精确闭式。** 最早就绪 r_j、issue 间隔 δ、Q 个 slots、服务带宽 β、服务后 response latency ℓ；slot 保持到 response：
\[
e_j=\max(r_j,e_{j-1}+\delta,c_{j-Q}),\quad
u_j=\max(e_j,u_{j-1})+q_j/\beta,\quad c_j=u_j+\ell.
\]
首次 issue=0，不存在的 predecessor 不约束。地址 hazard、跨 core 冲突、写后读和 barrier 加真实依赖，不能塞进 β。

当所有 r=0、q_j=q、无 hazard/外部竞争时，令 \(s=q/\beta,d=\max(\delta,s),L=s+\ell\)：
\[
T_N=(N-1)d+L+\left\lfloor\frac{N-1}{Q}\right\rfloor\max(0,L-Qd),\quad N\ge1.
\]
稳态 payload 带宽：
\[
BW_{payload}=\frac b{\max(\delta,q/\beta,(q/\beta+\ell)/Q)}
=\begin{cases}
b/\delta,&\delta\ge q/\beta,\ \delta\ge L/Q,\\
b\beta/q,&q/\beta\ge\delta,\ q/\beta\ge L/Q,\\
bQ/L,&L/Q\ge\delta,\ L/Q\ge q/\beta.
\end{cases}
\]
相等分支值一致；有限 N 用 T_N 保留 fill/drain。这是**声明的 FIFO 模型**中的精确式，不能冒充真实 Ascend 所有访问的精确物理定律。

C 个 active owners 通常各有自己的 issue/queue/hazard，并共享 HBM；逐 owner 建队列，在 Engine 中使用同一 HBM 资源。只有均匀独立请求且允许 pooled queue 的稳态模型，才可用 Q_total=CQ 和受共享端口约束的 δ_C 代入。active_cores 本身不能证明 CQ 个请求独立，也不能证明不同核的时相可忽略。

**读写转换。** 对一个 R/W 串行的已声明 memory channel：
\[
T_{service}=D_R/B_R+D_W/B_W+
n_{R\to W}\tau_{R\to W}+n_{W\to R}\tau_{W\to R}
+T_{row/bank/refresh}.
\]
切换次数由真实调度顺序产生，不由读写比例决定。不同 channel 可并行；MTE、共享 HBM、ready 和 credits 由 Engine 合成。未知 bank mapping/cache/controller policy 保留缺失项或明示假设，不拟合 η。这是该 channel 的服务清单，不是所有 kernel 的时间等式。

| 使用者 | 用同一 memory primitive 生成的内容 |
| --- | --- |
| Dense GEMM weights | N/K panels、真实 packing/scales、owner/reuse/layout、每次 reload |
| Paged KV | 页表读取→地址 ready→页内 K/V 段，page tails、head replication、并发限制 |
| QuantLightningIndexer | QK、weighted reduction、scales/score/top-k workspace；Fixpipe/MTE3 分账，跨 owner merge 和 Cube→Vector 依赖独立保留 |
| MoE gather/scatter | histogram/prefix 后真实 permutation、源/目标行与 metadata/scales，combine 重叠写/归约顺序 |

统一的是交易、队列与资源，不是把这些 kernel 全部当成 memory-only。
可复用 transfer.py 的有限 slot/response 生命周期与 transactions.py 的地址 hazard；后者当前向内存计 element_bytes，不能未经修改就宣称已计 wire-granule 放大。

**5. SRAM 驱动 tiling 的正确含义与 pipeline**

候选映射 π 对每个 pool p、owner u 必须满足整个程序的活跃量：
\[
\max_t\sum_{b\in Live(p,u,t;\pi)}Align_p(Bytes_b(\pi))
\le Capacity_{p,u}(H)-Reserved_{p,u}(I).
\]
不能平均总 SRAM 后判通过；必须含 packed operands/scales、解包操作数、持久 accumulator、临时输出、epilogue、所有 slots 和最后消费者。

简化 GEMM 的容量必要条件：
\[
d_1(B_A(M_t,K_t)+B_B(K_t,N_t))+S_{L1}\le L1,
\]
\[
d_A B_A^{compute}(m_t,k_t)+S_A\le L0A,\quad
d_B B_B^{compute}(k_t,n_t)+S_B\le L0B,
\]
\[
d_Cm_tn_tb_{acc}+S_C\le L0C,\quad
B_{convert}+B_{epilogue}+B_{reduce}+B_{state}\le UB.
\]
各 B 按真实 dtype/packing/alignment/scale blocks 求；L1 packed 与 L0 解包宽度不能混用。
d_* 是各 buffer 实际 slots，不是默认全部为 2。共享 pool 需检查同时 live 的所有 owner。
尾块 actual work 与分配尺寸分开；容量合法还须检查 lifetime/credit 是否造成死锁。

π 包含 (M_t,N_t,K_t,m_t,n_t,k_t,slots,loop_order,split,owner_map)。
候选来自 ISA 合法 extents、layout、pool 和实现规则；shape 参与尾块/任务数/复用/容量。
有 native dispatch 则编译其 guard；没有 native policy 时，只能声明 analytical mapping：
\[
\pi^*=\arg\min_{\pi\in Feasible(H,I,shape)}MakeSpan_{Engine}(P(\pi)).
\]
固定 tie-break，可用可证明 lower bound 剪枝。仅按 lower bound 排序则标为启发式，不能宣称实际最优/native reconstruction。I 决定使用哪种明确策略，所有策略仅生成同一 IR，失败不自动切换旧公式。

供应商 API 同时提供 shape、buffer space 和可选 SetFixSplit，说明容量约束与实现选择是两件事。[Matmul tiling](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0038.html)。
现有 matrix_mapping.py:125–139、181–265 有 footprint 筛选，但默认 candidate 列表与资源下界排名不能改名成容量导出的 native 定律。

**流水的闭式适用域。** 独立单服务资源、同长 n 个 tile、足够 buffer、无外部竞争：
\[
C_{i,p}=\max(C_{i-1,p},C_{i,p-1})+t_p,\qquad
T_n=\sum_pt_p+(n-1)\max_pt_p.
\]
n=1 必须是串行各阶段之和。全 tile 最多 s 个在途时，第一 stage 还依赖 \(C_{i-s,P}\)；
s=1 退化为 \(n\sum_pt_p\)。不等长 tail、expert skew、共享 HBM、有限 event/queue 使用真实 recurrence/DAG，不能继续用单个 max_pipe。

AIC↔AIV 用 produce→signal→wait/acquire→consume 表达；FFTS+ 或其他信号机制由 I/H 声明实际资源、事件容量与传播。consumer 等待由 producer 完成产生，不再加等待罚项。
实际需要的 GM flush/read 才计 bytes；不能把所有同步都假定为 GM。每个真实 handshake 收费，不能对每 tile 加整体 measured sync 常数。

保留 PipelineProgram、matrix_cycle.py、matrix_steady.py 的有限执行/压缩能力，但需清楚标识等价域。当前 steady 的外部竞争 fluid approximation 不等于任意竞争下与详细 tile 相同。最终每个实现只有一个已选程序表示，不保留为旧调用服务的多套默认选择链。

**6. Collective：先生成传输程序，再得到分段时间**

完整合同：
\[
T_{coll}=MakeSpan_{Engine}
(P_{coll}(op,dtype,reduce,I,\{M_{uv}\},\{chunks_{uv}\},\{ready_r\},Z);H,T).
\]
把这些信息全部塞进 TopologySpec 只是改名，不能证明原四个标量足够。
HCCL 官方文档明确存在自适应算法和 HOST/AI_CPU/AIV 等不同展开位置，实际资源占用不同。[HCCL_ALGO](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/envvar/envref_07_0079.html)，[HCCL_OP_EXPANSION_MODE](https://www.hiascend.com/document/detail/en/canncommercial/850/maintenref/envvar/envref_07_0096.html)。
源版本/支持域必须对应 capture，不能拿新文档的 policy 当历史 binary。

链路 e 的 payload MTU=P_e、alignment=a_e、每帧开销=h_e 时：
\[
W_e(x)=\sum_{j=0}^{\lceil x/P_e\rceil-1}
\left[a_e\left\lceil\frac{\min(P_e,x-jP_e)}{a_e}\right\rceil+h_e\right].
\]
x=0 为空和。实际协议若不同，按其合同生成帧。serialization=W_e/B_e；propagation、endpoint copy、reduction、control、credit 分开，同一 NIC/HCCS/HBM 不因不同通信域复制容量。
MTU/协议是部署/软件配置，物理端口能力属于 H/T。

**均匀 ring 窄域闭式。** R ranks，完整归约向量 M bytes，AllGather 每 rank 输入 M/R；
R 整除 M、所有 rank 同时 ready、独立等速边、无竞争、每轮整个 chunk、reduction 串行。
β 为 byte/time，α 为控制/传播程序成本，γ 为 reduction time/byte：

| collective | 限定模型时间 |
| --- | --- |
| ReduceScatter | \((R-1)\alpha+(R-1)M(1/\beta+\gamma)/R\) |
| AllGather | \((R-1)\alpha+(R-1)M/(R\beta)\) |
| AllReduce，RS→AG | \(2(R-1)\alpha+2(R-1)M/(R\beta)+(R-1)M\gamma/R\) |

α 不能来自 whole-kernel measured intercept，γ 来自真实 reduction 工作/资源率。
subchunk 后生成 hop readiness/credit recurrence，不盲乘 ChunkCount。
小/大消息 policy 来自 I；明示分析选择且某 cell 内候选为 a_i+b_iM 时，交点才是
(a_j-a_i)/(b_i-b_j)，还需检查合法域。源码阈值是软件事实，不是芯片峰值自然推导。

**AllToAllV。** 保留 directed M_uv。四 rank、四 chunk、总 payload=12、每 rank 发 3：
环状一对一交换最大接收量 3；0→1、1→0、2→0、3→0 最大接收量 9。
原四参数相同，receiver bottleneck 下界三倍。等收发总量也可因节点内/跨节点重排产生不同 slow-cut 流量。
仅明确逐轮全局 barrier 的 pairwise 模型能写
\(T=\sum_{round}\max_u(\alpha_{u,\pi(u)}+W(M_{u,\pi(u)})/\beta_{u,\pi(u)})\)；
非 barrier、多 hop/共享 edge 使用 Engine。一般 \(T\ge\max_e W_e/B_e\) 只是资源下界，不能当等式。

**MC2。** 保留 count/histogram/prefix、quant/scales/pack、发送/接收/notify、每 expert compute、combine/reduce/scatter、receiver readiness。
等长 c chunks、独立 dispatch/compute/combine 资源且 buffer 足够才有
d+k+g+(c−1)max(d,k,g)。dispatch/combine 共用链路时，仅该资源就要求至少 c(d+g)。
实际重叠由 DAG 得出，禁止 overlap coefficient。使用实际整数 expert/peer/valid/padded 行数逐 chunk 绑定。

新增多节点 A5 必须声明 rank→die→node→NIC、NIC 双向注入/接收、交换机共享边、路由、endpoint HBM、protocol credits、执行位置和并发工作。增加 node_count 或给 RoCE 套倍率不构成拓展性。

**7. 零目标 profiling 的组合证明及边界**

Canonical Subgraph 是 typed tensor/state/ownership ports 的生成器。
基础词表：projection/matmul、norm/elementwise/quant、cache/view/copy、attention、index/select、
recurrent/convolution、route/permute、expert compute、transport、residual/mix、input/output。
GQA/MLA、GDN/KDA、MoE/shared expert、SharedKV、draft 是其有状态组合。
沿用 ATTENTION/FFN/RESIDUAL registries；符号/数值 work 共用定义，不再手抄第二套 FLOPs/bytes。

对已支持语义、充分 X/Z、明确 H/I/T、终止且资源可行的程序，可结构归纳：

1. 基础节点从 geometry/layout/dtype 导出 work、访问段、primitive DAG，不读观测时长。
2. Serial/Fork/Join/Loop/StateLink 只连接端口、依赖、实例和真实共享资源，不增加拟合时间参数。
3. 唯一 Engine 执行有限组合程序，因而新增 shape/已知特性组合无需新增 profiling。

这证明无观测依赖的结构闭包和可执行性，**不证明真实硬件误差小于任意指定容差**。
误差还依赖 primitive 正确、充分状态和实现一致。
有限的是规则生成器与单次程序；无界 context 的 ceil/waves 可产生无穷多区域。
用共享表达式 DAG、惰性 guards、有界缓存，不枚举全模型分段笛卡尔积。

| 新组合 | 从结构推导 | 不可省略 |
| --- | --- | --- |
| GQA head ratio | local Q/KV、replication、QKV/O GEMM、QK/PV、head reuse | TP>KV heads 复制、window/mask、page/layout |
| MLA rank/head ratio | Q/latent KV projections、每 head absorption/expansion、cache | 独立 weight domains、真实 physical head padding |
| MoE E/top-k | router、\(\sum_e n_e=kT_{valid}\)、逐 expert GEMM、peer traffic | 相同 E/k/T 不决定 n_e；未知 route 显式 scenario/distribution，不能将平均值取整冒称真实执行 |
| hybrid schedule | 按 LayerPlan 顺序组合 attention/recurrent/residual | cache/state/conv、compression remainder、SharedKV owner、accepted state |
| TP/DP/EP/CP/SP | ownership/shard/replicate/pack、通信和 join | divisibility/tails、完整 EP cohort、空 sender 非空 receiver、CP horizon/merge、SP gather/trim |

新增真实 top-k/recurrent/fusion/protocol 机制需新增 registry 项，仍编译至相同 primitive/Engine。
不支持组合明确报错，不回旧 predictor。
同一 canonical 结构改 model label 不改预测；真实 weight/cache ownership 不得因去 label 被错误合并。

**8. 精确删除/替换清单：切换后的唯一代码面**

“替换”表示迁移有效机制后删除原旁路，没有 wrapper/兼容重导出。下列是待实施决议。

| 路径或接口 | 决议及接替职责 |
| --- | --- |
| prediction/service_regimes.py | 整个删除；structural guards 属于 compiler，不再有 observable timing regime |
| prediction/response_functions.py | 整个删除，含 affine/roofline 参数入口、knots/grid；接替为单位化 work AST |
| prediction/service_regime_domains.py | 整个删除；新 guard 检查不继承 empirical support/continuity/extrapolation schema |
| prediction/services.py | 删除模块及 ServiceCurve/ResourceService；有用 identity 逻辑迁入结构身份模块，不能绑定 timing |
| prediction/primitives.py | 删除 runtime PrimitiveStore/Timing lookup；原始观测及身份留在 observations 当前记录 |
| prediction/cost.py | 删除 lookup 与 service/regime/joint-native/calibration bookkeeping；只保留物理 work accounting |
| prediction/execution.py | 删除 RegimePolicy、measured floor、joint_native replacement、curve bookkeeping；只绑定已编译程序与 ready/stream/cohort |
| prediction/program.py | 删除 ResourceService factory 和 observation minimum/floor 语义、context 的 regime/joint_native 字段；保留真实 Stage/queue/control/propagation |
| prediction/invocation_binding.py；schema/invocation.py | 替换 elapsed minimum 和 match→timing 规则；使用真实 launch/control/signal primitive |
| prediction/body_defaults.py | 整个删除，包括 TargetGraphSet/TargetOperators 的隐式 defaults 携带协议；改为 CompiledGraph 显式固定合同 |
| prediction/operator_analysis.py:_target_ops；frontend.py:__call__ | 收敛为唯一 compilation entry；删 Body 专用 resolve/geometry/lowering/fusion 编排 |
| prediction/call_builders.py；calls_activation.py；call_grouped.py；call_grouped_source.py；gdn_composite.py | 保留真实 ABI adapters；重复 work/timing builders 改为调用 canonical invocation，不独立定价 |
| prediction/matrix_mapping.py 及各 matrix 默认选择 | 替换固定候选/多层默认策略入口；保留 footprint/owner/phase；一个 policy 产生一个确定程序 |
| prediction/efficiency.py | 删 _decompose_operator_pipes、fit_case_roofline_regimes、fit_efficiency_targets、_scale_service_regimes、evaluate_target_efficiency 的替代计时；MFU/MBU 只描述 ledger，SLA 改 H 后重跑 Engine |
| observations/calibration.py；observations/service_fitting.py | 删除预测参数生产/绑定接口；保留原始 ingest/只读 comparison 的必要职责 |
| observations/a5_comm_calibration.py | 删 CalibrationBundle/curve fitting、contention multiplier/派生时间；有用 summary 变为只读诊断 |
| observations/a5_comm_binding.py | 删 _fit_bundle、_A5CommPlan.minimum_ms、bind_service_curve、to_service_regime、curve recipe；流式 reader 迁到 ingest |
| observations/validation.py:fit_resource_rate | 删除 fitter；保留 service_observation/compare，不允许结果写回 H |
| schema/case.py；schema/hardware.py；schema/device_input.py | 旧 timing keys 严格拒绝；删 elapsed minima；typed schema 校验 rate/override 来源，不静默忽略旧 keys |
| backends/ascend.py；backends/profiles/ascend.json | 去 calibrated/effective-bandwidth 优先分支、任意 utilization、measured kernel minima；按来源保留真实 chip/topology facts，未知值不零计费冒称准确 |
| prediction/api.py；graph_analysis.py；operator_analysis.py；body.py；disaggregated.py；study.py | 去 primitives=，统一 CompiledGraph；arbitrary timing lowerer 改成 typed registry 扩展 |
| mindie_sim/__init__.py；prediction/__init__.py；__main__.py | 去旧 exports、--primitives、calibrate/bind-profiler/fit-rate/fit-service、效率拟合预测入口 |
| structure.py；structure_physical.py；cohort_comparison.py；advise_core.py；prediction/skew_analysis.py | 去隐式 body defaults、curve identity/二次 lowering；仅消费同一 IR |
| tools/replay_service_regime_cases.py；tools/verify_service_regimes.py | 删旧规则 replay/验证路径；保留的 capture adapter 改走 canonical replay+comparator |

保留 engine.py、mechanism.py、真实 source/family planners、owner/layout/quant、finite transfer、
topology/collective/hccl/mc2、native fusion、状态程序。
不同 vendor/arch adapter 和真实算法不等于多计时路径；不要为减少文件数抹掉真实机制。
所有 planner 最终产出同一 IR，已选实现失败必须报错。

tests/test_service_regimes*.py、test_service_response_*.py、test_resource_services.py 与相关 calibration/efficiency 测试，改测旧接口被拒绝及独立物理不变量；不能为了旧测试通过保留 shim。
保留 active fixtures 的原始协议/source identity；删除废弃 runtime bindings/退役输出，不删除外部 raw。
examples 与 architecture.md、observations.md、modeling-fidelity.md、migration.md、原 piecewise blueprint 同步撤销旧路径承诺。

**9. 直接切换顺序：一次合入，不因精度不足恢复旧路径**

1. 冻结现有语义/支持范围和 observation manifest，保存当前 dirty work 边界；列每个 source-bound family 的 work、buffer、call/kernel/fusion/collective 合同。
2. 在实现分支完成 typed 子图、参数所有权、统一 frontend、Memory/Matrix/Vector/Sync/Transport primitives，所有公开入口直接接新 compiler。
3. 同次切换删除第 8 节旧模块/接口/schema。开发过程不发布兼容双轨；不增加运行时新旧选择开关，不计算两次后取更接近观测的结果。
4. 修守恒/因果/结构问题，运行完整 maintained suite、安装包和 CPU/RAM 检查。独立数学/行为 oracle 不使用旧 latency 黄金值；支持缺口显式记录，禁止自动 envelope fallback。
5. 冻结代码/H/I/T hash，回放历史集与独立新域。超限保留失败并修机制，全量复测。精度下降阻止声称验收，不能成为恢复 knobs/floor 的理由。

建议职责落点：prediction/subgraphs/{registry,instantiate,state}.py，
prediction/piecewise/{expressions,guards,compiler,diagnostics}.py，复用已有 primitive mechanisms。
不新增第二个 piecewise Engine。模块尽量 <800 行，首 30 行带项目四项说明。
观测比较在预测进程之外，symbolic/numeric lowering 使用同一工作定义。

**10. diagnostics.py：可证实的物理归因与不可辨识必须分开**

当前 prediction/piecewise/diagnostics.py 只见于蓝图，未发现已实现模块。
新增模块分别输出 execution explanation 和 observation diagnosis：
前者可从程序精确得到，后者需要独立证据；显示模型内瓶颈不等于证明真实误差原因。

每节点 ledger 至少有：subgraph/template/kernel IDs、H/I/T hash、命中 guard/边界距离、
actual/allocated/padded work、payload/wire、owner/tile/tail、buffer peak/slot wait、
资源别名、signal/credit/stream 依赖、ready/start/end、critical chain/ties、资源忙碌区间、
参数来源/未建模项。普通执行 bounded trace；失败样例可固定输入重放详细 trace，
不把所有已完成事件常驻内存。

闭环：

1. 先核对 kernel/call/body/client、warmup/compile、统计量、rank 聚合、binary/implementation、
   shape/routing/page、graph/fusion。缺失为 unpaired/conditional；client TTFT 差额不能归 MTE。
2. 冻结预测后才读观测。保存源码/H/I/T/input hash 和 sample/source IDs；
   observation store 被移走或替换，预测结果必须不变。
3. 逐 case/part 输出 r=predicted−observed，保留 signed/absolute/repeated residual。
   建议沿用历史 10% relative criterion；absolute tolerance 由边界协议预先声明，不随失败放宽。
   所有 unsupported/unpaired/timeout 也留在报告分母。
4. 先查 work conservation 和源规则；证实漏 copy/tile/reduce/signal 或错误 fusion 时，
   标记 confirmed_structural_gap 并指向具体 primitive/guard。
5. 再以已有 phase/counter、物理坐标趋势、guard 两侧、跨 family 的一致性检验候选机制；
   每个假设必须预测至少一个区别于竞争假设的可观测后果。
6. 结构反事实（补已证实 reload/wait、改真实 tile）仍用 Engine 重算。
   sensitivity 是模型条件下的预测，不能当已测 causal effect；孤立 residual 不能直接相加当图贡献。
7. 不可辨识时输出候选集合/缺少证据。Jacobian \(J_{ik}=\partial prediction_i/\partial parameter_k\)
   的秩/相关性可揭示混淆；跨 guard 用结构对照。满秩也不证明全局唯一或机制正确。
8. 机制修复需新的规格/实现证据及独立反例，历史集全量复测，并验证未接触的新域。
   根据 heldout 失败设计过公式后，该批已是开发证据，不能再次标作 pristine heldout。

| 候选机制 | 区分证据 | 合格修复 | 禁止修复 |
| --- | --- | --- | --- |
| MTE burst/stride | payload/wire/request 数、地址/页/核间访问条件 | 取整/合并/复用/队列规则 | indexer family 独有 BW 倍率 |
| tiling/occupancy | dispatch/tile、block/core counts、边界趋势 | 候选生成、owner/tail/liveness | 按 batch 的 timing knots |
| AIC/AIV sync | producer ready、signal/wait/owner | 真实事件/credit 依赖 | fused-kernel observed intercept |
| collective | peer bytes、算法/展开位置、credit、慢 rank/链路证据 | route/round/framing/control/copy/reduce | A5 多节点统一折扣 |
| fusion | native boundaries、external live outputs、workspace | 修物化和边界 | 子图整体 scale |
| 数据依赖缺失 | histogram/selected pages/cache state | 充分状态或条件范围 | 按最接近测量倒推 route |

attribution_status 至少区分 confirmed_structural_gap、supported_hypothesis、unidentifiable、
input_boundary_mismatch、unsupported_semantics。
允许 unexplained_residual_ms 非零，**禁止回注预测**。
max/竞争/关键链切换使归因一般不唯一，不得为了凑总差值分摊“同步税”。

最小不可辨识证明：\(\hat t_C=\hat t_H=1,T=\max(t_C,t_H),T_{obs}=1.2\)；
(1.2,1)、(1,1.2)、(1.2,1.2) 都解释同一观测，原因不同。
只有同次执行的完整串行阶段与等待间隔都配对观测，才可精确相加分解残差；
多个隔离 kernel median 的和不具备该性质。

**11. 2,523 cases 与新增多节点 A5：验收合同和实际证据**

本次未运行历史预测脚本。已流式核对两份报告各有 2,523 cases、15 exclusions，并重新计算 report SHA256：

- /tmp/mindie-accuracy-audit-20261106/native_systemic2/report.json：
  43,347,919 bytes；858932e785e2e0436f8ad30fa60a2f9d541ca6093da13185cbb3f2b91432fe8f。
- /tmp/mindie-accuracy-audit-20261106/archive_qualified_incremental/report.json：
  28,954,856 bytes；657b6b85116c81fc246f26508458e94141fc009bb10afc0a3f24d6a03bf49a3d。

这是报告身份/条目数核验，**没有重哈希全部 raw 或重跑当前代码 accuracy**。
外部 systemic2-final-replay-independent-evidence.json 记载 9,930 parts、2,392 weighted capture passes、2,105 part failures；较早 report 的 weighted passes 为 2,363，不能拼接为“当前成绩”。
manifest.json 明示 Wave5 in-sample calibration。旧通过率不证明删除经验路径后的精度或零 profiling 泛化。

新验收 manifest 冻结 source path/hash、case/part identity、boundary/units、implementation/topology、
训练/开发污染历史、exclusions 理由。15 个 exclusions 逐项保留复核，不为通过率扩大排除。
新增多节点 A5 单列 cohort、与旧观测去重；同一 capture 的多个 parts 不是独立样本，
CPU synthetic control 不计设备准确率。

验收必须同时满足：

- 零依赖：prediction 不读时长/knots/efficiency override/store；新模型只加 config/必要结构 adapter，H/I/T 常量 hash 不变。
- 结构：FLOPs、payload/wire/scales/peer traffic、capacity、ready、完整 EP cohort 和 state transition 正确；不丢既有支持能力。
- 数值：每个可比 case 与 critical part 对预先声明 tolerance 评分；每个超限开机制问题。
  报告 chip×phase×family×shape×parallel×boundary 分层、P50/P95/max、signed bias、失败清单及未配对分母。
- 泛化：冻结后检验未见 model/head ratio/E/k/hybrid、tile/queue/page/MTU 两侧、新 TP/DP/EP/CP/SP 和多节点 A5 skew/readiness/NIC/HBM。
  没匹配 capture 的组合不能宣称 timing pass。
- 决策：配置排序、增长比、瓶颈转换；观测噪声内差异为 tie/未决，不能任意扩大预测范围换通过。
- 执行：完整 maintained suite、安装包、CPU/RAM 分开记录；routine RSS≤512 MiB、capture 流式读、长历史不积累 live tasks。configured CI 不算 completed CI。

必要独立反例：地址 granule 枚举、stride 周期求和、single-tile fill/drain、slots=1、
tail owner 不借空核、同 pipe demand 相加、两 HBM branch 竞争、同总量异 peer matrix、
MoE conservation、SharedKV 一次生产多次读、scope 不改同 kernel 程序、label 改名不改性能、
延迟输入不提前执行。

本次主审查者执行：1,600 组 granule/stride 对逐字节枚举、2,400 组 FIFO 闭式对直接递推、
320 组理想 pipeline 闭式对逐 stage 递推，以及 AllToAllV 接收热点反例；全部通过。
这些是数学内部验证，不是真实芯片验证。Astra 另做 FIFO 递推检查，不累计为独立硬件样本。
本次未运行 pytest、完整 maintained suite、CPU 模型 benchmark、设备 primitive 测量或新路径 accuracy campaign，因为本轮未实现 runtime 切换。

**12. 可证伪条件**

固定 H/I/T 后，新模型仍需要专属 rate/floor/knots，或每个新 shape 都要新时间参数，
即未达标。先检查充分状态、实现规则和 primitive 缺口，不允许改名绕过。
源规则/状态完整且守恒/因果正确，误差仍随访问/调度条件系统变化，说明 primitive 抽象不足；
补最小可区分反例的机制，再检验跨 family 转移。
参数未公开且不可辨识时，不能承诺实际精度；这属于证据/建模问题，不能用 per-case 拟合掩盖。
唯一 runtime 不回退，准确率失败保留，直到机制修复并通过独立验收。
