# mindie-sim 0.9.5

本版本发布 **算子—子图—模型—服务四级分段连续物理求解引擎（`mindie_sim/prediction/piecewise/` + `mindie_sim/prediction/subgraphs/`）与全算子/全模型家族因果组合体系**：

- **四级分段连续物理求解与多管道硬件服务包络（`mindie_sim/prediction/piecewise/{compiler,families,physical,geometry,composition,model_pipeline}.py`）**：
  - 实现 13 类算子物理机理（Cube GEMM、MLA/FlashAttention、Attention Indexer、GDN/KDA 门控线性注意力、Grouped GEMM、MoE Router、HCCL/HCOM 集合通信、KV Cache 与状态访问、RoPE、激活、归一化与残差、超连接、AIV 向量与量化）的分段连续物理定价，同步 `stage_work` / `all_work` 与硬件服务时间，确保单分支与多流并发包络（`contention_envelope` / `Engine`）在 $N=1$ 时严格守恒。
  - 引入局部 DP 副本批大小（`local_batch`）、EP 跨副本路由群大小（`ep_batch` / `cohort_batch`）与局部 Capture Bucket 约束（`clamp_graph_bucket`），保留异构 Ragged 请求分布并对歧义收缩显式抛出 `ModelingError`。
- **子图因果状态转移与多流重叠传导（`mindie_sim/prediction/subgraphs/{instantiate,builders_attention}.py`, `prediction/piecewise/model_pipeline.py`）**：
  - 在带 Lightning Indexer 的 MLA 层（`DeepSeek-V4.1`、`GLM-5.3` / `5.1`、`Kimi-K3`）中强制 `I0 -> I1 -> A1` 因果拓扑顺序与 `index_cache` / `selection` 状态端口就绪校验；`multistream_attention_indexer` 仅重叠 `A1` 前置投影阶段与索引计算，核心稀疏注意力严格等待 Top-K 索引就绪。
  - 支持 `multistream_shared_expert`（`F3 || F4`）、`multistream_kda_proj`（`A3` 双流输入投影）与 `multistream_window_compressed`（`K1 || A1`）多流硬件管线竞争包络，以及 `predict_framework(..., max_retained_steps=256)` 有界步级状态留存。
