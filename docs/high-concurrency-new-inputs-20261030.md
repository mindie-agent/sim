# 新增 26 组高并发采集的输入合同

本轮审计 `/Volumes/data/mindie-sim-data` 中新增的 17 + 9 个生产目录，生成 26 个可解析的完整模型输入模板。此前 6 个 A5 Qwen3.6 TP8 / Qwen3.8 TP4 输入保持原样，总计 32 个生产采集；空核、补采失败和扫描中间目录不混入这个计数。

这里完成的是结构、精度、并行方式和采集身份审计，**没有执行 body 预测，没有新采集或远端操作，也没有把 TPOT 当成 Forward 耗时**。Profiler 负责完整事件窗口、实际 q/B/context 和 body 对账。

## 可复用产物

- `data/verification/high-concurrency-new17-inputs-20261030.json`：17 个目录的文件 SHA256、来源、张量样本、精度角色、未解决合同和对应 case 路径。
- `data/verification/high-concurrency-new9-inputs-20261030.json`：3 个 A5 Qwen3.6 TP2 与 6 个 A5 Qwen3.5 图/Eager 对照输入，同样保留独立身份。
- `examples/cases/high-concurrency-20261030/new17/`、`new9/`：26 个独立模板，不覆盖原 6 个 case。
- 两份同名 `.producer.py`：串行读取元数据，每个 CSV 最多采样前 7,000 行；大文件的身份散列使用 1 MiB 流式块。

运行时合同为 `4499695a`。所有 26 个模板通过 `parse_case`，未通过减少层数或扩充硬件容量求可运行；A5 Qwen3.5 六个模板还通过公共 `target_operators` 的真实非融合链依赖检查。该检查只生成普通前端算子，不执行 Engine 或计时。

## 输入差异

下表的 `B×q` 是**有界前缀中的张量槽位**，不保证等于活跃请求数，也不表示完整采集的主要 batch。`·` 分隔不同客户端并发条件。

| 采集 | TP / EP | 客户端并发 | 样本中的 B×q | 结构 |
|---|---:|---|---|---|
| A3 Qwen3.5 35B | 4 / 1 | 144 · 320 | 160×4 · 320×4 | 40 层，30 GDN + 10 GQA |
| A3 Qwen3.5 397B | 16 / 1 | 48 · 136 | 64×4 · 160×4 | 60 层，45 GDN + 15 GQA，官方结构条件映射 |
| A3 Qwen3.6 27B | 4 / 1 | 64 · 128 | 64×4 · 128×4 | 64 层，48 GDN + 16 GQA |
| A3 Qwen3.8 27B | 4 / 1 | 64 · 128 | 64×4 · 128×4 | 64 层，48 GDN + 16 GQA，独立模型身份 |
| A3 V4 Flash | 8 / 8 | 4 · 8 | 前缀同时含 1×4096 与 1×2 | 43 层 MLA |
| A3 GLM Flash | 16 / 16 | livepre 4 · 16；普通 2 · 4 | 前缀出现 1×6，另有 Prefill 事件 | 45 层，34 KDA + 11 MLA |
| A5 GLM Flash | 8 / 1 | 3 · 96 · 1 | 4×4 · 96×4 · 1×4 | 45 层，34 KDA + 11 MLA |
| A5 Qwen3.6 TP2 | 2 / 1 | 1 · 12 · 96 | 1×4 · 12×4 · 96×4 | 64 层，48 GDN + 16 GQA |
| A5 Qwen3.5 Graph | 4 / 4 | 4 · 64 · 128 | 前 7,000 行仅见 1×4 初始波次 | 40 层，30 GDN + 10 GQA |
| A5 Qwen3.5 Eager | 4 / 4 | 16 · 32 · 8 | 前 7,000 行仅见 1×4 初始波次 | 同结构，独立运行模式 |

Case 的 `runtime.graph_mode="none"` 表示尚未注入捕获图的 padding 合同；真实模式保存在 `runtime.metadata.captured_execution_mode`。模板中的 workload 是有明确标记的占位输入，不能直接作为服务回放结果。Profiler 应按匹配窗口提供实际 q、活跃 B、图槽位和增长中的 context，并明确 body 是否包含 padding 工作。

### Qwen 397B 的官方结构与精度

保留了 [Qwen 官方原始 config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/8472618112abcbd45acbcdc58436aff4233c23f7/config.json)：revision `8472618112abcbd45acbcdc58436aff4233c23f7`，SHA256 `3ae7fa89c2f7d1354096418ddaf1331e9e0898a8ca11e804fb6dbaa087efb7da`。本地副本在 `examples/cases/high-concurrency-20261030/source-configs/`。

公共 adapter 构建 60 层、hidden4096、512 专家、top10、专家中间维度1024。采样张量独立支持 E512、hidden4096、TP16 下本地专家64维、GDN key1/value4/head128；层数来自官方 config，没有从前缀事件猜测。部署的 W8A8-MTP checkpoint 与该原始 BF16 结构的映射仍是条件声明，未伪称拿到了部署 config 的 SHA。旧 `qwen3.5-moe` 的 E128 预设没有被使用。

W8A8 是声明的表示模板。Qwen 各输入中能用形状与 BF16×BF16 矩阵签名直接匹配的 `gdn.qkvz`、`gdn.beta_alpha`、`gdn.o` 才获得 BF16 角色覆盖，CSV 行号与权重 N/K 保存在 manifest；没有通过耗时匹配精度。397B 三个角色均有匹配证据。公共 adapter 的 `model_coverage=partial` 和未建模字段列表原样保留，特别是原始 config 的 `rope_parameters`、状态精度等仍需单独审查，不宣称完整结构机制已验收。

### GLM Flash 与强制负载均衡

GLM Flash 使用已保留来源的 `mindie_sim/model_configs/glm-5.3-flash.json`：45 层 Glm5Next，而非 GLM5.3 的 78 层。结构文件与其中 HF revision / 原 checkpoint SHA 都在 manifest 中保留；该文件的 BF16 归一化说明不被当作本批精度依据。A5 显式使用 MXFP8 表示，A3 显式使用 W8A8，且 A3 KV 张量证实 INT8；KV scale 打包和 `KvQuantSparseFlashAttentionVllm` source 绑定仍列为缺项。

仅以下输入启用 `routing.policy="balanced"`：A3 Qwen3.5 35B 有本轮启动日志 `enable_force_eplb=True`；A5 GLM 有 summary recipe 与 serve.log；A5 Qwen3.5 Graph 有本轮启动日志，Eager 有用户直接声明和本轮启动脚本。A3 Qwen397 日志没有正向 forced-EPLB 声明，A3 GLM/V4 目前只有脚本配置、缺匹配生效日志，因此模板不冒称已观察到强制均衡。

A3 GLM livepre c16 的 manifest 包含两个不同时间窗口的 tp0 源目录：保留两者身份，但物理 rank 计数去重后是 16；不能将它算成 TP17，也不能在通信累计中重复统计两个窗口。

### A5 Qwen3.5 非融合与缓存合同

本轮脚本是 `.workstreams/w3-a5-hc-decode/cfg/serve_qwen35.sh` 与 `serve_qwen35_graph.sh`。脚本声明 TP4EP4、forced EPLB、`fusion_ops_gmmswigluquant=false`，分别使用 `--enforce-eager` 与 `FULL_DECODE_ONLY`。脚本还保留 `DequantSwigluQuantV2 clamp_limit=1e9` 和禁用 Triton SwiGLU-quant 的容器修改。图模式的 `evidence/qwen35_graph_startup_and_prefix_hit.txt:3` 进一步证实 TP4/EP、forced EPLB、fusion=false，line9 证实 ACLGraph replay 已启用；Eager 及容器补丁仍是 **configured/source_only** 证据。

普通前端已实际检查得到：`expert_up.quantize → balanced_grouped_gate_up → grouped_swiglu → expert_down.quantize → balanced_grouped_down`。这是独立转换与依赖拓扑检查，不声称通用分析程序已经逐阶段复现 CANN 的 `DequantSwigluQuantV2`；没有选用 A3 A8W4 MSD 融合。

`benchmark_summary.prefix_cache_reuse_ratio=0.0` 在该客户端实现中是占位字段，**不是 0% 命中率**。图模式启动/命中率日志 line10–14 实际记录93.1%–93.5%，其边界是 campaign server 日志区间；不能把它提升为每个 Forward 的实测命中率，更不能宣称100%。Eager 仍保留 campaign report 的独立声明，不冒用图模式日志。同样，脚本 synthetic acceptance `[0.8,0.64,0.512]` 是配置，不是实测接收率。A5 Qwen3.6 / GLM 的 sweep、profile 各窗口指标分别保留，不混为同一测量。

## 重建与下一步

在仓库根目录运行：

```sh
PYTHONPATH=. .venv/bin/python data/verification/high-concurrency-new17-inputs-20261030.producer.py
PYTHONPATH=. .venv/bin/python data/verification/high-concurrency-new9-inputs-20261030.producer.py
```

Profiler 可复用 manifest 的 case 路径、CSV 身份和精度角色，无需再次猜模型/TP/EP。完整 body 对账仍需它确认各窗口的实际 q/B/context、padding、主干与 MTP 边界、通信 wrapper 去重，以及显式 source binding。这里没有把缺少测量合同变成禁止分析建模的门槛，也没有将模板可解析冒充性能精度达标。
