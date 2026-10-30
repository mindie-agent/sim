# force-EPLB 的调用域与 DP 专家计数审计

结论：已核对的本地源码在 **prepare 后的本地物理 token 行**上生成强制路由 ID，且包含 **EP rank 偏移**，并非每个 DP 都从 expert 0 开始。`AllGather` 分支的 prepare 会先聚合输入，必须单独解释。当前 Sim `balanced` 是理想的全 cohort 整数均衡声明，不能称为精确复现这个源码表，更不能称为复现尚未确认身份的 0.30 镜像 helper。

本次没有修改 runtime、默认路由或 capture，也没有远端操作。以下行号及 SHA 已写入 `../../data/verification/force-eplb-dp-scope-audit-20261030.json`。源码基目录为 `/Users/maoxx241/code/vllm-ascend-v41-private`；独立 upstream helper 位于 `/Users/maoxx241/code/vllm-v0.27.1-k3-ops-validation`。

## 真实调用顺序

- `routed_experts.py:634–662`：先 `moe_comm_method.prepare`，再用返回的 hidden/router tensors 执行 `_select_experts`，然后交给 quant/expert implementation。
- `routed_experts.py:601–602`：开启 `enable_force_eplb` 后替换普通选择产生的 expert IDs；并不替换 router 权重。
- `force_eplb.py:59–86` 仅提前缓存 capture 大小的表；`:89–105` 在调用时使用 `topk_ids.shape[0]`，不是缓存最大行数。
- `force_eplb.py:130–142`：令每个物理来源 rank 的准备后行数为 `M_r`、top-k 为 `K`、EP 为 `P`、逻辑专家数为 `E`：

  ```text
  A_r = M_r * K
  cursor = j + (A_r + 1) * ep_rank
  destination = cursor % P
  local_bank = (cursor // P) % (E // P)
  expert_id = local_bank + destination * (E // P)
  ```

  所以专家是连续按目的 rank 分块放置，但轮询 cursor 先跨 rank、再跨该 rank 内 bank。
- 独立 upstream `fused_moe/config.py:1113–1121,1209–1246` 将 DP/PCP/TP flatten 成 EP rank（PCP1 时 `dp_rank*TP + tp_rank`）。不能把 offset 中的 rank 当成仅本地 TP rank。

## 三种 prepare 合同

| 通信分支 | `_select_experts` 看到的输入 | 哪些行实际进入专家执行 |
|---|---|---|
| AllToAll | `prepare_finalize.py:128–176`：本地 padding／TP 切片；SP 已切片则保留；没有前置 DP AllGather | 该分支 `mc2_mask=None`，包括物理 padding 行 |
| MC2 | `:271–312`：本地或 SP shard，必要时补到 DP-wide 统一长度；不是拼接全部 DP token | `token_dispatcher.py:140–144,185–186`：`global_bs=0` 时传 `x_active_mask`；非零 `global_bs` 分支不传该 mask |
| AllGather | `:381–420,455–507`：先 DP AG，SP 情况走 EP AG，再生成表 | 输入已被复制／聚合，不能把每个 rank 的整份表再作为独立源 populations 相加 |

特别地，MC2 的 offset 使用 **物理 M_r**，mask 决定哪些行的 ID 真正生效。用有效行数替代物理 M_r 计算 offset 也不正确。mask 的具体内容、skip-allreduce/global_bs 分支属于调用合同，不能从 output capacity 或客户端并发反推。

Kimi `kimi_k3.py:209–239,653–661` 确认模型先 SP 分片，MoE 收到该局部 hidden 与 router；V4 `deepseek_v4/model.py:382–405` 在启用 MoE SP 时也先分片，随后进入同一工厂／router 链。两者不是另一套专属强制路由算法。

## 数学控制：V4.1 不支持“6 banks × 8”的猜想

以 `E=384, EP=8, top-k=6` 为例：

- 每个物理来源 rank 均 8 行：共 384 assignments，每个目的 rank 接收 48 行。源码表让 rank0–6 各有 **47 个活跃 banks（46×1 + 1×2）**，rank7 有 **48×1**。理想均衡的 48×1 与源码存在小差异；没有退化成 6×8。
- 只有 rank0 的 8 行有效，其余来源均无有效行：共 48 assignments，每个目的 rank 接收 **6 banks × 1**。这与当前 source 条件 case 的专家行数相符，但来源活跃性仍是条件声明。
- 只有 rank0 的 1 行有效：共 6 assignments，目的 rank0–5 各 1 行，rank6–7 没有专家 payload。
- 不同 DP 有不同有效行数时，MC2 统一 padding 与 AllToAll 无 DP-wide padding 的 offset 会不同；两个分支即使总行数相同，bank 与 peer 矩阵也不一定相同。

## 更大的待确认边界：Kimi SP padding 与 MC2 mask

`E=16, EP=8, top-k=8`：

- 32 个有效 token 经 TP8 SP，每个 rank 有 4 个有效局部行：256 assignments，每个目的 rank 的两个 banks 各 16 行。
- q1/B1 若按 TP8 padding 后每 rank 物理 1 行：AllToAll 全行路由条件有 **64 assignments**；若 MC2 的 `global_bs=0` 且 mask 只保留真实的一个局部 token，则为 **8 assignments**。

这是两个明确的物理调用域，不是按误差挑选的候选。没有获取 deployed helper 和 mask 内容前，不能据此改变正式 case 的默认路由，或断言 capture 采用其中之一。

## 证据边界

- 已核对 force 表 SHA `bd85c5f4…`、router `aaea89c2…`、prepare `19d7d17c…`、dispatcher `86d316b1…`，与既有本地审计相符。
- 保留的 Kimi `source_image_883670fc` fixture 仅包含 caller／AttnRes，没有对应 router／prepare helper。启动记录 `enable_force_eplb=True` 只能确认开关，不能确认算法版本。
- 现有 V4.1 c64 compact receipt 可确认 `MoeDistributeDispatchV3/CombineV3` 活动，但该 transport 记录为 `no_tensor_candidates`，没有恢复 mask 内容或实际 group-list histogram。
- 8 个纯算术控制已保留完整 peer assignment 矩阵和每 rank bank 计数，全部通过 assignments 总数守恒；未把算子耗时或 group-list capacity 当成路由证据。

后续如要表达精确 source 条件，应显式接收每个物理来源的 prepared rows、有效 mask／valid rows、EP rank mapping、expert placement 和通信分支，再从同一 ID 规则同时导出 bank histogram 与 peer 矩阵。当前不增加自动选择或 fallback，也不改理想 `balanced` 声明。
