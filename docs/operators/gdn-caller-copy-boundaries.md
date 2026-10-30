# GDN decode caller 的拷贝边界与有限流水建议

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

本审计流式读取六批已有 A5 Qwen3.6 / Qwen3.8 高并发采集，不执行设备程序，不从残差推导启动延迟。完整描述符、逐组 count/mean/min/max、PMU、源文件 SHA256 保存在 审计 JSON，可用旁边的 `.producer.py` 重建。六批共 13,776 次 recurrence；每批 BA 两次、Q/K 两次、V 一次、convolution input 一次、Concat 一次、z Slice 一次、index-column Slice 一次，均与 recurrence 次数吻合。Zero fill 另计一次。

## 源代码和观测共同确认的内容

保留的 `gdn.py.txt:319–329` 从 QKVZ/BA 投影创建 strided views，BA `.contiguous()` 发生在 padding 裁切之前。`346–373` 先创建 caller output zero，执行 core，再把 `z` reshape 为 `[tokens * heads, head_dim]` 后送入 norm。`725–737` 明确把 core 结果写回 caller output。

多 token、多 head 的 z 行中间有 QKV 存储间隔：Qwen3.6 投影行宽 2048、选中 z 宽 768；Qwen3.8 分别为 4096、1536。合并 token/head 轴不能保留这些间隔，现有 `TensorLayout.reshape_view` 对六种真实形状均返回不能共享存储。采集中 recurrence 之后的 Slice 正好是 `[T,16,128]→[T,6,128]` 或 `[T,32,128]→[T,12,128]`，与 z reshape 的选中区域一致。

已据此修正 caller：经共享 `TensorLowering.reshape` materialize z，norm 依赖该结果，继续只读取选中的 payload；norm 本身既有读量保持。token 或 head 为 singleton、reshape 可保留地址时仍是免费 view。图 padding 的 z 投影保持 padded rows，recurrence 继续只执行真实 rows。没有删除已有 core-output assignment，也没有为同一次 z reshape 再造一份 copy。

六批全部相关 Slice / ConcatD / TensorMove 均报告 `AI_VECTOR_CORE`，但实际 BlockNum 并不统一：

| 几何 | BA | Conv input | Q/K | V | Concat | z Slice | Index column | Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 T16 | 1 | 16 | 8 | 16 | 16 | 16 | 1 | 6 |
| Qwen3.6 T160 | 1 | 54 | 54 | 54 | 53 | 54 | 1 | 60 |
| Qwen3.6 T512 | 2 | 64 | 64 | 64 | 42 | 64 | 1 | 64 |
| Qwen3.8 T16 | 1 | 64 | 16 | 16 | 32 | 16 | 1 | 12 |
| Qwen3.8 T128 | 1 | 64 | 64 | 64 | 42 | 64 | 1 | 64 |
| Qwen3.8 T512 | 3 | 64 | 64 | 64 | 42 | 64 | 1 | 64 |

z Slice 单次均值分别为 2.342、3.798、4.734、2.392、4.209、5.061 µs。它们是完整 kernel duration，不是读阶段时间，也不是可追加到 analytic duration 上的启动截距。

两批 T512 在 recurrence→norm 之间另有 `[T,H,D]` TensorMove 和 `[T*H*D]` TensorMove，各一次/recurrence。前者与 caller output assignment 的 geometry 一致，后者的确切 caller site 未由所保留代码/CSV 证明。较小四批未导出这两类 TensorMove。捕获只保留 kernel/op-statistic/step CSV，没有 runtime memcpy trace，因此不能据缺行宣布 source assignment 消失或判定它走 SDMA；更不能把大 shape 的两次 TensorMove 硬加到所有 shape。

`ND`/`NCL` 是导出格式标签，不直接等价于 Python view strides。审计只用源代码构造 stride，保留 CSV 标签供配对；不依据格式名填造 offset。INT32 index-column copy 与 BF16 payload copy 的 dtype、有效区域和一次调用边界分别保留。

## 物理管道证据与当前缺口

保留 CANN `arch3510-physical-services/cann/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h:53–97` 明确标注 VecCore 的 GM→UB 为 `PIPE_MTE2`；`99–123` 的 UB→GM 为 `PIPE_MTE3`。API 还区分 DataCopy 的 32-byte 单位和 DataCopyPad 的 byte 粒度。结合采集的 AI_VECTOR_CORE 与非零 AIV MTE2/MTE3 PMU，可确认 copy 有 AIV 读取、UB 中转、AIV 写回两个物理方向。

现有 `TensorLowering` 已正确产生选中 payload 的 read/write bytes，而默认 `ResourceEnvelope` 把它们作为可同时服务的聚合资源：没有实际 AIV owner、有限 UB slot 或读完后才可写的依赖，且读资源仍是 generic `mte2`。这是公共机制缺口，尤其会削弱 AIV 带宽扫描；不能把所有同名 copy 的实测时长反推成万能延迟来掩盖它。

本地保留的是 CANN basic API，未取得这批二进制的完整 Slice / ConcatD / TensorMove tiler。BlockNum、UB tile、slot 深度和实际 cache 命中不能由 PMU 占用率或上表几行反向拟合。因此以下是明确参数化的通用物理模型建议，不宣称重建 vendor dispatch。

## 共享 TensorCopy 的最小机制

1. 保留既有每个物理 materialization，只替换其执行 recipe。`source_layout(s)`、target layout、input/output dtype 决定 payload，不按 backing allocation 全量搬运；concat 的各 source segment 读一次、输出 payload 写一次。
2. 显式声明 tile bytes、slot 数与 owner selection。每个 tile 在相应 AIV 上依次执行 `read(HBM + mte2_aiv) → optional cast → write(HBM + mte3)`，UB slot 从读开始占用到写完成才释放；同 dtype copy 不凭空新增 Vector 算术，但仍占用物理 AIV 的 copy 管道。
3. owner 数限制为真实可用资源，logical programs 超过物理核时复用，不能将单核 BA/index copy 按 64 核总带宽定价。没有证实 dispatch 时显式使用声明的 analytic mapping，而不是按模型名或 shape 表选观测 BlockNum。
4. Split AIV profile 读仅使用 `mte2_aiv`，不再同时收取 generic `mte2`；统一/GPU 资源继续其显式配置，不隐式要求 Ascend UB。HBM 服务恒为 read+write，片上 padding 和 alignment 不直接扩大 HBM payload。
5. startup/minimum 只在外层 invocation 计一次。纯 copy 的 latency 由依赖和服务约束产生，不加 cadence、PMU scalar 时间或每 tile floor。大输入用有限 tile window；如果进一步压缩稳定循环，保留 detailed oracle 和 contention 近似标记。

最低验收应使用真实 Engine：单 tile 的读写时间相加（不能变成 max）；两 slot 可在同 owner 上允许下一 tile 读与前一 tile 写重叠；tail payload 守恒；1/64 active AIV 时没有借用闲核；late-input 前无服务；UB 全部释放；mte2_aiv/mte3/HBM 的独立扫描产生对应响应；dtype cast 只在需要时发生；现存 caller copy 数不因 recipe 替换变化。这些控制检验机制，不等于 whole-body 精度结论。

caller 本次仅补源代码必需的 z reshape materialization。由于它会增加真实应有的工作，不能承诺总残差必然变小；模型主体当前仍有其他未收敛机制，禁止保留抵消来制造达标。
