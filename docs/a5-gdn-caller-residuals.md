# A5 source-GDN caller 剩余误差与边界

本轮只分析 `7046935c` 已保存的 analytic-core 控制和已有 CSV，没有修改 runtime，
没有重新跑完整 body。新增报告为 `data/verification/a5-gdn-caller-residuals-20261030.json`。
13,776 次调用均由同流 BA 输出后的 contiguous copy 开始，到输出 norm/gate 结束；
使用 Decimal 对原始微秒时间戳做差，避免大 epoch 时间浮点舍入。

| 场景 | 预测 caller us | 实测 kernel activity us | 实测 caller span us | 内部 uncovered us | 原报告之外 TensorMove us |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 c4 | 15.516 | 58.387 | 60.617 | 2.230 | 0 |
| Qwen3.6 c40 | 73.044 | 131.350 | 133.871 | 2.521 | 0 |
| Qwen3.6 c128 | 154.749 | 211.252 | 212.200 | 0.948 | 7.840 |
| Qwen3.8 c4 | 24.539 | 75.256 | 77.603 | 2.347 | 0 |
| Qwen3.8 c32 | 118.138 | 174.919 | 177.562 | 2.644 | 0 |
| Qwen3.8 c128 | 301.969 | 363.843 | 364.825 | 0.982 | 8.731 |

实测 activity 包含状态索引 Slice 和下面描述的两个额外 TensorMove；预测尚无这些
调用。此处同流 kernel 没有时间重叠，因此 activity sum 与 interval union 数值相等。
预测图中15个算子也顺序执行，故图 span 与 isolated sum 相等；这是该调用的观察结果，
不是适用于整图的可加性假设。内部 uncovered 仅0.95–2.64us，不能把40–60us剩余
误差归为 Host 空泡。它也不能自动当作某项硬件固定延迟进行校准。

八个已建模 materialization 的 signed gap 为每次 −17.28 至 −28.24us，Q/K L2norm
合计 −3.63 至 −7.03us。它们都已进入预测图，当前 `TensorLowering` 主要保留有效
payload 读写，copy 同 dtype 时没有额外 vector 工作；A5 无通用 invocation floor。
这意味着问题是 copy/小核执行机制细度不足，不能再简单增加一组相同字节的 copy，
也不能把每个微核都乘倍率。要提高细度，应针对真实 wrapper 的独立调用、活跃 AIV、
最小搬运粒度、源 stride 和已证实的启动/尾阶段建模。

| 场景 | 八次 materialization gap us | Q/K norm gap us | gating gap us | convolution gap us | recurrence gap us |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 c4 | -17.282 | -3.627 | -4.516 | -6.693 | -2.604 |
| Qwen3.6 c40 | -25.937 | -4.776 | -9.787 | -10.441 | +4.651 |
| Qwen3.6 c128 | -28.235 | -5.887 | -12.281 | -11.302 | +23.401 |
| Qwen3.8 c4 | -20.491 | -4.188 | -7.079 | -6.917 | -2.560 |
| Qwen3.8 c32 | -27.781 | -5.408 | -12.302 | -14.484 | +16.399 |
| Qwen3.8 c128 | -27.008 | -7.033 | -19.261 | -32.392 | +47.878 |

高 batch 的 analytic recurrence 明显高估，抵消了其他路径的低估。改进 native
recurrence 后，caller 总误差可能暂时变差；不能据总误差上升撤销独立证据支持的修正。

明确可定位的遗漏有两类。第一，保留 `gdn.py.txt:138–143` 的
`cache_indices[:, 0].contiguous()` 在每次调用都对应 `(512,4) → (batch,1)` INT32
Slice，原报告已单列，尚未建模；其输入是状态索引表，不是 QKV payload。
第二，在 c128 的全部调用中，`aclnnInplaceCopy` 展开为 NCL TensorMove、NCL Slice、
ND TensorMove 三个实际 kernel，当前 `copy_into` 只发出一个复制算子。Qwen3.6 中
shape 分别是 `[512,6,128]`、`[512,16,128] → [512,6,128]`、`[393216]`；Qwen3.8
对应 `[512,12,128]`、`[512,32,128] → [512,12,128]`、`[786432]`。
新增报告保留完整名称、shape、format 与行号。源码 `gdn.py.txt:735–736` 证明这里有
caller-owned output assignment，但尚未保留 ACL 内部选择该三核分支的充分条件。
因此可明确记录 wrapper 展开遗漏，不能写成 `batch == 128` 的模型补丁；需要从物理
布局、stride、scratch/alias 与调用实现声明选择该分支。

Gating 当前保留既有 BF16 queue 参数、FP32 DMA 输出和实际64程序网格。
13个逻辑逐元素 vector 操作仍是粗略表示，尚未展开 exp/log/reciprocal 的实际服务、
scalar 控制及同步；旧 primitive 参数未校准这些部分。高 batch 的 −12.30/−19.26us
不能通过重新拟合 beta queue 抹平。Qwen3.8 c128 convolution 残差32.39us更值得优先
检查：源码 `causal_conv1d.h:1249–1262` 每个 sequence/channel 任务读取 query起止、
cache index、accepted count；模型尚未表示这些 GM scalar metadata 和控制服务。
源码中的多次 SetFlag/WaitFlag 也不能仅由有效 payload 字节推导其耗时。已有模型的
weights/history/输入 reread 与 state writeback 字节存在，应先核对新增服务而非重复
增加这些 payload。

Q/K norm 仍为 `gdn_prefill.decode_core` 的5次元素操作粗略模型。参考源码
`vllm-ascend-v4-default-comm/vllm_ascend/ops/triton/fla/l2norm.py` 明确包含 FP32 cast、
平方、跨列归约、rsqrt 与输出；但该镜像的 `kernel2_loop` 名称/dispatch 与 capture
的 `l2norm_fwd_kernel2` 尚未证明一致，不能将其 MBLOCK=69 或固定64程序当作本采集
事实。当前每个 norm 小于5us，按抓大放小原则排在上述重型路径之后。

六组完整 body 复测准备使用独立 `source-gdn-native/` 场景和
`data/verification/a5-source-gdn-body-20261030.producer.py`。方案只读 profiler 已完成的
六个 body receipt，保持 q/batch/context=既有物理声明，保留 `bc375e61` 基线，
不重复提取 CSV 或重跑基线。新增 `recurrent_implementation="arch35_fp32"`，
旧 analytic 控制场景及 SHA 保留；每 worker120s串行。准备本报告时尚未执行完整body。
新旧 runtime 还包含矩阵 AIC read pipe 等变化，因此完整 body delta 不会单独归因
于 recurrence。
