# A5 tiny Cube 的独立计时边界审计（2026-10-30）

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

本次只读审计 `/Volumes/data/mindie-sim-data/a5/a5_kernel_floor_tiny_cube_20261030/`，没有执行 producer 或启动设备。7 个条件各 50 次，共 350 个唯一物理 kernel，均与 `op_statistic.csv` 的数量和总时长精确一致。完整形状、数据类型、BlockNum、源文件 SHA256 和精确时间统计保存在 审计 JSON。观测角色仍为 `unassigned`，本报告不声称独立 heldout 精度或模型主体精度。

## 实际测量边界

生产脚本先创建输入和权重，20 次 warmup 后同步；随后每个条件独立进入 NPU-only profiler，在单个 Python eager loop 中调用 50 次算子，末尾同步。输出由每次调用返回，没有预分配 `out=`，没有图回放，也没有 Host submit-only 或 NPU event 计时器。量化接口 probe 选择了 `w8a8_pertoken`，没有回退到其他量化模式；CPU copy 仅确认完成，脚本未用数值 reference 校验正确性。

设备报告为 `Ascend950DT_9582`、32 AIC / 64 AIV；软件为 torch `2.10.0+cpu`、torch_npu `2.10.0.post7`、vllm `0.30.0`。本批 summary 未记录 CANN 版本，不能借上一批记录补填。每次 CSV 行是一个完整 device kernel duration；MIX_AIC 的 AIC/AIV PMU 时间可重叠，不能相加，也不能分别加两份 invocation floor。

所有矩阵均为 M=1、K=N。BF16 输入/输出都是 ND；量化为 ND INT8×INT8，FP32 `[N]` channel scale 和 FP32 `[1]` token scale，BF16 ND 输出。

| Kernel / K=N | 权重逻辑字节 | AIC / 混合 AIV BlockNum | Device min / P50 µs | Start cadence P50 µs | End→start gap P50 µs |
|---|---:|---:|---:|---:|---:|
| BF16 MatMulV3 / 64 | 8 KiB | 1 / 0 | 1.578 / 1.775 | 4.590 | 2.811 |
| BF16 MatMulV3 / 128 | 32 KiB | 2 / 0 | 1.889 / 1.979 | 5.622 | 3.581 |
| BF16 MatMulV3 / 256 | 128 KiB | 4 / 0 | 1.919 / 2.017 | 4.895 | 2.739 |
| BF16 MatMulV3 / 512 | 512 KiB | 8 / 0 | 2.284 / 2.422 | 6.174 | 3.760 |
| INT8 QuantBatchMatmulV3 / 128 | 16 KiB | 8 / 16 | 3.037 / 3.355 | 13.862 | 10.335 |
| INT8 QuantBatchMatmulV3 / 512 | 256 KiB | 32 / 64 | 5.322 / 6.374 | 16.572 | 9.730 |
| INT8 QuantBatchMatmulV3 / 4096 | 16 MiB | 32 / 64 | 8.379 / 9.672 | 15.863 | 6.149 |

用 `Decimal` 将原始 µs epoch 时间转换为整数 ns，先相减再转浮点。343 个同流相邻间隔全部非负；原 producer 先将 epoch 转为 float，生成了 3 个假负间隔。INT8 K512/K4096 的最小真实间隔都是正 22 ns。本次没有删掉首个较大间隔或其他采样；首间隔及去首间隔诊断另列，主统计保留全部样本。

上表 cadence/gap 包含框架、分配、提交、队列以及其他未分解间隔，不能唯一归因成 Host ACL 调用或硬件启动。与旧大矩阵微测相同，复用权重且 warmup 后测量只能说明 warm workload，不能证明每次 HBM 全量读取，也不能证明全部命中 L2。原 producer 关于“权重小于 128 KiB”“隔离纯启动”或“操作数全部命中 L2”的概括超出了实际证据：有 128 KiB、512 KiB、16 MiB 条件；非零加载、Cube、Fixpipe 和量化输出仍是 useful work。

## 参数吸收建议

普通非空 library Cube 可以把 **1.578 µs** 作为当前 A5 的保守工程 minimum，`startup=0`，完整物理调用仅执行一次 `max(program_duration, minimum)`。它来自最小 BF16 单核 MatMulV3 的 inclusive device minimum，不是纯硬件启动截距，也不是所有自定义 Cube 的物理下界。真实 empty AIC 的 0.439 µs minimum / 0.477 µs P50 继续保留在显式空核类；显式 measured-program class、规则和用户 override 应优先于普通类。

建议另设 **W8A8 per-token mixed matrix → BF16 output** 调用类，以 **3.037 µs** 为工程 minimum，`startup=0`。选择必须由 implementation / quantization / output contract 决定。该最小观测本身已使用 8 AIC + 16 AIV 并完成 scale/输出，不能直接外推到 W4、MXFP8、Grouped GEMM、NZ 权重、INT32 输出或其他 SKU。若将它作为这些域的先验，必须另行显式声明假设，不能标成实测支持。

K512 的 5.322 µs 和 K4096 的 8.379 µs 伴随 grid、工作量及传输改变，不能设成通用 INT8 固定底板；更不能把当前整步矩阵残差的 6–7 µs 填进去。精确形状的 P50 可定义独立 opt-in measured program class，但不能用 P50 与 min 的差值伪造 additive startup。现有显式算力、搬运、epilogue 程序仍保留，minimum 只做 inclusive envelope，不能追加一次 epilogue 或每 tile 收费。

与此前 [空核与 useful Vector 审计](a5-microbench-boundary-review.md) 合看：empty AIC/AIV、ordinary nonempty BF16 Vector、ordinary nonempty BF16 Cube、W8A8 mixed output 是不同边界。本报告未修改任何 hardware profile 或 runtime；Host 参数仍由独立直接 ACL 计时证据定义。

## 复现与小夹具

```sh
.venv/bin/python -m tools.analyze_a5_tiny_cube_boundaries \
  /Volumes/data/mindie-sim-data/a5/a5_kernel_floor_tiny_cube_20261030 \
  -o data/verification/a5-tiny-cube-boundary-review-20261030.json
.venv/bin/python -m pytest -q tests/test_a5_tiny_cube_boundaries.py tests/test_a5_microbench_boundaries.py
```

新夹具保留 BF16 K64 原始数据行 1–4，以及 INT8 K512 原始数据行 47–49（CSV header 不计）。它们分别检验首个提交间隔不能污染 device duration、混合 AIC/AIV 仍是一条任务以及 epoch 浮点制造的假负间隔。完整 350 行均在外部原始 capture 中审计，未用这些小夹具代替全量结论。
