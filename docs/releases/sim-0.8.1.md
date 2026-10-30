# MindIE Sim 0.8.1

0.8.1 在 0.8.0 因果建模引擎基础上新增整网与算子级四维效率度量（Cube MFU /
Vector MFU / MBU / Link Util / CV 融合重叠度）、反事实目标效率拟合（如内存受限
算子 0.80x MBU 屋脊线封顶）与硬件扩缩容 SLA 最大吞吐求解器，并完成了 A3/A5
全量 Decode（280 case）与 Prefill（15 case）长短序列校准及执行引擎热路径优化。

## 本版交付

- **算子级与整网四维效率剖析（`mindie_sim.prediction.efficiency`）**：
  - `analyze_operator_row` 与 `analyze_network_efficiency` 严格按算子物理工作量与
    实际执行服务（`executed_service`）拆解 `Cube_MFU`（按 `bf16` / `fp16` / `int8` /
    `fp8` / `w4a8` 分精度峰值）、`Vec_MFU`（独立于 Cube 的 AIV 向量峰值）、
    `MBU`（HBM 带宽利用率）、`Link_Util`（片内/片间互联带宽利用率）以及
    `CV_Overlap`（Cube-Vector 融合算子双流水重叠率，支持真实 PMU `aic_mac_time_us` /
    `aiv_vec_time_us` 与理论下界回退）。
- **多管道屋脊线目标拟合（`evaluate_target_efficiency` / `fit_efficiency_targets`）**：
  - 按 `t_target = fixed_ms + max(t_hbm / target_mbu, t_cube / target_cube_mfu, t_vec / target_vec_mfu, t_link / target_link_util)`
    计算反事实目标耗时，保留算子启动固定开销 `startup_ms` 与包含式 `minimum_ms` 下界，
    支持生成带确定性签名的 `service_regimes`。
- **反事实硬件扩缩容与 SLA 吞吐求解（`scale_case_hardware` / `solve_hardware_sla_throughput`）**：
  - 支持对 `hbm_bw_scale`、`cube_flops_scale`、`vector_ops_scale`、`link_bw_scale` 与
    `hbm_capacity_scale` 独立缩放（同时缩放设备速率、自定义服务与 `service_regimes`），
    在给定 Decode 步耗时 SLA（如 `<= 30ms`）与 VRAM 容量约束下联合求解最大可行并发
    `B*_SLA`、`B*_VRAM` 与吞吐增益。
- **Prefill / Decode 算子物理建模与执行器热路径加固**：
  - 完善 KDA recurrent / chunk、GDN recurrent / chunk、MXFP4 grouped MoE、MLA / GQA
    在跨 TP 与长序列（64K–128K）下的程序展开与多次迭代复用；
  - 消除 `InvocationTiming` 与 `RegimePolicy` 在逐算子匹配中的重复深拷贝与 JSON
    序列化开销，保证大图分析的内存与耗时有界。
