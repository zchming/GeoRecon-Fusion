# GeoRecon-Fusion

**Geometry-Reliability-Aware Hybrid Multi-View 3D Reconstruction**
基于视觉几何基础模型与经典几何融合的可靠性驱动多视图三维重建系统。

研究问题：经典 SfM/MVS 与视觉几何基础模型分别何时失败？能否利用推理时无需真值的几何可靠性进行选择或融合，并改善精度、完整度、成功率与计算成本之间的权衡？融合有效与否尚待实验验证。

## 当前进度：阶段 0

已完成环境与已有实验盘点、官方实现核查、数据候选与最小实验方案。**尚未实现统一系统，没有正式 Benchmark 或融合性能结果。** 当前仅提交阶段 0 交付物；本机已有学习实验与权重保留在原目录，未整体发布。

- [阶段 0 报告与后续验收标准](docs/stage0/report.md)
- [实际环境检查结果](docs/stage0/environment.json)
- [官方版本与来源核查](docs/stage0/official_checks.json)
- [历史学习实验的证据快照](docs/stage0/legacy_evidence.json)
- [完整项目要求与阶段边界](docs/project_brief.md)
- [此前的环境说明，含过时信息，仅供追溯](docs/stage0/legacy_environment_setup.md)

## 复查环境

在项目根目录，用装有科研依赖的 Python 运行：

```bash
python scripts/check_stage0_environment.py --output /tmp/georecon-environment.json
# 可选：官方站点与权重的 HEAD 可访问性检查，不下载权重
python scripts/check_stage0_environment.py --network --output /tmp/georecon-environment-network.json
```

脚本保存逐项状态；缺少 COLMAP 或模型依赖会明确记录，脚本完成不代表全部检查通过。当前机器的确切命令见报告。尚无可安装的 GeoRecon 包或端到端重建入口。

## 后续路线（均未开始）

1. 统一数据与最小评测链路。
2. COLMAP、VGGT 与官方 BA 的小样本 Baseline。
3. 固定 Benchmark 与失败分析。
4. 几何可靠性分析。
5. 选择/融合与消融。
6. 教学版 SfM（可按用户指示提前）。
7. 复现、可视化、成果与简历整理。

每阶段须得到用户明确指示后才开始。原始大数据、权重和大型生成产物不进入 Git。VGGT 官方已有跟踪、导出和 BA 均属于 Baseline，不作为原创贡献。
