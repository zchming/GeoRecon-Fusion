# DTU pose pilot v1：运行前固定协议

本文件与 `configs/benchmarks/dtu_pose_pilot_v1.json` 在模型运行前提交。数据提取结束后，`freeze.json` 固定全部输入 manifest、图片摘要、reference 摘要及三个方法配置摘要。执行器逐项校验，禁止静默重写已有运行。无测试集调参、自动重试或失败删行。

## 划分与条件

仅使用官方 SampleSet 内 scan1（dev，已在阶段2使用）和 scan6（test，未用于参数选择），light3。不是标准 DTU MVS 训练/测试划分，也不是全量 DTU Benchmark。每个场景四个条件，共8个实例、24个方法运行；同一对象的全部扰动版本留在同一划分。每个划分只有一个独立场景，四种条件彼此相关，不提供场景泛化置信区间或显著性结论。

| 条件 | 输入图像顺序 | 构造 |
|---|---|---|
| normal5 | 001,002,003,004,005 | 原始 rectified 图像 |
| reduced3 | 001,002,003 | 减少图像数量 |
| spaced3 | 001,010,020 | 扩大视图索引间隔；不把它预先等同于低重叠 |
| blur5 | 001,002,003,004,005 | 原始1600×1200 RGB上 Pillow GaussianBlur(radius=4)，保存PNG，再共同缩放 |

全部方法每个实例使用同一图片字节、顺序及 800×600 BILINEAR 公共输入。raw 继续官方518×392 crop，BA继续官方补黑边1024跟踪、518方形推理；这些方法必要差异保留，结果不解释成纯 BA 消融。无曝光扰动或“低纹理”标签：本轮没有标注依据。索引间隔只是一种构造，失败原因不能凭它直接断言低重叠。

数据来自 [DTU 官方 MVS 页面](https://roboimagedata.compute.dtu.dk/?page_id=36)，引用 Jensen et al., CVPR 2014。只作有界 HTTP ranges 提取，保持既有 archive size/ETag 校验、CRC与首次字节 SHA 记录；不下载整个6.3GB归档。每场景64MiB网络预算。数据、权重、vendor及大型输出不上传Git。

## 配置和预算

三个 baseline：官方 PyCOLMAP4.2.1 CPU SIFT/SfM，自估共享PINHOLE内参；官方固定VGGT1B raw；官方固定VGGT跟踪+BA，独立SIMPLE_PINHOLE无损规范化。模型/源码 pins沿用阶段2，未按scan6表现选择权重或参数。BA显式允许3/5张输入，复用阶段2已成功的forward后释放闲置VGGT模型调度，4096查询点、fine_tracking开启，其他优化设置不变。

所有方法每实例180秒worker预算；外层CLI最多210秒。8 CPU线程，seed42。每实例单次运行，最多24次；即使失败也不在看过测试结果后减少分辨率/点数重跑。GPU方法串行，allocator上限10.5GiB，给驱动与桌面留余量，物理GPU范围12GB。OOM和timeout均保留。Ceres默认CPU BA与CPU COLMAP按真实能力报告；不将结果描述成全GPU。

模型从本地加载；端到端计时包括进程、模型构造、权重校验、传输、推理、跟踪/BA与导出，区分纯forward和完整链路。逐运行记录GPU allocator峰值、reserved峰值、RSS；BA另有整卡100ms采样，raw没有整卡采样则标注缺测，不能用allocator冒充整卡值。下载/准备与评价时间分别保存，不能与推理计时混用。单机单次墙钟受文件缓存和cuDNN选择影响，不做高精度速度排名。

## 评价与失败判定

GT相机只由评价器读取，推理输入只有manifest与图片。图像覆盖率与场景完整输出成功率单独报告；success需status=success且所有输入相机均可用，不设事后精度阈值。部分注册记录partial及覆盖率；无模型、OOM、超时、进程错误均计入失败，不用占位相机替代。

报告每实例：全体输入相机对数量、可评价子集、相对旋转误差、有向平移方向误差、评价专用Sim(3)后的相机中心ATE（mm，至少三非共线中心）。平移反向为180°，中心位置误差与方向角不可混用。ATE不可计算时保留null及原因，不记为零。

AUC@5/10/20°分别针对旋转、平移方向及二者max联合误差。预注册定义为经验召回曲线精确积分：`sum(max(0,1-error/threshold))/全部输入对数`；缺失或方向未定义的相机对贡献0。每个实例先计算AUC，再按划分/方法平均，避免5图实例因相机对多而获得更高权重。不宣称此自定义pose pilot与其他论文的AUC插值实现可直接排名。

保存全部逐实例结果，再按dev/test分别汇总均值、中位数、计数、覆盖率与失败类型。成功子集的角度/ATE必须附可用样本数；不能用只成功的平均误差表示完整性能。四条件重复测量不视为四个独立对象。本轮不做DTU官方稠密几何accuracy/completeness/F-score：尚无可比COLMAP MVS链路，raw点图和BA稀疏点不能直接混为同一稠密任务。

## 验收与复现

先执行 `python scripts/stage3_benchmark.py audit` 核对阶段2的三份真实预测、产物SHA及独立GT评价；再执行`prepare`提取数据与冻结，提交冻结文件后执行`run`。脚本在项目已配置的三个独立venv间选择解释器。`run`默认拒绝覆盖或隐式恢复，`summarize`可只重建汇总而不重跑推理。

阶段3产物为冻结协议、逐实例结果、成本与失败分析。没有可靠性评分、置信度校准、选择/回退策略或融合；这些留待新的明确授权。
