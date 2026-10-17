# 阶段6：最小教学版双视图SfM验证

**本次授权的双视图教学子步骤已完成。** 实现SIFT匹配、Essential Matrix鲁棒估计、四组相对位姿选解、显式DLT及正深度/视差/重投影验证。没有PnP、增量注册、BA、自标定、稠密重建、新Benchmark或阶段7工作；不把此最小子步骤称作完整增量SfM。

## 阶段5复核与协议边界

stage5_audit.json核对了阶段5验收、冻结规则、所有记录的源码/输入/24个消融主产物哈希：工程与证据验收通过，稳定收益未证实。原方法和失败目录保持原状，没有把教学输出接入选择/融合/优化或重新排序模型。

配置、输入和实现提交 **9e8e030** 后才运行全部真实图像对与控制。使用本地既有DTU scan1图像，复用1600×1200→800×600的Pillow双线性像素中心变换；没有下载新数据。DTU原图来自既有Rectified目录，本教学模块采用已知无畸变K。

与阶段2/3 COLMAP自估内参协议不同：这里明确允许标定K作为算法输入，转换为Knew=A*K；真实R/t、点、尺度只由独立评价脚本读取。运行输入有严格字段白名单，只含两张图/观测及各自K；进程拒绝读取reference/calibration目录或stage6_truth。它不能被宣传为无标定或绝对无真值信息的方法，也不能与先前不同内参协议的模型直接作性能排名。

## 真实图像对

|图像对|SIFT关键点|互检匹配|E内点|保留三角化点|旋转误差°|平移方向误差°|状态/失败|
|---|---|---:|---:|---:|---:|---:|---|
|dtu_scan1_001_003|3809/4096|427|404|404|1.6422|1.9652|success / —|
|dtu_scan1_001_020|3809/1357|4|0|0|—|—|no_reconstruction / insufficient_matches|
|dtu_scan1_blur_001_003|504/678|101|86|86|1.0445|0.4970|success / —|

001/020仅4对互检匹配，按冻结20对门限停止，未换图、降低阈值或编造位姿；索引间隔本身不是已测量的重叠率。模糊对在本例方向误差更小，不代表模糊有益或稳健性更好：匹配子集和RANSAC都会变化，只有一个物体的少量教学图像对。

成功表示内部检查通过，不是GT正确率。已知K条件下的旋转/平移方向误差仍非零；低重投影误差不保证真实几何。两视图的尺度不可恢复，norm(t)=1、首相机I/0；不计算需要≥3非共线相机的ATE，不把误差缺失填零。

## 合成与退化验证

|控制|输入类型|结果|含义|
|---|---|---|---|
|blank_pair|images|no_reconstruction / insufficient_matches|关键点/匹配为0|
|identical_dtu_image|images|no_reconstruction / identical_correspondences_no_baseline|同一真实图像复制，不构成新真实场景|
|nonplanar_exact|correspondences|success / —|无噪声位姿/尺度约定与DLT数学检查|
|nonplanar_noise_outliers|correspondences|success / —|0.2像素噪声+32/160随机外点|
|planar|correspondences|no_reconstruction / homography_dominated_ambiguous_geometry|H主导，保守拒绝歧义|
|pure_rotation|correspondences|no_reconstruction / homography_dominated_ambiguous_geometry|没有可恢复的平移基线|
|tiny_baseline|correspondences|no_reconstruction / homography_dominated_ambiguous_geometry|本例H主导；单独数学测试也验证小视差风险|
|too_few_matches|correspondences|no_reconstruction / insufficient_matches|7对少于工程门限20，不是五点法的数学最小值|

8个控制的预期状态全部满足。无噪声非平面160点均接受，旋转/方向误差数值上为0，单位基线XYZ均距约8.8e-13；这是人为生成fixture的数学验证，不是真实重建性能。含噪声/外点控制接受128个真实匹配、保留假匹配0，旋转误差0.2735°、方向误差2.3793°，oracle尺度后的中位XYZ误差约0.3354个单位基线，不能省略噪声导致的结构误差。

平面/纯旋转/极小基线由H竞争门保守拒绝，不意味着H主导必然证明平面或旋转，也不表示所有此类情况只能失败。本模块未做单应分解或更复杂退化处理。

## 中间过程与可复算性

核心代码[teaching/two_view.py](../../src/georecon/teaching/two_view.py)展示互检、归一化E、四解计数、DLT和逐点掩码；[运行封装](../../src/georecon/teaching/run.py)负责严格输入、真值隔离和统一输出。[数学与产物 walkthrough](walkthrough.md)解释像素变换、E四解、C2=-Rᵀt、单位尺度及数组索引映射。

每例outputs/stage6_two_view_v1/<case>/保存prediction/steps/invocation/process、全部几何候选及稀疏点；图像模式另有features_matches.npz与matches.png（青=接受点、绿=E内点未接受、红=其余）。失败时不导出接受点云或已注册相机，但保留调试候选和完整原因。NPZ、图像与本地数据不入Git；逐例小JSON、配置、源码、测试、报告已归档。

check_stage6.py重新提取图像特征、匹配并重跑几何，与归档数组/步骤一致，逐点验证单位基线、旋转矩阵、双正深度、视差和两幅图误差；核验无位姿参考读取、输入和产物SHA、预期退出码。120项不同测试在可靠性环境（跳过1原生COLMAP）和COLMAP环境（跳过20个依赖OpenCV/SciPy的测试）交叉覆盖，所有跳过项在另一环境实际通过。

## 资源与限制

11例算法进程主体耗时合计1.3024s，最高进程RSS 0.1938GiB；控制器测量窗口合计4.0228s，含nvidia-smi采样开销，不当作纯算法速度。CPU线程8，OpenCL关闭，无CUDA/模型调用，新增GPU分配0。整卡采样使用503.0–508.0MiB（含桌面），未在暴露的compute列表发现教学子进程；WSL列表可能受限，不将整卡占用当作任务显存。详见cost_accounting.json和process.json。满足12GB约束，无新模型、权重或包安装。

已知局限：K必须给定且畸变已处理；只有两视图，尺度任意；RANSAC/互检并不消除所有歧义；H门可能过度拒绝；没有PnP、增量轨迹管理、BA、回环、稠密结构或大场景验证；真实示例仅同一DTU物体的少量图像对，合成对应测试没有验证图像特征检测性能。

## 截至目前的贡献

累计已有统一坐标/失败/资源协议和可审计实验管线，推理几何诊断、开发集冻结的选择/过滤/融合/联合优化及必要消融，并保留负结果。本步增加可解释的已知K双视图教学链、四解/DLT中间产物与退化验证，支持理解和展示几何基础。

SIFT、E分解、RANSAC、DLT、COLMAP/VGGT及现成数值库不是本项目原创；工程集成、验证工具与有界经验发现不能直接等同于算法新颖性或SOTA。阶段5稳定收益仍未证实，教学成功也不改变这一结论。[累计贡献与边界](../contributions.md)。

## 运行与停止边界

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=/home/zchming/.venvs/georecon-reliability/bin/python
# 依赖本地已有prepared input；输出必须是新目录。
$PY -m georecon.cli run-teaching-sfm --input data/stage6_pairs/dtu_scan1_001_003/input.json --config configs/teaching/stage6_two_view.json --output outputs/stage6_reproduction
# success退出0；普通无法重建退出1；输入/协议错误退出2。
$PY scripts/evaluate_stage6.py
$PY scripts/check_stage6.py
$PY -m unittest discover -s tests -p test_teaching_sfm.py -v
```

初始输入由prepare_stage6.py在本地既有DTU数据上一次性构建，输入锁存在时拒绝隐式重建；每个准备输入仅含K和图像/观测，真实位姿参考在单独data/stage6_truth。准备/评价访问参考不等于运行进程访问参考，输入假设均已明确。

本次仅完成阶段6最小双视图子步骤，停止于此；不自动实现完整增量SfM，不进入阶段7成果整理或UI。
