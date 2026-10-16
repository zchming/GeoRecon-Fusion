# VisionGeometryLab 科研基础环境

项目：`/home/zchming/projects/VisionGeometryLab`（WSL Linux 文件系统）
解释器：`/home/zchming/.venvs/visual-geometry/bin/python`

## 使用
```bash
cd /home/zchming/projects/VisionGeometryLab
source /home/zchming/.venvs/visual-geometry/bin/activate
python verify_environment.py
python -m pip check
python -m jupyter lab --no-browser
```
VS Code 使用 WSL: Ubuntu-24.04 打开此目录。已保存 `.vscode/settings.json` 和 `launch.json`；运行当前 Python 文件或按 F5 调试。Notebook 选择 Visual Geometry 内核。

## 安装选择
系统已装 Git、build-essential、CMake、Ninja、Python 3.12、venv、pip、Python 开发头文件和 pkg-config，保留现有版本。C/C++ 的 CMake/Ninja 编译运行检查通过。
PyTorch 使用官方稳定 CUDA 12.8 wheel 源，固定 torch 2.11.0，并由 pip 匹配 torchvision。网页选择器的静态标签版本有滞后，因此以正式 wheel 索引的可安装稳定版本为准。
驱动 617.42，GPU RTX 4070 SUPER 12282 MiB。未安装 Linux NVIDIA 驱动或系统 CUDA Toolkit。PyTorch 依赖可能包含名为 cuda-toolkit 的 Python 元包，它只在虚拟环境中选取运行库，不等于安装完整系统开发 Toolkit。
暂未安装 DUSt3R/VGGT，后续各自创建独立环境。

## Codex 后续工作目录
此聊天仍绑定原有生成目录，执行命令已显式使用 Linux 项目目录。
没有找到可验证的受支持接口修改“无项目任务文件夹”，未修改内部配置。
在 Codex 中添加本地项目，将文件夹设为 `/home/zchming/projects/VisionGeometryLab`；Windows 文件夹选择器可用 `\\wsl.localhost\Ubuntu-24.04\home\zchming\projects\VisionGeometryLab`。
按照官方文档，项目菜单 → Edit project → Add folder，添加目录后选择 Make primary。然后在该项目中开始新聊天，并先确认 pwd、WSL_DISTRO_NAME 和 Python 路径。
若希望所有科研项目集中管理，可将 `/home/zchming/projects` 添加为本地项目的主文件夹。

旧目录检查：本次开始时只有空 work/ 和 outputs/，没有代码或配置需要复制，旧目录保留。

官方参考：
- https://pytorch.org/get-started/locally/
- https://download.pytorch.org/whl/cu128
- https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html
- https://code.visualstudio.com/docs/remote/wsl
- https://learn.chatgpt.com/docs/projects
- https://learn.chatgpt.com/docs/reference/settings

验证结果以 verification.json、pip-check.txt 和 environment-versions.txt 为准。

## 最终验证结果
- PASS：全部指定包导入、pip check（No broken requirements found）。
- PASS：CUDA available=True，RTX 4070 SUPER，PyTorch CUDA runtime 12.8。
- PASS：512×512 GPU 矩阵乘法，与 CPU 结果比较通过；反向传播梯度有限。
- PASS：test_plot.png 已生成；Jupyter Notebook 内核执行通过。
- PASS：C/C++ CMake + Ninja 构建与运行。
- PASS：Windows VS Code 1.142.0，WSL Server 运行；扩展宿主 Jupyter 日志确认项目目录。所需扩展已安装。
- PASS：指定解释器 debugpy 启动与执行；已配置 F5 调试。尚未手动操作 GUI 断点。
- 未完成：Codex 无项目默认目录修改（无受支持工具或已核实设置）；此聊天原始绑定目录未改变，请按上文将 Linux 文件夹加入项目后继续。
- 网络记录：PyTorch 主包下载中断一次，pip 自动断点续传成功；最终安装验证通过。

复现时先创建新的 venv，按 requirements-torch.txt 安装 PyTorch，再安装 requirements-lock.txt 中其余包；CUDA wheel 需要上述官方索引。不要将此环境直接用于未来 DUSt3R/VGGT 依赖安装。
