# Windows REVE 服务

此环境面向原生 Windows、Python 3.11 与 CUDA 12.8。独立于项目历史 torch 2.0 环境；不要在此环境执行 `pip install -e .`，旧项目依赖会要求降级 torch。

## 安装

在 NCC 仓库根目录，使用已有 Python 3.11 创建 `.venv`：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-reve-windows.txt
.\.venv\Scripts\python.exe scripts/download_reve.py
```

下载脚本固定官方 `brain-bzh/reve-base` 与 `brain-bzh/reve-positions` 的提交，保留许可证并生成 SHA-256 清单。默认目录为 `checkpoints/backbones`，不要把权重提交到 Git。

本机验证环境的完整版本记录在 `requirements-reve-windows.lock.txt`。重建时可使用 `python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-reve-windows.lock.txt`。

## 本机验证记录（2026-10-10）

NCC 基线为 `mystery@9fae013`，RTX 5060，驱动 591.86，Python 3.11.16，torch 2.7.1+cu128。真实基座的合成数据检查通过：冷推理 188.6 ms，7 次热推理中位数 9.1 ms；均衡合成标签的线性头 3 步更新和 LoRA rank 4 / batch 1 单步更新均通过。PyTorch 峰值分配 330.7 MiB、预留 350 MiB（不含驱动和其他进程，不代表完整训练显存需求）。原始输入为 BCIGo 32 通道布局，现有预处理会过滤位置库不支持的通道。

直接 WebSocket 与 Vite `/ws/model` 代理均通过真实推理检查。后端相关测试 48 项通过。原始报告位于 `runs/windows-reve-smoke.json`、`runs/windows-reve-websocket.json` 与 `runs/windows-reve-vite-proxy.json`。尚未做真实 EEG 分类效果、长时间训练或自动模型接管验收。

## 验证与启动

```powershell
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe scripts/smoke_reve_windows.py
.\.venv\Scripts\python.exe scripts/serve_reve_model.py --device cuda --task smr_control --strategy none --no-lora --no-tcp
```

服务保持运行，在第二个终端验证协议：

```powershell
.\.venv\Scripts\python.exe scripts/smoke_reve_windows.py --url ws://127.0.0.1:8768/v1/model --output runs/windows-reve-websocket.json
```

检查脚本用合成 32 通道、250 Hz、2 秒窗经过现有预处理与真实基座推理，报告冷启动与热运行延迟；另运行均衡合成标签的线性头更新、rank 4 / batch 1 的 LoRA 单步反向传播并检查参数发生变化。报告落在 `runs/`。GPU 不可用时直接失败，不回退 CPU。此检查不衡量被试准确率、不导出或激活合成数据训练的头，也不代表任意 batch、rank 和长时训练都能在显存内运行。

未提供分类头时，服务使用零初始化头，输出均匀概率，并在 hello 中发送未训练提示。实际脑控必须使用与 encoder 和输入处理匹配、经过被试数据验证的分类头。下载基座不等于获得个体化 SMR 分类器。

## 与 Passive-BCI 一起运行

两个仓库放在同一父目录。停止旧的 Vite 后，在 Passive-BCI 根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-local.ps1
```

该脚本设置 NCC 路径、独立 Python 和 CUDA，启动网页。打开普通 SMR 页面或使用模型面板启动 REVE。`-Bootstrap` 表示使用基座和未训练头进行链路验证；不加载研究机历史被试的默认 LoRA 或头。已有训练好的模型时不要使用这一选项，改用模型面板选择匹配的头。

## Stieger 四分类部署（2026-10-11）

默认启动脚本现使用 `checkpoints/adapters/stieger2021_4class_reve_lora_r16_mi5init/best.pt`，来源为用户指定的 COS 对象。使用其 rank 16 / alpha 32 LoRA 和内置的四分类头，类别顺序为 `left_hand/right_hand/both_hand/rest`，不会额外加载历史 S02 头。`-LoraCheckpoint` 可显式指定其他兼容文件，`-Bootstrap` 保留未训练基座验收模式。

文件大小 289965156 字节；本地 MD5 与远端一致：`681883d9a88cd6c3de1564dccea0a645`；SHA-256：`bf81d7953bfac984ca4a35727c4d9bf63f451c7c33343a77bcd9c752901f4d20`。已核对全部 140 个合并参数，无缺失；分类头与 checkpoint 逐项一致。BCIGo 32 导和 Neuracle 59 导合成输入均通过 CUDA 推理，热推理中位数约 9.4 / 9.0 ms；网页 WebSocket 已确认返回该 adapter 版本和非均匀概率。

可复验命令：`.\.venv\Scripts\python.exe scripts/validate_reve_adapter.py checkpoints/adapters/stieger2021_4class_reve_lora_r16_mi5init/best.pt`。报告在 `runs/stieger-deployment.json` 和 `runs/stieger-websocket.json`。

部署沿用 NCC 原有 serving 预处理（2 秒、200 Hz、特征 L2 归一化）。训练 checkpoint 记录 `n_times=800`，但没有足够的采样率与预处理信息证明与当前 400 点输入等价；不能把其训练分数当成本机在线准确率。真实 EEG 分类效果及训练／推理输入约定仍需验证。
