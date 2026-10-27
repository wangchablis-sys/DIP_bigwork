# Practice 1 部署与运行说明

本文档说明如何在 Windows 10/11 上部署并运行 `practice_1` 中的极暗 RAW 图像增强程序，包括基础增强、SIED 单次推理、SIED 三次特征融合、粉色桌面界面，以及 PSNR、SSIM、LPIPS 指标计算。

## 1. 项目内容

| 文件 | 作用 | 是否需要 GPU |
|---|---|---|
| `enhance_raw_v1_basic_gray.py` | 线性拉伸或直方图均衡化的基础方法 | 否 |
| `enhance_raw_v2_SIED.py` | SIED 官方模型，默认 FP32、20 步 DDIM | 是，强烈建议 |
| `enhance_raw_v3_feature_ensemble.py` | SIED V3，默认 FP32、20 步、3 次特征融合 | 是，必须 |
| `image_enhancement_ui.py` | V3 的粉色桌面界面 | 是，必须 |
| `evaluate_metrics.py` | 计算 PSNR、SSIM 和 LPIPS | LPIPS 推荐 GPU，其余可用 CPU |
| `enhance_flow.png` | 方法流程图 | 否 |

V3 不是额外的事后修图网络。它对同一份 RAW 特征使用三个不同随机种子运行 DDIM，在 SIED 内部的 `pred_fea` 特征层取平均，最后只调用一次 RGB 解码器。

## 2. 数据格式要求

当前脚本只适用于本课程提供的无文件头 RAW：

- 数据类型：无符号 16 位整数，即 `uint16`；
- 分辨率：`3840 × 2160`；
- 单个 RAW 文件大小：`16,588,800` 字节；
- Bayer 排列：RGGB；
- 黑电平：512；
- 白电平：16383；
- 文件扩展名：`.raw`。

如果输入 RAW 的尺寸、位深或 Bayer 排列不同，不能直接使用当前脚本，否则会报“像素数不符合预期”或产生颜色错误。

## 3. 推荐目录结构

部署完成后，建议目录如下：

```text
practice_1/
├─ DEPLOYMENT.md
├─ enhance_raw_v1_basic_gray.py
├─ enhance_raw_v2_SIED.py
├─ enhance_raw_v3_feature_ensemble.py
├─ evaluate_metrics.py
├─ image_enhancement_ui.py
├─ enhance_flow.png
├─ SIED_official/                # 官方 SIED 代码，需要另外下载
│  ├─ models/
│  ├─ dataset/
│  ├─ inference.py
│  └─ ...
├─ ckpt/                         # 权重不上传 GitHub，需要另外下载
│  ├─ canon_0.0001.pt
│  ├─ canon_0.001.pt
│  ├─ canon_0.01.pt
│  ├─ sony_0.0001.pt
│  ├─ sony_0.001.pt
│  └─ sony_0.01.pt
├─ data/
│  └─ 0.0001/
│     ├─ 1.raw
│     ├─ 2.raw
│     └─ ...
├─ gt/                            # 对应的正常曝光参考图，可选
├─ output_v1/                     # V1 输出，自动创建
├─ output_v2/                     # V2 输出，自动创建
└─ output_v3/                     # V3/UI 输出，自动创建
```

模型权重、RAW 数据和生成结果体积很大，不应提交到普通 Git 仓库。部署到另一台电脑时，需要单独复制或下载。

## 4. 硬件和软件要求

### 4.1 最低要求

- Windows 10 或 Windows 11，64 位；
- Python 3.10 或 3.11，64 位；
- 至少 16 GB 内存；
- 至少预留 15 GB 磁盘空间；
- 运行 V2、V3 或 UI 时需要支持 CUDA 的 NVIDIA 显卡；
- 建议显存不少于 8 GB。整张推理显存不足时，脚本会自动改用分块推理。

V1 和 PSNR/SSIM 可以只使用 CPU。V3 采用 FP32 并进行三次采样，速度通常约为 V2 单次采样的 2.5～3 倍。

### 4.2 检查 Python

打开 PowerShell：

```powershell
python --version
```

如果同时安装了多个 Python，也可以使用：

```powershell
py -0p
```

不建议直接执行官方 `SIED_official/requirements.txt` 中的全部旧版依赖。它包含研究环境中大量与本项目推理无关的旧包，在较新的 Python 上可能发生版本冲突。本项目部署只安装实际用到的最小依赖。

## 5. 获取项目代码

如果通过 GitHub 部署，将下面的地址替换成实际仓库地址：

```powershell
Set-Location -LiteralPath "D:\VSCode\code"
git clone "https://github.com/你的用户名/DIP_bigwork.git"
Set-Location -LiteralPath "D:\VSCode\code\DIP_bigwork\practice_1"
```

如果项目已经在本机，只需进入目录：

```powershell
Set-Location -LiteralPath "D:\VSCode\code\DIP_bigwork\practice_1"
```

## 6. 创建 Python 虚拟环境

推荐在 `practice_1` 中创建独立环境：

```powershell
py -3.11 -m venv .venv
```

如果电脑没有 Python 3.11，但已有可用 Python：

```powershell
python -m venv .venv
```

激活环境：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

激活成功后，PowerShell 提示符前面会出现 `(.venv)`。

升级安装工具：

```powershell
python -m pip install --upgrade pip setuptools wheel
```

以后每次重新打开 PowerShell，都需要先进入目录并激活环境：

```powershell
Set-Location -LiteralPath "D:\VSCode\code\DIP_bigwork\practice_1"
.\.venv\Scripts\Activate.ps1
```

## 7. 安装依赖

### 7.1 安装 PyTorch

先安装与本机 NVIDIA 驱动兼容的 CUDA 版 PyTorch。安装命令应以 PyTorch 官方安装页面为准。已经正确安装 GPU 版 PyTorch 时，不要重复安装 CPU 版覆盖它。

安装后执行：

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
```

运行 V2/V3/UI 时，`CUDA可用`必须显示为 `True`。

### 7.2 安装本项目最小依赖

```powershell
python -m pip install numpy opencv-python pillow scikit-image lpips einops ema-pytorch accelerate tqdm packaging
```

检查依赖：

```powershell
python -c "import cv2, numpy, PIL, skimage, lpips, einops, ema_pytorch, accelerate; print('基础依赖检查通过')"
```

Tkinter 随标准 Windows Python 一起安装。可以这样检查：

```powershell
python -c "import tkinter; print('Tkinter检查通过')"
```

## 8. 下载 SIED 官方代码

V2、V3 和 UI 会默认在当前目录寻找 `SIED_official/models`。如果尚未下载，执行：

```powershell
Set-Location -LiteralPath "D:\VSCode\code\DIP_bigwork\practice_1"
git clone "https://github.com/JianghaiSCU/SIED.git" "SIED_official"
```

检查目录：

```powershell
Test-Path -LiteralPath ".\SIED_official\models"
```

应返回：

```text
True
```

## 9. 放置模型权重

从 SIED 官方项目提供的预训练模型地址下载权重，建议放入：

```text
practice_1\ckpt
```

创建目录：

```powershell
New-Item -ItemType Directory -Path ".\ckpt" -Force
```

权重必须与相机和照度匹配：

| 数据 | 权重文件 |
|---|---|
| Canon，0.0001 lux | `canon_0.0001.pt` |
| Canon，0.001 lux | `canon_0.001.pt` |
| Canon，0.01 lux | `canon_0.01.pt` |
| Sony，0.0001 lux | `sony_0.0001.pt` |
| Sony，0.001 lux | `sony_0.001.pt` |
| Sony，0.01 lux | `sony_0.01.pt` |

查看权重：

```powershell
Get-ChildItem -LiteralPath ".\ckpt" -Filter "*.pt" | Select-Object Name,Length
```

不能用 Canon 权重处理 Sony RAW，也不能把不同照度的数据全部交给同一个权重。

## 10. 准备输入数据

例如将低照度 RAW 放到：

```text
practice_1\data\0.0001
```

检查数量和大小：

```powershell
$rawDir = ".\data\0.0001"
$rawFiles = Get-ChildItem -LiteralPath $rawDir -Filter "*.raw" -File
$rawFiles.Count
$rawFiles | Select-Object -First 5 Name,Length
```

每个文件的 `Length` 应为 `16588800`。V1 在其他电脑上应显式传入 `--input`，不要依赖脚本中仅适用于原开发电脑的后备路径。

## 11. 运行 V1 基础增强

线性拉伸：

```powershell
python .\enhance_raw_v1_basic_gray.py `
  --input ".\data\0.0001" `
  --output ".\output_v1" `
  --method linear `
  --limit 5
```

直方图均衡化：

```powershell
python .\enhance_raw_v1_basic_gray.py `
  --input ".\data\0.0001" `
  --output ".\output_v1_he" `
  --method he `
  --limit 5
```

确认试跑结果正常后，删除 `--limit 5` 即可处理全部图片。V1 输出为全分辨率 JPG。

## 12. 运行 V2 SIED 单次推理

先只处理一张或少量图片：

```powershell
python .\enhance_raw_v2_SIED.py ".\data\0.0001" `
  --output ".\output_v2" `
  --ckpt ".\ckpt\canon_0.0001.pt" `
  --model-dir ".\SIED_official" `
  --camera canon `
  --lux 0.0001 `
  --steps 20 `
  --precision fp32 `
  --limit 1
```

确认成功后，将 `--limit 1` 改为 `--limit 0` 处理全部图片。已经存在的结果默认跳过；如需覆盖，增加：

```text
--overwrite
```

## 13. 运行 V3 三次特征融合

V3 是推荐的高质量版本：

```powershell
python .\enhance_raw_v3_feature_ensemble.py ".\data\0.0001" `
  --output ".\output_v3" `
  --ckpt ".\ckpt\canon_0.0001.pt" `
  --model-dir ".\SIED_official" `
  --camera canon `
  --lux 0.0001 `
  --steps 20 `
  --samples 3 `
  --seed 0 `
  --limit 1
```

默认参数含义：

- `--steps 20`：每次 DDIM 采样走 20 步；
- `--samples 3`：使用三个不同随机种子生成三份特征并求平均；
- `--seed 0`：固定实验起点，保证结果可复现；
- `--tile 512`：显存不足时使用的打包 RAW 分块尺寸；
- `--overlap 64`：分块重叠宽度。

V3 强制使用 FP32，没有 `--precision fp16` 参数。完整批量运行时，把 `--limit 1` 改成 `--limit 0`。

## 14. 启动粉色桌面界面

界面只调用 V3：

```powershell
python .\image_enhancement_ui.py
```

在界面中依次选择：

1. 一张 RAW 或包含 RAW 的文件夹；
2. 输出目录；
3. 正确的相机和照度；
4. 对应的 `.pt` 权重；
5. `SIED_official` 模型代码目录；
6. 默认使用 20 步、3 次融合；
7. 点击开始处理。

关闭界面不会修改原始 RAW。输出默认写入 `output_v3`。

## 15. 计算 PSNR、SSIM 和 LPIPS

比较两个文件夹：

```powershell
python .\evaluate_metrics.py ".\output_v3" ".\gt" `
  --lpips-net squeeze `
  --lpips-max-side 1024 `
  --batch-size 8 `
  --workers 4
```

只测试前 10 对：

```powershell
python .\evaluate_metrics.py ".\output_v3" ".\gt" --limit 10
```

暂时不计算 LPIPS：

```powershell
python .\evaluate_metrics.py ".\output_v3" ".\gt" --skip-lpips
```

评价脚本要求增强图和 GT 尺寸严格一致。它会自动尝试以下文件名配对：

```text
001_enhanced.png  <->  001.png
001_enhanced.png  <->  001_gt.png
001_enhanced.png  <->  001_gt_rgb.png
```

结果会显示在终端，并默认保存到增强结果目录的 `metrics.csv`。

第一次计算 LPIPS 可能需要联网下载 SqueezeNet、AlexNet 或 VGG 的预训练权重。之后会使用本地缓存。

## 16. 常见问题

### 16.1 `没有可用的 NVIDIA GPU`

检查：

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果 `nvidia-smi` 正常但 PyTorch 返回 `False`，通常是安装了 CPU 版 PyTorch，需要重装与驱动兼容的 CUDA 版。

### 16.2 `CUDA out of memory`

脚本会优先整张推理，显存不足时自动分块。如果 512 分块仍然不足，可以尝试：

```text
--tile 256 --overlap 64
```

同时关闭其他占用显存的软件。V3 的三次采样是顺序执行，不会同时保存三套模型，但 FP32 本身仍然需要较多显存。

### 16.3 `找不到官方模型代码`

确认下面的路径存在：

```powershell
Test-Path -LiteralPath ".\SIED_official\models"
```

也可以用 `--model-dir` 显式指定其他位置。

### 16.4 `找不到权重` 或权重与模型对不上

确认权重下载完整，并且是 SIED 官方 Stage 2 权重。检查大小：

```powershell
Get-Item -LiteralPath ".\ckpt\canon_0.0001.pt" |
Select-Object FullName,Length
```

不要将网页下载失败产生的 HTML 文件误当成 `.pt` 权重。

### 16.5 RAW 像素数错误

当前程序固定读取 `3840 × 2160 × uint16`。检查文件大小：

```powershell
Get-Item -LiteralPath ".\data\0.0001\1.raw" |
Select-Object Name,Length
```

正确大小应为 `16588800` 字节。

### 16.6 LPIPS 初始化失败

第一次运行需要下载骨干网络权重。检查网络后重试，或先使用：

```text
--skip-lpips
```

PSNR、SSIM 和 LPIPS 的方向分别为：PSNR 越高越好、SSIM 越接近 1 越好、LPIPS 越接近 0 越好。

### 16.7 PowerShell 不允许激活虚拟环境

仅对当前 PowerShell 窗口临时放行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 16.8 磁盘空间不足

模型权重约 2.6 GB，1000 张全分辨率 PNG 可能占用数 GB。开始批量处理前检查：

```powershell
Get-PSDrive -Name D |
Select-Object Name,
  @{Name="UsedGB";Expression={[math]::Round($_.Used / 1GB, 2)}},
  @{Name="FreeGB";Expression={[math]::Round($_.Free / 1GB, 2)}}
```

不要把 `output_v1`、`output_v2`、`output_v3`、RAW 数据或 `.pt` 权重添加到 Git 仓库。

## 17. 最小部署验收

按以下顺序验证：

```powershell
# 1. Python和CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. 最小依赖
python -c "import cv2, numpy, PIL, skimage, lpips, einops, ema_pytorch, accelerate; print('OK')"

# 3. 官方模型目录
Test-Path -LiteralPath ".\SIED_official\models"

# 4. 权重文件
Test-Path -LiteralPath ".\ckpt\canon_0.0001.pt"

# 5. 一张图试跑
python .\enhance_raw_v3_feature_ensemble.py ".\data\0.0001" `
  --ckpt ".\ckpt\canon_0.0001.pt" `
  --model-dir ".\SIED_official" `
  --limit 1
```

只有一张图试跑成功后，再处理全部数据。这样可以提前发现环境、权重、相机类型、照度或路径配置错误。

## 18. 退出虚拟环境

使用完毕后执行：

```powershell
deactivate
```

