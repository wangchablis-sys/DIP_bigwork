# Practice 2 部署与运行说明

本实践用于文档图像几何校正。程序根据原图及对应掩膜完成透视校正、文字方向校正、书缝拆页和弯曲文字行拉直，并输出校正结果、前后对比图和评价表。PaddleOCR 仅作为可选评价工具，不参与几何校正。

## 1. 文件说明

```text
practice_2/
├─ correct.py          # 主程序
├─ requirements.txt    # 完整依赖，包含可选OCR
├─ DEPLOYMENT.md       # 本文档
├─ data/               # 需要自行准备
│  ├─ images/          # 原始文档图片
│  └─ mask/            # 与原图同名的文档区域掩膜
└─ output/             # 运行后自动创建
```

程序支持 JPG、JPEG、PNG 和 BMP 原图。掩膜支持 PNG 或 JPG。原图和掩膜必须使用相同的文件名主体，例如：

```text
data/images/001.jpg
data/mask/001.png
```

掩膜中大于 127 的区域被视为文档区域，小于等于 127 的区域被视为背景。

## 2. 运行环境

- Windows 10/11，64 位；
- Python 3.10 或 3.11，64 位；
- 推荐内存不少于 8 GB；
- 基础几何校正只使用 CPU，不需要 NVIDIA GPU；
- 使用 `--ocr` 时，第一次运行需要联网下载 PaddleOCR 模型。

## 3. 进入项目目录

```powershell
Set-Location -LiteralPath "D:\VSCode\code\DIP_bigwork\practice_2"
```

## 4. 创建虚拟环境

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

以后重新打开 PowerShell 时，使用：

```powershell
Set-Location -LiteralPath "D:\VSCode\code\DIP_bigwork\practice_2"
.\.venv\Scripts\Activate.ps1
```

## 5. 安装依赖

### 5.1 只运行几何校正，不计算OCR

不使用 `--ocr` 时只需安装：

```powershell
python -m pip install numpy pillow opencv-python==5.0.0.93
```

检查：

```powershell
python -c "import cv2, numpy, PIL; print('OpenCV:', cv2.__version__); print('基础依赖检查通过')"
```

### 5.2 使用PaddleOCR评价

如果需要 `--ocr`，请严格按照以下顺序安装。PaddleOCR 会自动安装 `opencv-contrib-python`，它可能与本项目使用的 OpenCV 版本冲突，因此最后需要卸载并装回 `opencv-python`。

```powershell
python -m pip install paddlepaddle==3.3.1 paddleocr==3.7.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip uninstall -y opencv-contrib-python
python -m pip install --force-reinstall --no-deps opencv-python==5.0.0.93
python -m pip install "numpy>=2.3" pillow
```

检查：

```powershell
python -c "import cv2, paddle, paddleocr; print('OpenCV:', cv2.__version__); print('Paddle:', paddle.__version__); print('OCR依赖检查通过')"
```

也可以根据本项目锁定文件安装，但仍建议安装后检查 OpenCV：

```powershell
python -m pip install -r .\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip uninstall -y opencv-contrib-python
python -m pip install --force-reinstall --no-deps opencv-python==5.0.0.93
```

## 6. 准备数据

创建目录：

```powershell
New-Item -ItemType Directory -Path ".\data\images" -Force
New-Item -ItemType Directory -Path ".\data\mask" -Force
```

放置完成后检查数量：

```powershell
$images = Get-ChildItem -LiteralPath ".\data\images" -File
$masks = Get-ChildItem -LiteralPath ".\data\mask" -File
[PSCustomObject]@{
    Images = $images.Count
    Masks  = $masks.Count
}
```

检查前几个文件：

```powershell
Get-ChildItem -LiteralPath ".\data\images" -File | Select-Object -First 5 Name,Length
Get-ChildItem -LiteralPath ".\data\mask" -File | Select-Object -First 5 Name,Length
```

没有同名掩膜的原图会被跳过。数据目录也可以放在其他位置，运行时通过 `--data` 指定。

## 7. 试运行

先处理程序内置的5张代表图：

```powershell
python .\correct.py --samples
```

如果数据集中没有这5个指定文件名，程序会提示没有找到图像。此时可使用 `--only` 指定实际文件名：

```powershell
python .\correct.py --only "001.jpg" "002.jpg"
```

显式指定数据和输出目录：

```powershell
python .\correct.py `
  --data ".\data" `
  --out ".\output" `
  --only "001.jpg"
```

## 8. 处理全部图片

```powershell
python .\correct.py --data ".\data" --out ".\output"
```

脚本默认处理所有能够找到同名掩膜的图片。

## 9. 启用OCR评价

```powershell
python .\correct.py --data ".\data" --out ".\output" --ocr
```

OCR只计算校正前后的平均文字识别置信度，不影响校正结果。首次运行可能需要较长时间下载检测和识别模型。程序已经关闭 PaddleOCR 内置方向分类和文档矫正，避免与本项目的校正算法重复。

## 10. 保存调试图

```powershell
python .\correct.py --samples --debug
```

或者：

```powershell
python .\correct.py --only "001.jpg" --debug
```

调试图会保存文字行、边界和中间处理结果，便于分析失败样本。

## 11. 输出结构

```text
output/
├─ corrected/          # 最终校正图
├─ compare/            # 校正前后对比图
├─ debug/              # 仅使用 --debug 时生成
└─ metrics.csv         # 每张图片的处理方式和指标
```

`metrics.csv`主要字段：

| 字段 | 含义 |
|---|---|
| `name` | 原图文件名 |
| `pages` | 检测到并输出的页数 |
| `method` | 实际执行的校正步骤 |
| `deskew_deg` | 文字转正角度 |
| `straight_before` | 校正前文字行直线度 |
| `straight_after` | 校正后文字行直线度 |
| `ocr_before` | 校正前OCR平均置信度，仅 `--ocr` 时有值 |
| `ocr_after` | 校正后OCR平均置信度，仅 `--ocr` 时有值 |
| `seconds` | 单张处理时间 |
| `error` | 失败原因 |

只重跑部分图片时，程序会按文件名更新原有 `metrics.csv`，不会删除其他图片的历史记录。

## 12. 常见问题

### 12.1 `没有找到图像`

确认目录必须包含：

```text
指定数据目录/images
指定数据目录/mask
```

并确认原图存在同名掩膜。

### 12.2 OpenCV版本或导入冲突

查看版本：

```powershell
python -c "import cv2; print(cv2.__version__); print(cv2.__file__)"
python -m pip list | Select-String "opencv"
```

如果同时出现 `opencv-python` 与 `opencv-contrib-python`，执行：

```powershell
python -m pip uninstall -y opencv-contrib-python
python -m pip install --force-reinstall --no-deps opencv-python==5.0.0.93
```

### 12.3 PaddleOCR首次运行很慢

第一次需要下载模型。保持网络连接，并耐心等待。模型下载完成后再次运行会使用缓存。

### 12.4 OCR出现oneDNN或MKLDNN错误

本项目已经在导入 PaddleOCR 前关闭 Windows CPU 下容易出错的相关加速。如果仍报错，请确认使用的是项目当前版本的 `correct.py`，并在独立虚拟环境中重新安装依赖。

### 12.5 中文路径读取失败

程序使用 `numpy.fromfile` 和 `cv2.imdecode` 读取中文路径，正常情况下支持中文目录。如果仍失败，请确认路径存在、文件未损坏且扩展名正确。

## 13. 最小部署验收

```powershell
# 1. 环境检查
python -c "import cv2, numpy, PIL; print(cv2.__version__)"

# 2. 数据检查
Test-Path -LiteralPath ".\data\images"
Test-Path -LiteralPath ".\data\mask"

# 3. 单张试跑，将文件名替换为实际名称
python .\correct.py --only "001.jpg" --debug
```

确认 `output/corrected` 和 `output/compare` 中生成图片后，再运行全部数据。

## 14. 退出虚拟环境

```powershell
deactivate
```

