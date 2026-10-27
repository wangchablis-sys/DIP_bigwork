# Practice 3 部署与运行说明

本实践对 Kodak 24 张无损 PNG 图像进行 JPEG 有损压缩实验。程序分别使用质量参数 50、75、90 编码和解码，计算压缩率、PSNR、SSIM，以及压缩和解压缩时间，并自动输出 CSV 表格、率失真散点图和局部对比图。

## 1. 文件和目录结构

```text
practice_3/
├─ compress_kodak.py   # 主程序
├─ DEPLOYMENT.md       # 本文档
├─ kodak/              # 需要放置24张原始PNG
│  ├─ kodim01.png
│  ├─ kodim02.png
│  ├─ ...
│  └─ kodim24.png
└─ output/             # 运行后自动创建
```

程序固定查找 `kodak/kodim*.png`，并要求恰好找到24张图片。

## 2. 运行环境

- Windows 10/11，64 位；
- Python 3.10、3.11 或 3.12；
- 推荐内存不少于 4 GB；
- 不需要 NVIDIA GPU，全部计算使用 CPU；
- 建议保留至少 1 GB 可用磁盘空间。

## 3. 进入项目目录

```powershell
Set-Location -LiteralPath "D:\VSCode\code\DIP_bigwork\practice_3"
```

## 4. 创建虚拟环境

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

以后重新打开 PowerShell 时：

```powershell
Set-Location -LiteralPath "D:\VSCode\code\DIP_bigwork\practice_3"
.\.venv\Scripts\Activate.ps1
```

## 5. 安装依赖

```powershell
python -m pip install numpy opencv-python matplotlib scikit-image
```

检查：

```powershell
python -c "import cv2, numpy, matplotlib, skimage; print('依赖检查通过'); print('OpenCV:', cv2.__version__)"
```

## 6. 准备Kodak数据集

创建输入目录：

```powershell
New-Item -ItemType Directory -Path ".\kodak" -Force
```

将Kodak数据集24张无损PNG放入该目录，并保持标准文件名：

```text
kodim01.png
kodim02.png
...
kodim24.png
```

检查数量：

```powershell
$images = Get-ChildItem -LiteralPath ".\kodak" -Filter "kodim*.png" -File
$images.Count
$images | Select-Object Name,Length
```

数量必须为24。如果不是24，脚本会停止并报告实际数量。

## 7. 运行实验

```powershell
python .\compress_kodak.py
```

程序没有命令行参数。实验设置写在脚本顶部：

```python
QUALITIES = (50, 75, 90)
SAMPLE_QUALITY = 90
TIMING_REPEATS = 11
```

含义如下：

- 对每张图分别使用 JPEG 质量50、75、90；
- 对比图主要展示质量90，同时与质量50比较；
- 压缩和解压时间各测量11次并取中位数，降低偶然波动。

24张图、3档质量会生成72张JPEG。

## 8. 输出结果

运行完成后生成：

```text
output/
├─ jpg/
│  ├─ kodim01_q50.jpg
│  ├─ kodim01_q75.jpg
│  ├─ kodim01_q90.jpg
│  └─ ...
├─ figures/
│  ├─ rate_psnr.png
│  ├─ compare_kodim23_q90.png
│  └─ compare_kodim24_q90.png
├─ metrics.csv
└─ summary.csv
```

### `metrics.csv`

保存每张图片、每个质量等级的详细结果：

| 字段 | 含义 |
|---|---|
| `image` | 原PNG文件名 |
| `height`、`width` | 图像尺寸 |
| `quality` | JPEG质量参数 |
| `png_bytes` | 原PNG字节数 |
| `jpg_bytes` | 压缩后JPEG字节数 |
| `compression_ratio` | `JPG字节数 / PNG字节数` |
| `psnr` | 峰值信噪比，越高越好 |
| `ssim` | 结构相似度，越接近1越好 |
| `encode_ms` | JPEG压缩时间，中位数，毫秒 |
| `decode_ms` | JPEG解压时间，中位数，毫秒 |

本项目中的“压缩率”定义为：

$$
R=\frac{\text{JPEG文件字节数}}{\text{原PNG文件字节数}}
$$

因此这里的压缩率越小，表示压缩后的文件相对越小。

### `summary.csv`

按质量50、75、90分别统计24张图的平均压缩率、平均PSNR、平均SSIM、平均编码时间、平均解码时间和平均JPEG大小。

### 图表

- `rate_psnr.png`：横轴为压缩率，纵轴为PSNR；
- `compare_kodim23_q90.png`：Kodak第23张图片的全图和局部比较；
- `compare_kodim24_q90.png`：Kodak第24张图片的全图和局部比较。

## 9. 如何重新运行

再次执行：

```powershell
python .\compress_kodak.py
```

同名输出会被覆盖，原始 `kodak` 图片不会被修改。

## 10. 常见问题

### 10.1 `期望24张Kodak图，实际找到0张`

检查：

```powershell
Get-ChildItem -LiteralPath ".\kodak" -Filter "kodim*.png" -File
```

确认文件位于 `practice_3/kodak`，而不是其下的另一个嵌套文件夹。

### 10.2 图片数量不是24

检查是否缺少图片、文件名被修改，或者目录中存在额外的 `kodim*.png`。

### 10.3 Matplotlib中文警告或中文显示为方框

程序优先使用：

```text
C:\Windows\Fonts\msyh.ttc
```

即微软雅黑字体。确认该文件存在：

```powershell
Test-Path -LiteralPath "C:\Windows\Fonts\msyh.ttc"
```

字体缺失只影响图表中文字显示，不影响CSV数值。

### 10.4 OpenCV无法读取图片

确认PNG文件完整且能用普通图片查看器打开。查看前几个文件大小：

```powershell
Get-ChildItem -LiteralPath ".\kodak" -Filter "*.png" -File |
Select-Object -First 5 Name,Length
```

### 10.5 运行结果与其他电脑耗时不同

压缩和解压时间受CPU型号、后台程序、OpenCV版本和系统负载影响。程序取11次测量的中位数，只能降低波动，不能保证不同电脑结果完全相同。PSNR、SSIM和压缩文件大小应更稳定。

## 11. 最小部署验收

```powershell
# 1. 检查依赖
python -c "import cv2, numpy, matplotlib, skimage; print('OK')"

# 2. 检查输入数量
(Get-ChildItem -LiteralPath ".\kodak" -Filter "kodim*.png" -File).Count

# 3. 运行实验
python .\compress_kodak.py

# 4. 检查关键输出
Test-Path -LiteralPath ".\output\metrics.csv"
Test-Path -LiteralPath ".\output\summary.csv"
Test-Path -LiteralPath ".\output\figures\rate_psnr.png"
```

三个 `Test-Path` 都应返回 `True`。

## 12. 退出虚拟环境

```powershell
deactivate
```

