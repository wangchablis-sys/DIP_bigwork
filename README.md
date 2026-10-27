# 数字图像处理课程大作业

本仓库包含三个数字图像处理实践项目，分别涉及极暗 RAW 图像增强、文档图像几何校正和 JPEG 有损压缩分析。

## 项目结构

```text
DIP_bigwork/
├─ README.md
├─ .gitignore
├─ practice_1/         # 极暗RAW图像增强
├─ practice_2/         # 文档图像几何校正
└─ practice_3/         # Kodak JPEG压缩实验
```

## Practice 1：极暗 RAW 图像增强

主要内容：

- 线性拉伸和直方图均衡化基础方法；
- SIED扩散模型单次推理；
- FP32、20步DDIM、3次特征空间融合；
- 粉色桌面可视化界面；
- PSNR、SSIM和LPIPS评价。

主要文件：

- [`enhance_raw_v1_basic_gray.py`](./practice_1/enhance_raw_v1_basic_gray.py)
- [`enhance_raw_v2_SIED.py`](./practice_1/enhance_raw_v2_SIED.py)
- [`enhance_raw_v3_feature_ensemble.py`](./practice_1/enhance_raw_v3_feature_ensemble.py)
- [`image_enhancement_ui.py`](./practice_1/image_enhancement_ui.py)
- [`evaluate_metrics.py`](./practice_1/evaluate_metrics.py)
- [Practice 1部署文档](./practice_1/DEPLOYMENT.md)

V2、V3和界面需要NVIDIA GPU、CUDA版PyTorch、SIED官方代码和对应的预训练权重。

## Practice 2：文档图像几何校正

主要流程：

```text
掩膜定位 → 透视校正 → 文字转正 → 书缝拆页 → 弯曲文字行拉直 → 评价
```

支持：

- 成对读取原图与文档区域掩膜；
- 输出校正图和前后对比图；
- 保存文字行调试图；
- 统计文字行直线度；
- 可选使用PaddleOCR比较校正前后识别置信度。

主要文件：

- [`correct.py`](./practice_2/correct.py)
- [`requirements.txt`](./practice_2/requirements.txt)
- [Practice 2部署文档](./practice_2/DEPLOYMENT.md)

基础校正可在CPU运行，只有使用 `--ocr` 时才需要安装PaddleOCR。

## Practice 3：Kodak JPEG有损压缩实验

对Kodak 24张无损PNG分别使用JPEG质量50、75、90进行编码和解码，并统计：

- 压缩率；
- PSNR；
- SSIM；
- 压缩时间；
- 解压时间。

程序自动生成72张JPEG、详细CSV、汇总CSV、率失真图和局部对比图。

主要文件：

- [`compress_kodak.py`](./practice_3/compress_kodak.py)
- [Practice 3部署文档](./practice_3/DEPLOYMENT.md)

本实践只使用CPU。

## 环境建议

推荐使用：

- Windows 10/11；
- Python 3.10或3.11；
- 每个practice使用独立虚拟环境；
- Practice 1的SIED方法使用NVIDIA GPU；
- Practice 2和Practice 3可使用CPU。

不同实践的依赖版本可能不同，不建议三个实践共用同一个Python虚拟环境。

## 快速开始

克隆仓库：

```powershell
git clone "https://github.com/你的用户名/DIP_bigwork.git"
Set-Location -LiteralPath ".\DIP_bigwork"
```

然后阅读对应部署文档：

```text
practice_1/DEPLOYMENT.md
practice_2/DEPLOYMENT.md
practice_3/DEPLOYMENT.md
```

## 数据、模型和输出说明

由于体积较大，本仓库不包含：

- SIED RAW数据集；
- SIED预训练权重；
- 文档校正数据集及掩膜；
- Kodak原始图片；
- 各实践运行产生的输出图片、CSV和缓存；
- Python虚拟环境。

请按照各practice的部署文档单独准备数据和模型。不要使用普通Git上传 `.pt` 权重、大规模RAW数据或批量生成图片。

## 最简运行入口

### Practice 1

```powershell
Set-Location .\practice_1
python .\image_enhancement_ui.py
```

### Practice 2

```powershell
Set-Location .\practice_2
python .\correct.py --samples
```

### Practice 3

```powershell
Set-Location .\practice_3
python .\compress_kodak.py
```

每次进入新的practice前，请激活该目录自己的虚拟环境。

## Git仓库内容

建议提交：

- Python源代码；
- Markdown部署文档；
- 小型流程图；
- 依赖说明文件。

建议忽略：

```text
*.pt
*.pth
*.ckpt
*.raw
**/__pycache__/
**/output*/
**/data/
**/.venv/
```

## 参考

Practice 1使用SIED项目：

- 论文：*Learning to See in the Extremely Dark*, ICCV 2025；
- 官方代码：<https://github.com/JianghaiSCU/SIED>。

