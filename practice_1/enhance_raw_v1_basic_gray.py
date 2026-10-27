"""用线性矫正或直方图均衡化处理极暗 RAW，并输出彩色图片。

这 1000 张图是无文件头的 uint16 RAW，尺寸 3840x2160，黑电平 512，Bayer 排列为 RGGB。
像素几乎都挤在 512 附近，直接当 8 位图去均衡会得到一张全黑的图。
因此先减黑电平，再对整幅 Bayer 做同一种灰度变换。红、绿、蓝共用这一条映射，
颜色之间的比例先保留下来。随后按 RGGB 去马赛克，并用灰度世界白平衡校正偏色。
结果是 2160x3840 的三通道 JPG，和 GT 逐像素对齐。

线性矫正（默认）：
    s = (r - r1) / (r2 - r1) * 255
    r1、r2 取当前图像的低、高分位数。
    不用整幅图的最小、最大值，是因为个别热像素会把其余灰度压暗。

直方图均衡化：
    s = round(255 * (cdf(r) - cdf_min) / (N - cdf_min))

默认先找本脚本旁的 data/0.0001；没有这个文件夹时，改用课程目录里的 0.0001。结果写到 output_v1。

用法：
    python enhance_raw_v1_basic_gray.py
    python enhance_raw_v1_basic_gray.py --method he
    python enhance_raw_v1_basic_gray.py --limit 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "0.0001"
FALLBACK_INPUT = Path(r"C:\Users\Wyf\OneDrive\Desktop\本科课程\大作业\大二上\0.0001")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_v1"
WIDTH = 3840
HEIGHT = 2160
BLACK_LEVEL = 512


def read_bayer(path: Path) -> np.ndarray:
    """读取 RGGB RAW，减去黑电平，保留每个像素自己的颜色。"""
    raw = np.fromfile(path, dtype=np.uint16)
    expected = WIDTH * HEIGHT
    if raw.size != expected:
        raise ValueError(f"{path.name} 像素数为 {raw.size}，预期 {expected}（{WIDTH}x{HEIGHT}）")
    raw = raw.reshape(HEIGHT, WIDTH).astype(np.float32)
    return np.maximum(raw - BLACK_LEVEL, 0.0)


def linear_correct(bayer: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    """把窄灰度区间线性映射到 0~255。红绿蓝共用同一对分位数。"""
    low, high = np.percentile(bayer, [low_percentile, high_percentile])
    if high <= low + 1e-6:
        return np.zeros(bayer.shape, dtype=np.uint8)
    scaled = (bayer - low) / (high - low) * 255.0
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def histogram_equalize(bayer: np.ndarray) -> np.ndarray:
    """按累计分布函数把实际出现的灰度拉开到 0~255。红绿蓝共用同一条映射。"""
    levels = np.clip(np.rint(bayer), 0, None).astype(np.int32)
    histogram = np.bincount(levels.ravel()).astype(np.float64)
    cdf = np.cumsum(histogram)
    occupied = np.flatnonzero(histogram)
    if occupied.size == 0:
        return np.zeros(bayer.shape, dtype=np.uint8)
    cdf_min = cdf[occupied[0]]
    denom = cdf[-1] - cdf_min
    if denom <= 0:
        return np.zeros(bayer.shape, dtype=np.uint8)
    mapping = np.clip(np.rint((cdf - cdf_min) / denom * 255.0), 0, 255).astype(np.uint8)
    return mapping[levels]


def demosaic_color(bayer: np.ndarray) -> np.ndarray:
    """把 RGGB 恢复成 BGR，再用灰度世界校正偏色。"""
    if bayer.dtype != np.uint8 or bayer.ndim != 2:
        raise ValueError("去马赛克前的 Bayer 图必须是二维 uint8。")
    # 这批 RAW 第一行是 RG。OpenCV 按第二行命名，所以要用 BG 这一个转换。
    bgr = cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2BGR)
    means = bgr.reshape(-1, 3).astype(np.float32).mean(axis=0)
    scale = float(means.mean()) / np.maximum(means, 1.0)
    balanced = np.clip(bgr.astype(np.float32) * scale, 0, 255)
    return np.rint(balanced).astype(np.uint8)


def output_is_color(path: Path) -> bool:
    """已有结果必须是全分辨率彩色图。旧的三通道灰度图需要重做。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.shape != (HEIGHT, WIDTH, 3):
        return False
    sample = image[::32, ::32]
    return int(np.max(np.abs(sample[:, :, 0].astype(np.int16) - sample[:, :, 1]))) > 0 or int(
        np.max(np.abs(sample[:, :, 1].astype(np.int16) - sample[:, :, 2]))
    ) > 0


def save_jpg(path: Path, image: np.ndarray) -> None:
    """用 imencode 写文件，避免 Windows 中文路径下 cv2.imwrite 失败。"""
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise OSError(f"编码失败：{path}")
    encoded.tofile(path)


def resolve_input(path: Path) -> Path:
    """默认目录不存在时，改用已经放好的 1000 张 RAW。"""
    candidates = [path]
    if path == DEFAULT_INPUT:
        candidates.append(FALLBACK_INPUT)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.raw")):
            return candidate
    tried = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"找不到包含 .raw 的输入文件夹。已尝试：\n{tried}")


def collect_raws(input_dir: Path, limit: int | None) -> list[Path]:
    files = list(input_dir.glob("*.raw"))
    files.sort(key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)
    if limit is not None:
        files = files[:limit]
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="极暗 RAW 的彩色线性矫正 / 直方图均衡化")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="存放 .raw 的文件夹；本地 data/0.0001 不存在时使用课程目录中的 0.0001")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="输出文件夹；默认写到本脚本旁的 output_v1")
    parser.add_argument("--method", choices=["linear", "he"], default="linear",
                        help="linear：灰度线性矫正；he：直方图均衡化")
    parser.add_argument("--low", type=float, default=0.5, help="线性矫正的低分位数")
    parser.add_argument("--high", type=float, default=99.5, help="线性矫正的高分位数")
    parser.add_argument("--limit", type=int, help="只处理前 N 张，用于试跑")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已经生成的结果")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.low < args.high <= 100:
        raise ValueError("分位数需要满足 0 <= low < high <= 100。")

    input_dir = resolve_input(args.input.expanduser())

    output_dir = args.output.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_raws(input_dir, args.limit)
    if not files:
        print(f"没有找到 .raw 文件：{input_dir}")
        return

    method_name = "彩色线性矫正" if args.method == "linear" else "彩色直方图均衡化"
    print(f"方法：{method_name}")
    print(f"输入：{input_dir}（{len(files)} 张）")
    print(f"输出：{output_dir}")

    done = 0
    skipped = 0
    failed = 0
    start_all = time.perf_counter()
    for index, path in enumerate(files, start=1):
        destination = output_dir / f"{path.stem}.jpg"
        if destination.exists() and not args.overwrite and output_is_color(destination):
            skipped += 1
            continue
        start = time.perf_counter()
        try:
            bayer = read_bayer(path)
            if args.method == "linear":
                stretched = linear_correct(bayer, args.low, args.high)
            else:
                stretched = histogram_equalize(bayer)
            save_jpg(destination, demosaic_color(stretched))
        except Exception as exc:
            failed += 1
            print(f"[失败] {path.name}: {exc}")
            continue
        done += 1
        elapsed = time.perf_counter() - start
        print(f"[{index}/{len(files)}] {path.name} -> {destination.name}  {elapsed:.2f}秒")

    total = time.perf_counter() - start_all
    print(f"\n结束：新处理 {done}，跳过 {skipped}，失败 {failed}，耗时 {total:.1f}秒")


if __name__ == "__main__":
    main()
