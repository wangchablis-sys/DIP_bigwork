"""Kodak 数据集的 JPEG 有损压缩实验。

读取 24 张无损 PNG，按质量 50、75、90 编码并解码，统计压缩率、PSNR、SSIM
和压缩/解压缩时间。压缩率定义为 JPG 字节数除以原 PNG 字节数。
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


SCRIPT_DIR = Path(__file__).resolve().parent
KODAK_DIR = SCRIPT_DIR / "kodak"
OUTPUT_DIR = SCRIPT_DIR / "output"
QUALITIES = (50, 75, 90)
SAMPLE_QUALITY = 90
TIMING_REPEATS = 11
ZOOM_CROPS = {
    "kodim23.png": (540, 120, 200, 200),
    "kodim24.png": (500, 30, 220, 160),
}


def configure_chinese_font() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False


def encode_jpeg(bgr: np.ndarray, quality: int) -> np.ndarray:
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"JPEG 编码失败，质量 {quality}")
    return buffer


def decode_jpeg(buffer: np.ndarray) -> np.ndarray:
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("JPEG 解码失败")
    return image


def median_elapsed_ms(func, repeats: int = TIMING_REPEATS):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2], result


def evaluate_image(path: Path, quality: int) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"无法读取 {path}")
    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    encode_jpeg(bgr, quality)
    decode_jpeg(encode_jpeg(bgr, quality))
    encode_ms, buffer = median_elapsed_ms(lambda: encode_jpeg(bgr, quality))
    decode_ms, restored_bgr = median_elapsed_ms(lambda: decode_jpeg(buffer))
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

    png_bytes = path.stat().st_size
    jpg_bytes = int(buffer.size)
    return {
        "image": path.name,
        "height": int(bgr.shape[0]),
        "width": int(bgr.shape[1]),
        "quality": quality,
        "png_bytes": png_bytes,
        "jpg_bytes": jpg_bytes,
        "compression_ratio": jpg_bytes / png_bytes,
        "psnr": float(peak_signal_noise_ratio(original_rgb, restored_rgb, data_range=255)),
        "ssim": float(
            structural_similarity(original_rgb, restored_rgb, channel_axis=2, data_range=255)
        ),
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
        "original_rgb": original_rgb,
        "restored_rgb": restored_rgb,
        "buffer": buffer,
    }


def write_metrics(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "image",
        "height",
        "width",
        "quality",
        "png_bytes",
        "jpg_bytes",
        "compression_ratio",
        "psnr",
        "ssim",
        "encode_ms",
        "decode_ms",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_summary(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "quality",
        "mean_compression_ratio",
        "mean_psnr",
        "mean_ssim",
        "mean_encode_ms",
        "mean_decode_ms",
        "mean_jpg_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for quality in QUALITIES:
            group = [row for row in rows if row["quality"] == quality]
            writer.writerow(
                {
                    "quality": quality,
                    "mean_compression_ratio": float(np.mean([row["compression_ratio"] for row in group])),
                    "mean_psnr": float(np.mean([row["psnr"] for row in group])),
                    "mean_ssim": float(np.mean([row["ssim"] for row in group])),
                    "mean_encode_ms": float(np.mean([row["encode_ms"] for row in group])),
                    "mean_decode_ms": float(np.mean([row["decode_ms"] for row in group])),
                    "mean_jpg_bytes": float(np.mean([row["jpg_bytes"] for row in group])),
                }
            )


def format_kb(num_bytes: int) -> str:
    return f"{num_bytes / 1000:.1f} kB"


def draw_marker(axis, box: tuple[int, int, int, int]) -> None:
    x, y, width, height = box
    rectangle = plt.Rectangle(
        (x, y), width, height, fill=False, edgecolor="#ff2d2d", linewidth=1.2
    )
    axis.add_patch(rectangle)


def save_comparison(result: dict, low_quality: dict, output_path: Path) -> None:
    name = result["image"]
    crop = ZOOM_CROPS[name]
    x, y, width, height = crop
    original = result["original_rgb"]
    restored = result["restored_rgb"]
    coarse = low_quality["restored_rgb"]

    figure, axes = plt.subplots(2, 3, figsize=(12.6, 7.4))
    panels = [
        (axes[0, 0], original, f"原 PNG\n{format_kb(result['png_bytes'])}"),
        (axes[0, 1], restored, f"JPEG Q={result['quality']}\n{format_kb(result['jpg_bytes'])}"),
        (axes[0, 2], coarse, f"JPEG Q={low_quality['quality']}\n{format_kb(low_quality['jpg_bytes'])}"),
        (axes[1, 0], original[y : y + height, x : x + width], "原图局部"),
        (axes[1, 1], restored[y : y + height, x : x + width], f"Q={result['quality']} 局部"),
        (axes[1, 2], coarse[y : y + height, x : x + width], f"Q={low_quality['quality']} 局部"),
    ]
    for axis, image, title in panels:
        axis.imshow(image, interpolation="nearest")
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    draw_marker(axes[0, 0], crop)
    draw_marker(axes[0, 1], crop)
    draw_marker(axes[0, 2], crop)

    ratio = result["compression_ratio"]
    figure.suptitle(
        f"{name}    压缩率 {ratio:.3f}    PSNR {result['psnr']:.2f} dB    SSIM {result['ssim']:.4f}",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_rate_psnr_curve(rows: list[dict], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    colors = {50: "#d95f02", 75: "#7570b3", 90: "#1b9e77"}
    for quality in QUALITIES:
        group = [row for row in rows if row["quality"] == quality]
        axis.scatter(
            [row["compression_ratio"] for row in group],
            [row["psnr"] for row in group],
            s=28,
            alpha=0.45,
            color=colors[quality],
            label=f"Q={quality} 各图",
        )
        mean_ratio = float(np.mean([row["compression_ratio"] for row in group]))
        mean_psnr = float(np.mean([row["psnr"] for row in group]))
        axis.scatter(
            [mean_ratio],
            [mean_psnr],
            s=90,
            color=colors[quality],
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        label_offset = {50: (-72, -28), 75: (12, -34), 90: (10, 12)}[quality]
        axis.annotate(
            f"Q={quality}\n平均 {mean_ratio:.3f}, {mean_psnr:.2f} dB",
            (mean_ratio, mean_psnr),
            textcoords="offset points",
            xytext=label_offset,
            fontsize=9,
        )
    means = []
    for quality in QUALITIES:
        group = [row for row in rows if row["quality"] == quality]
        means.append(
            (
                float(np.mean([row["compression_ratio"] for row in group])),
                float(np.mean([row["psnr"] for row in group])),
            )
        )
    axis.plot(
        [item[0] for item in means],
        [item[1] for item in means],
        color="black",
        linewidth=1.0,
        label="三档平均",
    )
    axis.set_xlabel("压缩率（JPG 字节 / PNG 字节）")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Kodak 24 张图的压缩率与 PSNR")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def print_summary(rows: list[dict]) -> None:
    print(f"{'质量':>6} {'压缩率':>8} {'PSNR':>8} {'SSIM':>8} {'压缩ms':>8} {'解压ms':>8}")
    for quality in QUALITIES:
        group = [row for row in rows if row["quality"] == quality]
        print(
            f"{quality:6d} "
            f"{np.mean([row['compression_ratio'] for row in group]):8.3f} "
            f"{np.mean([row['psnr'] for row in group]):8.2f} "
            f"{np.mean([row['ssim'] for row in group]):8.4f} "
            f"{np.mean([row['encode_ms'] for row in group]):8.2f} "
            f"{np.mean([row['decode_ms'] for row in group]):8.2f}"
        )


def main() -> None:
    configure_chinese_font()
    images = sorted(KODAK_DIR.glob("kodim*.png"))
    if len(images) != 24:
        raise RuntimeError(f"期望 24 张 Kodak 图，实际找到 {len(images)} 张：{KODAK_DIR}")

    jpg_dir = OUTPUT_DIR / "jpg"
    figure_dir = OUTPUT_DIR / "figures"
    jpg_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    cached = {}
    for path in images:
        for quality in QUALITIES:
            result = evaluate_image(path, quality)
            jpg_path = jpg_dir / f"{path.stem}_q{quality}.jpg"
            jpg_path.write_bytes(result["buffer"].tobytes())
            cached[(path.name, quality)] = result
            rows.append(result)
            print(
                f"{path.name} Q={quality:<2d}  "
                f"压缩率 {result['compression_ratio']:.3f}  "
                f"PSNR {result['psnr']:.2f}  SSIM {result['ssim']:.4f}"
            )

    metric_rows = [{key: value for key, value in row.items() if key not in {"original_rgb", "restored_rgb", "buffer"}} for row in rows]
    write_metrics(metric_rows, OUTPUT_DIR / "metrics.csv")
    write_summary(metric_rows, OUTPUT_DIR / "summary.csv")
    save_rate_psnr_curve(metric_rows, figure_dir / "rate_psnr.png")

    for name in ("kodim23.png", "kodim24.png"):
        sample = cached[(name, SAMPLE_QUALITY)]
        low_quality = cached[(name, 50)]
        save_comparison(
            sample,
            low_quality,
            figure_dir / f"compare_{Path(name).stem}_q{SAMPLE_QUALITY}.png",
        )

    print()
    print_summary(metric_rows)
    print(f"\n结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
