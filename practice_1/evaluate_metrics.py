"""计算图像增强结果的 PSNR、SSIM 和 LPIPS。

支持两种用法：

1. 比较一对图片：
   python evaluate_metrics.py "增强图.jpg" "GT图.jpg"

2. 批量比较两个文件夹：
   python evaluate_metrics.py "增强结果文件夹" "GT文件夹"

批量模式会自动尝试以下配对：
   001_enhanced.jpg <-> 001.jpg
   001_enhanced.jpg <-> 001_gt.jpg
   001_enhanced.jpg <-> 001_gt_rgb.jpg

结果会显示在终端，并保存为 enhanced 文件夹中的 metrics.csv。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class MetricResult:
    enhanced: str
    gt: str
    psnr: float
    ssim: float
    lpips: float
    seconds: float


@dataclass
class PreparedPair:
    result: MetricResult
    enhanced_lpips: np.ndarray | None
    gt_lpips: np.ndarray | None


def read_rgb(path: Path) -> np.ndarray:
    """读取图片，返回 HxWx3、RGB、0～1 范围的 float32 数组。"""
    # np.fromfile + imdecode 可以正确读取包含中文的 Windows 路径。
    encoded = np.fromfile(path, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取图片：{path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0


def normalized_key(path: Path, is_gt: bool) -> str:
    """移除常见后缀，生成用于自动配对的文件名。"""
    key = path.stem.lower().strip()
    if is_gt:
        suffixes = ("_gt_rgb", "-gt-rgb", "_ground_truth", "-ground-truth", "_gt", "-gt")
    else:
        suffixes = (
            "_03_enhanced", "-03-enhanced", "_enhanced", "-enhanced",
            "_result", "-result", "_output", "-output",
        )
    for suffix in suffixes:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    # 忽略空格、横线和下划线的区别。
    return re.sub(r"[\s_-]+", "", key)


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def pair_images(enhanced_path: Path, gt_path: Path) -> list[tuple[Path, Path]]:
    """处理单文件或文件夹，并返回增强图和GT的配对。"""
    if enhanced_path.is_file() and gt_path.is_file():
        return [(enhanced_path, gt_path)]

    if enhanced_path.is_file() != gt_path.is_file():
        raise ValueError("两个路径必须同时是图片，或者同时是文件夹。")
    if not enhanced_path.is_dir():
        raise FileNotFoundError(f"找不到增强结果路径：{enhanced_path}")
    if not gt_path.is_dir():
        raise FileNotFoundError(f"找不到GT路径：{gt_path}")

    enhanced_images = collect_images(enhanced_path)
    gt_images = collect_images(gt_path)
    if not enhanced_images:
        raise ValueError(f"增强结果文件夹中没有图片：{enhanced_path}")
    if not gt_images:
        raise ValueError(f"GT文件夹中没有图片：{gt_path}")

    gt_by_key: dict[str, list[Path]] = {}
    for path in gt_images:
        gt_by_key.setdefault(normalized_key(path, is_gt=True), []).append(path)

    pairs: list[tuple[Path, Path]] = []
    unmatched: list[Path] = []
    ambiguous: list[tuple[Path, list[Path]]] = []
    for enhanced in enhanced_images:
        key = normalized_key(enhanced, is_gt=False)
        candidates = gt_by_key.get(key, [])
        if len(candidates) == 1:
            pairs.append((enhanced, candidates[0]))
        elif len(candidates) == 0:
            unmatched.append(enhanced)
        else:
            ambiguous.append((enhanced, candidates))

    if unmatched:
        print(f"警告：有 {len(unmatched)} 张增强图没有找到GT，将跳过。")
        for path in unmatched[:5]:
            print(f"  未配对：{path.name}")
    if ambiguous:
        print(f"警告：有 {len(ambiguous)} 张增强图对应多个GT，将跳过。")
        for path, candidates in ambiguous[:5]:
            print(f"  多重匹配：{path.name} -> {[p.name for p in candidates]}")
    if not pairs:
        raise ValueError(
            "没有成功配对。请检查文件名，例如：001_enhanced.jpg 与 001_gt.jpg。"
        )
    return pairs


def resize_for_lpips(
    enhanced: np.ndarray,
    gt: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """只为LPIPS同步缩小两张图；0表示保持原始尺寸。"""
    if max_side <= 0:
        return enhanced, gt
    height, width = gt.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return enhanced, gt
    scale = max_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    enhanced_small = cv2.resize(enhanced, new_size, interpolation=cv2.INTER_AREA)
    gt_small = cv2.resize(gt, new_size, interpolation=cv2.INTER_AREA)
    return enhanced_small, gt_small


def to_lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """RGB [0,1] -> NCHW [-1,1]。"""
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).float().mul(2.0).sub(1.0)
    return tensor.to(device)


def prepare_pair(
    enhanced_path: Path,
    gt_path: Path,
    lpips_max_side: int,
    compute_lpips: bool,
) -> PreparedPair:
    """CPU阶段：读图并计算PSNR/SSIM，同时准备较小的LPIPS输入。"""
    start = time.perf_counter()
    enhanced = read_rgb(enhanced_path)
    gt = read_rgb(gt_path)

    if enhanced.shape != gt.shape:
        raise ValueError(
            f"尺寸不一致：增强图 {enhanced.shape}，GT {gt.shape}。"
            "必须保证两张图严格对齐，程序不会自动缩放后计算PSNR/SSIM。"
        )

    psnr_value = float(peak_signal_noise_ratio(gt, enhanced, data_range=1.0))
    ssim_value = float(
        structural_similarity(gt, enhanced, channel_axis=2, data_range=1.0)
    )

    if not compute_lpips:
        enhanced_lpips = None
        gt_lpips = None
    else:
        enhanced_lpips, gt_lpips = resize_for_lpips(
            enhanced, gt, max_side=lpips_max_side
        )

    result = MetricResult(
        enhanced=str(enhanced_path), gt=str(gt_path),
        psnr=psnr_value, ssim=ssim_value, lpips=math.nan,
        seconds=time.perf_counter() - start,
    )
    return PreparedPair(result, enhanced_lpips, gt_lpips)


def evaluate_lpips_batch(
    batch: list[PreparedPair],
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """GPU阶段：把多对图片一次送入显卡，减少逐张调用开销。"""
    if not batch:
        return
    enhanced_array = np.stack([item.enhanced_lpips for item in batch])
    gt_array = np.stack([item.gt_lpips for item in batch])

    enhanced_tensor = torch.from_numpy(enhanced_array).permute(0, 3, 1, 2)
    gt_tensor = torch.from_numpy(gt_array).permute(0, 3, 1, 2)
    enhanced_tensor = enhanced_tensor.float().mul(2.0).sub(1.0).to(device)
    gt_tensor = gt_tensor.float().mul(2.0).sub(1.0).to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        values = model(enhanced_tensor, gt_tensor).reshape(-1).cpu().numpy()
    shared_time = (time.perf_counter() - start) / len(batch)
    for item, value in zip(batch, values):
        item.result.lpips = float(value)
        item.result.seconds += shared_time


def safe_mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return math.nan
    if any(math.isinf(value) for value in values):
        return math.inf
    return float(np.mean(values))


def write_csv(path: Path, results: list[MetricResult], lpips_max_side: int) -> None:
    lpips_name = "LPIPS_full" if lpips_max_side <= 0 else f"LPIPS_max{lpips_max_side}"
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["enhanced", "gt", "PSNR_dB", "SSIM", lpips_name, "seconds"])
        for result in results:
            writer.writerow([
                result.enhanced,
                result.gt,
                result.psnr,
                result.ssim,
                result.lpips,
                result.seconds,
            ])
        writer.writerow([
            "AVERAGE",
            "",
            safe_mean([r.psnr for r in results]),
            safe_mean([r.ssim for r in results]),
            safe_mean([r.lpips for r in results]),
            safe_mean([r.seconds for r in results]),
        ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算增强图片相对于GT的PSNR、SSIM和LPIPS")
    parser.add_argument("enhanced", nargs="?", help="增强图片或增强结果文件夹")
    parser.add_argument("gt", nargs="?", help="对应GT图片或GT文件夹")
    parser.add_argument(
        "--lpips-max-side",
        type=int,
        default=1024,
        help="LPIPS最长边，默认1024以节省内存；0表示原始尺寸",
    )
    parser.add_argument(
        "--lpips-net",
        choices=["squeeze", "alex", "vgg"],
        default="squeeze",
        help="LPIPS骨干网络；默认squeeze体积小、CPU速度快",
    )
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="暂不计算LPIPS，只计算PSNR和SSIM（无需下载模型权重）",
    )
    parser.add_argument("--limit", type=int, help="只评价前N对，方便测试")
    parser.add_argument("--csv", help="CSV保存路径；默认增强图旁边或增强文件夹内")
    parser.add_argument("--workers", type=int, default=4,
                        help="CPU并行读图及计算PSNR/SSIM的线程数，默认4")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="GPU一次计算的LPIPS图片数，默认8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enhanced_text = args.enhanced or input("请粘贴增强图片或文件夹路径：").strip()
    gt_text = args.gt or input("请粘贴对应GT图片或GT文件夹路径：").strip()
    enhanced_path = Path(enhanced_text.strip().strip('"')).expanduser()
    gt_path = Path(gt_text.strip().strip('"')).expanduser()

    pairs = pair_images(enhanced_path, gt_path)
    if args.limit is not None:
        pairs = pairs[: max(0, args.limit)]
    if not pairs:
        print("没有需要评价的图片。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备：{device}")
    print(f"成功配对：{len(pairs)} 对")
    if args.lpips_max_side > 0:
        print(
            f"PSNR/SSIM按原尺寸计算；LPIPS将最长边缩放到不超过"
            f"{args.lpips_max_side}像素。"
        )
    else:
        print("三个指标均按原始尺寸计算。")

    # 三种网络均为LPIPS官方支持；SqueezeNet权重更小，适合CPU批量评价。
    model: torch.nn.Module | None = None
    try:
        if not args.skip_lpips:
            model = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
    except Exception as exc:
        weight_info = {
            "squeeze": (
                "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
                "squeezenet1_1-b8a52dc0.pth",
            ),
            "alex": (
                "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
                "alexnet-owt-7be5be79.pth",
            ),
            "vgg": (
                "https://download.pytorch.org/models/vgg16-397923af.pth",
                "vgg16-397923af.pth",
            ),
        }
        weight_url, weight_name = weight_info[args.lpips_net]
        cache_path = Path(torch.hub.get_dir()) / "checkpoints" / weight_name
        raise RuntimeError(
            "LPIPS模型初始化失败。第一次运行需要联网下载预训练权重；"
            "请检查网络后重试。也可以用浏览器下载下面的文件，再放到指定位置。\n"
            f"下载地址：{weight_url}\n"
            f"保存位置：{cache_path}\n"
            f"原始错误：{exc}"
        ) from exc
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(f"LPIPS批大小：{max(1, args.batch_size)}")
    results: list[MetricResult] = []
    failed = 0
    start_all = time.perf_counter()

    workers = max(1, min(args.workers, 8, len(pairs)))
    batch_size = max(1, args.batch_size if device.type == "cuda" else 1)
    pending: list[PreparedPair] = []

    def finish_batch() -> None:
        if not pending:
            return
        if model is not None:
            evaluate_lpips_batch(pending, model, device)
        for item in pending:
            result = item.result
            results.append(result)
            print(
                f"[{len(results)+failed}/{len(pairs)}] {Path(result.enhanced).name} | "
                f"PSNR={result.psnr:.4f} dB | SSIM={result.ssim:.4f} | "
                f"LPIPS={'未计算' if math.isnan(result.lpips) else f'{result.lpips:.4f}'} | "
                f"{result.seconds:.2f}秒"
            )
        pending.clear()

    # 分块提交CPU任务，避免一次把1000张全读入内存。
    chunk_size = max(batch_size, workers * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk_start in range(0, len(pairs), chunk_size):
            chunk = pairs[chunk_start: chunk_start + chunk_size]
            future_map = {
                pool.submit(
                    prepare_pair, enhanced, gt, args.lpips_max_side, model is not None
                ): enhanced
                for enhanced, gt in chunk
            }
            for future in as_completed(future_map):
                enhanced = future_map[future]
                try:
                    item = future.result()
                    # 极少数尺寸不同的图片不能放在同一个Tensor批次中。
                    if (
                        pending
                        and item.enhanced_lpips is not None
                        and pending[0].enhanced_lpips.shape != item.enhanced_lpips.shape
                    ):
                        finish_batch()
                    pending.append(item)
                    if len(pending) >= batch_size:
                        finish_batch()
                except Exception as exc:
                    failed += 1
                    print(f"[{len(results)+failed}/{len(pairs)}] 失败：{enhanced.name} -> {exc}")
        finish_batch()

    if not results:
        raise RuntimeError("所有图片均评价失败，请检查上面的错误信息。")

    if args.csv:
        csv_path = Path(args.csv.strip().strip('"')).expanduser()
    elif enhanced_path.is_dir():
        csv_path = enhanced_path / "metrics.csv"
    else:
        csv_path = enhanced_path.parent / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda result: result.enhanced.lower())
    write_csv(csv_path, results, args.lpips_max_side)

    mean_psnr = safe_mean([r.psnr for r in results])
    mean_ssim = safe_mean([r.ssim for r in results])
    mean_lpips = safe_mean([r.lpips for r in results])
    elapsed = time.perf_counter() - start_all

    print("\n========== 平均结果 ==========")
    print(f"PSNR  = {mean_psnr:.4f} dB（越大越好）")
    print(f"SSIM  = {mean_ssim:.4f}（越接近1越好）")
    if math.isnan(mean_lpips):
        print("LPIPS = 未计算（本次使用了 --skip-lpips）")
    else:
        print(f"LPIPS = {mean_lpips:.4f}（越接近0越好）")
    print(f"成功 {len(results)} 对，失败 {failed} 对，总耗时 {elapsed:.1f}秒")
    print(f"详细结果：{csv_path}")


if __name__ == "__main__":
    main()
