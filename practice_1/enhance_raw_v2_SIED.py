"""用 SIED 官方预训练模型增强极暗 RAW。

可以处理一张，也可以处理一个文件夹里的任意张数。
流程：黑电平校正 -> Bayer 四通道打包 -> FP16 推理 -> PNG。
曝光、去噪和颜色恢复都由模型完成。

官方训练和推理使用 20 步、FP32。默认结果写到 output_v2。
没有用 --ckpt 或 --ckpt-dir 指定权重时，运行后会要求输入 .pt 文件路径。

示例：
  python enhance_raw_v2_SIED.py data/0.0001 --ckpt ckpt/canon_0.0001.pt --steps 20 --limit 50
  python enhance_raw_v2_SIED.py data/0.0001/1.raw --ckpt ckpt/canon_0.0001.pt
  python enhance_raw_v2_SIED.py data/0.001 --ckpt-dir ckpt --camera sony --lux 0.001
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


HEIGHT = 2160
WIDTH = 3840
BLACK_LEVEL = 512
WHITE_LEVEL = 16383
OFFICIAL_STEPS = 20
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_v2"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "SIED_official"


def normalize_raw(raw: np.ndarray) -> np.ndarray:
    """黑电平校正，再按白电平缩放到大约 [0, 1]。"""
    return np.maximum(raw.astype(np.float32) - BLACK_LEVEL, 0) / (WHITE_LEVEL - BLACK_LEVEL)


def pack_raw(raw: np.ndarray) -> np.ndarray:
    """与官方 dataloader.pack_raw 相同的 Bayer 打包顺序。"""
    image = normalize_raw(raw)
    return np.concatenate(
        (
            image[0::2, 0::2, None],
            image[0::2, 1::2, None],
            image[1::2, 1::2, None],
            image[1::2, 0::2, None],
        ),
        axis=2,
    )


def raw_bayer_planes(raw: np.ndarray) -> np.ndarray:
    """全分辨率四通道 Bayer，通道顺序与 pack_raw 相同。

    每个传感器位置只在自己的颜色通道上保留真实值，其余通道为 0。
    因此边缘仍在原来的像素上，不会被插值抹开。
    """
    image = normalize_raw(raw)
    planes = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
    planes[0::2, 0::2, 0] = image[0::2, 0::2]
    planes[0::2, 1::2, 1] = image[0::2, 1::2]
    planes[1::2, 1::2, 2] = image[1::2, 1::2]
    planes[1::2, 0::2, 3] = image[1::2, 0::2]
    return planes


def read_raw(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    expected = HEIGHT * WIDTH
    if raw.size != expected:
        raise ValueError(f"像素数为 {raw.size}，预期 {expected}（{WIDTH}x{HEIGHT}）")
    return raw.reshape(HEIGHT, WIDTH)


def sort_key(path: Path) -> tuple:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem.lower())


def collect_raws(input_path: Path, limit: int, include_gt: bool) -> list[Path]:
    """一张文件就处理一张；一个文件夹就处理其中全部 RAW，再用 --limit 截取前 N 张。"""
    if input_path.is_file():
        if input_path.suffix.lower() != ".raw":
            raise ValueError("请选择 .raw 文件，或包含 .raw 的文件夹。")
        files = [input_path]
    elif input_path.is_dir():
        files = [path for path in input_path.glob("*.raw") if path.is_file()]
    else:
        raise FileNotFoundError(f"找不到路径：{input_path}")

    if not include_gt:
        files = [path for path in files if "gt" not in path.stem.lower()]
    files.sort(key=sort_key)
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"没有找到要处理的低光 .raw 文件：{input_path}")
    return files


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("重叠必须小于分块尺寸。")
    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_weight(
    tile_h: int,
    tile_w: int,
    top: int,
    left: int,
    full_h: int,
    full_w: int,
    overlap: int,
) -> np.ndarray:
    """重叠区线性过渡；贴着整图边缘的一边保持权重 1。"""
    overlap = max(0, min(overlap, tile_h // 2, tile_w // 2))
    weight_y = np.ones(tile_h, dtype=np.float32)
    weight_x = np.ones(tile_w, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        if top > 0:
            weight_y[:overlap] = ramp
        if top + tile_h < full_h:
            weight_y[-overlap:] = ramp[::-1]
        if left > 0:
            weight_x[:overlap] = ramp
        if left + tile_w < full_w:
            weight_x[-overlap:] = ramp[::-1]
    return weight_y[:, None, None] * weight_x[None, :, None]


def import_model(model_dir: Path):
    if not (model_dir / "models").is_dir():
        raise FileNotFoundError(f"找不到官方模型代码：{model_dir}")
    model_dir_text = str(model_dir)
    if model_dir_text not in sys.path:
        sys.path.insert(0, model_dir_text)
    from models import GaussianDiffusion, Unet

    return GaussianDiffusion, Unet


def build_model(model_dir: Path):
    gaussian_diffusion, unet = import_model(model_dir)
    network = unet(channels=3, dim=128, dim_mults=(1, 2, 4, 8), flash_attn=False)
    return gaussian_diffusion(
        network,
        timesteps=1000,
        sampling_timesteps=20,
        training_stage="stage2",
    )


def load_ema(model, ckpt_path: Path) -> None:
    print(f"正在加载权重：{ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    ema_state = checkpoint["ema"]
    state = {
        key[len("ema_model."):]: value
        for key, value in ema_state.items()
        if key.startswith("ema_model.")
    }
    del checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"权重和模型对不上。缺少 {len(missing)} 项，多余 {len(unexpected)} 项。"
            f"\n缺少示例：{missing[:5]}\n多余示例：{unexpected[:5]}"
        )
    model.eval()


def pad_to_multiple(raw: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    height, width = raw.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        raw = F.pad(raw, (0, pad_w, 0, pad_h), mode="reflect")
    return raw, height, width


def fast_sample(model, raw: torch.Tensor, steps: int) -> torch.Tensor:
    """整张打包 RAW 只编码一次，再做 DDIM。不跑用不到的 GT 颜色分支。"""
    net = model.stage1_Net
    pyramid = net.RAW_encoder(raw)
    fea16 = pyramid["down16"]
    raw_fea = torch.sigmoid(net.RAW_illu(net.RAW_channel_down(fea16)))
    model.sampling_timesteps = steps
    pred_fea = model.ddim_sample(model.normalize(raw_fea))
    restored = net.RGB_decoder(net.RGB_channel_up(pred_fea) + fea16, pyramid)
    return F.pixel_shuffle(restored, 2).clamp(0, 1)


def enhance_tensor(model, raw: torch.Tensor, steps: int, precision: str) -> torch.Tensor:
    padded, height, width = pad_to_multiple(raw, 256)
    if precision == "fp32":
        with torch.autocast(device_type="cuda", enabled=False):
            predicted = fast_sample(model, padded, steps)
    else:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predicted = fast_sample(model, padded, steps)
    return predicted.float()[:, :, : height * 2, : width * 2]


def enhance_tile(model, tile: np.ndarray, device: torch.device, steps: int, precision: str) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(tile)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)
    predicted = enhance_tensor(model, tensor, steps, precision)
    return predicted.squeeze(0).float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def enhance_image_tiled(
    model, packed: np.ndarray, tile: int, overlap: int, steps: int,
    device: torch.device, precision: str,
) -> np.ndarray:
    packed_h, packed_w, _ = packed.shape
    out_h, out_w = packed_h * 2, packed_w * 2
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weights = np.zeros((out_h, out_w, 1), dtype=np.float32)
    tile_h = min(tile, packed_h)
    tile_w = min(tile, packed_w)

    ys = tile_starts(packed_h, tile_h, overlap)
    xs = tile_starts(packed_w, tile_w, overlap)
    total = len(ys) * len(xs)
    done = 0
    for y in ys:
        for x in xs:
            patch = packed[y:y + tile_h, x:x + tile_w]
            enhanced = enhance_tile(model, patch, device, steps, precision)
            top, left = y * 2, x * 2
            block_h, block_w = enhanced.shape[:2]
            weight = blend_weight(block_h, block_w, top, left, out_h, out_w, overlap * 2)
            canvas[top:top + block_h, left:left + block_w] += enhanced * weight
            weights[top:top + block_h, left:left + block_w] += weight
            done += 1
            print(f"    分块 {done}/{total}", flush=True)
    return canvas / np.maximum(weights, 1e-6)


def enhance_image(
    model, packed: np.ndarray, steps: int, tile: int, overlap: int,
    device: torch.device, precision: str,
) -> np.ndarray:
    """优先整张推理。只有显存不够时才退回分块。"""
    tensor = torch.from_numpy(np.ascontiguousarray(packed)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)
    try:
        predicted = enhance_tensor(model, tensor, steps, precision)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        print("    显存不足，这一张改为分块处理。", flush=True)
        return enhance_image_tiled(model, packed, tile, overlap, steps, device, precision)
    return predicted.squeeze(0).float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def save_png(path: Path, rgb: np.ndarray) -> None:
    image = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise OSError(f"保存失败：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def resolve_ckpt(args: argparse.Namespace) -> Path:
    """优先用命令行给出的权重；都没给时，要求用户输入 .pt 路径。"""
    if args.ckpt is not None:
        return args.ckpt.expanduser()
    if args.ckpt_dir is not None:
        return args.ckpt_dir.expanduser() / f"{args.camera}_{args.lux}.pt"
    try:
        text = input("请输入权重文件路径（.pt）：").strip().strip('"')
    except EOFError:
        text = ""
    if not text:
        raise FileNotFoundError("没有提供权重文件。请加上 --ckpt，或在终端里输入 .pt 路径。")
    return Path(text).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 SIED 官方模型增强任意张极暗 RAW")
    parser.add_argument("input", nargs="?", help=".raw 文件，或存放多张 .raw 的文件夹")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="PNG 输出文件夹；默认写到本脚本旁的 output_v2")
    parser.add_argument("--ckpt", type=Path, help=".pt 权重文件。不填且未指定 --ckpt-dir 时，运行后会要求输入")
    parser.add_argument("--ckpt-dir", type=Path, help="权重文件夹。指定后按相机和照度拼出文件名，不再询问")
    parser.add_argument("--camera", choices=["canon", "sony"], default="canon")
    parser.add_argument("--lux", choices=["0.0001", "0.001", "0.01"], default="0.0001")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="SIED 官方代码目录")
    parser.add_argument("--limit", type=int, default=0, help="只处理排序后的前 N 张；0 表示全部")
    parser.add_argument("--steps", type=int, default=OFFICIAL_STEPS, help="扩散采样步数。官方训练和推理使用 20")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32", help="官方推理是 fp32。fp16 更快，但 20 步迭代会累积误差")
    parser.add_argument("--tile", type=int, default=512, help="仅显存不足时使用的分块边长，需能被 256 整除")
    parser.add_argument("--overlap", type=int, default=64, help="打包后的重叠像素，64 对应成图 128 像素")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="重新生成已经存在的 PNG")
    parser.add_argument("--include-gt", action="store_true", help="也处理文件名中带 gt 的 RAW")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tile % 256 != 0:
        raise ValueError("tile 必须是 256 的倍数，官方模型按这个尺寸对齐。")
    if not 1 <= args.steps <= 1000:
        raise ValueError("steps 必须在 1 到 1000 之间。")
    if args.limit < 0:
        raise ValueError("limit 不能小于 0。")
    if not torch.cuda.is_available():
        raise RuntimeError("没有可用的 NVIDIA GPU。")

    input_text = args.input or input("请粘贴 .raw 文件或文件夹路径：").strip()
    input_path = Path(input_text.strip().strip('"')).expanduser()
    files = collect_raws(input_path, args.limit, args.include_gt)
    output_dir = args.output.expanduser()
    ckpt_path = resolve_ckpt(args)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到权重：{ckpt_path}")

    device = torch.device("cuda")
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"待处理 {len(files)} 张，整张推理，采样 {args.steps} 步，{args.precision}")
    print(f"权重：{ckpt_path}")
    print(f"输出目录：{output_dir}")
    if args.steps != OFFICIAL_STEPS:
        print(f"注意：SIED 官方采样是 {OFFICIAL_STEPS} 步，这次用的是 {args.steps} 步。")

    model = build_model(args.model_dir)
    load_ema(model, ckpt_path)
    model.to(device)
    torch.backends.cudnn.benchmark = True
    use_tf32 = args.precision == "fp16"
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if not use_tf32:
        torch.set_float32_matmul_precision("highest")

    succeeded = 0
    skipped = 0
    failed = 0
    pending: list[tuple[str, Future]] = []
    started_all = time.perf_counter()

    def collect_finished(wait: bool) -> None:
        nonlocal succeeded, failed
        while pending and (wait or pending[0][1].done()):
            name, future = pending.pop(0)
            try:
                future.result()
            except Exception as exc:
                failed += 1
                print(f"    保存失败 {name}：{exc}", flush=True)
            else:
                succeeded += 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, path in enumerate(files, start=1):
            collect_finished(wait=False)
            output_path = output_dir / f"{path.stem}_enhanced.png"
            if output_path.exists() and not args.overwrite:
                skipped += 1
                print(f"[{index}/{len(files)}] 跳过已存在 {output_path.name}", flush=True)
                continue
            image_seed = args.seed + (int(path.stem) if path.stem.isdigit() else index)
            torch.manual_seed(image_seed)
            torch.cuda.manual_seed_all(image_seed)
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    enhanced = enhance_image(
                        model, pack_raw(read_raw(path)), args.steps, args.tile, args.overlap,
                        device, args.precision,
                    )
            except Exception as exc:
                failed += 1
                print(f"[{index}/{len(files)}] 失败 {path.name}：{exc}", flush=True)
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue
            while len(pending) >= 2:
                collect_finished(wait=True)
            pending.append((output_path.name, pool.submit(save_png, output_path, enhanced)))
            print(
                f"[{index}/{len(files)}] {path.name} | 均值 {enhanced.mean():.3f} | "
                f"{time.perf_counter() - started:.2f}秒",
                flush=True,
            )
        collect_finished(wait=True)

    elapsed = time.perf_counter() - started_all
    print(
        f"\n结束：成功 {succeeded}，跳过 {skipped}，失败 {failed}，共 {len(files)} 张，"
        f"耗时 {elapsed:.1f}秒"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
