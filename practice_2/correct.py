"""文档图像几何校正。

流程：掩膜四角透视 -> 文字转正 -> 书缝拆页 -> 文字行拉直或边界网格 -> 评价。

依赖见 requirements.txt。paddleocr 装完后如果 cv2 版本变成 4.x，
按 requirements.txt 里的顺序把 opencv-python 5 装回去。

用法（在 practice_2 目录下）：
    python correct.py --samples          # 5 张代表图
    python correct.py                    # 全部 200 张
    python correct.py --ocr              # 加上 PaddleOCR 置信度
    python correct.py --samples --debug  # 额外保存文字行调试图
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

SAMPLES = [
    "150.jpg",
    "dewarp20260728_rotate_019.jpg",
    "dewarp20260728_perspective_013.jpg",
    "dewarp20260728_curved_017.jpg",
    "dewarp20260728_fold_022.jpg",
]

# 分析与输出的长边。几何在这个尺度上估计，结果也按这个尺度保存。
WORK_LONG = 1500


def imread_unicode(path: Path, color: bool) -> np.ndarray | None:
    """cv2.imread 不能读中文路径，改用 imdecode。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    return cv2.imdecode(data, flag)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, image, [cv2.IMWRITE_JPEG_QUALITY, 92] if ext in {".jpg", ".jpeg"} else [])
    if not ok:
        raise RuntimeError(f"编码失败: {path}")
    buf.tofile(str(path))


def to_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def resize_long(img: np.ndarray, mask: np.ndarray, long_side: int) -> tuple[np.ndarray, np.ndarray, float]:
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 0.999:
        return img, mask, 1.0
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    sm = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return small, sm, scale


def crop_to_mask(img: np.ndarray, mask: np.ndarray, pad: int = 6) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 127)
    if xs.size < 80:
        return img, mask
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(img.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(img.shape[1], int(xs.max()) + pad + 1)
    return img[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def find_corners(mask: np.ndarray) -> np.ndarray | None:
    """凸包上用 x+y、x-y 的极值取四个外角，避开书脊向内的凹陷。"""
    binm = (mask > 127).astype(np.uint8)
    if int(binm.sum()) < 200:
        return None
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float32)
    if len(hull) < 4:
        return None
    s = hull[:, 0] + hull[:, 1]
    d = hull[:, 0] - hull[:, 1]
    corners = np.stack([
        hull[np.argmin(s)],
        hull[np.argmax(d)],
        hull[np.argmax(s)],
        hull[np.argmin(d)],
    ]).astype(np.float32)
    if cv2.contourArea(corners) < 1000:
        return None
    dists = [float(np.linalg.norm(corners[i] - corners[(i + 1) % 4])) for i in range(4)]
    # 有一条边短到接近一个点时，单应性会把几个像素拉成整页。
    if min(dists) < 5 or min(dists) / max(dists) < 0.08:
        return None
    return corners


def perspective_warp(img: np.ndarray, mask: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    tl, tr, br, bl = corners
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    width = max(width, 32)
    height = max(height, 32)
    if width * height > img.shape[0] * img.shape[1] * 4:
        return None
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    warped_mask = cv2.warpPerspective(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    return crop_to_mask(warped, warped_mask)


def rotate_bound(img: np.ndarray, mask: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """angle 为逆时针角度。背景填白，掩膜填 0，并裁掉空白。"""
    if abs(angle) < 0.15:
        return crop_to_mask(img, mask)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw = int(round(h * sin + w * cos))
    nh = int(round(h * cos + w * sin))
    matrix[0, 2] += nw / 2.0 - center[0]
    matrix[1, 2] += nh / 2.0 - center[1]
    rot = cv2.warpAffine(img, matrix, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    rot_m = cv2.warpAffine(mask, matrix, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    return crop_to_mask(rot, rot_m)


def binarize_text(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """压平书缝阴影后把字变成白色前景。"""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 21)
    norm = gray.astype(np.float32) / (blur.astype(np.float32) + 1.0) * 255.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    block = 41 if min(gray.shape[:2]) > 200 else 21
    bw = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)
    bw[mask < 127] = 0
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return bw


def projection_peakiness(bw: np.ndarray) -> float:
    proj = (bw > 127).sum(axis=1).astype(np.float64)
    total = proj.sum()
    if total < 50:
        return 0.0
    p = proj / total
    return float(np.dot(p, p))


def rotate_binary(bw: np.ndarray, angle: float) -> np.ndarray:
    """只用于方向打分。背景填 0，避免白边被当成文字。"""
    if abs(angle) < 0.15:
        return bw
    h, w = bw.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw = int(round(h * sin + w * cos))
    nh = int(round(h * cos + w * sin))
    matrix[0, 2] += nw / 2.0 - center[0]
    matrix[1, 2] += nh / 2.0 - center[1]
    rot = cv2.warpAffine(bw, matrix, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    return rot


def deskew(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """用文字水平投影的尖锐程度选择方向，先在 0/90/180/270 里选，再 ±15° 微调。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = binarize_text(gray, mask)
    probe, _, _ = resize_long(bw, mask, 700)

    def score(angle: float) -> float:
        return projection_peakiness(rotate_binary(probe, angle))

    coarse_scores = {angle: score(float(angle)) for angle in (0, 90, 180, 270)}
    # 0° 和 180° 的投影几乎一样，差别不大时保持较小旋转，避免把正的页面倒过去。
    if coarse_scores[180] <= coarse_scores[0] * 1.08:
        coarse_scores[180] = -1.0
    if coarse_scores[270] <= coarse_scores[90] * 1.08:
        coarse_scores[270] = -1.0
    coarse = max(coarse_scores, key=coarse_scores.get)
    best_angle = float(coarse)
    best_score = -1.0
    for delta in range(-12, 13, 2):
        angle = float(coarse + delta)
        value = score(angle)
        if value > best_score:
            best_score = value
            best_angle = angle
    out_img, out_mask = rotate_bound(img, mask, best_angle)
    return out_img, out_mask, best_angle


def split_gutter(img: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """中间有足够深的竖直暗槽时，按书缝切成左右页。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valid = mask > 127
    h, w = gray.shape
    if w < 80:
        return [(img, mask)]
    values = gray.astype(np.float32)
    values[~valid] = np.nan
    with np.errstate(invalid="ignore"):
        col = np.nanmean(values, axis=0)
    good = np.isfinite(col)
    if int(good.sum()) < w * 0.5:
        return [(img, mask)]
    idx = np.arange(w)
    col = np.interp(idx, idx[good], col[good]).astype(np.float32)
    col = cv2.GaussianBlur(col.reshape(1, -1), (0, 0), 11).ravel()
    lo, hi = int(w * 0.30), int(w * 0.70)
    x = lo + int(np.argmin(col[lo:hi]))
    left_ref = float(np.median(col[max(0, lo - 50):lo])) if lo > 8 else float(col[lo])
    right_ref = float(np.median(col[hi:min(w, hi + 50)]))
    dip = (left_ref + right_ref) / 2.0 - float(col[x])
    if dip < 16 or x < w * 0.28 or x > w * 0.72:
        return [(img, mask)]
    pages = []
    for x0, x1 in ((0, x - 3), (x + 3, w)):
        if x1 - x0 < w * 0.22:
            return [(img, mask)]
        pages.append(crop_to_mask(img[:, x0:x1], mask[:, x0:x1]))
    return pages


def text_components(bw: np.ndarray) -> list[tuple[float, float, int, int]]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(bw, 8)
    h, w = bw.shape
    raw = []
    heights = []
    for i in range(1, count):
        x, y, cw, ch, area = stats[i]
        if area < 12 or ch < 6 or cw < 2:
            continue
        if ch > h * 0.18 or cw > w * 0.45 or area > 0.02 * h * w:
            continue
        raw.append((float(centroids[i, 0]), float(centroids[i, 1]), int(cw), int(ch)))
        heights.append(ch)
    if len(heights) < 12:
        return []
    med = float(np.median(heights))
    return [c for c in raw if med * 0.40 <= c[3] <= med * 2.8]


def group_lines(chars: list[tuple[float, float, int, int]], page_w: int) -> list[list[tuple[float, float, int, int]]]:
    """从左到右串字。下一个字要够近，且高度差不超过约一个字高，这样弧线也能跟着走。"""
    if len(chars) < 8:
        return []
    med_h = float(np.median([c[3] for c in chars]))
    chars = sorted(chars, key=lambda c: (c[0], c[1]))
    used = np.zeros(len(chars), dtype=bool)
    x_gap = med_h * 5.0
    y_gate = med_h * 0.90
    lines = []
    for i, _ in enumerate(chars):
        if used[i]:
            continue
        line = [chars[i]]
        used[i] = True
        cursor = i
        while True:
            lx, ly = line[-1][0], line[-1][1]
            best = -1
            best_dx = 1e9
            for j in range(cursor + 1, len(chars)):
                if used[j]:
                    continue
                dx = chars[j][0] - lx
                if dx > x_gap:
                    break
                if dx < med_h * 0.15:
                    continue
                if abs(chars[j][1] - ly) <= y_gate and dx < best_dx:
                    best = j
                    best_dx = dx
            if best < 0:
                break
            used[best] = True
            line.append(chars[best])
            cursor = best
        xs = [p[0] for p in line]
        if len(line) >= 6 and max(xs) - min(xs) >= page_w * 0.28:
            lines.append(line)
    lines.sort(key=lambda ln: float(np.mean([p[1] for p in ln])))
    return lines


def straightness(img: np.ndarray, mask: np.ndarray) -> tuple[float | None, list]:
    """字符中心相对直线的垂直残差标准差，越小越平直。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bw = binarize_text(gray, mask)
    lines = group_lines(text_components(bw), img.shape[1])
    if len(lines) < 3:
        return None, lines
    scores = []
    weights = []
    for line in lines:
        xs = np.array([p[0] for p in line], np.float64)
        ys = np.array([p[1] for p in line], np.float64)
        coef = np.polyfit(xs, ys, 1)
        resid = ys - np.polyval(coef, xs)
        scores.append(float(np.std(resid)))
        weights.append(len(line))
    return float(np.average(scores, weights=weights)), lines


def _line_model(line: list[tuple[float, float, int, int]], page_h: int):
    xs = np.array([p[0] for p in line], np.float64)
    ys = np.array([p[1] for p in line], np.float64)
    x0, x1 = float(xs.min()), float(xs.max())
    span = max(x1 - x0, 1.0)
    xn = (xs - x0) / span
    coef = np.polyfit(xn, ys, 2)
    y_hat = np.polyval(coef, xn)
    med_h = float(np.median([p[3] for p in line]))
    if float(np.std(ys - y_hat)) > med_h:
        linear = np.polyfit(xn, ys, 1)
        coef = np.array([0.0, linear[0], linear[1]])
    y0 = float(np.polyval(coef, 0.0))
    y1 = float(np.polyval(coef, 1.0))
    ym = float(np.polyval(coef, 0.5))
    bow = ym - 0.5 * (y0 + y1)
    if abs(bow) > 0.16 * page_h:
        return None
    return x0, x1, coef, float(ys.mean())


def textline_remap(img: np.ndarray, mask: np.ndarray, lines: list) -> tuple[np.ndarray, np.ndarray] | None:
    """输出图上水平的文字行，到原图二次曲线的位置去取样。"""
    h, w = img.shape[:2]
    models = []
    for line in lines:
        model = _line_model(line, h)
        if model is not None:
            models.append(model)
    if len(models) < 4:
        return None
    models.sort(key=lambda m: m[3])

    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    ys = np.arange(h, dtype=np.float64)
    crossed = 0
    filled = 0
    for x in range(w):
        pairs = []
        for x0, x1, coef, y_mean in models:
            if x < x0 or x > x1:
                continue
            t = (x - x0) / max(x1 - x0, 1.0)
            pairs.append((y_mean, float(np.polyval(coef, t))))
        if len(pairs) < 3:
            continue
        pairs.sort(key=lambda p: p[0])
        y_out = np.array([p[0] for p in pairs], np.float64)
        y_src = np.array([p[1] for p in pairs], np.float64)
        # 去重，保证插值自变量递增
        keep = np.concatenate([[True], np.diff(y_out) > 1.0])
        y_out, y_src = y_out[keep], y_src[keep]
        if len(y_out) < 3:
            continue
        if np.any(np.diff(y_src) <= 0.5):
            crossed += 1
            continue
        src = np.interp(ys, y_out, y_src)
        below = ys < y_out[0]
        above = ys > y_out[-1]
        src[below] = ys[below] + (y_src[0] - y_out[0])
        src[above] = ys[above] + (y_src[-1] - y_out[-1])
        map_y[:, x] = src.astype(np.float32)
        filled += 1
    if filled < w * 0.35 or crossed > w * 0.15:
        return None
    # 没有文字的列沿用最近一列的纵向位移，避免接缝
    identity = np.arange(h, dtype=np.float32)[:, None]
    known = np.zeros(w, dtype=bool)
    # 重新标记：与恒等映射不同，或 filled 的列。上面未写入的列仍是恒等。
    shift = map_y - identity
    known = np.max(np.abs(shift), axis=0) > 0.2
    if int(known.sum()) < w * 0.35:
        return None
    xs_known = np.flatnonzero(known)
    for x in range(w):
        if known[x]:
            continue
        nearest = xs_known[np.argmin(np.abs(xs_known - x))]
        map_y[:, x] = map_y[:, nearest]
    map_y = cv2.GaussianBlur(map_y, (9, 1), 0)
    shift = map_y - identity
    if float(np.percentile(np.abs(shift), 95)) > 0.18 * h:
        return None
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    out_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, out_mask


def path_forward(cnt: np.ndarray, i: int, j: int) -> np.ndarray:
    if i <= j:
        return cnt[i:j + 1]
    return np.concatenate([cnt[i:], cnt[:j + 1]], axis=0)


def edge_between(cnt: np.ndarray, i: int, j: int, top: bool) -> np.ndarray:
    forward = path_forward(cnt, i, j)
    backward = path_forward(cnt, j, i)[::-1]
    if abs(float(forward[:, 1].mean()) - float(backward[:, 1].mean())) > 5:
        if top:
            path = forward if forward[:, 1].mean() < backward[:, 1].mean() else backward
        else:
            path = forward if forward[:, 1].mean() > backward[:, 1].mean() else backward
    else:
        path = forward if len(forward) <= len(backward) else backward
    if path[0, 0] > path[-1, 0]:
        path = path[::-1]
    return path


def boundary_info(mask: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    binm = (mask > 127).astype(np.uint8)
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    corners = find_corners(mask)
    if corners is None or len(cnt) < 20:
        return None

    def nearest(pt: np.ndarray) -> int:
        d = np.sum((cnt.astype(np.float32) - pt) ** 2, axis=1)
        return int(np.argmin(d))

    tl, tr, br, bl = [nearest(p) for p in corners]
    top = edge_between(cnt, tl, tr, top=True)
    bot = edge_between(cnt, bl, br, top=False)
    if len(top) < 8 or len(bot) < 8:
        return None

    def deviation(path: np.ndarray) -> float:
        a = path[0].astype(np.float64)
        b = path[-1].astype(np.float64)
        v = b - a
        length = np.linalg.norm(v) + 1e-6
        w = path.astype(np.float64) - a
        cross = np.abs(w[:, 0] * v[1] - w[:, 1] * v[0]) / length
        return float(cross.max())

    ratio = max(deviation(top), deviation(bot)) / float(mask.shape[0])
    return ratio, top, bot


def resample_arc(path: np.ndarray, n: int) -> np.ndarray:
    pts = path.astype(np.float64)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if dist[-1] < 1:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0, dist[-1], n)
    x = np.interp(target, dist, pts[:, 0])
    y = np.interp(target, dist, pts[:, 1])
    return np.stack([x, y], axis=1)


def boundary_remap(img: np.ndarray, mask: np.ndarray, top: np.ndarray, bot: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = img.shape[:2]
    upper = resample_arc(top, w)
    lower = resample_arc(bot, w)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    map_x = (1.0 - t) * upper[:, 0].astype(np.float32)[None, :] + t * lower[:, 0].astype(np.float32)[None, :]
    map_y = (1.0 - t) * upper[:, 1].astype(np.float32)[None, :] + t * lower[:, 1].astype(np.float32)[None, :]
    if float(np.mean(np.diff(map_x, axis=1) > 0)) < 0.9:
        return None
    if float(np.mean(np.diff(map_y, axis=0) > 0)) < 0.9:
        return None
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    out_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, out_mask


def correct_page(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, float | None, float | None, list]:
    before, lines = straightness(img, mask)
    method = "perspective"
    chosen = img
    chosen_mask = mask
    debug_lines = lines

    if len(lines) >= 4:
        mapped = textline_remap(img, mask, lines)
        if mapped is not None:
            after_img, after_mask = mapped
            after, after_lines = straightness(after_img, after_mask)
            improved = (
                before is not None
                and after is not None
                and after < before * 0.95
                and before - after > 0.25
            )
            if improved:
                chosen, chosen_mask = after_img, after_mask
                method = "textline"
                debug_lines = after_lines
                before_out, after_out = before, after
                return chosen, chosen_mask, method, before_out, after_out, debug_lines

    info = boundary_info(mask)
    if info is not None and info[0] >= 0.015 and method == "perspective":
        mapped = boundary_remap(img, mask, info[1], info[2])
        if mapped is not None:
            b_img, b_mask = mapped
            b_after, _ = straightness(b_img, b_mask)
            # 边界网格把已经较直的文字拉弯时，退回透视结果。
            worse = before is not None and b_after is not None and b_after > before * 1.08
            if not worse:
                chosen, chosen_mask = b_img, b_mask
                method = "boundary"

    after, after_lines = straightness(chosen, chosen_mask)
    if method == "textline":
        debug_lines = after_lines
    return chosen, chosen_mask, method, before, after, debug_lines if method != "perspective" else lines


def hstack_white(images: list[np.ndarray], gap: int = 12) -> np.ndarray:
    height = max(im.shape[0] for im in images)
    resized = []
    for im in images:
        if im.shape[0] != height:
            scale = height / im.shape[0]
            im = cv2.resize(im, (max(1, int(im.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
        resized.append(im)
    if len(resized) == 1:
        return resized[0]
    bar = np.full((height, gap, 3), 255, np.uint8)
    out = resized[0]
    for im in resized[1:]:
        out = np.concatenate([out, bar, im], axis=1)
    return out


def make_compare(original: np.ndarray, corrected: np.ndarray, height: int = 720) -> np.ndarray:
    def fit(im: np.ndarray) -> np.ndarray:
        scale = height / im.shape[0]
        return cv2.resize(im, (max(1, int(im.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)

    left, right = fit(original), fit(corrected)
    bar = np.full((height, 8, 3), 255, np.uint8)
    canvas = np.concatenate([left, bar, right], axis=1)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = None
    for candidate in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"):
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 28)
            break
    if font is None:
        font = ImageFont.load_default()
    draw.rectangle((8, 8, 130, 48), fill=(0, 0, 0))
    draw.text((16, 12), "校正前", fill=(255, 255, 255), font=font)
    draw.rectangle((left.shape[1] + 16, 8, left.shape[1] + 138, 48), fill=(0, 0, 0))
    draw.text((left.shape[1] + 24, 12), "校正后", fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_lines(img: np.ndarray, lines: list) -> np.ndarray:
    vis = img.copy()
    for line in lines:
        pts = np.array([[[int(p[0]), int(p[1])]] for p in line], np.int32)
        cv2.polylines(vis, [pts], False, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def nanmean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return float(np.mean(nums))


def process_pair(img_path: Path, mask_path: Path, debug_dir: Path | None) -> dict:
    img = imread_unicode(img_path, color=True)
    mask = imread_unicode(mask_path, color=False)
    if img is None or mask is None:
        raise RuntimeError("读图失败")
    mask = to_binary_mask(mask)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 角点在缩小掩膜上找，再除以 scale 回到原图，透视仍作用在原图上。
    _, small_mask, scale = resize_long(img, mask, 1200)
    corners = find_corners(small_mask)
    if corners is None:
        raise RuntimeError("页面外轮廓太扁或太碎，四角透视不可靠")
    corners = corners / scale
    warped = perspective_warp(img, mask, corners)
    if warped is None:
        raise RuntimeError("透视变换失败")
    page, page_mask = warped
    page, page_mask, _ = resize_long(page, page_mask, WORK_LONG)
    page, page_mask, angle = deskew(page, page_mask)

    parts = split_gutter(page, page_mask)
    corrected_pages = []
    methods = []
    befores = []
    afters = []
    debug_vis = []
    for index, (part, part_mask) in enumerate(parts):
        out, out_mask, method, before, after, lines = correct_page(part, part_mask)
        corrected_pages.append(out)
        methods.append(method)
        befores.append(before)
        afters.append(after)
        if debug_dir is not None:
            debug_vis.append(draw_lines(out, lines))

    final = hstack_white(corrected_pages)
    compare = make_compare(img, final)
    if debug_dir is not None and debug_vis:
        imwrite_unicode(debug_dir / f"{img_path.stem}.jpg", hstack_white(debug_vis))
    return {
        "image": final,
        "compare": compare,
        "method": "+".join(methods),
        "pages": len(corrected_pages),
        "angle": angle,
        "straight_before": nanmean(befores),
        "straight_after": nanmean(afters),
    }


def list_pairs(data: Path) -> list[tuple[Path, Path]]:
    image_dir = data / "images"
    mask_dir = data / "mask"
    pairs = []
    for path in sorted(image_dir.iterdir(), key=lambda p: p.name):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        mask = mask_dir / f"{path.stem}.png"
        if not mask.exists():
            alt = mask_dir / f"{path.stem}.jpg"
            mask = alt if alt.exists() else mask
        if mask.exists():
            pairs.append((path, mask))
    return pairs


def fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{value:.4f}"


class OcrScorer:
    def __init__(self) -> None:
        self.engine = None
        self.kind = ""

    def load(self) -> None:
        import os
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_onednn"] = "0"
        # 必须在 import paddleocr 之前关掉。Windows CPU 上 oneDNN 会在 PIR 里直接报错。
        os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        # 国内从百度 BOS 下模型。默认走 HuggingFace，识别模型容易卡住。
        os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
        from paddleocr import PaddleOCR
        try:
            self.engine = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
            )
            self.kind = "v3"
        except TypeError:
            self.engine = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False, use_gpu=False)
            self.kind = "v2"
        print(f"PaddleOCR 已加载（{self.kind}），内置矫正已关闭，只用来打分。")

    def score(self, image: np.ndarray) -> float | None:
        if self.engine is None:
            return None
        probe, _, _ = resize_long(image, np.full(image.shape[:2], 255, np.uint8), 960)
        if self.kind == "v3":
            result = self.engine.predict(probe)
            return _scores_v3(result)
        result = self.engine.ocr(probe, cls=False)
        return _scores_v2(result)


def _scores_v2(result) -> float | None:
    scores = []
    if not result:
        return None
    for page in result:
        if not page:
            continue
        for line in page:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                rec = line[1]
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    scores.append(float(rec[1]))
    if not scores:
        return None
    return float(np.mean(scores))


def _scores_v3(result) -> float | None:
    scores = []
    pages = result if isinstance(result, (list, tuple)) else [result]
    for page in pages:
        rec = None
        if hasattr(page, "rec_scores"):
            rec = page.rec_scores
        elif isinstance(page, dict):
            rec = page.get("rec_scores")
        if rec is None and hasattr(page, "json"):
            try:
                rec = page.json.get("res", {}).get("rec_scores")
            except Exception:
                rec = None
        if rec is not None:
            scores.extend(float(s) for s in rec)
    if not scores:
        return None
    return float(np.mean(scores))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeoCorrection 文档几何校正")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="含 images 和 mask 的数据目录")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "output", help="结果目录")
    parser.add_argument("--samples", action="store_true", help="只处理 5 张代表图")
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些文件名")
    parser.add_argument("--ocr", action="store_true", help="计算校正前后的 PaddleOCR 平均置信度")
    parser.add_argument("--debug", action="store_true", help="保存文字行叠加图")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = list_pairs(args.data)
    if args.only:
        wanted = set(args.only)
        pairs = [p for p in pairs if p[0].name in wanted]
    elif args.samples:
        wanted = set(SAMPLES)
        pairs = [p for p in pairs if p[0].name in wanted]
    if not pairs:
        raise SystemExit(f"没有找到图像，请检查 {args.data}")

    corrected_dir = args.out / "corrected"
    compare_dir = args.out / "compare"
    debug_dir = args.out / "debug" if args.debug else None
    for folder in (corrected_dir, compare_dir, debug_dir):
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)

    ocr = None
    if args.ocr:
        ocr = OcrScorer()
        ocr.load()

    rows = []
    print(f"共 {len(pairs)} 张")
    for index, (img_path, mask_path) in enumerate(pairs, start=1):
        t0 = time.perf_counter()
        try:
            result = process_pair(img_path, mask_path, debug_dir)
            imwrite_unicode(corrected_dir / f"{img_path.stem}.jpg", result["image"])
            imwrite_unicode(compare_dir / f"{img_path.stem}.jpg", result["compare"])
            ocr_before = ocr_after = None
            if ocr is not None:
                original = imread_unicode(img_path, color=True)
                ocr_before = ocr.score(original)
                ocr_after = ocr.score(result["image"])
            elapsed = time.perf_counter() - t0
            row = {
                "name": img_path.name,
                "pages": result["pages"],
                "method": result["method"],
                "deskew_deg": f"{result['angle']:.1f}",
                "straight_before": fmt(result["straight_before"]),
                "straight_after": fmt(result["straight_after"]),
                "ocr_before": fmt(ocr_before),
                "ocr_after": fmt(ocr_after),
                "seconds": f"{elapsed:.2f}",
                "error": "",
            }
            print(
                f"[{index}/{len(pairs)}] {img_path.name}  {result['method']}  "
                f"pages={result['pages']}  angle={result['angle']:.1f}  "
                f"straight {row['straight_before'] or '-'} -> {row['straight_after'] or '-'}  "
                f"{elapsed:.1f}s"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            row = {
                "name": img_path.name,
                "pages": "",
                "method": "",
                "deskew_deg": "",
                "straight_before": "",
                "straight_after": "",
                "ocr_before": "",
                "ocr_after": "",
                "seconds": f"{elapsed:.2f}",
                "error": str(exc),
            }
            print(f"[{index}/{len(pairs)}] {img_path.name} 失败: {exc}")
            traceback.print_exc()
        rows.append(row)

    csv_path = args.out / "metrics.csv"
    # 只重跑部分图片时，按文件名更新原表，避免把其余 200 张的记录冲掉。
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            old_rows = list(csv.DictReader(f))
        merged = {row["name"]: row for row in old_rows}
        for row in rows:
            merged[row["name"]] = row
        rows = [merged[name] for name in sorted(merged)]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"指标已写入 {csv_path}")
    print(f"校正图在 {corrected_dir}")
    print(f"对比图在 {compare_dir}")


if __name__ == "__main__":
    main()
