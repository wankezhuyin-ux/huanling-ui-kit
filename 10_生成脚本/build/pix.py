#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""像素搬运：保留概念图原始像素，只重建 alpha + 擦文字。

擦字原理（row-median）：
  这类国风 UI 的底色在**同一行内几乎恒定**，只沿纵向渐变。
  所以对每一行取 ROI 内的中值色作为该行底色，偏离超过阈值的像素即笔画（白字 + 深描边），
  用该行底色填平即可。中值天然抗干扰：文字宽度 < 行宽 40%。
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kit import rule_mask  # noqa: E402


# ----------------------------------------------------------------- 形态学
def _pad(m):
    return np.pad(m, 1, mode="constant", constant_values=False)


def dil(m, r=1):
    cur = m
    for _ in range(r):
        p = _pad(cur)
        cur = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | p[1:-1, 1:-1])
    return cur


def ero(m, r=1):
    cur = m
    for _ in range(r):
        p = _pad(cur)
        cur = (p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:] & p[1:-1, 1:-1])
    return cur


def largest(m, min_area=200):
    """最大连通域（4 邻域 BFS，numpy 加速近似：反复标记）"""
    h, w = m.shape
    lab = np.zeros((h, w), np.int32)
    cid = 0
    best, best_area = 0, 0
    ys, xs = np.where(m)
    if len(ys) == 0:
        return m
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0] or not m[y0, x0]:
            continue
        cid += 1
        stack = [(y0, x0)]
        lab[y0, x0] = cid
        n = 0
        while stack:
            cy, cx = stack.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cid
                    stack.append((ny, nx))
        if n > best_area:
            best_area, best = n, cid
    if best_area < min_area:
        return np.zeros_like(m)
    return lab == best


# ----------------------------------------------------------------- 擦字
def strip_text(im, rects, thr=16, pad=2, noise=1.0, seed=0, allow=None, ywin=9, gold=False):
    """用「行中值 + 纵向平滑」的底色填平 ROI 内的文字。

    allow: 允许擦除的区域（一般是 alpha 轮廓内缩 npx），避免把牌子的描边当笔画擦掉。
    ywin:  行底色沿纵向的中值窗口，消除横向条纹。
    """
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    h, w = a.shape[:2]
    zone = np.zeros((h, w), bool)
    for (x0, y0, x1, y1) in rects:
        zone[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    if allow is not None:
        zone &= np.asarray(allow, bool)
    if not zone.any():
        return im, np.zeros((h, w), bool)

    # 1) 每行底色（zone 内中值）
    base = a.copy()
    for y in range(h):
        cols = np.where(zone[y])[0]
        if len(cols) == 0:
            continue
        base[y, cols] = np.median(a[y, cols], axis=0)

    # 2) 偏离底色的像素 = 笔画
    dev = np.abs(a - base).max(2)
    m = (dev > thr) & zone
    if gold:
        # 保护高饱和高亮度的金色装饰（菱形 / 卷草 / 云头），文字是白或深棕
        from kit import hsv3 as _hsv
        hh, ss, vv, _ = _hsv(im)
        keep = (ss > 0.33) & (vv > 0.55) & (vv < 0.97)
        m &= ~keep
    m = dil(m, pad)
    # 闭运算补断笔
    m = ero(dil(m, 2), 2)
    m &= zone

    # 3) 用「未被判为笔画」的像素重算每行底色，再沿纵向中值平滑
    rowcol = np.zeros((h, 3), np.float32)
    valid = np.zeros(h, bool)
    for y in range(h):
        cols = np.where(zone[y] & ~m[y])[0]
        if len(cols) < 3:
            continue
        rowcol[y] = np.median(a[y, cols], axis=0)
        valid[y] = True
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return im, m
    for y in range(h):
        if not valid[y]:
            j = idx[np.argmin(np.abs(idx - y))]
            rowcol[y] = rowcol[j]
    sm = rowcol.copy()
    r = ywin // 2
    for y in range(h):
        seg = rowcol[max(0, y - r):min(h, y + r + 1)]
        sm[y] = np.median(seg, axis=0)

    fill = a.copy()
    for y in range(h):
        fill[y, zone[y]] = sm[y]
    rng = np.random.default_rng(seed)
    fill = fill + rng.normal(0, noise, fill.shape)

    soft = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.9)), np.float32) / 255.0
    out = a * (1 - soft[..., None]) + fill * soft[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"), m


# ----------------------------------------------------------------- alpha
def fill_holes(m):
    """从边界 flood fill 背景，其余即内部孔洞 → 全部填实"""
    h, w = m.shape
    free = ~m
    seen = np.zeros((h, w), bool)
    stack = []
    for x in range(w):
        for y in (0, h - 1):
            if free[y, x] and not seen[y, x]:
                seen[y, x] = True
                stack.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if free[y, x] and not seen[y, x]:
                seen[y, x] = True
                stack.append((y, x))
    while stack:
        cy, cx = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return m | (free & ~seen)


def mask_by_rules(im, box, rules, grow=2, shrink=1, open_r=2, min_area=400):
    crop = im.crop(box)
    m = rule_mask(crop, rules)
    if m.sum() == 0:
        return None
    m = dil(m, 1)
    m = largest(m, min_area)
    m = fill_holes(m)
    m = dil(ero(m, open_r), open_r)
    m = largest(m, min_area // 2)
    m = fill_holes(m)
    m = dil(m, grow)
    m = ero(m, shrink)
    return m


def alpha_from_mask(m, grow=2, blur=2.2, lo=0.36, hi=0.72):
    mm = dil(m, grow)
    a = np.asarray(Image.fromarray((mm * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(blur)), np.float32) / 255.0
    a = (a - lo) / max(hi - lo, 1e-6)
    return np.clip(a, 0, 1)


def compose(im, box, alpha_fn, text_rects=(), thr=26, protect=None, alpha_grow=2, alpha_blur=2.2):
    """裁剪 box → 擦字（相对坐标）→ 套 alpha → 裁边"""
    crop = im.crop(box).convert("RGB")
    x0, y0 = box[0], box[1]
    if text_rects:
        rel = [(rx0 - x0, ry0 - y0, rx1 - x0, ry1 - y0) for (rx0, ry0, rx1, ry1) in text_rects]
        crop, _ = strip_text(crop, rel, thr=thr, protect=protect)
    W, H = crop.size
    m = alpha_fn(crop, W, H)
    if m is None:
        return None
    a = alpha_from_mask(m, alpha_grow, alpha_blur)
    if protect is not None:
        a = a * np.asarray(protect, np.float32)
    arr = np.asarray(crop).copy()
    arr = np.dstack([arr, (a * 255).astype(np.uint8)])
    out = Image.fromarray(arr, "RGBA")
    return trim(out, 3)


def trim(im, thr=3):
    a = np.asarray(im)[..., 3]
    ys, xs = np.where(a > thr)
    if len(xs) == 0:
        return im
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


# ----------------------------------------------------------------- 形状生成器
def shape_rrect(rx=0.26, dy_top=0.0, dy_bot=0.0):
    def f(crop, W, H):
        im = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(im)
        r = min(W, H) * rx
        d.rounded_rectangle([0, H * dy_top, W - 1, H - 1 - H * dy_bot], radius=r, fill=255)
        return np.asarray(im) > 127
    return f


def shape_full():
    def f(crop, W, H):
        im = Image.new("L", (W, H), 0)
        ImageDraw.Draw(im).rectangle([0, 0, W - 1, H - 1], fill=255)
        return np.asarray(im) > 127
    return f


def shape_para(k=0.55, arrow=0.86):
    """左斜切 + 右箭头"""
    def f(crop, W, H):
        im = Image.new("L", (W, H), 0)
        kk = H * k
        pts = [(kk, 0), (W - H * (1 - arrow) - 1, 0), (W - 1, H * 0.5),
               (W - H * (1 - arrow) - 1, H - 1), (0, H - 1)]
        ImageDraw.Draw(im).polygon(pts, fill=255)
        return np.asarray(im) > 127
    return f


def shape_rules(rules):
    def f(crop, W, H):
        m = rule_mask(crop, rules)
        if m.sum() == 0:
            return None
        m = dil(m, 1)
        m = largest(m, 400)
        m = dil(ero(m, 2), 2)
        return largest(m, 200)
    return f
