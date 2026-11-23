#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""seg.py —— v8 内容驱动分割

核心思路（不重绘任何像素）：
  1) 金色描边本身就是一条闭合轮廓 —— 直接用它 fill_holes 得到整块牌匾的真实外轮廓，
     端头的折角台阶、菱形装饰、双层描边全部天然包含，不再用参数化 hexagon 去猜。
  2) 轮廓外 3px 的颜色用「向内扩散」重建（decontaminate），彻底消除浅色背景晕边。
"""
import numpy as np
from PIL import Image
import pix


def hsv(a):
    a = np.asarray(a).astype(np.float32)
    r, g, b = a[..., 0] / 255.0, a[..., 1] / 255.0, a[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    h = np.zeros_like(mx)
    m = d > 1e-6
    rr = m & (mx == r)
    gg = m & (mx == g) & ~rr
    bb = m & (mx == b) & ~rr & ~gg
    dd = np.where(m, d, 1.0)
    h[rr] = (((g - b) / dd)[rr] % 6.0)
    h[gg] = ((b - r) / dd)[gg] + 2.0
    h[bb] = ((r - g) / dd)[bb] + 4.0
    return h * 60.0, s, mx


def gold(a, s_min=0.24, v_min=0.40):
    h, s, v = hsv(a)
    return (h > 20) & (h < 82) & (s > s_min) & (v > v_min)


def sub(m):
    """取最大连通域"""
    return pix.largest(m.astype(np.uint8), min_area=60).astype(bool)


def plate_from_gold(a, minfrac=0.12):
    """用金色描边闭合轮廓求整块外轮廓"""
    g = sub(gold(a))
    if g.sum() < 30:
        return None, g
    gg = pix.dil(g, 1)
    f = pix.fill_holes(gg)
    p = sub(f)
    if p.sum() < minfrac * a.shape[0] * a.shape[1]:
        return None, g
    return pix.ero(p, 1) | g, g


def plate_from_rules(a, mode, minfrac=0.10):
    """无金框时：按明暗规则取主体"""
    v = np.asarray(a).astype(np.float32).max(2) / 255.0
    m = v < 0.62 if mode == "dark" else v > 0.55
    m = pix.fill_holes(pix.dil(sub(m), 1))
    if m.sum() < minfrac * a.shape[0] * a.shape[1]:
        return None
    return m


def decontam(a, m, band=3):
    """把轮廓边缘 band 像素的颜色用「最近的内部像素」扩散填充，去掉背景晕边"""
    a = np.asarray(a).astype(np.float32).copy()
    inner = pix.ero(m, band)
    if inner.sum() < 20:
        inner = pix.ero(m, 1)
    known = inner.copy()
    cur = a * inner[..., None]
    H, W = m.shape
    for _ in range(band + 2):
        todo = m & ~known
        if not todo.any():
            break
        acc = np.zeros_like(cur)
        cnt = np.zeros((H, W), np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = slice(max(0, dy), H + min(0, dy))
            y2 = slice(max(0, -dy), H + min(0, -dy))
            xx = slice(max(0, dx), W + min(0, dx))
            x2 = slice(max(0, -dx), W + min(0, -dx))
            acc[yy, xx] += cur[y2, x2]
            cnt[yy, xx] += known[y2, x2]
        good = todo & (cnt > 0)
        cur[good] = acc[good] / cnt[good][..., None]
        known |= good
    return np.clip(cur, 0, 255)


def alpha_of(m, soft=1):
    a = m.astype(np.float32)
    if soft:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8), "L").filter(
            __import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(soft)), np.float32) / 255.0
    return a


def extract(crop, mode="dark", ring=0, use_gold=True, autotrim=4, mask=None):
    a = np.asarray(crop.convert("RGB"))
    if mask is not None:
        m = mask
    else:
        v = a.astype(np.float32).max(2) / 255.0
        m = v < 0.62 if mode == "dark" else v > 0.55
        if use_gold:
            m = m | gold(a)
        m = sub(pix.dil(m, 1))
        m = pix.fill_holes(m)
    if ring:
        m = m & ~pix.ero(m, ring)
    col = decontam(a, m, band=3)
    al = alpha_of(m, 1.0)
    rgba = np.dstack([col.astype(np.uint8), (al * 255).astype(np.uint8)])
    im = Image.fromarray(rgba, "RGBA")
    return pix.trim(im, autotrim)


# ------------------------------------------------------------ 擦字 v2
def strip2(im, rects, thr=14, pad=2, allow=None, gold=False, seed=0, nois=1.1):
    """行中值判笔画 + 邻域扩散填充（保留底色渐变，不再刷成一片死色）"""
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    h, w = a.shape[:2]
    zone = np.zeros((h, w), bool)
    for (x0, y0, x1, y1) in rects:
        zone[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    if allow is not None:
        zone &= np.asarray(allow, bool)
    if not zone.any():
        return im

    base = a.copy()
    for y in range(h):
        cs = np.where(zone[y])[0]
        if len(cs):
            base[y, cs] = np.median(a[y, cs], axis=0)
    m = (np.abs(a - base).max(2) > thr) & zone
    if gold:
        _, s, v = hsv(a)
        m &= ~((s > 0.30) & (v > 0.50))
    m = pix.ero(pix.dil(m, pad), 1) & zone
    if not m.any():
        return im

    # 1) 未知像素先做「已知邻域加权扩散」填满
    known = ~m
    cur = a * known[..., None]
    for _ in range(120):
        if known.all():
            break
        acc = np.zeros_like(cur)
        cnt = np.zeros((h, w), np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = slice(max(0, dy), h + min(0, dy)); y2 = slice(max(0, -dy), h + min(0, -dy))
            xx = slice(max(0, dx), w + min(0, dx)); x2 = slice(max(0, -dx), w + min(0, -dx))
            acc[yy, xx] += cur[y2, x2]; cnt[yy, xx] += known[y2, x2]
        good = (~known) & (cnt > 0)
        if not good.any():
            break
        cur[good] = acc[good] / cnt[good][..., None]
        known |= good

    # 2) 调和松弛，让填充与周围渐变连上
    cur[~m] = a[~m]
    for _ in range(60):
        nb = np.zeros_like(cur)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = slice(max(0, dy), h + min(0, dy)); y2 = slice(max(0, -dy), h + min(0, -dy))
            xx = slice(max(0, dx), w + min(0, dx)); x2 = slice(max(0, -dx), w + min(0, -dx))
            nb[yy, xx] += cur[y2, x2]
        cur[m] = nb[m] / 4.0
        cur[~m] = a[~m]

    rng = np.random.default_rng(seed)
    cur[m] += rng.normal(0, nois, (int(m.sum()), 3))
    soft = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.8)), np.float32)[..., None] / 255.0
    out = a * (1 - soft) + np.clip(cur, 0, 255) * soft
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


from PIL import ImageFilter  # noqa: E402


def size_ok(m, margin=2):
    return not (m[:margin].any() or m[-margin:].any() or m[:, :margin].any() or m[:, -margin:].any())


def auto_extract(src, box, texts=(), mode="dark", ring=0, goldprot=False, pad=26,
                 max_try=4, use_gold=True, vt=None, circ=False):
    """自动外扩裁切窗，直到主体不贴边（彻底解决「缺角」）"""
    W, H = src.size
    x0, y0, x1, y1 = box
    for i in range(max_try):
        b = (max(0, x0 - pad * i), max(0, y0 - pad * i), min(W, x1 + pad * i), min(H, y1 + pad * i))
        crop = src.crop(b)
        a = np.asarray(crop)
        v = a.astype(np.float32).max(2) / 255.0
        mm = v < (vt if vt else 0.62) if mode == "dark" else v > (vt if vt else 0.55)
        if use_gold:
            mm = mm | gold(a)
        mm = pix.fill_holes(sub(pix.dil(mm, 1)))
        if circ:
            hh, ww = mm.shape
            yy, xx = np.mgrid[0:hh, 0:ww]
            cy, cx = (hh - 1) / 2.0, (ww - 1) / 2.0
            r = min(hh, ww) / 2.0 - 0.5
            mm = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        if ring or circ or not size_ok(mm, 2) or i == max_try - 1:
            break
    # 擦字
    if texts:
        rel = [(t[0] - b[0], t[1] - b[1], t[2] - b[0], t[3] - b[1]) for t in texts]
        allow = pix.ero(mm, 6) if not ring else None
        crop = strip2(crop, rel, thr=14, pad=2, allow=allow, gold=goldprot)
    # alpha
    if ring:
        mm = mm & ~pix.ero(mm, ring)
    col = decontam(np.asarray(crop), mm, band=3)
    al = alpha_of(mm, 1.0)
    rgba = np.dstack([col.astype(np.uint8), (al * 255).astype(np.uint8)])
    return pix.trim(Image.fromarray(rgba, "RGBA"), 4)


def bg_flood(a, tol=10, iters=400):
    """从裁切窗四边漫水，按「相邻像素色差」判定连通，得到背景区（可跨越渐变）"""
    a = np.asarray(a).astype(np.float32)
    h, w = a.shape[:2]
    bg = np.zeros((h, w), bool)
    bg[0, :] = bg[-1, :] = bg[:, 0] = bg[:, -1] = True
    for _ in range(iters):
        prev = bg.copy()
        cand = pix.dil(bg, 1) & ~bg
        if not cand.any():
            break
        ok = np.zeros((h, w), bool)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(np.roll(a, dy, 0), dx, 1)
            nb = np.roll(np.roll(prev, dy, 0), dx, 1)
            d = np.abs(a - sh).max(2)
            ok |= (d < tol) & cand & nb
        bg |= ok
        if np.array_equal(bg, prev):
            break
    return bg
