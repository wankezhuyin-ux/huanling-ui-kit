#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""final9.py —— v9 定稿生成器

四条铁律：
  1. 像素只搬运，不重绘；
  2. 形状 = 明暗/金色规则 ∪ 之后剔除「与画框四边连通的背景」（cv2 floodFill 邻域自适应），
     因此端头折角、双层描边、菱形装饰全在原位，不需要猜几何；
  3. 文字用 cv2.inpaint(TELEA) 修复，不再刷死底色；
  4. 轮廓外 3px 颜色向内扩散（去背景晕边），alpha 1px 羽化。
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pix  # noqa: E402
import seg  # noqa: E402

ROOT = os.path.join(HERE, "..", "src")
OUT = os.path.join(HERE, "..", "out", "v9")
os.makedirs(OUT, exist_ok=True)
SRC = {"A": "A_battle.jpg", "B": "B_pet.jpg", "C": "C_city.jpg"}
BAND = 0
EDGE_ERO = 0
_C = {}
POS = {}


def img(k):
    if k not in _C:
        _C[k] = Image.open(os.path.join(ROOT, SRC[k])).convert("RGB")
    return _C[k]


# ---------------------------------------------------------------- masks
def bgmask(a, d=10):
    im = np.ascontiguousarray(a.copy())
    h, w = im.shape[:2]
    fl = np.zeros((h + 2, w + 2), np.uint8)
    seeds = []
    for x in range(0, w, 6):
        seeds += [(x, 0), (x, h - 1)]
    for y in range(0, h, 6):
        seeds += [(0, y), (w - 1, y)]
    for s in seeds:
        if fl[s[1] + 1, s[0] + 1]:
            continue
        cv2.floodFill(im, fl, s, (1, 1, 1), (d,) * 3, (d,) * 3, 4)
    return (im[..., 0] == 1) & (im[..., 1] == 1) & (im[..., 2] == 1)


def pick(m):
    """闭运算 → 取最大连通域 → 填满内部空洞（保证描边内侧是实心的）"""
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    out = (lab == i).astype(np.uint8)
    if st[i, cv2.CC_STAT_AREA] < 80:
        return m.astype(bool)
    ff = out.copy()
    mm = np.zeros((out.shape[0] + 2, out.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mm, (0, 0), 1)
    out[(ff == 0) & (out == 0)] = 1
    return out.astype(bool)


def canny_plate(a, lo=30, hi=90, cl=5, pad=2):
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g, lo, hi)
    e[:pad] = 0; e[-pad:] = 0; e[:, :pad] = 0; e[:, -pad:] = 0
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cl, cl)))
    return pick(e > 0)


def plate_mask(a, mode, vt=None, extra=None, cut_bg=True, d=10):
    v = a.astype(np.float32).max(2) / 255.0
    if mode == "dark":
        m = v < (vt or 0.62)
    elif mode == "bright":
        m = v > (vt or 0.55)
    else:
        m = np.ones_like(v, bool)
    m = m | seg.gold(a)
    if extra is not None:
        m = m | extra(a)
    if cut_bg:
        m = m & ~bgmask(a, d)
    return pick(m)


def green(a):
    h, s, v = seg.hsv(a)
    return (h > 95) & (h < 180) & (s > 0.20) & (v > 0.40)


# ---------------------------------------------------------------- text
def strip(a, rects, fg, gold=False, thr=14, pad=2, pol=None):
    h, w = a.shape[:2]
    zone = np.zeros((h, w), bool)
    for (x0, y0, x1, y1) in rects:
        zone[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    if not zone.any():
        return a
    af = a.astype(np.float32)
    base = af.copy()
    for y in range(h):
        cs = np.where(zone[y])[0]
        if len(cs):
            base[y, cs] = np.median(af[y, cs], axis=0)
    if pol == "dark":
        dev = (base - af).max(2)
    elif pol == "light":
        dev = (af - base).max(2)
    else:
        dev = np.abs(af - base).max(2)
    m = (dev > thr) & zone
    if gold:
        _, s, v = seg.hsv(a)
        m &= ~((s > 0.30) & (v > 0.50))
    m &= pix.ero(fg, 4)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1,) * 2)
    mm = cv2.dilate(m.astype(np.uint8), k)
    if not mm.any():
        return a
    out = cv2.inpaint(a, mm, 4, cv2.INPAINT_TELEA)
    return np.asarray(cv2.bilateralFilter(out, 5, 18, 6))


# ---------------------------------------------------------------- alpha
def decontam(a, m, band=3):
    return seg.decontam(a, m, band=band)


def build(name, k, box, texts=(), mode="dark", ring=0, gold=False, circ=False,
          mirror=None, pad=18, noexp=False, vt=None, shape=None, only=None, nobg=False,
          rects=None, xgreen=False, pol=None):
    src = img(k)
    W, H = src.size
    x0, y0, x1, y1 = box
    p = 0 if noexp else pad
    b = (max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p))
    crop = src.crop(b)
    a = np.asarray(crop.convert("RGB"))
    ch, cw = a.shape[:2]

    # 限定在实测框内，防止把旁边立绘/背景一起卷进来
    region = np.zeros((ch, cw), bool)
    rx0, ry0 = max(0, x0 - b[0] - 2), max(0, y0 - b[1] - 2)
    rx1, ry1 = min(cw, x1 - b[0] + 2), min(ch, y1 - b[1] + 2)
    region[ry0:ry1, rx0:rx1] = True

    if circ:
        yy, xx = np.mgrid[0:ch, 0:cw]
        cyc, cxc = (ry0 + ry1) / 2.0, (rx0 + rx1) / 2.0
        r = min(ry1 - ry0, rx1 - rx0) / 2.0 - 0.5
        m = ((yy - cyc) ** 2 + (xx - cxc) ** 2) <= r * r
    elif shape == "rect":
        m = region.copy()
    elif mode == "bg":
        m = pick(~bgmask(a) & region)
    else:
        m = plate_mask(a, mode, vt=vt, extra=green if only == "green" else None,
                       cut_bg=not nobg) & region
        if name.startswith("bar"):
            m = pick(~bgmask(a) & region)

    if xgreen:
        hh, ss, vv = seg.hsv(a)
        m = m & ~((hh > 58) & (hh < 118) & (ss > 0.26))
    if rects:
        rr = np.zeros((ch, cw), bool)
        for (ax, ay, bx, by) in rects:
            rr[max(0, ay - b[1]):max(0, by - b[1]), max(0, ax - b[0]):max(0, bx - b[0])] = True
        m = pick(m & rr)
    if m.sum() < 0.03 * region.sum():
        v = a.astype(np.float32).max(2) / 255.0
        m = pick(((v < (vt or 0.62)) if mode == "dark" else (v > (vt or 0.55))) | seg.gold(a))
    if texts:
        rel = [(t[0] - b[0], t[1] - b[1], t[2] - b[0], t[3] - b[1]) for t in texts]
        a = strip(a, rel, m, gold=gold, pol=pol)

    if ring:
        m = m & ~pix.ero(m, ring)
    if EDGE_ERO:
        m = pix.ero(m, EDGE_ERO)
    col = a.astype(np.float32) if BAND <= 0 else decontam(a, m, band=BAND)
    al = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)
    al = np.clip((al - 0.36) / 0.28, 0, 1)  # 收紧羽化，保住细尖角
    rgba = np.dstack([col.astype(np.uint8), (np.clip(al, 0, 1) * 255).astype(np.uint8)])
    al8 = rgba[..., 3]
    ys, xs = np.where(al8 > 4)
    if len(xs):
        POS[name] = [int(b[0] + xs.min()), int(b[1] + ys.min()), k]
    im = pix.trim(Image.fromarray(rgba, "RGBA"), 4)
    if ring:
        im = edge_clean(im, 2)
        im = corner_clean(im, 44)
        im = round_corners(im, 9)
    if mirror:
        if isinstance(mirror, (tuple, list)):
            im = mirror_cap(im, mirror[0], mirror[1])
        else:
            im = mirror_cap(im, mirror)
    im.save(os.path.join(OUT, name + ".png"))
    print("  %-18s %dx%d" % (name, *im.size))
    return im


def corner_clean(im, n=40):
    """面板四角：按「纸色」判前景，把圆角外的场景像素抹透明（纸边的金线也保留）"""
    a = np.asarray(im).astype(np.float32)
    h, w = a.shape[:2]
    rgb = a[..., :3]
    v = rgb.max(2) / 255.0
    s = (rgb.max(2) - rgb.min(2)) / np.maximum(rgb.max(2), 1)
    paper = ((v > 0.60) & (s < 0.40)) | ((v > 0.80) & (s < 0.55))
    m = cv2.GaussianBlur(paper.astype(np.float32), (0, 0), 1.0)
    m = np.clip((m - 0.25) / 0.5, 0, 1)
    for ys, xs in ((slice(0, n), slice(0, n)), (slice(0, n), slice(w - n, w)),
                   (slice(h - n, h), slice(0, n)), (slice(h - n, h), slice(w - n, w))):
        a[ys, xs, 3] = a[ys, xs, 3] * m[ys, xs]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")


def round_corners(im, r=16):
    """把面板四角的圆角外侧抹成透明（原图圆角外是场景，会露出青绿色块）"""
    a = np.asarray(im).copy()
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    keep = np.ones((h, w), bool)
    for cy, cx in ((r, r), (r, w - 1 - r), (h - 1 - r, r), (h - 1 - r, w - 1 - r)):
        dx = np.clip(np.abs(xx - cx) - 0, 0, None)
        dy = np.clip(np.abs(yy - cy) - 0, 0, None)
        # 只在角区域判定
    # 逐角处理，避免全局圆形误伤
    def corner(cy, cx, sy, sx):
        ys = slice(0, r + 1) if sy < 0 else slice(h - r - 1, h)
        xs = slice(0, r + 1) if sx < 0 else slice(w - r - 1, w)
        sub = a[ys, xs]
        hh, ww = sub.shape[:2]
        yy2, xx2 = np.mgrid[0:hh, 0:ww]
        cy2 = 0 if sy < 0 else hh - 1
        cx2 = 0 if sx < 0 else ww - 1
        d = np.sqrt((yy2 - cy2) ** 2 + (xx2 - cx2) ** 2)
        alpha = np.clip((r - d) / 1.4 + 0.5, 0, 1)[..., None]
        sub[..., 3:4] = (sub[..., 3:4] * alpha).astype(np.uint8)
        a[ys, xs] = sub
    for sy in (-1, 1):
        for sx in (-1, 1):
            corner(r, r, sy, sx)
    return Image.fromarray(a, "RGBA")


def edge_clean(im, n=2):
    """把最外 n 圈像素替换成「向内 n+1 像素」的颜色，去掉贴合处留的场景描边"""
    a = np.asarray(im).copy()
    h, w = a.shape[:2]
    for i in range(n):
        a[i, :] = a[n + 1, :]
        a[h - 1 - i, :] = a[h - n - 2, :]
        a[:, i] = a[:, n + 1]
        a[:, w - 1 - i] = a[:, w - n - 2]
    return Image.fromarray(a, "RGBA")


def mirror_cap(im, side, frac=0.15):
    a = np.asarray(im).copy()
    h, w = a.shape[:2]
    n = max(6, int(w * frac))
    if side == "right":
        a[:, w - n:] = np.fliplr(a[:, :n])
    else:
        a[:, :n] = np.fliplr(a[:, w - n:])
    return Image.fromarray(a, "RGBA")


JOBS = [
    ("banner_scene", "A", (226, 90, 668, 162), [(300, 94, 606, 154)], dict(gold=True)),
    ("plate_name_self", "A", (152, 172, 410, 236), [(166, 182, 396, 232)], dict(gold=True, noexp=True, nobg=True, mirror=("left", 0.16), xgreen=True, rects=[(174, 22, 342, 72), (180, 66, 300, 102)])),
    ("plate_name_enemy", "A", (494, 172, 782, 236), [(536, 182, 770, 232)], dict(gold=True, noexp=True, mirror="right")),
    ("plate_char", "C", (172, 16, 344, 108), [(172, 16, 352, 72), (174, 66, 320, 108)], dict(gold=True, noexp=True, nobg=True, mirror=("left", 0.16), xgreen=True, rects=[(174, 22, 342, 72), (180, 66, 300, 102)])),
    ("tab_nav_on", "A", (256, 1660, 401, 1719), [(262, 1664, 394, 1714)], dict(gold=True, pol="dark")),
    ("tab_nav_off", "A", (58, 1662, 196, 1716), [(64, 1668, 188, 1710)], dict(gold=True, pol="dark")),
    ("tab_on", "B", (272, 914, 550, 994), [(278, 918, 546, 990)], dict(gold=True)),
    ("plate_pet", "B", (526, 662, 772, 734), [(528, 664, 774, 736)], dict(mode="bright", gold=True, mirror="right")),
    ("plate_skill", "B", (279, 1040, 516, 1108), [(288, 1044, 508, 1104)], dict(mode="bright", gold=True)),
    ("tag_vertical", "C", (21, 196, 112, 406), [(30, 206, 102, 366)], dict(gold=True, noexp=True)),
    ("panel_paper", "A", (8, 1152, 888, 1352), [(24, 1160, 872, 1348)], dict(shape="rect", ring=18, noexp=True)),
    ("panel_main", "B", (8, 897, 845, 1630), [], dict(shape="rect", ring=14, noexp=True)),
    ("panel_infobar", "C", (8, 1318, 820, 1462), [(140, 1324, 476, 1366), (596, 1340, 812, 1410)], dict(noexp=True, shape="rect", gold=True)),
    ("bar_track", "A", (160, 265, 374, 286), [(182, 266, 368, 284)], dict(vt=0.60)),
    ("bar_hp", "A", (160, 265, 374, 286), [], dict(only="green", vt=0.60)),
    ("btn_gold", "A", (62, 1395, 366, 1464), [(164, 1406, 334, 1462)], dict()),
    ("btn_x2", "A", (407, 1396, 565, 1463), [(420, 1400, 558, 1460)], dict(gold=True)),
    ("btn_log", "A", (606, 1394, 839, 1465), [(694, 1400, 808, 1462)], dict(gold=True)),
    ("btn_train", "B", (255, 1502, 592, 1586), [(330, 1506, 510, 1584)], dict(mode="bright", gold=True)),
    ("btn_func", "C", (792, 190, 862, 258), [], dict(circ=True)),
    ("avatar_self", "A", (6, 150, 170, 314), [], dict(circ=True)),
    ("icon_skill", "B", (68, 1026, 268, 1226), [], dict(circ=True, mode="bright")),
]


def run(only=None):
    for name, k, box, tx, kw in JOBS:
        if only and name not in only:
            continue
        build(name, k, box, tx, **kw)


if __name__ == "__main__":
    run(sys.argv[1:] or None)
    import json
    json.dump(POS, open(os.path.join(HERE, "..", "out", "pos.json"), "w"), indent=1)
