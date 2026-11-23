#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_plates.py —— 重修两条「墨笔名条」

原图里名条一端压在立绘/头像下面，抠出来一定缺一个笔锋；
旧做法在最终 RGBA 上镜像覆盖，会把透明留白也镜像过去，产生硬边脏块。

这里：
  1. 用「墨青（h 150~200, s>0.18, v<0.58）∪ 金线」分离名条；
  2. 被压住的一侧直接裁掉，只保留干净的一侧；
  3. 擦字：行中值判笔画 + TELEA inpaint；
  4. 补笔锋：把干净那端的笔锋水平翻转接上去（带 alpha）。
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pix
import seg

ROOT = os.path.join(HERE, "..", "src")
OUT = os.path.join(HERE, "..", "out", "v9")
POS = {}


def plate_mask(a):
    h, s, v = seg.hsv(a)
    plate = (v < 0.60) & (s > 0.16) & (h > 145) & (h < 205)
    m = plate.astype(np.uint8)
    # 纯白高光（头像反光）不算底板
    halo = (v > 0.80) & (s < 0.11)
    m[halo] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lab == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8)
    # 只补「小洞」（字腔、纹理孔），大块空洞保留透明
    inv = (m == 0).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n2):
        x, y, ww, hh, area = st2[i]
        if area <= 520 and x > 0 and y > 0 and x + ww < m.shape[1] and y + hh < m.shape[0]:
            m[lab2 == i] = 1
    return m.astype(bool)


def strip_text(a, rect, fg, thr=13):
    h, w = a.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in rect]
    zone = np.zeros((h, w), bool)
    zone[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    af = a.astype(np.float32)
    base = af.copy()
    for y in range(h):
        cs = np.where(zone[y])[0]
        if len(cs):
            base[y, cs] = np.median(af[y, cs], axis=0)
    m = (np.abs(af - base).max(2) > thr) & zone & pix.ero(fg, 4)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mm = cv2.dilate(m.astype(np.uint8), k)
    if not mm.any():
        return a
    out = cv2.inpaint(a, mm, 4, cv2.INPAINT_TELEA)
    return np.asarray(cv2.bilateralFilter(out, 5, 16, 5))


def make(name, k, box, text, keep, tip, tfrom):
    src = Image.open(os.path.join(ROOT, {"A": "A_battle.jpg", "C": "C_city.jpg"}[k])).convert("RGB")
    crop = src.crop(box)
    a = np.asarray(crop).copy()
    m = plate_mask(a)
    lo, hi = keep[0] - box[0], keep[1] - box[0]
    keepm = np.zeros_like(m)
    keepm[:, max(0, lo):max(0, hi)] = True
    m &= keepm

    if text:
        a = strip_text(a, (text[0] - box[0], text[1] - box[1],
                           text[2] - box[0], text[3] - box[1]), m)

    m = pix.dil(m, 2)  # 向外补 2px，把原图的软边找回来（否则比原图窄一圈）
    al = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.7)
    al = np.clip((al - 0.34) / 0.30, 0, 1)
    rgba = np.dstack([a, (al * 255).astype(np.uint8)])

    ys, xs = np.where(al > 0.3)
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 补端头：把干净端的笔锋「叠」上去（不延长宽度，避免接缝硬线）
    if tip > 4:
        hh, ww = rgba.shape[:2]
        nn = min(tip, ww // 3)
        if tfrom == "left":
            patch, x0 = rgba[:, :nn][:, ::-1], ww - nn
        else:
            patch, x0 = rgba[:, ww - nn:][:, ::-1], 0
        basep = rgba[:, x0:x0 + nn].astype(np.float32)
        top = patch.astype(np.float32)
        aa = top[..., 3:4] / 255.0
        rgba[:, x0:x0 + nn, 0:3] = np.clip(top[..., :3] * aa + basep[..., :3] * (1 - aa), 0, 255).astype(np.uint8)
        rgba[:, x0:x0 + nn, 3:4] = np.clip(top[..., 3:4] + basep[..., 3:4] * (1 - aa), 0, 255).astype(np.uint8)

    im = pix.trim(Image.fromarray(rgba, "RGBA"), 2)
    im.save(os.path.join(OUT, name + ".png"))
    # 记录落点（左上角在原截图中的坐标）
    a2 = np.asarray(im)[..., 3]
    ys2, xs2 = np.where(a2 > 6)
    if len(xs2):
        POS[name] = [int(box[0] + xs.min() + xs2.min()), int(box[1] + ys.min() + ys2.min())]
    print("  %-18s %dx%d" % (name, *im.size))


JOBS_PC = [
    ("plate_char", "C", (138, 10, 360, 92), (176, 18, 350, 80), (184, 358), 34, "right"),
]

JOBS = [
    ("plate_name_enemy", "A", (490, 168, 782, 238), (532, 178, 744, 234),
     (496, 742), 46, "left"),
    ("plate_name_self", "A", (150, 170, 412, 238), (196, 180, 404, 234),
     (192, 410), 6, "right"),
]

if __name__ == "__main__":
    only = sys.argv[1:] or None
    SRC2 = {"A": "A_battle.jpg", "C": "C_city.jpg"}
    for j in (JOBS_PC if (only and "plate_char" in only) else []) + JOBS:
        if only and j[0] not in only:
            continue
        make(*j)
    import json
    json.dump(POS, open(os.path.join(HERE, "..", "out", "plates_pos.json"), "w"), indent=1)
    print(POS)
