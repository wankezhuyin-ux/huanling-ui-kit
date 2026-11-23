#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_paper.py —— 单独重出 paper_fill.png

面板类只出「边框环」，内部要用一块可平铺的纸面垫底。
这里从概念图 A 的战斗日志面板内部取一块干净纸面（避开文字），
按行取中值去掉残迹，加一点点噪点避免平铺出现可视接缝。

    python3 make_paper.py [--dst ../../02_面板]
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "src"))
OUT = os.path.abspath(os.path.join(HERE, "..", "out", "v9"))

# A_battle.jpg 里日志面板内部的一块空白纸面
INNER_BOX = (700, 1200, 860, 1300)


def make_paper(src, box=INNER_BOX, seed=3):
    a = np.asarray(Image.open(src).convert("RGB").crop(box)).astype(np.float32)
    h = a.shape[0]
    for y in range(h):
        a[y] = np.median(a[y], axis=0)
    rng = np.random.default_rng(seed)
    a = a + rng.normal(0, 1.8, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "A_battle.jpg"))
    ap.add_argument("--dst", default=None)
    args = ap.parse_args()
    im = make_paper(args.src)
    dst = args.dst or OUT
    os.makedirs(dst, exist_ok=True)
    p = os.path.join(dst, "paper_fill.png")
    im.save(p)
    print("  paper_fill        %dx%d -> %s" % (im.size[0], im.size[1], p))
