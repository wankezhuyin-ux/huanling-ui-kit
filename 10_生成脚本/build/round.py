#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""round.py —— 把「方形底」的圆形件修成真正的圆形 alpha

07_写实件 里的导航图标/头像/技能圆是从截图里按方块裁的，
四角还带着导航条底色，进引擎会出现一圈方形底。
这里按圆盘半径重做 alpha（保留原像素，只改透明度），并留 1px 抗锯齿。
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.abspath(os.path.join(HERE, "..", "..", "07_写实件"))
FILES = {
    "icon_nav_洞府.png": 0.44,
    "icon_nav_历练.png": 0.44,
    "icon_nav_灵兽.png": 0.44,
    "icon_nav_行囊.png": 0.44,
    "avatar_char.png": 0.455,
    "avatar_enemy.png": 0.455,
    "icon_skill.png": 0.46,
}


def circle_alpha(w, h, f):
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = min(w, h) * f
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    a = np.clip((r - d) / 1.2 + 0.5, 0, 1)  # 1px 抗锯齿
    return a.astype(np.float32)


def main():
    for name, f in FILES.items():
        p = os.path.join(DST, name)
        im = Image.open(p).convert("RGBA")
        w, h = im.size
        a = np.asarray(im).astype(np.float32)
        old = a[..., 3] / 255.0
        new = np.minimum(old, circle_alpha(w, h, f))
        a[..., 3] = new * 255
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA").save(p)
        print("  %-22s %dx%d r=%.0f" % (name, w, h, min(w, h) * f))


if __name__ == "__main__":
    main()
