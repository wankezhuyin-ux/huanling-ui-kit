#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare9 —— 生成「原图 / 抠件」逐项对照，并做一次贴回原位的还原验证"""
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final9 as F

DST = os.path.abspath(os.path.join(HERE, "..", "..", "幻灵归虚UI复刻套件_v9"))
OUT = os.path.join(HERE, "..", "out", "v9")
POS = json.load(open(os.path.join(HERE, "..", "out", "pos.json")))
FN = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 16)


def sheet():
    names = [j[0] for j in F.JOBS]
    rows = []
    for n in names:
        p = os.path.join(OUT, n + ".png")
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGBA")
        x, y = POS.get(n, (0, 0))[:2]
        rows.append((n, im, x, y, POS[n][2]))
    W = 1180
    y = 14
    blocks = []
    for n, im, x, y0, pg in rows:
        sc = 1 if im.width > 520 else 2
        big = im.resize((im.width * sc, im.height * sc), Image.LANCZOS)
        h = max(big.height, 26)
        blocks.append((n, im, big, h))
    H = sum(b[3] + 40 for b in blocks) + 20
    canvas = Image.new("RGB", (W, H), (28, 28, 30))
    d = ImageDraw.Draw(canvas)
    yy = 10
    for n, im, big, h in blocks:
        d.text((14, yy), n, font=FN, fill=(255, 232, 170))
        yy += 22
        bg = Image.new("RGBA", big.size, (44, 46, 50, 255))
        bg.alpha_composite(big)
        canvas.paste(bg.convert("RGB"), (14, yy))
        # 原图对应区域
        src = F.img(POS[n][2])
        cx0, cy0 = POS[n][0], POS[n][1]
        crop = src.crop((cx0, cy0, cx0 + im.width, cy0 + im.height)).resize(big.size, Image.LANCZOS)
        canvas.paste(crop, (14 + big.width + 24, yy))
        d.text((14 + big.width + 24, yy + big.height + 2), "原图（含文字）", font=FN, fill=(150, 150, 160))
        yy += h + 40
    canvas.save(os.path.join(DST, "对比_原图vs抠件.jpg"), quality=92)
    print("sheet", canvas.size)


def pasteback():
    """把每件贴回原图原位，看是否与原图重合"""
    for k in ("A", "B", "C"):
        base = F.img(k).copy()
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        used = 0
        for n, (x, y, pg) in POS.items():
            if pg != k:
                continue
            p = os.path.join(OUT, n + ".png")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGBA")
            if im.width > 900:
                continue
            layer.alpha_composite(im, (x, y))
            used += 1
        out = base.convert("RGBA")
        out.alpha_composite(layer)
        out.convert("RGB").save(os.path.join(DST, "还原验证_%s.jpg" % k), quality=90)
        print(k, used)


if __name__ == "__main__":
    sheet()
    pasteback()
