#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sync9 —— 同步 v9 定稿到交付目录，重算切边 / 索引"""
import os
import json
import shutil
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "out", "v9")
DST = os.path.abspath(os.path.join(HERE, "..", ".."))
OLD = DST   # 交接包里不再有旧目录，回退到同一处

# 文件, 子目录, 中文名, 左,右,上,下, 用途
SPEC = [
    ("banner_scene.png", "01_底板", "场景名横幅", 112, 112, 0, 0, "战斗页顶部"),
    ("plate_name_self.png", "01_底板", "我方名条", 26, 26, 0, 0, "战斗页左上（左端被立绘压住，已镜像补全）"),
    ("plate_name_enemy.png", "01_底板", "敌方名条", 26, 26, 0, 0, "战斗页右上"),
    ("plate_char.png", "01_底板", "双层角色牌", 0, 0, 0, 0, "主城左上（上层名条+下层境界）"),
    ("tab_nav_on.png", "01_底板", "导航选中页签", 26, 26, 0, 0, "底部导航当前项"),
    ("tab_nav_off.png", "01_底板", "导航未选页签", 24, 24, 0, 0, "底部导航其它项"),
    ("tab_on.png", "01_底板", "页内选中页签", 30, 30, 0, 0, "灵兽/行囊内部分页"),
    ("plate_pet.png", "01_底板", "灵兽名牌", 26, 26, 0, 0, "灵兽页（右端原图被面板压住，已镜像补全）"),
    ("plate_skill.png", "01_底板", "技能名牌", 26, 26, 0, 0, "灵兽页"),
    ("tag_vertical.png", "01_底板", "竖挂牌", 0, 0, 0, 0, "主城左侧「洞府」"),
    ("panel_paper.png", "02_面板", "纸面大面板(边框环)", 34, 34, 34, 34, "战斗日志 / 通用弹窗"),
    ("panel_main.png", "02_面板", "主面板(边框环)", 48, 48, 48, 48, "灵兽页主面板"),
    ("panel_infobar.png", "02_面板", "底部信息条", 0, 0, 0, 0, "主城底部（含山水底纹）"),
    ("paper_fill.png", "02_面板", "纸面填充块", 0, 0, 0, 0, "与边框环搭配平铺"),
    ("bar_hp.png", "03_进度条", "血条填充", 18, 24, 0, 0, "战斗页"),
    ("bar_track.png", "03_进度条", "血条底槽", 18, 24, 0, 0, "战斗页"),
    ("btn_gold.png", "04_按钮", "金色主按钮", 118, 40, 0, 0, "自动战斗 / 培养"),
    ("btn_x2.png", "04_按钮", "次按钮小", 20, 20, 0, 0, "×2"),
    ("btn_log.png", "04_按钮", "次按钮长", 92, 26, 0, 0, "战报"),
    ("btn_train.png", "04_按钮", "培养按钮", 92, 92, 0, 0, "灵兽页"),
    ("btn_func.png", "04_按钮", "功能圆钮", 22, 22, 22, 22, "邮件 / 活动"),
    ("avatar_self.png", "05_圆形", "头像盘", 0, 0, 0, 0, "战斗页左下"),
    ("icon_skill.png", "05_圆形", "技能圆", 0, 0, 0, 0, "灵兽页"),
]


def main():
    rows = []
    for fn, sub, cn, l, r, t, b, use in SPEC:
        s = os.path.join(SRC, fn)
        if not os.path.exists(s):
            print("!! 缺", fn)
            continue
        d = os.path.join(DST, sub)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(s, os.path.join(d, fn))
        w, h = Image.open(s).size
        l, r = min(l, w // 3), min(r, w // 3)
        t, b = min(t, h // 3), min(b, h // 3)
        rows.append(dict(file=fn, dir=sub, name=cn, w=w, h=h, l=l, r=r, t=t, b=b, use=use))

    # 写实件沿用（立绘 / 图标 / 宠物原画，未做任何改动）
    os.makedirs(os.path.join(DST, "07_写实件"), exist_ok=True)
    src7 = os.path.join(OLD, "07_写实件")
    if os.path.isdir(src7):
        for f in os.listdir(src7):
            if f.lower().endswith((".png", ".jpg")):
                shutil.copy2(os.path.join(src7, f), os.path.join(DST, "07_写实件", f))

    json.dump(rows, open(os.path.join(DST, "九宫格切边.json"), "w"), ensure_ascii=False, indent=2)
    md = ["# 幻灵归虚 · UI 复刻套件 v9\n",
          "> 像素来自概念图原图（1x），未重绘；形状由原图轮廓决定，文字用 inpainting 擦除。\n",
          "| 文件 | 尺寸 | 九宫格 左/右/上/下 | 用途 |", "|---|---|---|---|"]
    for r in rows:
        md.append("| `%s/%s` %s | %d×%d | %d/%d/%d/%d | %s |" % (
            r["dir"], r["file"], r["name"], r["w"], r["h"], r["l"], r["r"], r["t"], r["b"], r["use"]))
    open(os.path.join(DST, "索引.md"), "w").write("\n".join(md) + "\n")
    print("synced", len(rows))


if __name__ == "__main__":
    main()
