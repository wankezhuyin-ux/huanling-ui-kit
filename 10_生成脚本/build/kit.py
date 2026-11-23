"""幻灵归虚 UI v3 处理库：裁切 / 去字 / 补全 / 归一化。"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageFont

SS = 4  # 超采样倍数


# ---------------------------------------------------------------- 基础
def np2im(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def im2np(im):
    return np.asarray(im.convert('RGB')).astype(np.float32)


def dilate_mask(m, r):
    """圆盘膨胀（numpy 移位法）。"""
    if r <= 0:
        return m
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            out |= _shift(m, dx, dy)
    return out


def _shift(m, dx, dy):
    out = np.zeros_like(m)
    h, w = m.shape
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    out[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def blur_mask(m, r):
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8), 'L')
                      .filter(ImageFilter.GaussianBlur(r))) > 127


# ---------------------------------------------------------------- 去字填充

def _side_mean(out, mask, y, x, band, direction):
    """沿行取 x 一侧 band 个非空洞像素均值。"""
    h, w = mask.shape
    vals = []
    xx = x + direction
    while len(vals) < band and 0 <= xx < w:
        if not mask[y, xx]:
            vals.append(out[y, xx])
        xx += direction
    return np.mean(vals, 0) if vals else None


def _side_mean_v(out, mask, x, y, band, direction):
    h, w = mask.shape
    vals = []
    yy = y + direction
    while len(vals) < band and 0 <= yy < h:
        if not mask[yy, x]:
            vals.append(out[yy, x])
        yy += direction
    return np.mean(vals, 0) if vals else None


def inpaint_holes(im, mask, mode='h', band=6, jitter=1.8, seed=0):
    """把 mask 区域按行(h)/列(v)用两侧已知像素插值填掉，并加轻微噪声。
    适合底色是低频渐变的件（名条/按钮/面板/横幅）。"""
    a = im2np(im)
    h, w = mask.shape
    out = a.copy()
    rng = np.random.default_rng(seed)
    idxs = np.where(mask.any(1))[0] if mode == 'h' else np.where(mask.any(0))[0]
    if mode == 'h':
        for y in range(h):
            row = mask[y]
            if not row.any():
                continue
            xs = np.where(row)[0]
            # 找连续段
            cuts = np.where(np.diff(xs) > 1)[0]
            segs = np.split(xs, cuts + 1)
            for sg in segs:
                x0, x1 = sg[0], sg[-1]
                n = x1 - x0 + 1
                t = np.linspace(0, 1, n)[:, None]
                # 左右各取 band 个已知像素；遇到边界或相邻空洞就退化为单侧
                lv = _side_mean(out, mask, y, x0, band, -1)
                rv = _side_mean(out, mask, y, x1, band, +1)
                if lv is None and rv is None:
                    continue
                if lv is None:
                    out[y, x0:x1 + 1] = rv
                elif rv is None:
                    out[y, x0:x1 + 1] = lv
                else:
                    out[y, x0:x1 + 1] = lv * (1 - t) + rv * t
    else:
        for x in range(w):
            col = mask[:, x]
            if not col.any():
                continue
            ys = np.where(col)[0]
            cuts = np.where(np.diff(ys) > 1)[0]
            segs = np.split(ys, cuts + 1)
            for sg in segs:
                y0, y1 = sg[0], sg[-1]
                n = y1 - y0 + 1
                t = np.linspace(0, 1, n)[:, None]
                tv = _side_mean_v(out, mask, x, y0, band, -1)
                bv = _side_mean_v(out, mask, x, y1, band, +1)
                if tv is None and bv is None:
                    continue
                if tv is None:
                    out[y0:y1 + 1, x] = bv
                elif bv is None:
                    out[y0:y1 + 1, x] = tv
                else:
                    out[y0:y1 + 1, x] = tv * (1 - t) + bv * t
    # 轻度垂直模糊抹平接缝 + 噪声保持颗粒感
    m3 = Image.fromarray((mask * 255).astype(np.uint8), 'L').filter(ImageFilter.GaussianBlur(2.2))
    soft = Image.fromarray(out.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.1))
    res = Image.composite(soft, np2im(out), m3)
    arr = im2np(res)
    noise = rng.normal(0, jitter, arr.shape)
    m4 = np.asarray(Image.fromarray((mask * 255).astype(np.uint8), 'L')
                    .filter(ImageFilter.GaussianBlur(3))) / 255.0
    arr = arr + noise * m4[..., None]
    return np2im(arr)


def text_mask(im, kind='white', thr=None):
    """检测文字像素。white=白字带描边(取亮且低饱和)；dark=深字。"""
    a = im2np(im)
    v = a.max(2) / 255.0
    mn = a.min(2) / 255.0
    s = np.where(v > 0, (v - mn) / np.maximum(v, 1e-6), 0)
    if kind == 'white':
        t = thr if thr is not None else 0.72
        warm = (a[..., 0] - a[..., 2]) > 16   # 偏暖=金/纸，不是白字
        m = (v > t) & (s < 0.20) & ~warm
    elif kind == 'dark':
        t = thr if thr is not None else 0.30
        m = v < t
    elif kind == 'ink':  # 深棕描边字
        t = thr if thr is not None else 0.45
        m = (v < t) & (s > 0.15)
    else:
        raise ValueError(kind)
    return m


def strip_text(im, kind='white', grow=3, mode='both', thr=None, protect=None, roi=None):
    """roi=(x0,y0,x1,y1) 局部坐标，只在框内检测文字，保护装饰。"""
    m = text_mask(im, kind, thr)
    if roi is not None:
        x0, y0, x1, y1 = [int(v) for v in roi]
        keep = np.zeros_like(m)
        keep[max(0, y0):y1, max(0, x0):x1] = True
        m &= keep
    if grow:
        m = dilate_mask(m, grow)
    if protect is not None:
        m &= ~protect
    if not m.any():
        return im
    if mode == 'both':
        a = inpaint_holes(im, m, mode='h')
        b = inpaint_holes(im, m, mode='v')
        out = (im2np(a) * 0.5 + im2np(b) * 0.5)
        return np2im(out)
    return inpaint_holes(im, m, mode=mode)


# ---------------------------------------------------------------- 形状掩码
def capsule_mask(w, h, r=None):
    r = r if r is not None else h / 2
    m = Image.new('L', (w * SS, h * SS), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, w * SS - 1, h * SS - 1], radius=r * SS, fill=255)
    return np.asarray(m.resize((w, h), Image.LANCZOS)) / 255.0


def circle_mask(w, h, r, cx=None, cy=None, feather=1.0):
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    m = Image.new('L', (w * SS, h * SS), 0)
    ImageDraw.Draw(m).ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], fill=255)
    m = m.resize((w, h), Image.LANCZOS)
    if feather:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(m) / 255.0


def apply_alpha(im, alpha):
    im = im.convert('RGBA')
    a = np.asarray(im.getchannel('A')).astype(np.float32) / 255.0
    im.putalpha(Image.fromarray((a * alpha * 255).clip(0, 255).astype(np.uint8), 'L'))
    return im


def pad(im, n, color=(0, 0, 0, 0)):
    out = Image.new('RGBA', (im.width + 2 * n, im.height + 2 * n), color)
    out.alpha_composite(im.convert('RGBA'), (n, n))
    return out


def mirror_avg(im, keep_left=None):
    """左右镜像平均，强制对称。keep_left: 保留左侧比例(0-1)，其余镜像。"""
    im = im.convert('RGBA')
    a = np.asarray(im).astype(np.float32)
    f = a[:, ::-1]
    w = a.shape[1]
    half = keep_left if keep_left is not None else 0.5
    cut = int(w * half)
    out = a.copy()
    out[:, :cut] = a[:, :cut]
    out[:, cut:] = f[:, cut:]
    return np2im_rgba(out)


def np2im_rgba(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA')


def upscale(im, k):
    return im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)


def trim_alpha(im, thr=6):
    a = np.asarray(im.convert('RGBA').getchannel('A'))
    ys, xs = np.where(a > thr)
    if len(xs) == 0:
        return im
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


# ---------------------------------------------------------------- 自动包围盒
def hsv3(im):
    a = im2np(im)
    mx = a.max(2) / 255.0
    mn = a.min(2) / 255.0
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    r, g, b = a[..., 0] / 255.0, a[..., 1] / 255.0, a[..., 2] / 255.0
    hh = np.zeros_like(mx)
    m = d > 1e-6
    idx = m & (mx == r); hh[idx] = ((g[idx] - b[idx]) / d[idx]) % 6
    idx = m & (mx == g) & (mx != r); hh[idx] = ((b[idx] - r[idx]) / d[idx]) + 2
    idx = m & (mx == b) & (mx != r) & (mx != g); hh[idx] = ((r[idx] - g[idx]) / d[idx]) + 4
    return (hh * 60) % 360, s, mx, a


def rule_mask(im, rules):
    """rules: [(hmin,hmax,smin,smax,vmin,vmax), ...] 任一命中即真。"""
    h, s, v, a = hsv3(im)
    m = np.zeros(h.shape, bool)
    for (h0, h1, s0, s1, v0, v1) in rules:
        m |= (h >= h0) & (h <= h1) & (s >= s0) & (s <= s1) & (v >= v0) & (v <= v1)
    return m


def largest_cc(m, min_area=200):
    """返回最大连通域掩码（RLE 扫描线 + BFS）。"""
    h, w = m.shape
    lab = np.zeros((h, w), np.int32)
    cur = 0
    best_lab, best_area = 0, 0
    ys, xs = np.where(m)
    seen = np.zeros((h, w), bool)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        seen[y0, x0] = True
        area = 0
        while stack:
            y, x = stack.pop()
            lab[y, x] = cur
            area += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if area > best_area:
            best_area, best_lab = area, cur
    return lab == best_lab, best_area


def gap_mask(m, axis=0, min_gap=3):
    """axis=0 保留列方向主段（列投影）；axis=1 保留行方向主段。"""
    proj = m.any(axis=axis)
    n = len(proj)
    keep = np.zeros(n, bool)
    st = None
    runs = []
    for i, b in enumerate(proj):
        if b and st is None:
            st = i
        if not b and st is not None:
            runs.append((st, i)); st = None
    if st is not None:
        runs.append((st, n))
    # 合并间隔 < min_gap-1 的段
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    if not merged:
        return np.zeros_like(m)
    a, b = max(merged, key=lambda t: t[1] - t[0])
    keep[a:b] = True
    if axis == 0:
        return m & keep[None, :]
    return m & keep[:, None]


# ---------------------------------------------------------------- 轮廓提取
def rowspan_alpha(m, grow=1, smooth=0.9, min_px=2, med=3, col_clip=True):
    """按行的最左/最右像素生成实心掩码，适合横幅/名条/血条/按钮这类横行元件。"""
    if col_clip:
        m = gap_mask(gap_mask(m, 0, 10), 1, 10)
    h, w = m.shape
    a = np.zeros((h, w), np.float32)
    rows = np.zeros(h, bool)
    for y in range(h):
        xs = np.where(m[y])[0]
        if len(xs) < min_px:
            continue
        # 只取该行最长连续段，排除背景零散命中
        cuts = np.where(np.diff(xs) > 1)[0]
        segs = np.split(xs, cuts + 1)
        best = max(segs, key=len)
        if len(best) < max(min_px, w * 0.22):
            continue
        a[y, best[0]:best[-1] + 1] = 1.0
        rows[y] = True
    # 行方向补缝（中间隔一两行的空洞）
    keep = np.zeros(h, bool)
    ys = np.where(rows)[0]
    if len(ys) == 0:
        return a
    prev = ys[0]
    keep[prev] = True
    for y in ys[1:]:
        if y - prev <= 3:
            keep[prev:y + 1] = True
        prev = y
    a[~keep] = 0
    # 中值滤波平滑锯齿
    if med:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8), 'L')
                       .filter(ImageFilter.MedianFilter(med))) / 255.0
    m2 = a > 0.5
    if grow:
        m2 = dilate_mask(m2, grow)
    a = m2.astype(np.float32)
    if smooth:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8), 'L')
                       .filter(ImageFilter.GaussianBlur(smooth))) / 255.0
    return a


def rect_alpha(w, h, x0, y0, x1, y1, r=0, smooth=1.0):
    m = Image.new('L', (w * SS, h * SS), 0)
    ImageDraw.Draw(m).rounded_rectangle([x0 * SS, y0 * SS, x1 * SS - 1, y1 * SS - 1],
                                        radius=r * SS, fill=255)
    m = m.resize((w, h), Image.LANCZOS)
    if smooth:
        m = m.filter(ImageFilter.GaussianBlur(smooth))
    return np.asarray(m) / 255.0


def col_detect_edge(im, axis='y', lo=None, hi=None, thr=0.5):
    """沿 axis 找亮度突变位置，返回区间。用于面板/底栏这类贴边件。"""
    a = im2np(im).mean(2)
    prof = a.mean(1) if axis == 'y' else a.mean(0)
    grad = np.abs(np.diff(prof))
    if hi is None:
        return prof
    return prof, grad


def diff_mask(im, roi, thr=26, axis=1):
    """以行(axis=1)/列中值色为底色，偏差大的像素判为文字。渐变底最稳。"""
    a = im2np(im)
    h, w = a.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in roi]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    sub = a[y0:y1, x0:x1]
    base = np.median(sub, axis=axis, keepdims=True)
    d = np.abs(sub - base).max(2)
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = d > thr
    return m


def strip_text_diff(im, roi, thr=26, grow=3, mode='both'):
    m = diff_mask(im, roi, thr, axis=1)
    m = dilate_mask(m, grow)
    if not m.any():
        return im
    if mode == 'both':
        h = inpaint_holes(im, m, mode='h')
        v = inpaint_holes(im, m, mode='v')
        return np2im(im2np(h) * 0.5 + im2np(v) * 0.5)
    return inpaint_holes(im, m, mode=mode)


def poly_alpha(w, h, pts, smooth=0.8):
    """多边形 alpha（超采样）。pts 为 (x,y) 列表。"""
    m = Image.new('L', (w * SS, h * SS), 0)
    ImageDraw.Draw(m).polygon([(x * SS, y * SS) for x, y in pts], fill=255)
    m = m.resize((w, h), Image.LANCZOS)
    if smooth:
        m = m.filter(ImageFilter.GaussianBlur(smooth))
    return np.asarray(m) / 255.0


def hexplate_alpha(w, h, cut=None, smooth=0.8):
    """两端斜切的六边形牌。"""
    cut = cut if cut is not None else h * 0.42
    pts = [(0, h / 2), (cut, 0), (w - cut, 0), (w, h / 2), (w - cut, h), (cut, h)]
    return poly_alpha(w, h, pts, smooth)
