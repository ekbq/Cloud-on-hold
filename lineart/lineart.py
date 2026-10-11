"""Photo -> marcam-style ink line art -> traced SVG.

Pipeline: subject cutout (rembg) -> XDoG ink lines + solid black for the darkest
regions -> speck cleanup -> heavy subject outline -> potrace to a single filled path.
"""
import sys
import cv2
import numpy as np
import potrace
from PIL import Image, ImageOps
from rembg import new_session, remove

LONG_SIDE = 1024
MODEL = "models/edge_detection_dexined_2024sep.onnx"
EDGE_THR = 0.75
EDGE_SCALE = 1.5
DARK_PCT = 12


def load(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = LONG_SIDE / max(im.size)
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)


def subject_mask(im, session):
    m = np.array(remove(im, session=session, only_mask=True))
    m = (m > 128).astype(np.uint8) * 255
    # keep only sizeable blobs, fill small holes
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 0.01 * m.size:
            keep[lab == i] = 255
    return cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


_SESS = None


def dexined(rgb):
    """Edge probability map (0..1) from DexiNed, run in overlapping 480x640 tiles."""
    global _SESS
    if _SESS is None:
        _SESS = cv2.dnn.readNet(MODEL)
    TH, TW, ST_H, ST_W = 480, 640, 360, 480
    h, w = rgb.shape[:2]
    ph, pw = max(h, TH), max(w, TW)
    bgr = np.full((ph, pw, 3), 255, np.uint8)
    bgr[:h, :w] = rgb[:, :, ::-1]
    acc = np.zeros((ph, pw), np.float32)
    ys = list(range(0, ph - TH + 1, ST_H)) + [ph - TH]
    xs = list(range(0, pw - TW + 1, ST_W)) + [pw - TW]
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            t = bgr[y:y + TH, x:x + TW].astype(np.float32) - np.array([103.5, 116.2, 123.6], np.float32)
            _SESS.setInput(np.ascontiguousarray(t.transpose(2, 0, 1)[None]))
            out = _SESS.forward("block_cat")[0, 0]
            p = 1 / (1 + np.exp(-out))
            acc[y:y + TH, x:x + TW] = np.maximum(acc[y:y + TH, x:x + TW], p)
    return acc[:h, :w]


def xdog(gray, sigma=1.0, k=1.6, p=22.0, eps=0.0, phi=12.0):
    g = gray.astype(np.float32) / 255.0
    a = cv2.GaussianBlur(g, (0, 0), sigma)
    b = cv2.GaussianBlur(g, (0, 0), sigma * k)
    d = (1 + p) * a - p * b
    out = np.where(d >= eps, 1.0, 1.0 + np.tanh(phi * (d - eps)))
    return out  # 1 = white, toward 0 = ink


def remove_specks(ink, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return keep


def make_ink(im, mask):
    rgb = np.array(im)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # local contrast so features in dim light still register
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    # edge-preserving smoothing kills fur/fleece texture noise, keeps contours
    smooth = cv2.bilateralFilter(gray, 9, 40, 7)
    smooth = cv2.bilateralFilter(smooth, 9, 40, 7)

    # run the edge model at higher resolution for finer, thinner lines
    big = cv2.resize(rgb, None, fx=EDGE_SCALE, fy=EDGE_SCALE, interpolation=cv2.INTER_CUBIC)
    edges = cv2.resize(dexined(big), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    lines = (edges > EDGE_THR).astype(np.uint8) * 255
    lines = remove_specks(lines, 40)

    # solid black for the darkest masses (hair, pupils, open mouth, stripes)
    dark_thr = np.percentile(smooth[mask > 0], DARK_PCT)
    dark = ((smooth < dark_thr) & (mask > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    ink = cv2.bitwise_or(lines, dark)
    ink = cv2.bitwise_and(ink, mask)
    ink = remove_specks(ink, 30)

    # heavy outer outline around the subject
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = mask.shape
    outline = np.zeros_like(mask)
    cv2.drawContours(outline, cnts, -1, 255, 4)
    # don't draw outline along the image border (subject cropped by frame)
    outline[:3, :] = outline[-3:, :] = 0
    outline[:, :3] = outline[:, -3:] = 0
    return cv2.bitwise_or(ink, outline)


def to_svg(ink, path):
    h, w = ink.shape
    bmp = potrace.Bitmap(ink == 0)  # potracer fills pixels darker than blacklevel
    plist = bmp.trace(turdsize=8, alphamax=1.0, opticurve=True, opttolerance=0.2)
    parts = []
    for curve in plist:
        s = curve.start_point
        d = [f"M {s.x:.2f} {s.y:.2f}"]
        for seg in curve.segments:
            if seg.is_corner:
                d.append(f"L {seg.c.x:.2f} {seg.c.y:.2f} L {seg.end_point.x:.2f} {seg.end_point.y:.2f}")
            else:
                d.append(f"C {seg.c1.x:.2f} {seg.c1.y:.2f}, {seg.c2.x:.2f} {seg.c2.y:.2f}, {seg.end_point.x:.2f} {seg.end_point.y:.2f}")
        parts.append(" ".join(d) + " Z")
    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
                f'<rect width="100%" height="100%" fill="#fff"/>\n'
                f'<path fill="#000" fill-rule="evenodd" d="{" ".join(parts)}"/>\n</svg>\n')


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    session = new_session("isnet-general-use")
    im = load(src)
    mask = subject_mask(im, session)
    ink = make_ink(im, mask)
    Image.fromarray(255 - ink).save(out + ".png")
    to_svg(ink, out + ".svg")
    print(out, im.size, f"ink {100 * (ink > 0).mean():.1f}%")
