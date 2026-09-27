#!/usr/bin/env python3
"""
relief_build.py - ComfyUI depth map + silhouette  ->  machinable relief  ->  STL

    python relief_build.py --depth depth.png --mask silhouette.png --photo normalized.png \
                           --mode portrait --panel 203x254 --relief 5.0 --out projects/rex/rex_v1

Modes
  bas       figure stands proud of a flat panel      (the raven)
  sunken    figure cut down into the panel face      (cavo-rilievo)
  portrait  bas-relief tuned for faces and fur

Outputs  <out>.stl  <out>_field.npy  <out>_preview.png
All lengths in MILLIMETRES.  See README.md.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import tooling as T
from lib import flow as FL
from lib import fields as FD
from lib import mesh as MS

# alpha  = gradient compression strength (higher squashes large steps harder)
# detail = how much photo-derived surface texture rides on the depth-derived form
# ced_*  = flow smoothing; rho is roughly the feature width to follow, in pixels
PRESETS = {
    'bas':      dict(alpha=6.0,  detail=0.60, ced_iters=42, ced_rho=6.5, ced_alpha=0.005,
                     groove=1.00, grain=0.08, hp_sigma=22, base_lift=0.28),
    'sunken':   dict(alpha=6.0,  detail=0.60, ced_iters=42, ced_rho=6.5, ced_alpha=0.005,
                     groove=1.00, grain=0.08, hp_sigma=22, base_lift=0.28),
    # A face carries ~80 mm of true depth that must land inside ~5 mm of relief, so
    # compression does the heavy lifting and the luminance detail layer is dialled
    # back - skin should be modelled, not engraved.
    'portrait': dict(alpha=11.0, detail=0.28, ced_iters=34, ced_rho=5.0, ced_alpha=0.006,
                     groove=0.75, grain=0.05, hp_sigma=26, base_lift=0.22),
}


def build(a):
    t0 = time.time()
    P = dict(PRESETS[a.mode])
    if a.preset:
        P.update(json.load(open(a.preset)))
        print("preset       %s" % a.preset)
    for k in ('alpha', 'detail', 'ced_iters', 'ced_rho'):
        if getattr(a, k, None) is not None:
            P[k] = getattr(a, k)

    depth = FD.load_gray(a.depth)
    H0, W0 = depth.shape
    side = a.grid
    rs = lambda im: cv2.resize(im.astype(np.float32), (side, side),
                               interpolation=cv2.INTER_AREA).astype(np.float64)
    mask_img = rs(FD.load_gray(a.mask, (H0, W0))) if a.mask else None
    photo = rs(FD.load_gray(a.photo, (H0, W0))) if a.photo else None
    depth = rs(depth)

    pw, ph = a.panel
    mm_per_px = max(pw, ph) / side

    mb = (mask_img > 0.5).astype(np.uint8) if mask_img is not None else (depth > a.depth_threshold).astype(np.uint8)
    mb = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mb = cv2.morphologyEx(mb, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sd_mm = FD.signed_field(mb) * mm_per_px
    m = sd_mm > 0
    if m.sum() < 500:
        sys.exit("ERROR: silhouette covers almost nothing. Check --mask / --depth-threshold.")
    d_mm = np.maximum(sd_mm, 0.0)

    r_eff = T.effective_radius(a.relief, a.tool_r, a.tool_taper)
    print("grid         %d px   %.4f mm/px   subject %.1f%% of panel" % (side, mm_per_px, 100 * m.mean()))
    print("tool         tip R %.3f, taper %.2f deg -> effective R %.3f mm at %.1f mm depth"
          % (a.tool_r, a.tool_taper, r_eff, a.relief))
    print("             narrowest cuttable feature ~%.2f mm wide" % (2 * r_eff))

    # repair the depth halo that sits around any segmented silhouette
    halo = 9
    interior = cv2.erode(m.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo + 1,) * 2)).astype(bool)
    fill = cv2.inpaint((np.clip(depth, 0, 1) * 255).astype(np.uint8),
                       ((m & ~interior) | ~m).astype(np.uint8), halo + 4,
                       cv2.INPAINT_TELEA).astype(np.float32) / 255.
    depth = np.where(interior, depth, fill)
    lo, hi = np.percentile(depth[m], [1, 99])
    form = cv2.bilateralFilter(np.clip((depth - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32),
                               9, 0.08, 9).astype(np.float64)

    det = np.zeros_like(form)
    guide = form
    if photo is not None:
        # inpaint outward instead of blurring a masked image: blurring leaves a
        # luminance cliff on the silhouette, which the groove weighting below then
        # reads as a deep carved line all the way around the subject
        pin = cv2.inpaint((np.clip(photo, 0, 1) * 255).astype(np.uint8), (~m).astype(np.uint8),
                          12, cv2.INPAINT_TELEA).astype(np.float32)
        eq = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(
            np.clip(pin, 0, 255).astype(np.uint8)).astype(np.float32) / 255.
        guide = eq.astype(np.float64)
        sm = cv2.bilateralFilter(eq, 9, 0.07, 9)
        hp = sm - cv2.GaussianBlur(sm, (0, 0), P['hp_sigma'])
        # in a carving a dark line IS a cut groove, while an equally bright line is
        # only a highlight - so the two signs are deliberately not weighted the same
        det = np.where(hp < 0, hp * P['groove'], hp * P['grain'])
        det = cv2.GaussianBlur(np.clip(det, -0.5, 0.5), (0, 0), 2.4).astype(np.float64)
        det = FL.ced(det, guide, m, P['ced_iters'], P['ced_rho'], P['ced_alpha'])
        tt = np.clip(d_mm / 2.1, 0, 1)
        det *= (tt * tt * (3 - 2 * tt))

    fe = np.where(m, form, cv2.GaussianBlur(np.where(m, form, 0).astype(np.float32), (0, 0), 25))
    gx = np.zeros_like(fe); gy = np.zeros_like(fe)
    gx[:, :-1] = np.diff(fe, axis=1); gy[:-1, :] = np.diff(fe, axis=0)
    dx = np.zeros_like(det); dy = np.zeros_like(det)
    dx[:, :-1] = np.diff(det, axis=1); dy[:-1, :] = np.diff(det, axis=0)
    gx += dx * P['detail']; gy += dy * P['detail']
    gx, gy = FD.compress_gradients(gx, gy, P['alpha'])
    h = FD.poisson(FD.divergence(gx, gy))
    h -= h[m].min(); h /= max(h[m].max(), 1e-9)

    top = np.clip(np.where(m, P['base_lift'] + (1 - P['base_lift']) * h, 0.0), 0, None)
    top /= top.max()
    top = FL.ced(top, guide, m, max(P['ced_iters'] // 5, 4), P['ced_rho'] * 1.4, 0.02, dt=0.16, sigma=1.3)
    top = np.clip(np.where(m, top, 0.0), 0, None); top /= top.max()

    TAN = np.tan(np.deg2rad(a.max_slope))
    if a.mode == 'sunken':
        lo_i, hi_i = top[m].min(), top[m].max()
        Hr = np.zeros_like(top); Hr[m] = (top[m] - lo_i) / max(hi_i - lo_i, 1e-9)
        z = np.where(m, -a.relief * (1.0 - Hr), 0.0)
        z_floor, z_top = -a.relief, 0.0
    else:
        z = FD.softmin(FD.nearest_inside(top, m) * a.relief, TAN * sd_mm, 6.0)
        z = np.clip(z, 0, None)
        z *= a.relief / max(z.max(), 1e-9)
        z_floor, z_top = 0.0, a.relief

    raw = np.percentile(T.max_slope_deg(z, mm_per_px)[m], 99.5)
    z = T.slope_clamp(z, a.max_slope, mm_per_px, a.relief)
    cl = np.percentile(T.max_slope_deg(z, mm_per_px)[m], 99.5)
    zf, _ = T.reach_filter(z, a.tool_r, a.tool_taper, mm_per_px, a.relief)
    lost = (zf - z)[m]
    z = np.clip(np.maximum(z, zf), z_floor, z_top)
    z[sd_mm < -0.02] = 0.0

    print("slope        p99.5 %.1f -> %.1f deg  (clamped at %.0f)" % (raw, cl, a.max_slope))
    print("unreachable  mean %.4f  max %.4f mm   >0.05 mm on %.2f%%"
          % (lost.mean(), lost.max(), 100 * (lost > 0.05).mean()))
    print("z range      %.3f .. %.3f mm" % (z.min(), z.max()))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out + '_field.npy', z)
    FD.shade(z, a.out + '_preview.png')

    nx = int(round(pw / mm_per_px)); ny = int(round(ph / mm_per_px))
    x0, y0 = (side - nx) // 2, (side - ny) // 2
    sub = z[max(y0, 0):max(y0, 0) + ny, max(x0, 0):max(x0, 0) + nx]
    Hg = MS.resample(sub, a.quads, pw, ph)
    base = a.base + (a.relief if a.mode == 'sunken' else 0.0)
    V, F = MS.grid_solid(Hg, pw, ph, base)
    MS.write_stl(V, F, a.out + '.stl',
                 "relief_pipeline %s %.0fx%.0f relief %.2f" % (a.mode, pw, ph, a.relief))
    print("part         %.2f x %.2f x %.2f mm    [%.1fs]" % (pw, ph, base + Hg.max(), time.time() - t0))


def panel_type(s):
    try:
        w, h = s.lower().split('x'); return float(w), float(h)
    except Exception:
        raise argparse.ArgumentTypeError("panel must look like 180x180 or 203x254 (mm)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--depth', required=True, help='Stage 3 depth map PNG')
    p.add_argument('--mask', help='Stage 4 silhouette PNG (white subject on black)')
    p.add_argument('--photo', help='Stage 1 normalized photo - adds detail, steers the fur flow')
    p.add_argument('--mode', choices=list(PRESETS), default='portrait')
    p.add_argument('--preset', help='JSON file of overrides from presets/')
    p.add_argument('--panel', type=panel_type, default=(203.0, 254.0), help='mm WxH (default 8x10in)')
    p.add_argument('--relief', type=float, default=5.0)
    p.add_argument('--base', type=float, default=10.0, help='solid panel under the relief, mm')
    p.add_argument('--max-slope', type=float, default=65.0)
    p.add_argument('--tool-r', type=float, default=0.79375, help='finishing tool TIP radius, mm')
    p.add_argument('--tool-taper', type=float, default=5.36, help='taper half-angle, deg')
    p.add_argument('--grid', type=int, default=900, help='working grid, px per long edge')
    p.add_argument('--quads', type=int, default=760, help='mesh quads along the long edge')
    p.add_argument('--depth-threshold', type=float, default=0.10, help='only used without --mask')
    p.add_argument('--alpha', type=float); p.add_argument('--detail', type=float)
    p.add_argument('--ced-iters', type=int, dest='ced_iters')
    p.add_argument('--ced-rho', type=float, dest='ced_rho')
    p.add_argument('--out', default='relief_out')
    build(p.parse_args())


if __name__ == '__main__':
    main()
