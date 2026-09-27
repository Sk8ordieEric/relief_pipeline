#!/usr/bin/env python3
"""
inlay_build.py - silhouette  ->  drafted press-fit inlay  ->  two STLs + DXF

    python inlay_build.py --mask silhouette.png --panel 180x180 \
                          --vbit 60 --pocket 3.0 --offset 0.4 --out projects/oak_raven/inlay

Cuts a pocket in board A and a matching plug on board B, both with walls drafted at
the V-bit's half angle. The plug is undersized laterally so it seats DEEPER rather
than looser - that is what makes the joint tolerate machine backlash.

    epsilon = S * tan(delta)      lateral undersize, per side
    Delta   = epsilon / tan(delta) = S
    -> the plug sinks exactly S, and you face off exactly S past board A's surface.

Outputs  <out>_pocket.stl  <out>_plug.stl  <out>_contour.dxf  <out>_preview.png
All lengths in MILLIMETRES.  See README.md.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import tooling as T
from lib import fields as FD
from lib import mesh as MS


def build(a):
    t0 = time.time()
    delta = a.vbit / 2.0                      # half angle = draft from vertical
    tan_d = np.tan(np.deg2rad(delta))
    F = a.pocket
    S = a.offset
    plug_h = F - S - a.glue_void
    if plug_h <= 0.3:
        sys.exit("ERROR: plug height would be %.2f mm. Increase --pocket or reduce --offset." % plug_h)
    eps = S * tan_d
    min_stroke = 2 * F * tan_d + 2 * a.tip_flat

    print("V-bit        %.0f deg included  ->  draft %.1f deg from vertical" % (a.vbit, delta))
    print("pocket       %.2f mm deep       plug %.2f mm tall" % (F, plug_h))
    print("offset S     %.3f mm            -> undersize %.3f mm per side" % (S, eps))
    print("face off     %.3f mm past board A's surface" % S)
    print("min stroke   %.2f mm  (below this the V never reaches full depth)" % min_stroke)

    mask = FD.load_gray(a.mask)
    side = a.grid
    pw, ph = a.panel
    mm_per_px = max(pw, ph) / side
    mb = cv2.resize((mask > 0.5).astype(np.uint8) * 255, (side, side),
                    interpolation=cv2.INTER_NEAREST)
    mb = (mb > 127).astype(np.uint8)

    # Guarantee both halves are machinable by the same tool: an opening followed by a
    # closing with a disc of radius r forces every corner - convex AND concave - to a
    # radius of at least r. Without this, the pocket's inside corners are rounded by
    # the tool while the plug's outside corners stay sharp, and the plug will not seat.
    r_px = max(1, int(round(a.tip_radius / mm_per_px)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1,) * 2)
    before = int(mb.sum(dtype=np.int64))
    mb = cv2.morphologyEx(mb, cv2.MORPH_OPEN, disc)
    mb = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, disc)
    after = int(mb.sum(dtype=np.int64))
    if before:
        print("corner prep  rounded to r >= %.2f mm   area changed %+.2f%%"
              % (a.tip_radius, 100.0 * (after - before) / before))
    if after < 500:
        sys.exit("ERROR: nothing survived corner prep. The artwork is too fine for this tool.")

    sd_mm = FD.signed_field(mb) * mm_per_px
    inside = sd_mm > 0

    # how much of the inlay never reaches full depth
    shallow = inside & (sd_mm < F * tan_d)
    frac = 100.0 * shallow.sum() / max(inside.sum(), 1)
    print("coverage     %.1f%% of the inlay area reaches FULL depth   (%.1f%% stays shallow)"
          % (100 - frac, frac))
    if frac > 35:
        print("             ^ that is a lot. Use a narrower V-bit, a shallower pocket,")
        print("               or bolder artwork with thicker strokes.")

    # ---- board A: pocket. Opens upward, so a plain height field. ----------
    depth_A = np.minimum(F, np.maximum(sd_mm, 0.0) / tan_d)
    zA = a.board_a - depth_A

    # ---- board B: plug, in MACHINING orientation (narrow end up, no undercut) ----
    # tip ring sits at sd = eps + plug_h*tan(delta); the wall flares outward-downward
    # to the base ring at sd = eps, which lands on the cleared floor.
    sd_tip = sd_mm - (eps + plug_h * tan_d)
    drop = np.minimum(plug_h, np.maximum(-sd_tip, 0.0) / tan_d)
    zB = a.board_b - drop

    if a.board_a < F + 3.0:
        print("WARNING      board A is %.1f mm; only %.1f mm left under a %.1f mm pocket"
              % (a.board_a, a.board_a - F, F))
    print("board B      %.2f mm stock - %.2f mm plug = %.2f mm backing to face off"
          % (a.board_b, plug_h, a.board_b - plug_h))
    print("total facing %.2f mm  (backing + S)" % (a.board_b - plug_h + S))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    FD.shade(zA, a.out + '_pocket_preview.png')
    FD.shade(zB, a.out + '_plug_preview.png')

    for tag, z, thick in (('pocket', zA, a.board_a), ('plug', zB, a.board_b)):
        nx = int(round(pw / mm_per_px)); ny = int(round(ph / mm_per_px))
        x0, y0 = max((side - nx) // 2, 0), max((side - ny) // 2, 0)
        Hg = MS.resample(z[y0:y0 + ny, x0:x0 + nx], a.quads, pw, ph)
        V, Fc = MS.grid_solid(Hg, pw, ph, 0.0)
        MS.write_stl(V, Fc, '%s_%s.stl' % (a.out, tag),
                     "relief_pipeline inlay %s vbit%.0f pocket%.2f" % (tag, a.vbit, F))

    cs = FD.contours_of(mb)
    MS.write_dxf(cs, a.out + '_contour.dxf', mm_per_px, side)
    print("done         %.1fs" % (time.time() - t0))


def panel_type(s):
    try:
        w, h = s.lower().split('x'); return float(w), float(h)
    except Exception:
        raise argparse.ArgumentTypeError("panel must look like 180x180 (mm)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mask', required=True, help='Stage 4 silhouette PNG (white subject on black)')
    p.add_argument('--panel', type=panel_type, default=(180.0, 180.0))
    p.add_argument('--vbit', type=float, default=60.0, help='V-bit INCLUDED angle, deg (30/60/90)')
    p.add_argument('--pocket', type=float, default=3.0, help='pocket depth in board A, mm')
    p.add_argument('--offset', type=float, default=0.4, help='S - depth offset / press allowance, mm')
    p.add_argument('--glue-void', type=float, default=0.3, help='clearance left under the plug tip, mm')
    p.add_argument('--board-a', type=float, default=19.0, help='pocket board thickness, mm')
    p.add_argument('--board-b', type=float, default=6.35, help='plug board thickness, mm (1/4in)')
    p.add_argument('--tip-radius', type=float, default=0.8,
                   help='radius every corner is forced to, mm - use the smallest endmill radius')
    p.add_argument('--tip-flat', type=float, default=0.2, help='V-bit tip flat, mm')
    p.add_argument('--grid', type=int, default=900)
    p.add_argument('--quads', type=int, default=700)
    p.add_argument('--out', default='inlay_out')
    build(p.parse_args())


if __name__ == '__main__':
    main()
