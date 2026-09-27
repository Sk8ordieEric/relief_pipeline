"""Watertight solids from height fields, verified before anything is written.

The verification is not decoration. A mesh that is not closed will import into Fusion
looking fine and then produce silently wrong toolpaths, so nothing here writes an STL
that fails the checks.
"""
import os, struct, sys
import numpy as np
import cv2


def grid_solid(Hg, panel_w, panel_h, base, flip_y=True):
    """Build a closed solid: height-field top, skirt walls, ring-and-fan bottom.

    Hg      (PY, PX) heights in mm, measured up from `base`
    returns (vertices, faces)

    flip_y because image row 0 is the TOP of the picture, which must land at MAX Y in
    CAD. Getting this wrong gives you a mirrored carving and no error message.
    """
    if flip_y:
        Hg = Hg[::-1, :]
    PY, PX = Hg.shape
    NQX, NQY = PX - 1, PY - 1
    xs = np.linspace(0, panel_w, PX)
    ys = np.linspace(0, panel_h, PY)
    X, Y = np.meshgrid(xs, ys)
    top = np.stack([X.ravel(), Y.ravel(), (base + Hg).ravel()], 1)

    per = ([(0, j) for j in range(PX)] + [(i, NQX) for i in range(1, PY)]
           + [(NQY, j) for j in range(NQX - 1, -1, -1)] + [(i, 0) for i in range(NQY - 1, 0, -1)])
    per = np.array(per, dtype=np.int64)
    NP = len(per)
    bot = np.stack([xs[per[:, 1]], ys[per[:, 0]], np.zeros(NP)], 1)
    V = np.vstack([top, bot, [[panel_w / 2, panel_h / 2, 0.0]]])
    O_BOT, I_CTR = len(top), len(top) + NP

    i0 = np.arange(NQY)[:, None] * PX + np.arange(NQX)[None, :]
    A, B = i0.ravel(), (i0 + 1).ravel()
    C, D = (i0 + PX + 1).ravel(), (i0 + PX).ravel()
    F_top = np.concatenate([np.stack([A, B, C], 1), np.stack([A, C, D], 1)])
    k = np.arange(NP); k1 = (k + 1) % NP
    tk = per[k, 0] * PX + per[k, 1]; tk1 = per[k1, 0] * PX + per[k1, 1]
    bk, bk1 = O_BOT + k, O_BOT + k1
    F_wall = np.concatenate([np.stack([tk, bk, bk1], 1), np.stack([tk, bk1, tk1], 1)])
    F_bot = np.stack([np.full(NP, I_CTR), bk1, bk], 1)
    return V, np.vstack([F_top, F_wall, F_bot]).astype(np.int64)


def verify(V, F, label="mesh", quiet=False):
    """Every edge exactly twice, no repeated direction, Euler characteristic 2,
    positive signed volume. Returns (ok, volume)."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.minimum(e[:, 0], e[:, 1]) * len(V) + np.maximum(e[:, 0], e[:, 1])
    uk, cnt = np.unique(key, return_counts=True)
    dk = e[:, 0] * len(V) + e[:, 1]
    ndup = len(dk) - len(np.unique(dk))
    chi = len(V) - len(uk) + len(F)
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum('ij,ij->i', p0, np.cross(p1, p2)).sum() / 6.0
    ok = bool((cnt == 2).all()) and ndup == 0 and chi == 2 and vol > 0
    if not quiet:
        print("%-9s %d verts  %d tris | edges not-twice %d | dupes %d | Euler %d | vol %.1f mm^3"
              % (label, len(V), len(F), int((cnt != 2).sum()), ndup, chi, vol))
        print("%-9s WATERTIGHT: %s" % ("", "PASS" if ok else "FAIL"))
    return ok, vol


def write_stl(V, F, path, header="relief_pipeline"):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    ok, _ = verify(V, F, os.path.basename(path))
    if not ok:
        sys.exit("ERROR: mesh failed verification - %s not written." % path)
    n = np.cross(p1 - p0, p2 - p0)
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-30)
    rec = np.zeros(len(F), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'] = n; rec['v'] = np.stack([p0, p1, p2], 1)
    with open(path, 'wb') as f:
        f.write(header.encode()[:79].ljust(80, b'\0'))
        f.write(struct.pack('<I', len(F)))
        f.write(rec.tobytes())
    print("%-9s %.1f MB" % ("", os.path.getsize(path) / 1e6))


def resample(z, quads_long, panel_w, panel_h):
    """Decimate a working grid to the mesh resolution. INTER_AREA is a proper
    anti-aliased reduction; bilinear or nearest would alias fine detail."""
    if panel_w >= panel_h:
        PX = quads_long + 1
        PY = max(2, int(round(quads_long * panel_h / panel_w)) + 1)
    else:
        PY = quads_long + 1
        PX = max(2, int(round(quads_long * panel_w / panel_h)) + 1)
    return cv2.resize(z.astype(np.float32), (PX, PY), interpolation=cv2.INTER_AREA).astype(np.float64)


def write_dxf(contours, path, mm_per_px, panel_h_px, scale=1.0):
    """Minimal ASCII DXF with one closed LWPOLYLINE per contour, in millimetres.

    Y is flipped so the DXF matches CAD orientation rather than image orientation.
    """
    ents = []
    for c in contours:
        pts = [(x * mm_per_px * scale, (panel_h_px - y) * mm_per_px * scale) for x, y in c]
        ents.append("0\nLWPOLYLINE\n8\n0\n90\n%d\n70\n1\n" % len(pts)
                    + "".join("10\n%.4f\n20\n%.4f\n" % p for p in pts))
    with open(path, 'w') as f:
        f.write("0\nSECTION\n2\nENTITIES\n" + "".join(ents) + "0\nENDSEC\n0\nEOF\n")
    print("%-9s %s  (%d contours)" % ("", os.path.basename(path), len(contours)))
