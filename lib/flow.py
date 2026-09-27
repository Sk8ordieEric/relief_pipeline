"""Coherence-enhancing anisotropic diffusion (Weickert).

Smooths hard ALONG coherent linear structure and barely at all ACROSS it. On fur,
feathers or hair this turns noisy strand texture into flowing grouped masses while
keeping the separations between them crisp.

That is not only an aesthetic choice. A 1/16" tapered ball is ~2.4 mm wide at depth,
so individual hairs are physically uncuttable - grouped masses are the only form of
fur the machine can reproduce. On skin there is little coherent structure, so this
does almost nothing, which is correct.
"""
import numpy as np
import cv2


def structure_tensor(img, sigma, rho):
    """sigma = noise scale for the gradient; rho = integration scale, roughly the
    feature width to follow, in pixels."""
    s = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
    return (cv2.GaussianBlur(gx * gx, (0, 0), rho).astype(np.float64),
            cv2.GaussianBlur(gx * gy, (0, 0), rho).astype(np.float64),
            cv2.GaussianBlur(gy * gy, (0, 0), rho).astype(np.float64))


def eig2x2(Jxx, Jxy, Jyy):
    """Returns mu1 >= mu2 and the unit eigenvector of mu1, which points ACROSS the
    structure (the direction of fastest change)."""
    tr, dif = Jxx + Jyy, Jxx - Jyy
    rad = np.sqrt(np.maximum(dif * dif + 4.0 * Jxy * Jxy, 0.0))
    mu1, mu2 = 0.5 * (tr + rad), 0.5 * (tr - rad)
    vx, vy = Jxy, mu1 - Jxx
    n = np.hypot(vx, vy)
    flat = n < 1e-12
    return (mu1, mu2,
            np.where(flat, 1.0, vx / np.where(flat, 1, n)),
            np.where(flat, 0.0, vy / np.where(flat, 1, n)))


def ced(h, guide, mask, iters, rho, alpha, dt=0.18, sigma=1.1):
    """Explicit coherence-enhancing diffusion.

    h      field to smooth
    guide  image whose structure steers the diffusion (normally the photo)
    mask   diffuse only inside this boolean mask, or None
    alpha  diffusivity ACROSS structure - small keeps the separations crisp
    """
    if iters <= 0:
        return h
    h = h.astype(np.float64).copy()
    Jxx, Jxy, Jyy = structure_tensor(guide, sigma, rho)
    mu1, mu2, vx, vy = eig2x2(Jxx, Jxy, Jyy)
    coh = (mu1 - mu2) ** 2
    pos = coh[coh > 0]
    C = max(float(np.percentile(pos, 60)) if pos.size else 1e-18, 1e-18)
    lam_across = np.full_like(coh, alpha)
    lam_along = alpha + (1.0 - alpha) * np.exp(-C / (coh + 1e-30))
    # (vx, vy) is ACROSS the structure; (-vy, vx) is ALONG it
    Dxx = lam_across * vx * vx + lam_along * vy * vy
    Dxy = lam_across * vx * vy - lam_along * vy * vx
    Dyy = lam_across * vy * vy + lam_along * vx * vx
    for _ in range(iters):
        hy, hx = np.gradient(h)
        _, dax = np.gradient(Dxx * hx + Dxy * hy)
        day, _ = np.gradient(Dxy * hx + Dyy * hy)
        step = dt * (dax + day)
        h += np.where(mask, step, 0.0) if mask is not None else step
    return h


def across_flow_crisp(h, guide, rho, amount, fade=1.0):
    """Unsharp ACROSS the flow only. Makes the separation between adjacent feathers
    or coat masses read as a carved line rather than a soft swell, without adding any
    roughness along the flow."""
    if amount <= 0:
        return h
    Jxx, Jxy, Jyy = structure_tensor(guide, 1.1, rho)
    _, _, vx, vy = eig2x2(Jxx, Jxy, Jyy)
    hy, hx = np.gradient(h)
    dn = vx * hx + vy * hy
    dny, dnx = np.gradient(dn)
    return h - amount * fade * (vx * dnx + vy * dny)
