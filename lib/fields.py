"""Distance fields, Poisson integration and image loading. ALL LENGTHS IN MM."""
import sys
import numpy as np
import cv2
from scipy.fft import dctn, idctn
from scipy.ndimage import distance_transform_edt


def load_gray(path, shape=None):
    """Read any depth/mask/photo as float 0..1, handling 8- and 16-bit."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit("ERROR: cannot read %s" % path)
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)
    img /= 65535.0 if img.max() > 255 else 255.0
    if shape is not None and img.shape != shape:
        img = cv2.resize(img.astype(np.float32), (shape[1], shape[0]),
                         interpolation=cv2.INTER_AREA).astype(np.float64)
    return np.clip(img, 0, 1)


def smooth_contour(p, sig):
    """Gaussian-smooth a closed contour, wrapping at the seam."""
    n = len(p)
    k = int(max(3, round(sig * 6))) | 1
    g = cv2.getGaussianKernel(k, sig).ravel()
    out = np.empty_like(p)
    for d in range(2):
        ext = np.concatenate([p[-k:, d], p[:, d], p[:k, d]])
        out[:, d] = np.convolve(ext, g, mode='same')[k:k + n]
    return out


def contours_of(mb, min_area=40, min_pts=12, contour_sigma=2.0):
    """Smoothed sub-pixel outer contours of a binary mask, as float arrays."""
    cs, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        c = c.reshape(-1, 2).astype(np.float64)
        if cv2.contourArea(c.astype(np.float32)) < min_area or len(c) < min_pts:
            continue
        out.append(smooth_contour(c, contour_sigma))
    return out


def signed_field(mb, ss=4, contour_sigma=2.0):
    """Sub-pixel signed distance in PIXELS, positive inside.

    Both transforms come from ONE supersampled raster of the smoothed contour.
    OpenCV's DIST_L2 chamfer approximation quantises badly enough that any edge
    rollover built on it machines as a visible comb of ribs along the silhouette -
    this exists specifically to avoid that.
    """
    h, w = mb.shape
    polys = [(c * ss).round().astype(np.int32) for c in contours_of(mb, contour_sigma=contour_sigma)]
    if not polys:
        sys.exit("ERROR: no usable contour found in the mask. Check the silhouette image.")
    big = np.zeros((h * ss, w * ss), np.uint8)
    cv2.fillPoly(big, polys, 1)
    sd = (distance_transform_edt(big) - distance_transform_edt(1 - big)).astype(np.float32)
    return cv2.resize(sd, (w, h), interpolation=cv2.INTER_AREA) / ss


def poisson(div):
    """Solve grad^2 h = div with Neumann boundaries, via DCT-II."""
    M, N = div.shape
    F = dctn(div, type=2, norm='ortho')
    i, j = np.arange(M)[:, None], np.arange(N)[None, :]
    d = 2 * (np.cos(np.pi * i / M) - 1) + 2 * (np.cos(np.pi * j / N) - 1)
    d[0, 0] = 1.0
    H = F / d
    H[0, 0] = 0.0
    return idctn(H, type=2, norm='ortho')


def compress_gradients(gx, gy, alpha):
    """Gradient-domain range compression, s(|g|) = log(1 + alpha*|g|) / (alpha*|g|).

    This is what lets ~80 mm of real depth (a face, nose-tip to ear) land inside 5 mm
    of relief without going flat: big steps are squashed hard, small detail is left
    almost untouched.
    """
    mag = np.hypot(gx, gy)
    s = np.ones_like(mag)
    nz = mag > 1e-9
    s[nz] = np.log1p(alpha * mag[nz]) / (alpha * mag[nz])
    return gx * s, gy * s


def divergence(gx, gy):
    div = np.zeros_like(gx)
    div[:, 1:] += np.diff(gx, axis=1); div[:, 0] += gx[:, 0]
    div[1:, :] += np.diff(gy, axis=0); div[0, :] += gy[0, :]
    return div


def softmin(a, b, k):
    """Smooth minimum - rounds the crease where a drafted wall meets a top face."""
    return np.minimum(a, b) - np.log1p(np.exp(-k * np.abs(a - b))) / k


def nearest_inside(field, mask):
    """Extend `field` outward by nearest-inside value, so a softmin against a wall is
    well defined outside the subject."""
    _, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
    return field[iy, ix]


def shade(z, path, gain=1.6):
    """Write a lit preview of a height field."""
    hn = ((z - z.min()) / max(z.max() - z.min(), 1e-9)).astype(np.float32)
    gx = cv2.Sobel(hn, cv2.CV_32F, 1, 0, ksize=3) * gain
    gy = cv2.Sobel(hn, cv2.CV_32F, 0, 1, ksize=3) * gain
    nz = 1 / np.sqrt(gx * gx + gy * gy + 1)
    nx, ny = -gx * nz, -gy * nz
    img = np.zeros(hn.shape, np.float32)
    for L, w in (((-.48, -.58, .66), .80), ((.56, -.30, .77), .24), ((.08, .66, .75), .18)):
        L = np.array(L) / np.linalg.norm(L)
        img += w * np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    spec = np.clip(nx * -.34 + ny * -.40 + nz * .85, 0, 1) ** 26
    cv2.imwrite(path, (np.clip(.16 + .90 * img + .26 * spec, 0, 1) * 255).astype(np.uint8))
