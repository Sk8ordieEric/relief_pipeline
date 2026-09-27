"""The cutter's real geometry. ALL LENGTHS IN MM.

The single most important fact in here: a tapered ball nose is NOT a ball. It is a
sphere of radius R tangent to a cone, so its effective radius grows with depth. A
1/16" tapered ball with a 0.794 mm tip behaves like a 1.19 mm radius tool 5 mm down.
Model it as a sphere and you will leave uncut material in every groove.
"""
import numpy as np


def taper_profile(d, R, taper_deg):
    """Height of the cutter's surface above its tip, at radial distance d from the axis.

    Tangency between sphere and cone sits at d_t = R*cos(tau), z_t = R*(1 - sin(tau)).
    """
    tau = np.deg2rad(taper_deg)
    d_t, z_t = R * np.cos(tau), R * (1.0 - np.sin(tau))
    d = np.asarray(d, dtype=np.float64)
    inner = R - np.sqrt(np.maximum(R * R - np.minimum(d, R) ** 2, 0.0))
    return np.where(d <= d_t, inner, z_t + (d - d_t) / np.tan(tau))


def effective_radius(depth_mm, R, taper_deg):
    """How wide the tool actually is, `depth_mm` above its tip."""
    tau = np.deg2rad(taper_deg)
    d_t, z_t = R * np.cos(tau), R * (1.0 - np.sin(tau))
    return d_t + (depth_mm - z_t) * np.tan(tau) if depth_mm > z_t else d_t


def shift(a, dy, dx, fill):
    o = np.full_like(a, fill)
    H, W = a.shape
    ys0, ys1 = max(0, -dy), min(H, H - dy)
    xs0, xs1 = max(0, -dx), min(W, W - dx)
    yd0, yd1 = max(0, dy), min(H, H + dy)
    xd0, xd1 = max(0, dx), min(W, W + dx)
    o[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
    return o


def reach_filter(h_mm, R, taper_deg, mm_per_px, max_depth):
    """The surface this cutter can actually produce.

        tip locus  z = max over d of [ h(x+d) - s(|d|) ]   tool clears the part
        swept      m = min over d of [ z(x+d) + s(|d|) ]   what the tool sweeps

    Whatever this raises is a concavity the tool physically cannot enter. Leaving it
    in the model means leaving uncut material on the part, which reads as roughness.
    Returns (filtered_surface, effective_radius_used).
    """
    r_max = effective_radius(max_depth, R, taper_deg)
    Rpx = int(np.ceil(r_max / mm_per_px))
    offs = []
    for dy in range(-Rpx, Rpx + 1):
        for dx in range(-Rpx, Rpx + 1):
            d_mm = mm_per_px * np.hypot(dx, dy)
            if d_mm <= r_max:
                offs.append((dy, dx, float(taper_profile(d_mm, R, taper_deg))))
    z = np.full_like(h_mm, -np.inf)
    for dy, dx, s in offs:
        z = np.maximum(z, shift(h_mm, dy, dx, -np.inf) - s)
    m = np.full_like(h_mm, np.inf)
    for dy, dx, s in offs:
        m = np.minimum(m, shift(z, dy, dx, np.inf) + s)
    return np.where(np.isfinite(m), m, h_mm), r_max


def slope_clamp(h_mm, max_deg, mm_per_px, reach_mm):
    """Infimal convolution with a cone: h'(x) = min over d of [h(x+d) + tan(theta)*|d|].

    The result is tan(theta)-Lipschitz, so no facet exceeds max_deg from horizontal.
    Around a silhouette this produces a clean constant-angle draft; in the interior it
    shaves only what was too sharp to cut.
    """
    t = np.tan(np.deg2rad(max_deg))
    r_max = reach_mm / t + 2 * mm_per_px
    Rpx = int(np.ceil(r_max / mm_per_px))
    out = h_mm.copy()
    for dy in range(-Rpx, Rpx + 1):
        for dx in range(-Rpx, Rpx + 1):
            d_mm = mm_per_px * np.hypot(dx, dy)
            if d_mm <= r_max:
                out = np.minimum(out, shift(h_mm, dy, dx, np.inf) + t * d_mm)
    return out


def max_slope_deg(h_mm, mm_per_px):
    gy, gx = np.gradient(h_mm, mm_per_px)
    return np.rad2deg(np.arctan(np.hypot(gx, gy)))


def scallop_height(stepover_mm, tool_radius_mm):
    """Cusp left between adjacent passes of a ball-nosed cutter."""
    return stepover_mm ** 2 / (8.0 * tool_radius_mm)


def chip_thinning(ae_mm, dia_mm):
    """Radial chip thinning factor: 2*sqrt(ae/D * (1 - ae/D)).

    At light radial engagement the actual chip per tooth is far below the programmed
    feed per tooth, which is how a cutter ends up rubbing instead of cutting.
    """
    r = np.clip(ae_mm / dia_mm, 0.0, 1.0)
    return 2.0 * np.sqrt(r * (1.0 - r))
