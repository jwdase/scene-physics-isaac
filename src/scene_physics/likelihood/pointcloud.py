"""3DP3-style point-cloud likelihood (port of Scene-Physics ``likelihood/likelihoods_functions.py``).

For every *observed* pixel with valid depth:

    inlier  = max over a (2f+1)x(2f+1) window of rendered pixels of
              sum_xyz log N(obs - rendered; 0, variance) - log(H*W)      + log(1 - outlier_prob)
    outlier = log(outlier_prob) - log(outlier_volume)
    score   = logaddexp(inlier, outlier)

and the image score is the sum over pixels. Rendered pixels outside the image or with no hit use the
same -100 sentinel as the JAX code; observed pixels with no depth contribute 0. The functions here
are numerically the same computation, written as a Warp kernel so it runs on the ray caster's output
in place, with a deterministic per-world reduction (no floating-point atomics).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

SENTINEL = -100.0


@wp.kernel(enable_backward=False)
def _pixel_scores(
    observed: wp.array(dtype=wp.vec3f),  # (H*W,)
    rendered: wp.array2d(dtype=wp.vec3f),  # (N, H*W)
    height: int,
    width: int,
    f: int,
    inv_two_var: float,
    log_norm: float,  # 1.5*log(2*pi*var) + log(H*W)
    log_inlier_prior: float,  # log(1 - outlier_prob)
    outlier_score: float,  # log(outlier_prob) - log(outlier_volume)
    out: wp.array2d(dtype=wp.float32),  # (N, H*W)
):
    w, p = wp.tid()
    o = observed[p]
    if wp.isnan(o[0]):
        out[w, p] = 0.0
        return
    i = p / width
    j = p - i * width
    best = float(-1.0e30)
    for di in range(-f, f + 1):
        for dj in range(-f, f + 1):
            ii = i + di
            jj = j + dj
            r = wp.vec3f(-100.0, -100.0, -100.0)
            if ii >= 0 and ii < height and jj >= 0 and jj < width:
                rr = rendered[w, ii * width + jj]
                if not wp.isnan(rr[0]):
                    r = rr
            d = o - r
            lp = -wp.dot(d, d) * inv_two_var - log_norm
            best = wp.max(best, lp)
    inlier = best + log_inlier_prior
    hi = wp.max(inlier, outlier_score)
    out[w, p] = hi + wp.log(wp.exp(inlier - hi) + wp.exp(outlier_score - hi))


@wp.kernel(enable_backward=False)
def _row_sums(scores: wp.array2d(dtype=wp.float32), width: int, out: wp.array2d(dtype=wp.float64)):
    w, row = wp.tid()
    acc = float(0.0)
    base = row * width
    for j in range(width):
        acc += float(scores[w, base + j])
    out[w, row] = wp.float64(acc)


class PointCloudLikelihood:
    """Scores ``N`` rendered point clouds against one observed cloud (all on one device)."""

    def __init__(self, observed: np.ndarray | wp.array, height: int, width: int, *, variance: float = 0.001,
                 outlier_prob: float = 0.001, outlier_volume: float = 1.0, filter_size: int = 3,
                 device: str | None = None):
        self.device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
        self.height, self.width = height, width
        if isinstance(observed, wp.array):
            obs = observed.reshape((height * width,)) if observed.ndim != 1 else observed
            if obs.dtype != wp.vec3f:
                obs = wp.array(obs.numpy().reshape(-1, 3).astype(np.float32), dtype=wp.vec3f, device=self.device)
        else:
            obs = wp.array(np.ascontiguousarray(np.asarray(observed, dtype=np.float32).reshape(-1, 3)),
                           dtype=wp.vec3f, device=self.device)
        self.observed = obs
        self.filter_size = int(filter_size)
        self.inv_two_var = 1.0 / (2.0 * variance)
        self.log_norm = 1.5 * math.log(2.0 * math.pi * variance) + math.log(height * width)
        self.log_inlier_prior = math.log(1.0 - outlier_prob)
        self.outlier_score = math.log(outlier_prob) - math.log(outlier_volume)
        self.baseline = float(self.score(self.observed.reshape((1, height * width)))[0])

    def pixel_scores(self, rendered: wp.array) -> wp.array:
        n = rendered.shape[0]
        out = wp.empty((n, self.height * self.width), dtype=wp.float32, device=self.device)
        wp.launch(
            _pixel_scores,
            dim=(n, self.height * self.width),
            inputs=[self.observed, rendered, self.height, self.width, self.filter_size, self.inv_two_var,
                    self.log_norm, self.log_inlier_prior, self.outlier_score],
            outputs=[out],
            device=self.device,
        )
        return out

    def score(self, rendered: wp.array) -> np.ndarray:
        """``rendered``: ``(N, H*W)`` vec3 world points (NaN = no hit). Returns ``(N,)`` float64 sums."""
        if not isinstance(rendered, wp.array):
            rendered = wp.array(np.ascontiguousarray(np.asarray(rendered, dtype=np.float32).reshape(-1, self.height * self.width, 3)),
                                dtype=wp.vec3f, device=self.device)
        pix = self.pixel_scores(rendered)
        rows = wp.empty((rendered.shape[0], self.height), dtype=wp.float64, device=self.device)
        wp.launch(_row_sums, dim=(rendered.shape[0], self.height), inputs=[pix, self.width], outputs=[rows],
                  device=self.device)
        return rows.numpy().sum(axis=1)

    def delta(self, rendered: wp.array) -> np.ndarray:
        """Score minus the perfect-match score (``score(observed, observed)``), as the old ``still()``."""
        return self.score(rendered) - self.baseline


def reference_score(observed: np.ndarray, rendered: np.ndarray, variance: float = 0.001, outlier_prob: float = 0.001,
                    outlier_volume: float = 1.0, filter_size: int = 3) -> float:
    """Plain numpy transcription of the JAX implementation (for tests)."""
    h, w, _ = observed.shape
    f = filter_size
    rend = np.where(np.isnan(rendered), SENTINEL, rendered)
    padded = np.full((h + 2 * f, w + 2 * f, 3), SENTINEL, dtype=np.float64)
    padded[f:f + h, f:f + w] = rend
    total = 0.0
    log_norm = 1.5 * np.log(2 * np.pi * variance) + np.log(h * w)
    outlier = np.log(outlier_prob) - np.log(outlier_volume)
    for i in range(h):
        for j in range(w):
            o = observed[i, j]
            if np.isnan(o).any():
                continue
            win = padded[i:i + 2 * f + 1, j:j + 2 * f + 1].reshape(-1, 3)
            lp = -((o - win) ** 2).sum(-1) / (2 * variance) - log_norm
            inlier = lp.max() + np.log(1 - outlier_prob)
            total += np.logaddexp(inlier, outlier)
    return float(total)
