"""
collision.py — Soft Collision & Boundary System  (UPGRADED — All-Time Best Edition)
════════════════════════════════════════════════════════════════════════════════════
Features:
  • Smooth exponential boundary repulsion (no hard resets)
  • Per-axis margin control
  • Optional spherical aquarium mode
  • Optimised vectorised NumPy operations
"""

import numpy as np


class CollisionSystem:
    """
    Applies soft boundary repulsion to boid position/velocity arrays.
    Works in both cubic (default) and spherical containment modes.
    """

    def __init__(self,
                 bounds:    float = 15.0,
                 margin:    float = 2.5,
                 strength:  float = 6.0,
                 spherical: bool  = False):
        self.bounds    = bounds
        self.margin    = margin
        self.strength  = strength
        self.spherical = spherical

    # ─── Public API ────────────────────────────────────────────────────────────

    def apply(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        if self.spherical:
            return self._apply_sphere(pos, vel, dt)
        return self._apply_cube(pos, vel, dt)

    # ─── Cubic containment ────────────────────────────────────────────────────

    def _apply_cube(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        B = self.bounds
        M = self.margin
        S = self.strength

        for axis in range(3):
            # ── Near–min wall ────────────────────────────────────────────────
            near_min  = pos[:, axis] < (-B + M)
            depth_min = (-B + M) - pos[near_min, axis]
            depth_min = np.clip(depth_min, 0, M)
            # Quadratic + exponential blend for smooth and forceful response
            force_min = (depth_min / M) ** 2 * S + np.exp(depth_min / (M * 0.4)) * S * 0.5
            vel[near_min, axis] += force_min * dt

            # ── Near–max wall ────────────────────────────────────────────────
            near_max  = pos[:, axis] > (B - M)
            depth_max = pos[near_max, axis] - (B - M)
            depth_max = np.clip(depth_max, 0, M)
            force_max = (depth_max / M) ** 2 * S + np.exp(depth_max / (M * 0.4)) * S * 0.5
            vel[near_max, axis] -= force_max * dt

        return vel

    # ─── Spherical containment ────────────────────────────────────────────────

    def _apply_sphere(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        R = self.bounds
        M = self.margin
        S = self.strength

        dist   = np.linalg.norm(pos, axis=1, keepdims=True) + 1e-6
        over   = dist > (R - M)
        if over.any():
            depth = dist - (R - M)
            depth = np.where(over, np.clip(depth / M, 0, 1), 0)
            inward = -pos / dist                           # toward center
            force  = depth ** 2 * S + np.exp(depth) * S * 0.3
            vel   += inward * force * dt

        return vel
