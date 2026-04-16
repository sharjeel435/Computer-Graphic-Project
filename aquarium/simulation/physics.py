"""
physics.py — Anti-Gravity Physics Engine  (UPGRADED — All-Time Best Edition)
═════════════════════════════════════════════════════════════════════════════
Environmental forces applied each frame:
  • Zero-gravity default (stochastic micro-drift only)
  • Optional standard gravity toggle (G key)
  • Vortex attractors  (V key spawns / clears)
  • Current streams    (directional flow zones)
  • Thermal columns    (upward buoyancy pockets)
  • Ocean swell        (slow sinusoidal global push)
"""

import numpy as np
import math


class AntiGravityPhysics:
    """
    Applies environmental forces to any position/velocity array pair.
    All operations are fully vectorised (NumPy) for O(N) cost.
    """

    def __init__(self, bounds: float = 15.0):
        self.bounds     = bounds
        self.gravity_on = False
        self.g_constant = 9.81

        # Dynamic features (modifiable at runtime)
        self.vortices      = []   # list of [center(3), strength, active]
        self.current_dir   = np.array([0.5, 0.0, 0.2], dtype=np.float32)
        self.current_str   = 0.4  # base current strength
        self.thermal_cols  = [    # (x, z, radius, strength)
            (-5.0, -5.0, 2.5, 1.2),
            ( 6.0,  4.0, 2.0, 1.0),
            (-2.0,  8.0, 2.0, 0.9),
        ]
        self.swell_amp  = 0.12    # global ocean-swell amplitude
        self.swell_freq = 0.15    # cycles/s

        # Runtime clock (for swell phase)
        self._t = 0.0

    # ─── Main apply ───────────────────────────────────────────────────────────

    def apply(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        self._t += dt
        N = pos.shape[0]

        # ── Optional Gravity ─────────────────────────────────────────────────
        if self.gravity_on:
            vel[:, 1] -= self.g_constant * dt

        # ── Micro-turbulence (anti-gravity chaos) ────────────────────────────
        turb = np.random.randn(N, 3).astype(np.float32) * 0.04 * dt
        vel += turb

        # ── Ocean Current (slow, space-varying) ──────────────────────────────
        # Varies with depth: stronger near surface
        depth_factor = np.clip((pos[:, 1] + self.bounds) / (2 * self.bounds),
                               0.1, 1.0)[:, np.newaxis]
        vel += (self.current_dir * self.current_str * dt) * depth_factor

        # ── Ocean Swell (global sinusoidal push) ─────────────────────────────
        swell = math.sin(self._t * self.swell_freq * 2 * math.pi) * self.swell_amp
        vel[:, 1] += swell * dt

        # ── Thermal Columns (upward buoyancy pockets) ─────────────────────────
        for tx, tz, tr, ts in self.thermal_cols:
            dx = pos[:, 0] - tx
            dz = pos[:, 2] - tz
            r  = np.sqrt(dx**2 + dz**2) + 1e-6
            inside = r < tr
            # Gaussian bell strength
            strength = ts * np.exp(-((r[inside] / tr) ** 2)) * dt
            vel[inside, 1] += strength

        # ── Vortex Attractors ────────────────────────────────────────────────
        for center, strength, active in self.vortices:
            if not active:
                continue
            to_c  = center - pos                        # N×3
            dist  = np.linalg.norm(to_c, axis=1, keepdims=True) + 1e-6
            # Tangential swirl component
            toward = to_c / dist
            swirl  = np.cross(toward, np.array([0, 1, 0], dtype=np.float32))
            vel   += (toward * 0.6 + swirl * 0.4) * strength * dt / dist

        return vel

    # ─── Runtime controls ─────────────────────────────────────────────────────

    def toggle_gravity(self) -> bool:
        self.gravity_on = not self.gravity_on
        return self.gravity_on

    def add_vortex(self, center: np.ndarray, strength: float = 3.0):
        """Spawn a new vortex at the given world position."""
        self.vortices.append([center.copy().astype(np.float32), strength, True])

    def clear_vortices(self):
        self.vortices.clear()

    def reverse_current(self):
        """Flip the global current direction."""
        self.current_dir *= -1.0

    def set_current_strength(self, s: float):
        self.current_str = float(s)
