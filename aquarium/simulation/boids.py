"""
boids.py — Swarm Intelligence  (UPGRADED — All-Time Best Edition)
══════════════════════════════════════════════════════════════════
Craig Reynolds' Boids algorithm with:
  • Separation / Alignment / Cohesion  (vectorised NumPy)
  • Predator-prey fear response (jelly predator → neon/clown flee)
  • Speed burst when frightened
  • Altitude comfort zone (species prefer certain Y bands)
  • Smooth velocity interpolation (inertia damping)
"""

import numpy as np

# ─── Species Profiles ────────────────────────────────────────────────────────

SPECIES = {
    # name:  color (RGB), max_speed, boid_radius, weights (sep/ali/coh),
    #        wave magnitude, bioluminescent, preferred Y range, is_predator
    "silver": dict(color=(0.75, 0.85, 0.95), speed=3.2, radius=2.5,
                   w_sep=1.8, w_ali=2.2, w_coh=1.9,
                   wave=0.06, bio=False,
                   y_band=(-5.0, 10.0), predator=False),

    "neon":  dict(color=(0.10, 0.90, 1.00), speed=3.8, radius=3.0,
                  w_sep=1.9, w_ali=1.3, w_coh=1.0,
                  wave=0.08, bio=True,
                  y_band=(-4.0, 8.0), predator=False),

    "clown": dict(color=(1.00, 0.50, 0.10), speed=2.9, radius=3.5,
                  w_sep=1.6, w_ali=1.1, w_coh=1.3,
                  wave=0.06, bio=False,
                  y_band=(-8.0, 2.0), predator=False),

    "angel": dict(color=(0.70, 0.20, 1.00), speed=4.4, radius=2.6,
                  w_sep=2.1, w_ali=1.6, w_coh=0.8,
                  wave=0.12, bio=True,
                  y_band=(-2.0, 10.0), predator=False),

    "manta": dict(color=(0.20, 0.40, 0.90), speed=2.1, radius=5.5,
                  w_sep=1.0, w_ali=0.9, w_coh=1.6,
                  wave=0.04, bio=False,
                  y_band=( 2.0, 12.0), predator=False),

    "jelly": dict(color=(1.00, 0.30, 0.80), speed=1.6, radius=2.2,
                  w_sep=2.6, w_ali=0.5, w_coh=0.6,
                  wave=0.22, bio=True,
                  y_band=(-6.0, 6.0),  predator=True),
}

# Prey species that flee from jellyfish
PREY_OF_JELLY = {"neon", "clown"}
FEAR_RADIUS    = 6.0     # distance within which prey detects predator
FEAR_ACCEL     = 6.0     # flee force magnitude


class BoidFlock:
    """One flock of a single species, fully vectorised simulation."""

    def __init__(self, species_name: str, count: int, bounds: float = 15.0):
        sp = SPECIES[species_name]
        self.species  = sp
        self.name     = species_name
        self.count    = count
        self.bounds   = bounds

        rng = np.random.default_rng()
        self.pos = rng.uniform(-bounds * 0.75, bounds * 0.75,
                               (count, 3)).astype(np.float32)
        # Restrict initial Y to preferred band
        ylo, yhi = sp["y_band"]
        self.pos[:, 1] = rng.uniform(ylo, yhi, count).astype(np.float32)

        d = rng.standard_normal((count, 3)).astype(np.float32)
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
        self.vel = d * sp["speed"]

        self.yaw   = np.zeros(count, dtype=np.float32)
        self.pitch = np.zeros(count, dtype=np.float32)

        # Per-boid fear level 0..1 (updated externally)
        self.fear  = np.zeros(count, dtype=np.float32)

    # ─── Internal step ────────────────────────────────────────────────────────

    def step(self, dt: float):
        pos  = self.pos
        vel  = self.vel
        r    = self.species["radius"]
        ws   = self.species["w_sep"]
        wa   = self.species["w_ali"]
        wc   = self.species["w_coh"]
        spd  = self.species["speed"]
        ylo, yhi = self.species["y_band"]
        N    = self.count

        # Pairwise differences (N×N×3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2) + 1e-6

        mask     = (dist < r)     & (dist > 0)
        sep_mask = (dist < r*0.5) & (dist > 0)

        # ── Boids rules ─────────────────────────────────────────────────────
        sep_force = np.where(sep_mask[..., np.newaxis],
                             diff / (dist[..., np.newaxis] ** 2 + 1e-9),
                             0.0).sum(axis=1)

        n_count   = mask.sum(axis=1, keepdims=True).astype(np.float32) + 1e-6
        ali_avg   = np.where(mask[..., np.newaxis],
                             vel[np.newaxis, :, :], 0.0).sum(axis=1)
        ali_force = ali_avg / n_count - vel

        coh_avg   = np.where(mask[..., np.newaxis],
                             pos[np.newaxis, :, :], 0.0).sum(axis=1)
        coh_force = coh_avg / n_count - pos

        # ── Anti-gravity stochastic drift ────────────────────────────────────
        drift = np.random.randn(N, 3).astype(np.float32) * 0.12

        # ── Altitude comfort zone (soft spring toward preferred Y) ────────────
        y_mid  = (ylo + yhi) * 0.5
        y_err  = y_mid - pos[:, 1]
        alt_force = np.zeros((N, 3), dtype=np.float32)
        alt_force[:, 1] = y_err * 0.40

        # ── Fear flee force (set from outside if near predator) ──────────────
        fear_force = np.zeros((N, 3), dtype=np.float32)
        if self.fear.any():
            # Flee away from group centre as a proxy; direction updated by main
            fear_force = -vel * self.fear[:, np.newaxis] * 1.5

        # ── Combine ──────────────────────────────────────────────────────────
        accel = (ws * sep_force
                 + wa * ali_force
                 + wc * coh_force
                 + drift
                 + alt_force
                 + fear_force)

        # Velocity update with inertia
        target_vel = vel + accel * dt
        vel        = vel + (target_vel - vel) * min(dt * 6.0, 1.0)

        # Speed clamp (frightened fish sprint!)
        max_spd = spd * (1.8 + self.fear.max() * 1.5)
        speeds  = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-6
        vel     = vel / speeds * np.clip(speeds, spd * 0.35, max_spd)

        # ── Soft boundary repulsion ──────────────────────────────────────────
        for axis in range(3):
            near_min = pos[:, axis] < (-self.bounds + 2.0)
            near_max = pos[:, axis] > ( self.bounds - 2.0)
            vel[near_min, axis] += 3.0 * dt * np.exp(
                (-self.bounds + 2.0 - pos[near_min, axis]) / 1.5)
            vel[near_max, axis] -= 3.0 * dt * np.exp(
                (pos[near_max, axis] - (self.bounds - 2.0)) / 1.5)

        pos = np.clip(pos + vel * dt, -self.bounds, self.bounds)

        self.vel = vel.astype(np.float32)
        self.pos = pos.astype(np.float32)

        # Orientation for model matrix
        self.yaw   = np.arctan2(vel[:, 0], vel[:, 2]).astype(np.float32)
        self.pitch = np.arcsin(
            np.clip(vel[:, 1] / (speeds[:, 0] + 1e-6), -1, 1)
        ).astype(np.float32)

        # Decay fear over time
        self.fear = np.maximum(self.fear - dt * 0.8, 0.0).astype(np.float32)

    def apply_fear(self, predator_positions: np.ndarray):
        """
        Given the positions of predators (M×3), compute per-boid fear level.
        Called once per frame from the main loop for prey flocks.
        """
        if predator_positions.shape[0] == 0:
            return
        # Min distance from each boid to any predator
        # pos: N×3, pred: M×3  →  diff: N×M×3
        diff = self.pos[:, np.newaxis, :] - predator_positions[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2).min(axis=1)  # N,
        fear_raw = np.clip(1.0 - dist / FEAR_RADIUS, 0.0, 1.0)

        # Also add a flee force (push away from nearest predator direction)
        nearest_idx = np.linalg.norm(
            diff, axis=2).argmin(axis=1)                  # N, index into M
        flee_dir = self.pos - predator_positions[nearest_idx]  # N×3
        norms    = np.linalg.norm(flee_dir, axis=1, keepdims=True) + 1e-6
        flee_dir = flee_dir / norms
        self.vel += flee_dir * fear_raw[:, np.newaxis] * FEAR_ACCEL * 0.016

        self.fear = np.maximum(self.fear, fear_raw.astype(np.float32))
