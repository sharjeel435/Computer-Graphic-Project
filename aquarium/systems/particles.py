"""
particles.py — GPU-Friendly Dual-Type Particle System  (UPGRADED — All-Time Best Edition)
══════════════════════════════════════════════════════════════════════════════════════════
Two particle layers:
  Type 0 — Bubbles  : ring-outline sprites that rise from the sea floor
  Type 1 — Sparkles : ember/star sprites spawned near bioluminescent fish

GPU upload via glBufferSubData each frame (minimal CPU→GPU traffic).
Layout per particle: x y z size alpha type  (6 floats = 24 bytes)
"""

import ctypes
import numpy as np
from OpenGL.GL import *


class ParticleSystem:
    """
    Manages two particle pools in a single interleaved VBO.
    Pool 0: bubbles   (count = max_bubbles)
    Pool 1: sparkles  (count = max_sparkles)
    """

    def __init__(self,
                 max_bubbles:  int = 2500,
                 max_sparkles: int = 800):
        self.max_b  = max_bubbles
        self.max_s  = max_sparkles
        self.total  = max_bubbles + max_sparkles
        self.active = True

        rng = np.random.default_rng(42)

        # ── Bubble state ──────────────────────────────────────────────────────
        self.b_pos  = np.column_stack([
            rng.uniform(-14, 14, max_bubbles),
            rng.uniform(-14,  0, max_bubbles),
            rng.uniform(-14, 14, max_bubbles),
        ]).astype(np.float32)
        self.b_vel      = np.zeros((max_bubbles, 3), dtype=np.float32)
        self.b_vel[:,1] = rng.uniform(0.6, 3.2, max_bubbles)   # rise speed
        self.b_vel[:,0] = rng.uniform(-0.25, 0.25, max_bubbles)
        self.b_vel[:,2] = rng.uniform(-0.25, 0.25, max_bubbles)
        self.b_life     = rng.uniform(0, 7.0, max_bubbles).astype(np.float32)
        self.b_maxlife  = rng.uniform(4.5, 9.0, max_bubbles).astype(np.float32)
        self.b_size     = rng.uniform(6.0, 28.0, max_bubbles).astype(np.float32)
        self.b_wobbleph = rng.uniform(0, 2*np.pi, max_bubbles).astype(np.float32)

        # ── Sparkle state ─────────────────────────────────────────────────────
        self.s_pos  = rng.uniform(-14, 14, (max_sparkles, 3)).astype(np.float32)
        self.s_vel  = (rng.standard_normal((max_sparkles, 3)) * 0.4).astype(np.float32)
        self.s_life = rng.uniform(0, 2.0, max_sparkles).astype(np.float32)
        self.s_maxlife = rng.uniform(0.8, 2.5, max_sparkles).astype(np.float32)
        self.s_size    = rng.uniform(4.0, 14.0, max_sparkles).astype(np.float32)

        # ── Build single VAO / VBO ────────────────────────────────────────────
        # Layout: x y z  size  alpha  type   (6 floats, stride = 24 bytes)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.total * 6 * 4,   # 6 floats × 4 bytes
                     None,
                     GL_DYNAMIC_DRAW)

        stride = 6 * 4   # 24 bytes
        # Attrib 0: position (3 floats, offset 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Attrib 1: size (1 float, offset 12)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        # Attrib 2: alpha (1 float, offset 16)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(2)
        # Attrib 3: type (1 float, offset 20)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(3)

        glBindVertexArray(0)

    # ─── Update ───────────────────────────────────────────────────────────────

    def update(self, dt: float, sparkle_origins=None):
        """
        sparkle_origins: (M×3) array of world positions where sparkles
                         should be re-spawned (bioluminescent fish positions).
        """
        if not self.active:
            return

        # ── Bubbles ───────────────────────────────────────────────────────────
        self.b_life += dt
        self.b_pos  += self.b_vel * dt

        # Horizontal sine wobble (each bubble has its own phase)
        self.b_pos[:, 0] += (np.sin(self.b_life * 2.1 + self.b_wobbleph) * 0.018).astype(np.float32)
        # Slight acceleration as bubbles rise (buoyancy)
        self.b_vel[:, 1] += 0.05 * dt

        # Clamp rise speed
        self.b_vel[:, 1] = np.minimum(self.b_vel[:, 1], 4.5)

        # Respawn dead bubbles
        dead_b = self.b_life >= self.b_maxlife
        if dead_b.any():
            nd = int(dead_b.sum())
            rng = np.random.default_rng()
            self.b_pos[dead_b]     = np.column_stack([
                rng.uniform(-14, 14, nd).astype(np.float32),
                np.full(nd, -13.5, dtype=np.float32),
                rng.uniform(-14, 14, nd).astype(np.float32),
            ])
            self.b_vel[dead_b, 0]  = rng.uniform(-0.25, 0.25, nd).astype(np.float32)
            self.b_vel[dead_b, 1]  = rng.uniform(0.60,  3.20, nd).astype(np.float32)
            self.b_vel[dead_b, 2]  = rng.uniform(-0.25, 0.25, nd).astype(np.float32)
            self.b_life[dead_b]    = 0.0
            self.b_maxlife[dead_b] = rng.uniform(4.5, 9.0, nd).astype(np.float32)
            self.b_size[dead_b]    = rng.uniform(6.0, 28.0, nd).astype(np.float32)
            self.b_wobbleph[dead_b]= rng.uniform(0, 2*np.pi, nd).astype(np.float32)

        # ── Sparkles ──────────────────────────────────────────────────────────
        self.s_life += dt
        self.s_pos  += self.s_vel * dt
        # Slow decay of sparkle velocity
        self.s_vel  *= (1.0 - 0.8 * dt)

        dead_s = self.s_life >= self.s_maxlife
        if dead_s.any():
            nd  = int(dead_s.sum())
            rng = np.random.default_rng()
            if sparkle_origins is not None and sparkle_origins.shape[0] > 0:
                # Respawn near a random bioluminescent fish
                chosen = sparkle_origins[rng.integers(0, sparkle_origins.shape[0], nd)]
                noise  = rng.standard_normal((nd, 3)).astype(np.float32) * 0.5
                self.s_pos[dead_s] = (chosen + noise).astype(np.float32)
            else:
                self.s_pos[dead_s] = rng.uniform(-14, 14, (nd, 3)).astype(np.float32)

            self.s_vel[dead_s]     = (rng.standard_normal((nd, 3)) * 0.35).astype(np.float32)
            self.s_life[dead_s]    = 0.0
            self.s_maxlife[dead_s] = rng.uniform(0.8, 2.5, nd).astype(np.float32)
            self.s_size[dead_s]    = rng.uniform(4.0, 14.0, nd).astype(np.float32)

        # ── Build interleaved upload buffer ──────────────────────────────────
        b_alpha  = np.clip(1.0 - (self.b_life / self.b_maxlife), 0.1, 1.0).astype(np.float32)
        s_alpha  = np.clip(1.0 - (self.s_life / self.s_maxlife), 0.0, 1.0).astype(np.float32)

        b_type   = np.zeros(self.max_b, dtype=np.float32)     # type 0 = bubble
        s_type   = np.ones (self.max_s, dtype=np.float32)     # type 1 = sparkle

        b_data   = np.column_stack([self.b_pos, self.b_size, b_alpha, b_type])
        s_data   = np.column_stack([self.s_pos, self.s_size, s_alpha, s_type])
        data     = np.vstack([b_data, s_data]).astype(np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)

    # ─── Draw ─────────────────────────────────────────────────────────────────

    def draw(self, shader, color=(0.55, 0.80, 1.0), night=False):
        if not self.active:
            return
        shader.use()
        shader.set_vec3("particleColor", color)
        shader.set_bool("nightMode",     night)

        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)      # additive for glow effect
        glDepthMask(GL_FALSE)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.total)
        glBindVertexArray(0)

        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

    def toggle(self):
        self.active = not self.active
        return self.active
