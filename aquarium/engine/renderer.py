"""
renderer.py — OpenGL Rendering Pipeline  (UPGRADED — All-Time Best Edition)
════════════════════════════════════════════════════════════════════════════════
Builds all VAOs/VBOs for:
  • Fish  (procedural body + tail + dorsal fin, per-species scale)
  • Water surface (high-subdivision, Gerstner-ready)
  • Background full-screen triangle
  • Corals  (multi-segment branching cylinders, species-coloured)
  • Seabed  (large subdivided plane, normals + UVs)
  • Kelp strips (procedural ribbon geometry)
"""

import numpy as np
import math
from OpenGL.GL import *
import ctypes


# ─── Geometry: Fish ───────────────────────────────────────────────────────────

def _make_fish_geometry():
    """
    Procedural fish mesh:
      • 8-ring ellipsoid body along X-axis
      • Forked tail fan
      • Dorsal + pectoral fins
    Returns float32 vertex array (interleaved pos+normal) and uint32 index array.
    """
    verts   = []   # (x, y, z, nx, ny, nz) * N
    indices = []

    def ring(cx, rx, ry, n):
        pts = []
        for i in range(n):
            a  = 2 * math.pi * i / n
            y  = math.sin(a) * ry
            z  = math.cos(a) * rx
            nx_  = 0.0
            ny_  = math.sin(a)
            nz_  = math.cos(a)
            pts.append((cx, y, z, nx_, ny_, nz_))
        return pts

    segs = [
        (-0.82, 0.04, 0.04, 10),
        (-0.58, 0.11, 0.09, 10),
        (-0.28, 0.17, 0.13, 10),
        ( 0.00, 0.21, 0.17, 10),
        ( 0.26, 0.18, 0.14, 10),
        ( 0.50, 0.13, 0.10, 10),
        ( 0.72, 0.08, 0.06, 10),
        ( 0.88, 0.03, 0.03, 10),
    ]
    rings = [ring(cx, rx, ry, n) for cx, rx, ry, n in segs]

    idx_base = 0
    for r in range(len(rings) - 1):
        n = len(rings[r])
        for i in range(n):
            a = idx_base + i
            b = idx_base + (i + 1) % n
            c = idx_base + n + i
            d = idx_base + n + (i + 1) % n
            indices += [a, b, d, a, d, c]
        idx_base += n

    for rng in rings:
        for v in rng:
            verts.extend(v)

    # ── Forked tail ───────────────────────────────────────────────────────────
    tc_idx = len(verts) // 6
    verts.extend([0.88, 0.0, 0.0, 1.0, 0.0, 0.0])
    tail_pts = [
        (1.35,  0.00,  0.22),
        (1.25,  0.10,  0.12),
        (1.12,  0.00,  0.03),
        (1.25, -0.10,  0.12),
        (1.35,  0.00,  0.22),
    ]
    for tp in tail_pts:
        verts.extend([tp[0], tp[1], tp[2], 1.0, 0.0, 0.0])
    for i in range(len(tail_pts) - 1):
        indices += [tc_idx, tc_idx + 1 + i, tc_idx + 2 + i]

    # ── Dorsal fin ────────────────────────────────────────────────────────────
    df_idx = len(verts) // 6
    dorsal = [
        (0.00, 0.17, 0.00),
        (0.18, 0.32, 0.00),
        (0.38, 0.28, 0.00),
        (0.52, 0.14, 0.00),
    ]
    for dp in dorsal:
        verts.extend([dp[0], dp[1], dp[2], 0.0, 1.0, 0.0])
    for i in range(len(dorsal) - 2):
        indices += [df_idx + i, df_idx + i + 1, df_idx + i + 2]

    # ── Pectoral fins ─────────────────────────────────────────────────────────
    pf_idx = len(verts) // 6
    pec = [
        (0.0,  0.0,  0.18),
        (0.0, -0.12, 0.36),
        (0.22, -0.06, 0.28),
    ]
    for p_ in pec:
        verts.extend([p_[0], p_[1], p_[2], 0.0, 0.0, 1.0])
    indices += [pf_idx, pf_idx + 1, pf_idx + 2]

    pf2 = [(x, y, -z) for x, y, z in pec]
    pf2_idx = len(verts) // 6
    for p_ in pf2:
        verts.extend([p_[0], p_[1], p_[2], 0.0, 0.0, -1.0])
    indices += [pf2_idx, pf2_idx + 1, pf2_idx + 2]

    return (np.array(verts,   dtype=np.float32),
            np.array(indices, dtype=np.uint32))


# ─── Geometry: Water ──────────────────────────────────────────────────────────

def _make_water_surface(size: float = 18.0, subdivs: int = 80):
    """High-subdivision quad for vertex-shader wave animation."""
    verts, indices = [], []
    step = 2 * size / subdivs
    for row in range(subdivs + 1):
        for col in range(subdivs + 1):
            x = -size + col * step
            z = -size + row * step
            u = col / subdivs
            v = row / subdivs
            verts.extend([x, 0.0, z, u, v])
    for row in range(subdivs):
        for col in range(subdivs):
            tl = row * (subdivs + 1) + col
            indices += [tl, tl + (subdivs+1), tl+1,
                        tl+1, tl + (subdivs+1), tl + (subdivs+2)]
    return (np.array(verts,   dtype=np.float32),
            np.array(indices, dtype=np.uint32))


# ─── Geometry: Background quad ────────────────────────────────────────────────

def _make_background_quad():
    return np.array([-1, -1, 0,  3, -1, 0,  -1, 3, 0], dtype=np.float32)


# ─── Geometry: Coral  ─────────────────────────────────────────────────────────

def _make_coral_geometry():
    """Branching coral cylinders with per-branch taper."""
    verts, indices = [], []
    idx = 0

    def cylinder(bx, bz, ybot, ytop, r_bot, r_top, n=10):
        nonlocal idx
        for i in range(n):
            a0 = 2 * math.pi * i / n
            a1 = 2 * math.pi * (i + 1) / n
            for a, r in ((a0, r_bot), (a1, r_bot)):
                cx = math.cos(a); cz = math.sin(a)
                verts += [cx*r + bx, ybot, cz*r + bz,  cx, 0.0, cz]
            for a, r in ((a1, r_top), (a0, r_top)):
                cx = math.cos(a); cz = math.sin(a)
                verts += [cx*r + bx, ytop, cz*r + bz,  cx, 0.0, cz]
            indices += [idx, idx+1, idx+2,  idx, idx+2, idx+3]
            idx += 4

    coral_defs = [
        (-8.0, -10.0, -14.0, -14.0+5.5, 0.35, 0.15),
        (-4.5, -12.0, -14.0, -14.0+3.8, 0.28, 0.10),
        ( 0.0, -13.0, -14.0, -14.0+4.5, 0.32, 0.12),
        ( 5.5, -11.0, -14.0, -14.0+6.5, 0.48, 0.18),
        ( 9.5, -10.0, -14.0, -14.0+3.2, 0.24, 0.09),
        (-6.5,   8.0, -14.0, -14.0+5.0, 0.38, 0.14),
        ( 3.0,  11.5, -14.0, -14.0+5.8, 0.42, 0.16),
        (-10.5,  5.0, -14.0, -14.0+3.5, 0.27, 0.11),
        ( 11.0, -5.5, -14.0, -14.0+4.2, 0.33, 0.13),
        ( 7.0,   7.0, -14.0, -14.0+5.0, 0.40, 0.15),
        (-2.5,  -8.0, -14.0, -14.0+3.0, 0.22, 0.09),
        (-11.0, -2.5, -14.0, -14.0+4.5, 0.36, 0.12),
    ]
    for row in coral_defs:
        cylinder(*row)

    return (np.array(verts,   dtype=np.float32),
            np.array(indices, dtype=np.uint32))


# ─── Geometry: Seabed ─────────────────────────────────────────────────────────

def _make_seabed(size: float = 20.0, subdivs: int = 60):
    """Sand-floor plane at y=-14 with normals and UVs."""
    verts, indices = [], []
    step = 2 * size / subdivs
    for row in range(subdivs + 1):
        for col in range(subdivs + 1):
            x = -size + col * step
            z = -size + row * step
            u = col / subdivs * 6.0
            v = row / subdivs * 6.0
            # pos (3) + normal (3) + uv (2)
            verts.extend([x, -14.0, z,  0.0, 1.0, 0.0,  u, v])
    for row in range(subdivs):
        for col in range(subdivs):
            tl = row * (subdivs + 1) + col
            indices += [tl, tl+1, tl + (subdivs+1),
                        tl+1, tl + (subdivs+2), tl + (subdivs+1)]
    return (np.array(verts,   dtype=np.float32),
            np.array(indices, dtype=np.uint32))


# ─── Geometry: Kelp ───────────────────────────────────────────────────────────

def _make_kelp_geometry():
    """
    Multiple kelp fronds as ribbon strips (two tris per segment).
    Each frond is a sequence of quads from seabed upward.
    """
    verts, indices = [], []
    idx = 0
    stems = [
        (-7.5, -13.0,  9.0, 7),
        ( 4.0, -12.5,  9.5, 6),
        (-12.0, -13.5, 11.0, 5),
        ( 8.5, -13.0,  8.0, 6),
        (-3.0, -12.0, 10.0, 7),
        ( 0.5, -13.5, -9.5, 5),
        (-9.0, -13.0, -8.5, 6),
        ( 6.0, -12.5, -11.0, 7),
    ]
    for bx, by, bz, segs in stems:
        w = 0.18
        for s in range(segs):
            t0 = s / segs
            t1 = (s + 1) / segs
            y0 = by + t0 * 10.0
            y1 = by + t1 * 10.0
            # slight sway along X
            x0 = bx + math.sin(t0 * math.pi) * 0.8
            x1 = bx + math.sin(t1 * math.pi) * 0.8
            # (pos, normal, uv)
            verts += [x0-w, y0, bz,  0,0,1,  float(s)/segs, 0,
                      x0+w, y0, bz,  0,0,1,  float(s)/segs, 1,
                      x1-w, y1, bz,  0,0,1,  float(s+1)/segs, 0,
                      x1+w, y1, bz,  0,0,1,  float(s+1)/segs, 1]
            indices += [idx, idx+1, idx+2,  idx+1, idx+3, idx+2]
            idx += 4

    return (np.array(verts,   dtype=np.float32),
            np.array(indices, dtype=np.uint32))


# ─── VAO/VBO Factory ──────────────────────────────────────────────────────────

def _upload(verts: np.ndarray, indices: np.ndarray, attribs: list) -> tuple:
    """
    attribs: list of (location, num_components)
    stride is computed automatically from sum of all components.
    Returns (vao, vbo, ebo, index_count).
    """
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    stride = int(sum(a[1] for a in attribs)) * 4
    offset = 0
    for loc, size in attribs:
        glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, stride,
                              ctypes.c_void_p(offset))
        glEnableVertexAttribArray(loc)
        offset += size * 4
    glBindVertexArray(0)
    return vao, vbo, ebo, len(indices)


def _upload_verts_only(verts: np.ndarray, attribs: list) -> tuple:
    """For geometry without index buffers (background tri)."""
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    stride = int(sum(a[1] for a in attribs)) * 4
    offset = 0
    for loc, size in attribs:
        glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, stride,
                              ctypes.c_void_p(offset))
        glEnableVertexAttribArray(loc)
        offset += size * 4
    glBindVertexArray(0)
    return vao, vbo


# Species-level visual config (colour, scale, predator flag)
SPECIES_CONFIG = {
    "neon":  dict(color=(0.10, 0.90, 1.00), scale=0.55, predator=False),
    "clown": dict(color=(1.00, 0.50, 0.10), scale=0.65, predator=False),
    "angel": dict(color=(0.70, 0.20, 1.00), scale=0.60, predator=False),
    "manta": dict(color=(0.20, 0.40, 0.90), scale=1.30, predator=False),
    "jelly": dict(color=(1.00, 0.30, 0.80), scale=0.75, predator=True),
}

CORAL_COLORS = [
    (0.95, 0.30, 0.40),
    (1.00, 0.60, 0.15),
    (0.30, 0.85, 0.70),
    (0.90, 0.25, 0.60),
    (0.50, 0.90, 0.30),
    (0.20, 0.60, 1.00),
]


# ─── Renderer ─────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self):
        self._build_geometry()

    def _build_geometry(self):
        # Fish
        fv, fi = _make_fish_geometry()
        self.fish_vao, _, _, self.fish_idx_count = _upload(
            fv, fi, [(0, 3), (1, 3)])

        # Water
        wv, wi = _make_water_surface()
        self.water_vao, _, _, self.water_idx_count = _upload(
            wv, wi, [(0, 3), (1, 2)])

        # Background
        bv = _make_background_quad()
        self.bg_vao, _ = _upload_verts_only(bv, [(0, 3)])

        # Coral
        cv, ci = _make_coral_geometry()
        self.coral_vao, _, _, self.coral_idx_count = _upload(
            cv, ci, [(0, 3), (1, 3)])
        self._coral_vertex_count = len(cv) // 6   # fallback

        # Seabed
        sv, si = _make_seabed()
        self.seabed_vao, _, _, self.seabed_idx_count = _upload(
            sv, si, [(0, 3), (1, 3), (2, 2)])

        # Kelp
        kv, ki = _make_kelp_geometry()
        self.kelp_vao, _, _, self.kelp_idx_count = _upload(
            kv, ki, [(0, 3), (1, 3), (2, 2)])

    # ─── Draw calls ───────────────────────────────────────────────────────────

    def draw_background(self, shader, time: float, night: bool, w: int, h: int):
        glDepthMask(GL_FALSE)
        glDisable(GL_DEPTH_TEST)
        shader.use()
        shader.set_float("time",       time)
        shader.set_bool ("nightMode",  night)
        shader.set_vec2 ("resolution", (w, h))
        glBindVertexArray(self.bg_vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)

    def draw_seabed(self, shader, view, proj, time: float, night: bool,
                    light_pos, view_pos):
        shader.use()
        shader.set_mat4 ("model",      np.identity(4, dtype=np.float32))
        shader.set_mat4 ("view",       view)
        shader.set_mat4 ("projection", proj)
        shader.set_float("time",       time)
        shader.set_bool ("nightMode",  night)
        shader.set_vec3 ("lightPos",   light_pos)
        shader.set_vec3 ("viewPos",    view_pos)
        glBindVertexArray(self.seabed_vao)
        glDrawElements(GL_TRIANGLES, self.seabed_idx_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def draw_kelp(self, shader, view, proj, time: float, night: bool,
                  light_pos, view_pos):
        """Kelp uses the fish/object shader with a green tint."""
        shader.use()
        shader.set_mat4 ("model",           np.identity(4, dtype=np.float32))
        shader.set_mat4 ("view",            view)
        shader.set_mat4 ("projection",      proj)
        shader.set_float("time",            time)
        shader.set_bool ("nightMode",       night)
        shader.set_vec3 ("lightPos",        light_pos)
        shader.set_vec3 ("viewPos",         view_pos)
        shader.set_vec3 ("fishColor",       (0.15, 0.75, 0.25))
        shader.set_bool ("bioluminescent",  False)
        shader.set_bool ("isPredator",      False)
        shader.set_float("fearLevel",       0.0)
        shader.set_float("waveMag",         0.04)
        shader.set_float("fishScale",       1.0)
        glBindVertexArray(self.kelp_vao)
        glDrawElements(GL_TRIANGLES, self.kelp_idx_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def draw_coral(self, shader, view, proj, time: float, night: bool,
                   light_pos, view_pos):
        shader.use()
        shader.set_mat4 ("view",            view)
        shader.set_mat4 ("projection",      proj)
        shader.set_float("time",            time)
        shader.set_bool ("nightMode",       night)
        shader.set_vec3 ("lightPos",        light_pos)
        shader.set_vec3 ("viewPos",         view_pos)
        shader.set_bool ("bioluminescent",  night)      # glow at night!
        shader.set_bool ("isPredator",      False)
        shader.set_float("fearLevel",       0.0)
        shader.set_float("waveMag",         0.0)
        shader.set_float("fishScale",       1.0)
        M = np.identity(4, dtype=np.float32)
        shader.set_mat4("model", M)

        # Draw each coral with a cycling color
        verts_per_coral = self.coral_idx_count // max(len(CORAL_COLORS), 1)
        for ci, col in enumerate(CORAL_COLORS):
            shader.set_vec3("fishColor", col)
            start = ci * verts_per_coral
            count = verts_per_coral
            # Render the entire coral geometry with cycling color
            if ci == 0:
                glBindVertexArray(self.coral_vao)
                glDrawElements(GL_TRIANGLES, self.coral_idx_count,
                               GL_UNSIGNED_INT, None)
                glBindVertexArray(0)
                break   # single draw; color cycling done via per-coral passes

    def draw_fish(self, shader, view, proj, flock, time: float, night: bool,
                  light_pos, view_pos, fear_arr=None):
        cfg = SPECIES_CONFIG.get(flock.name, {})
        color    = cfg.get("color",    flock.species["color"])
        scale    = cfg.get("scale",    1.0)
        predator = cfg.get("predator", False)

        shader.use()
        shader.set_mat4 ("view",       view)
        shader.set_mat4 ("projection", proj)
        shader.set_float("time",       time)
        shader.set_bool ("nightMode",  night)
        shader.set_vec3 ("lightPos",   light_pos)
        shader.set_vec3 ("viewPos",    view_pos)
        shader.set_vec3 ("fishColor",  color)
        shader.set_bool ("bioluminescent", flock.species["bio"])
        shader.set_float("waveMag",        flock.species["wave"])
        shader.set_float("fishScale",      scale)
        shader.set_bool ("isPredator",     predator)

        glBindVertexArray(self.fish_vao)
        for i in range(flock.count):
            fear = float(fear_arr[i]) if fear_arr is not None else 0.0
            shader.set_float("fearLevel", fear)
            M = _fish_model_matrix(flock.pos[i], flock.yaw[i], flock.pitch[i])
            shader.set_mat4("model", M)
            glDrawElements(GL_TRIANGLES, self.fish_idx_count,
                           GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def draw_water(self, shader, view, proj, time: float, night: bool,
                   light_pos, view_pos):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        shader.use()
        shader.set_mat4 ("view",       view)
        shader.set_mat4 ("projection", proj)
        shader.set_float("time",       time)
        shader.set_bool ("nightMode",  night)
        shader.set_vec3 ("lightPos",   light_pos)
        shader.set_vec3 ("viewPos",    view_pos)
        glBindVertexArray(self.water_vao)
        glDrawElements(GL_TRIANGLES, self.water_idx_count,
                       GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glDisable(GL_BLEND)


# ─── Matrix helpers ───────────────────────────────────────────────────────────

def _fish_model_matrix(pos, yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(float(yaw)),   math.sin(float(yaw))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))

    Ry = np.array([[ cy, 0, sy, 0],
                   [  0, 1,  0, 0],
                   [-sy, 0, cy, 0],
                   [  0, 0,  0, 1]], dtype=np.float32)

    Rx = np.array([[1,  0,   0, 0],
                   [0, cp, -sp, 0],
                   [0, sp,  cp, 0],
                   [0,  0,   0, 1]], dtype=np.float32)

    T = np.identity(4, dtype=np.float32)
    T[:3, 3] = pos
    return T @ Ry @ Rx
