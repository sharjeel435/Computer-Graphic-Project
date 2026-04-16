"""
camera.py — 6-DOF First-Person Camera Controller  (UPGRADED — All-Time Best Edition)
══════════════════════════════════════════════════════════════════════════════════════
Features:
  • WASD + QE free-fly movement
  • LShift speed boost
  • Mouse look with configurable sensitivity
  • Smooth momentum / inertia (velocity lerp)
  • FOV zoom in/out (Z key or scroll)
  • Camera shake support (triggered by external events, e.g. vortex spawn)
  • View / Projection matrix generation (pure NumPy, no GLU)
"""

import numpy as np
import pygame
import math


class Camera:
    def __init__(self,
                 width:    int,
                 height:   int,
                 position: tuple = (0.0, 0.0, 14.0),
                 yaw:      float = -90.0,
                 pitch:    float = -8.0):

        self.pos   = np.array(position, dtype=np.float64)
        self.yaw   = yaw
        self.pitch = pitch

        # Optics
        self.fov_default = 60.0
        self.fov         = self.fov_default
        self.fov_target  = self.fov_default
        self.near        = 0.1
        self.far         = 250.0

        # Movement
        self.speed_normal = 7.0
        self.speed_boost  = 20.0
        self.speed        = self.speed_normal
        self.sensitivity  = 0.11

        # Smooth momentum
        self._vel      = np.zeros(3, dtype=np.float64)
        self._momentum = 0.12    # 0 = no inertia, 1 = infinite glide

        # Camera shake
        self._shake_amt     = 0.0
        self._shake_decay   = 4.0   # per second

        self.width  = width
        self.height = height
        self._update_vectors()

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _update_vectors(self):
        yr = math.radians(self.yaw)
        pr = math.radians(self.pitch)
        f  = np.array([
            math.cos(yr) * math.cos(pr),
            math.sin(pr),
            math.sin(yr) * math.cos(pr),
        ], dtype=np.float64)
        self.front  = f / np.linalg.norm(f)
        world_up    = np.array([0.0, 1.0, 0.0])
        self.right  = np.cross(self.front, world_up)
        self.right /= np.linalg.norm(self.right)
        self.up     = np.cross(self.right, self.front)
        self.up    /= np.linalg.norm(self.up)

    # ─── Public API ───────────────────────────────────────────────────────────

    def process_keyboard(self, dt: float, keys):
        boost  = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        self.speed = self.speed_boost if boost else self.speed_normal

        intent = np.zeros(3, dtype=np.float64)
        if keys[pygame.K_w]: intent += self.front
        if keys[pygame.K_s]: intent -= self.front
        if keys[pygame.K_a]: intent -= self.right
        if keys[pygame.K_d]: intent += self.right
        if keys[pygame.K_q]: intent += self.up
        if keys[pygame.K_e]: intent -= self.up

        if np.linalg.norm(intent) > 0:
            intent /= np.linalg.norm(intent)

        target_vel = intent * self.speed

        # Smooth momentum (lerp velocity toward target)
        alpha = 1.0 - self._momentum
        self._vel = self._vel + (target_vel - self._vel) * min(alpha * dt * 12.0, 1.0)

        # Camera shake offset
        shake = np.zeros(3)
        if self._shake_amt > 0.001:
            shake = (np.random.randn(3) * self._shake_amt).astype(np.float64)
            self._shake_amt -= self._shake_decay * dt
            self._shake_amt  = max(0.0, self._shake_amt)

        self.pos += (self._vel + shake * 0.3) * dt

    def process_mouse(self, dx: float, dy: float):
        self.yaw   += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        self.pitch  = max(-89.0, min(89.0, self.pitch))
        self._update_vectors()

    def process_scroll(self, dy: float):
        """Zoom FOV in/out with mouse scroll."""
        self.fov_target = max(20.0, min(100.0, self.fov_target - dy * 3.0))

    def update(self, dt: float):
        """Call each frame to smooth FOV transitions."""
        self.fov += (self.fov_target - self.fov) * min(dt * 8.0, 1.0)

    def set_zoom(self, zoomed: bool):
        self.fov_target = 28.0 if zoomed else self.fov_default

    def add_shake(self, intensity: float = 0.3):
        self._shake_amt = min(self._shake_amt + intensity, 1.0)

    def reset(self):
        self.pos[:] = [0.0, 0.0, 14.0]
        self.yaw   = -90.0
        self.pitch = -8.0
        self._vel[:]  = 0.0
        self._shake_amt = 0.0
        self.fov_target = self.fov_default
        self._update_vectors()

    def resize(self, width: int, height: int):
        self.width  = width
        self.height = height

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.pos, self.pos + self.front, self.up)

    def projection_matrix(self) -> np.ndarray:
        aspect = self.width / max(self.height, 1)
        return _perspective(math.radians(self.fov), aspect, self.near, self.far)


# ─── Math Helpers ─────────────────────────────────────────────────────────────

def _look_at(eye, center, up) -> np.ndarray:
    f = center - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.identity(4, dtype=np.float32)
    M[0, :3] =  r
    M[1, :3] =  u
    M[2, :3] = -f
    T = np.identity(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T


def _perspective(fov_rad: float, aspect: float,
                 near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_rad * 0.5)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] =  f / aspect
    M[1, 1] =  f
    M[2, 2] =  (far + near) / (near - far)
    M[2, 3] =  (2.0 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M
