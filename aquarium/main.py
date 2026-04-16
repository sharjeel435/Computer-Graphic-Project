"""
main.py — Anti-Gravity Virtual Aquarium Engine  (ALL-TIME BEST EDITION)
════════════════════════════════════════════════════════════════════════
Author  : Sharjeel Safdar
Course  : Computer Graphics (CG) Final Project
Engine  : Python + PyOpenGL + GLSL + Pygame

Features:
  ✔ Anti-Gravity Physics Engine (currents, thermals, swells, vortex)
  ✔ Boids Swarm AI — 5 species with predator-prey fear response
  ✔ GPU GLSL Shaders — fish, water (Fresnel+caustics), background,
                        seabed (sand ripple+caustics), particles
  ✔ Dual Particle System (bubbles + bioluminescent sparkles)
  ✔ Real-Time Blinn-Phong + Bioluminescent + Chromatic glow lighting
  ✔ 6-DOF Free-Look Camera with smooth momentum + FOV zoom
  ✔ Procedural Kelp / Seaweed geometry
  ✔ Textured Sand Seabed with projected caustics
  ✔ Multi-colored Branching Coral Colony
  ✔ Volumetric God-Rays  /  Animated Nebula Night Sky + Moon
  ✔ Day / Night Mode  (N)
  ✔ Gravity toggle  (G)
  ✔ Bubble toggle   (B)
  ✔ Vortex spawn    (V)  — camera shake on spawn
  ✔ Zoom / Focus    (Z)
  ✔ Pause           (P)
  ✔ Camera reset    (R)
  ✔ Help overlay    (H)
  ✔ Adaptive HUD (FPS, fish count, mode, controls, species legend)
"""

import sys
import math
import time as _time

import pygame
from pygame.locals import *
from OpenGL.GL import *

import numpy as np

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, ".")

from engine.shader    import ShaderLibrary
from engine.camera    import Camera
from engine.renderer  import Renderer
from simulation.boids  import BoidFlock, PREY_OF_JELLY
from simulation.physics import AntiGravityPhysics
from systems.particles  import ParticleSystem
from systems.collision  import CollisionSystem


# ─── Constants ────────────────────────────────────────────────────────────────

WIN_W, WIN_H  = 1280, 720
TITLE         = "🌌 Anti-Gravity Virtual Aquarium  |  Sharjeel Safdar  |  CG Final"
TARGET_FPS    = 60
BOUNDS        = 15.0
VERSION       = "2.0 — All-Time Best Edition"

FLOCK_CONFIG = [
    ("neon",  18),
    ("clown", 14),
    ("angel", 12),
    ("manta",  8),
    ("jelly", 10),
]

# ─── Splash Screen ────────────────────────────────────────────────────────────

def _print_splash():
    sep = "═" * 64
    print(f"\n{sep}")
    print("  🌌  Anti-Gravity Virtual Aquarium Engine")
    print(f"  Version : {VERSION}")
    print("  Author  : Sharjeel Safdar  |  CG Final Project")
    print("  Engine  : PyOpenGL + GLSL 3.30 + Pygame")
    print(sep)
    print("  Controls:")
    print("    WASD + QE         →  Free-fly camera")
    print("    LShift + WASD     →  Boost speed")
    print("    MOUSE             →  Look around")
    print("    G                 →  Toggle gravity")
    print("    N                 →  Day / Night mode")
    print("    B                 →  Toggle bubbles")
    print("    V                 →  Spawn vortex at camera")
    print("    Z                 →  Zoom / Focus toggle")
    print("    P                 →  Pause simulation")
    print("    H                 →  Toggle help overlay")
    print("    R                 →  Reset camera")
    print("    ESC               →  Quit")
    print(sep)
    total = sum(c for _, c in FLOCK_CONFIG)
    print(f"  Spawning {total} fish across {len(FLOCK_CONFIG)} species …")
    print(f"{sep}\n")


# ─── HUD Drawing ──────────────────────────────────────────────────────────────

def _panel(surf: pygame.Surface, rect, alpha=160, border_col=(0, 180, 255, 80)):
    s = pygame.Surface(rect[2:], pygame.SRCALPHA)
    s.fill((4, 8, 26, alpha))
    pygame.draw.rect(s, border_col, s.get_rect(), 1)
    surf.blit(s, rect[:2])


def _txt(surf, fnt, text, pos, col):
    img = fnt.render(text, True, col)
    surf.blit(img, pos)


def _draw_hud(screen: pygame.Surface, fonts, state: dict):
    """Render all HUD panels onto a transparent Pygame surface."""
    fb, fs = fonts["big"], fonts["sm"]
    W, H   = screen.get_size()

    overlay = pygame.Surface((W, H), pygame.SRCALPHA)

    # ── Colours ───────────────────────────────────────────────────────────────
    CYAN   = (0, 220, 255)
    WHITE  = (210, 235, 255)
    YELLOW = (255, 230,  70)
    GREEN  = ( 70, 255, 130)
    RED    = (255,  70, 100)
    ORANGE = (255, 160,  40)
    GREY   = (120, 140, 160)

    # ── Top-left Status Panel ─────────────────────────────────────────────────
    pw, ph = 330, 220
    _panel(overlay, (10, 10, pw, ph))

    _txt(overlay, fb, "🌌 ANTI-GRAVITY AQUARIUM", (18, 14), CYAN)
    _txt(overlay, fs, f"Version  : {VERSION}",           (18, 40), GREY)
    _txt(overlay, fs, f"FPS      : {state['fps']:5.1f}", (18, 60), WHITE)
    _txt(overlay, fs, f"Fish     : {state['fish']}",     (18, 80), WHITE)
    _txt(overlay, fs, f"Time     : {state['elapsed']:6.1f} s", (18, 100), WHITE)
    _txt(overlay, fs, f"Vortices : {state['vortex_count']}",   (18, 120), ORANGE)

    mode_col = RED if state["night"] else YELLOW
    _txt(overlay, fs, f"Mode     : {'🌙 NIGHT' if state['night'] else '☀  DAY'}",
         (18, 145), mode_col)
    grav_col = (255, 100, 50) if state["grav"] else GREEN
    _txt(overlay, fs, f"Gravity  : {'ON  ⬇' if state['grav'] else 'OFF 🚀'}",
         (18, 165), grav_col)
    bub_col = CYAN if state["bubbles"] else GREY
    _txt(overlay, fs, f"Bubbles  : {'ON  🔵' if state['bubbles'] else 'OFF'}",
         (18, 185), bub_col)
    if state["paused"]:
        _txt(overlay, fb, "⏸  PAUSED", (18, 205), YELLOW)

    # ── Controls Panel (bottom-left) ──────────────────────────────────────────
    if state["show_help"]:
        controls = [
            ("WASD+QE",   "Move camera"),
            ("LShift",    "Speed boost"),
            ("MOUSE",     "Look around"),
            ("G",         "Toggle gravity"),
            ("N",         "Day / Night"),
            ("B",         "Bubbles on/off"),
            ("V",         "Spawn vortex"),
            ("Z",         "Zoom toggle"),
            ("P",         "Pause"),
            ("H",         "Hide help"),
            ("R",         "Reset camera"),
            ("ESC",       "Quit"),
        ]
        cw, ch = 292, 14 + len(controls) * 20 + 8
        cy = H - ch - 10
        _panel(overlay, (10, cy, cw, ch))
        _txt(overlay, fb, "CONTROLS", (18, cy + 6), CYAN)
        for i, (k, v) in enumerate(controls):
            _txt(overlay, fs, f"  {k:<10}{v}", (18, cy + 26 + i * 20), WHITE)

    # ── Species Legend (bottom-right) ─────────────────────────────────────────
    species = [
        ("Neon   (Biolum.)", (0,   220, 255)),
        ("Clown",            (255, 130,  30)),
        ("Angel  (Biolum.)", (180,  50, 255)),
        ("Manta  (gentle)",  ( 50, 100, 230)),
        ("Jelly  (PRED.)",   (255,  70, 200)),
    ]
    lw, lh = 200, 24 + len(species) * 24 + 8
    lx = W - lw - 10
    ly = H - lh - 10
    _panel(overlay, (lx, ly, lw, lh))
    _txt(overlay, fb, "SPECIES", (lx + 8, ly + 6), CYAN)
    for i, (name, col) in enumerate(species):
        pygame.draw.circle(overlay, col, (lx + 14, ly + 28 + i * 24), 6)
        _txt(overlay, fs, name, (lx + 25, ly + 21 + i * 24), WHITE)

    # ── Zoom indicator ────────────────────────────────────────────────────────
    if state["zoomed"]:
        _txt(overlay, fb, "🔍 ZOOM", (W // 2 - 40, 14), CYAN)

    screen.blit(overlay, (0, 0))


# ─── Main Application ─────────────────────────────────────────────────────────

class AquariumApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)

        flags = DOUBLEBUF | OPENGL | pygame.RESIZABLE
        self.screen_2d = pygame.display.set_mode((WIN_W, WIN_H), flags)

        # ── OpenGL global state ──────────────────────────────────────────────
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glViewport(0, 0, WIN_W, WIN_H)
        glClearColor(0.0, 0.015, 0.06, 1.0)

        # ── Core subsystems ──────────────────────────────────────────────────
        self.shaders   = ShaderLibrary()
        self.camera    = Camera(WIN_W, WIN_H)
        self.renderer  = Renderer()
        self.physics   = AntiGravityPhysics(BOUNDS)
        self.collision = CollisionSystem(BOUNDS, margin=2.5, strength=6.0)
        self.particles = ParticleSystem(max_bubbles=2500, max_sparkles=800)

        # ── Flocks ───────────────────────────────────────────────────────────
        self.flocks     = [BoidFlock(sp, cnt, BOUNDS) for sp, cnt in FLOCK_CONFIG]
        self.total_fish = sum(cnt for _, cnt in FLOCK_CONFIG)

        # Identify predator flocks and prey flocks
        self.pred_flocks = [f for f in self.flocks if f.species.get("predator", False)]
        self.prey_flocks = [f for f in self.flocks if f.name in PREY_OF_JELLY]

        # ── Application state ────────────────────────────────────────────────
        self.night_mode  = False
        self.paused      = False
        self.zoomed      = False
        self.show_help   = True
        self.clock       = pygame.time.Clock()
        self.start_time  = _time.time()
        self.mouse_locked= True
        self._w          = WIN_W
        self._h          = WIN_H

        # Light (orbiting sun/moon)
        self.light_pos   = np.array([5.0, 20.0, 5.0], dtype=np.float32)

        # FPS rolling average
        self._fps_buf    = []

        # ── Fonts ────────────────────────────────────────────────────────────
        pygame.font.init()
        self.fonts = {
            "big": pygame.font.SysFont("consolas", 15, bold=True),
            "sm" : pygame.font.SysFont("consolas", 13),
        }

        # Mouse capture
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    # ─── Input ────────────────────────────────────────────────────────────────

    def _handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == QUIT:
                return False

            if ev.type == KEYDOWN:
                k = ev.key
                if k == K_ESCAPE:
                    return False
                if k == K_n:
                    self.night_mode = not self.night_mode
                if k == K_g:
                    self.physics.toggle_gravity()
                if k == K_b:
                    self.particles.toggle()
                if k == K_p:
                    self.paused = not self.paused
                if k == K_z:
                    self.zoomed = not self.zoomed
                    self.camera.set_zoom(self.zoomed)
                if k == K_h:
                    self.show_help = not self.show_help
                if k == K_r:
                    self.camera.reset()
                if k == K_v:
                    # Spawn vortex at camera position, clear after 8 vortices
                    if len(self.physics.vortices) >= 8:
                        self.physics.clear_vortices()
                    self.physics.add_vortex(
                        self.camera.pos.astype(np.float32), strength=3.5)
                    self.camera.add_shake(0.35)

            if ev.type == MOUSEMOTION:
                if self.mouse_locked:
                    self.camera.process_mouse(ev.rel[0], ev.rel[1])

            if ev.type == MOUSEWHEEL:
                self.camera.process_scroll(ev.y)

            if ev.type == VIDEORESIZE:
                self._w, self._h = ev.w, ev.h
                glViewport(0, 0, ev.w, ev.h)
                self.camera.resize(ev.w, ev.h)

        return True

    # ─── Update ───────────────────────────────────────────────────────────────

    def _update(self, dt: float, t: float):
        self.camera.process_keyboard(dt, pygame.key.get_pressed())
        self.camera.update(dt)

        # Animate light orbit
        self.light_pos[0] = 14.0 * math.sin(t * 0.18)
        self.light_pos[2] = 14.0 * math.cos(t * 0.18)
        self.light_pos[1] = 22.0 + 6.0 * math.sin(t * 0.09)

        if self.paused:
            return

        # ── Collect predator positions for fear system ───────────────────────
        pred_pos = None
        if self.pred_flocks:
            pred_pos = np.vstack([f.pos for f in self.pred_flocks])

        # ── Step each flock ──────────────────────────────────────────────────
        for flock in self.flocks:
            flock.step(dt)
            flock.vel = self.physics.apply(flock.pos, flock.vel, dt)
            flock.vel = self.collision.apply(flock.pos, flock.vel, dt)

            # Apply fear if this species is prey
            if flock.name in PREY_OF_JELLY and pred_pos is not None:
                flock.apply_fear(pred_pos)

        # ── Collect bioluminescent fish positions for sparkles ────────────────
        bio_positions = np.vstack([
            f.pos for f in self.flocks if f.species.get("bio", False)
        ]) if any(f.species.get("bio", False) for f in self.flocks) else None

        # ── Update particles ──────────────────────────────────────────────────
        self.particles.update(dt, sparkle_origins=bio_positions)

    # ─── Render ───────────────────────────────────────────────────────────────

    def _render(self, t: float):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view  = self.camera.view_matrix()
        proj  = self.camera.projection_matrix()
        night = self.night_mode
        lp    = self.light_pos
        vp    = self.camera.pos.astype(np.float32)

        # 1. Background (fullscreen, no depth write)
        self.renderer.draw_background(
            self.shaders.background, t, night, self._w, self._h)

        # 2. Sand seabed
        self.renderer.draw_seabed(
            self.shaders.seabed, view, proj, t, night, lp, vp)

        # 3. Kelp / seaweed
        self.renderer.draw_kelp(
            self.shaders.fish, view, proj, t, night, lp, vp)

        # 4. Coral colony
        self.renderer.draw_coral(
            self.shaders.fish, view, proj, t, night, lp, vp)

        # 5. Fish (per flock, with fear array)
        for flock in self.flocks:
            fear_arr = flock.fear if hasattr(flock, "fear") else None
            self.renderer.draw_fish(
                self.shaders.fish, view, proj, flock,
                t, night, lp, vp, fear_arr=fear_arr)

        # 6. Water surface (alpha-blended, drawn after opaque geometry)
        self.renderer.draw_water(
            self.shaders.water, view, proj, t, night, lp, vp)

        # 7. Particles (additive blend, no depth write)
        bub_col = (0.30, 0.72, 1.00) if not night else (0.50, 0.28, 1.00)
        self.particles.draw(self.shaders.particle, color=bub_col, night=night)

    # ─── HUD overlay ──────────────────────────────────────────────────────────

    def _blit_hud(self, t: float):
        self._fps_buf.append(self.clock.get_fps())
        if len(self._fps_buf) > 45:
            self._fps_buf.pop(0)
        avg_fps = sum(self._fps_buf) / len(self._fps_buf)

        hud_surf = pygame.Surface((self._w, self._h), pygame.SRCALPHA)
        _draw_hud(hud_surf, self.fonts, {
            "fps":         avg_fps,
            "fish":        self.total_fish,
            "elapsed":     t,
            "night":       self.night_mode,
            "grav":        self.physics.gravity_on,
            "bubbles":     self.particles.active,
            "paused":      self.paused,
            "zoomed":      self.zoomed,
            "show_help":   self.show_help,
            "vortex_count":len(self.physics.vortices),
        })

        # Blit via glDrawPixels (simple 2D overlay onto OpenGL surface)
        hud_data = pygame.image.tostring(hud_surf, "RGBA", True)
        glWindowPos2d(0, 0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawPixels(self._w, self._h, GL_RGBA, GL_UNSIGNED_BYTE, hud_data)
        glDisable(GL_BLEND)

    # ─── Main Loop ────────────────────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            dt = min(self.clock.tick(TARGET_FPS) / 1000.0, 0.05)
            t  = _time.time() - self.start_time

            running = self._handle_events()
            self._update(dt, t)
            self._render(t)
            self._blit_hud(t)

            pygame.display.flip()

        pygame.quit()
        sys.exit(0)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _print_splash()
    AquariumApp().run()
