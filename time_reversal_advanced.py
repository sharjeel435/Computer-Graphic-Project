"""
Anti-Gravity Aquarium Engine
============================

A fresh computer-graphics project idea replacing the previous time-reversal
demo. It presents an interactive virtual aquarium with boid schooling, predator
avoidance, anti-gravity vortex forces, procedural caustic lighting, particles,
depth layers, glow, and a polished real-time HUD.
"""

import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pygame


WIN_W, WIN_H = 1280, 720
WORLD_W, WORLD_H = 1800, 980
TARGET_FPS = 60
TITLE = "Anti-Gravity Aquarium Engine - Computer Graphics Demo"


def clamp(value, low, high):
    return max(low, min(high, value))


def length(v):
    return float(np.linalg.norm(v))


def normalize(v):
    n = length(v)
    if n < 1e-5:
        return np.zeros(2, dtype=np.float32)
    return (v / n).astype(np.float32)


def limit(v, max_len):
    n = length(v)
    if n > max_len:
        return (v / n * max_len).astype(np.float32)
    return v


def draw_text(surface, font, text, color, pos):
    surface.blit(font.render(text, True, color), pos)


@dataclass
class Fish:
    pos: np.ndarray
    vel: np.ndarray
    color: tuple
    size: float
    species: str = "school"
    trail: deque = field(default_factory=lambda: deque(maxlen=28))

    def update(self, aquarium, dt):
        acc = np.zeros(2, dtype=np.float32)

        if self.species == "school":
            acc += aquarium.schooling_force(self)
            acc += aquarium.predator_avoidance(self)
            max_speed = 118.0
        else:
            acc += aquarium.predator_force(self)
            max_speed = 92.0

        acc += aquarium.vortex_force(self.pos)
        acc += aquarium.boundary_force(self.pos)
        acc += np.array([math.sin(time.time() * 1.7 + self.pos[1] * 0.02), 0.0], dtype=np.float32) * 6.0

        self.vel = limit(self.vel + acc * dt, max_speed)
        self.pos += self.vel * dt
        self.pos[0] %= WORLD_W
        self.pos[1] = clamp(self.pos[1], 80, WORLD_H - 80)
        self.trail.append(self.pos.copy())

    def draw(self, surface, camera, t):
        p = camera.world_to_screen(self.pos)
        if p[0] < -80 or p[0] > WIN_W + 80 or p[1] < -80 or p[1] > WIN_H + 80:
            return

        if len(self.trail) > 2:
            pts = [camera.world_to_screen(q) for q in self.trail]
            for i in range(len(pts) - 1):
                alpha = i / max(1, len(pts) - 1)
                c = tuple(int(ch * (0.18 + 0.45 * alpha)) for ch in self.color)
                pygame.draw.line(surface, c, pts[i], pts[i + 1], 1)

        direction = normalize(self.vel)
        if length(direction) < 0.1:
            direction = np.array([1, 0], dtype=np.float32)
        side = np.array([-direction[1], direction[0]], dtype=np.float32)

        body = self.size * camera.zoom
        nose = p + direction * body * 1.55
        tail = p - direction * body * 1.20
        top = p + side * body * 0.62
        bottom = p - side * body * 0.62
        fin = tail - direction * body * 0.36

        glow = tuple(min(255, int(c * 1.55)) for c in self.color)
        pygame.draw.polygon(surface, glow, [nose, top, tail, bottom], 2)
        pygame.draw.polygon(surface, self.color, [nose, top, tail, bottom])
        pygame.draw.polygon(surface, tuple(int(c * 0.7) for c in self.color), [tail, fin + side * body * 0.55, fin - side * body * 0.55])

        eye = nose - direction * body * 0.35 + side * body * 0.16
        pygame.draw.circle(surface, (245, 250, 255), eye.astype(int), max(1, int(body * 0.13)))


@dataclass
class Bubble:
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    life: float
    max_life: float

    def update(self, dt):
        self.life -= dt
        self.pos += self.vel * dt
        self.vel[0] += math.sin(time.time() * 4.0 + self.pos[1] * 0.03) * 4.0 * dt

    def draw(self, surface, camera):
        if self.life <= 0:
            return
        p = camera.world_to_screen(self.pos)
        alpha = self.life / self.max_life
        r = max(1, int(self.radius * camera.zoom * (0.6 + alpha * 0.4)))
        color = (120, 210, 255)
        pygame.draw.circle(surface, color, p.astype(int), r, 1)


@dataclass
class Coral:
    base: np.ndarray
    height: float
    color: tuple
    phase: float

    def draw(self, surface, camera, t):
        root = camera.world_to_screen(self.base)
        points = [root]
        segments = 8
        for i in range(1, segments + 1):
            k = i / segments
            sway = math.sin(t * 1.4 + self.phase + k * 2.0) * 12.0 * k
            world = self.base + np.array([sway, -self.height * k], dtype=np.float32)
            points.append(camera.world_to_screen(world))
        if len(points) > 1:
            pygame.draw.lines(surface, self.color, False, points, max(2, int(5 * camera.zoom)))
        for branch in (0.35, 0.55, 0.75):
            idx = int(branch * segments)
            start = points[idx]
            side = -1 if idx % 2 else 1
            end = start + np.array([side * 34 * camera.zoom, -28 * camera.zoom], dtype=np.float32)
            pygame.draw.line(surface, tuple(min(255, int(c * 1.2)) for c in self.color), start, end, max(1, int(3 * camera.zoom)))


class Camera:
    def __init__(self):
        self.pos = np.array([WORLD_W * 0.5 - WIN_W * 0.5, 120], dtype=np.float32)
        self.zoom = 1.0

    def world_to_screen(self, p):
        return ((p - self.pos) * self.zoom).astype(np.float32)

    def screen_to_world(self, p):
        return np.array(p, dtype=np.float32) / self.zoom + self.pos

    def update(self, keys, dt):
        speed = 420 / self.zoom
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.pos[0] -= speed * dt
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.pos[0] += speed * dt
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.pos[1] -= speed * dt
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.pos[1] += speed * dt
        self.pos[0] = clamp(self.pos[0], 0, WORLD_W - WIN_W / self.zoom)
        self.pos[1] = clamp(self.pos[1], 0, WORLD_H - WIN_H / self.zoom)


class AquariumApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.SCALED)
        self.clock = pygame.time.Clock()
        self.camera = Camera()
        self.font = pygame.font.SysFont("consolas", 16, bold=True)
        self.small = pygame.font.SysFont("consolas", 13)
        self.running = True
        self.paused = False
        self.night_mode = False
        self.show_help = True
        self.vortex_active = False
        self.vortex_pos = np.array([WORLD_W * 0.5, WORLD_H * 0.5], dtype=np.float32)
        self.fps_history = deque(maxlen=90)
        self.fish = []
        self.bubbles = []
        self.corals = []
        self.caustic_phase = random.random() * 10
        self._build_scene()

    def _build_scene(self):
        random.seed(7)
        np.random.seed(7)
        self.fish.clear()
        palette = [
            (90, 220, 255),
            (255, 90, 180),
            (255, 210, 80),
            (120, 255, 150),
            (185, 150, 255),
        ]
        for _ in range(130):
            p = np.array([random.uniform(80, WORLD_W - 80), random.uniform(130, WORLD_H - 180)], dtype=np.float32)
            v = normalize(np.random.randn(2).astype(np.float32)) * random.uniform(35, 110)
            self.fish.append(Fish(p, v, random.choice(palette), random.uniform(8, 15), "school"))
        for _ in range(4):
            p = np.array([random.uniform(120, WORLD_W - 120), random.uniform(180, WORLD_H - 260)], dtype=np.float32)
            v = normalize(np.random.randn(2).astype(np.float32)) * 60
            self.fish.append(Fish(p, v, (255, 80, 65), random.uniform(20, 28), "predator"))
        for x in np.linspace(60, WORLD_W - 60, 48):
            self.corals.append(Coral(np.array([x + random.uniform(-20, 20), WORLD_H - 35], dtype=np.float32), random.uniform(45, 135), random.choice([(255, 95, 145), (120, 230, 190), (255, 170, 80)]), random.random() * 6.28))

    def schooling_force(self, fish):
        neighbors = []
        for other in self.fish:
            if other is fish or other.species != "school":
                continue
            d = length(other.pos - fish.pos)
            if d < 95:
                neighbors.append((other, d))
        if not neighbors:
            return np.zeros(2, dtype=np.float32)

        center = np.mean([o.pos for o, _ in neighbors], axis=0)
        avg_vel = np.mean([o.vel for o, _ in neighbors], axis=0)
        separation = np.zeros(2, dtype=np.float32)
        for other, d in neighbors:
            if d < 34:
                separation -= (other.pos - fish.pos) / max(d, 1.0)

        cohesion = normalize(center - fish.pos) * 42
        alignment = normalize(avg_vel) * 34
        separation = normalize(separation) * 82
        return cohesion + alignment + separation

    def predator_avoidance(self, fish):
        force = np.zeros(2, dtype=np.float32)
        for other in self.fish:
            if other.species != "predator":
                continue
            delta = fish.pos - other.pos
            d = length(delta)
            if d < 180:
                force += normalize(delta) * (210 * (1 - d / 180))
        return force

    def predator_force(self, predator):
        nearest = None
        nearest_d = 999999.0
        for other in self.fish:
            if other.species != "school":
                continue
            d = length(other.pos - predator.pos)
            if d < nearest_d:
                nearest, nearest_d = other, d
        if nearest is None:
            return np.zeros(2, dtype=np.float32)
        return normalize(nearest.pos - predator.pos) * 60

    def boundary_force(self, pos):
        margin = 120
        force = np.zeros(2, dtype=np.float32)
        if pos[1] < margin:
            force[1] += 90
        if pos[1] > WORLD_H - margin:
            force[1] -= 100
        return force

    def vortex_force(self, pos):
        if not self.vortex_active:
            return np.zeros(2, dtype=np.float32)
        delta = self.vortex_pos - pos
        d = max(length(delta), 1.0)
        if d > 420:
            return np.zeros(2, dtype=np.float32)
        tangent = np.array([-delta[1], delta[0]], dtype=np.float32)
        pull = normalize(delta) * 150 * (1 - d / 420)
        swirl = normalize(tangent) * 230 * (1 - d / 420)
        return pull + swirl

    def emit_bubbles(self, amount=1):
        for _ in range(amount):
            if len(self.bubbles) > 360:
                self.bubbles.pop(0)
            p = np.array([random.uniform(20, WORLD_W - 20), WORLD_H + random.uniform(0, 90)], dtype=np.float32)
            v = np.array([random.uniform(-10, 10), random.uniform(-80, -35)], dtype=np.float32)
            life = random.uniform(5.5, 12)
            self.bubbles.append(Bubble(p, v, random.uniform(3, 12), life, life))

    def update(self, dt):
        if self.paused:
            return
        self.emit_bubbles(3)
        for fish in self.fish:
            fish.update(self, dt)
        for bubble in self.bubbles[:]:
            bubble.update(dt)
            if bubble.life <= 0 or bubble.pos[1] < -80:
                self.bubbles.remove(bubble)

    def draw_background(self, t):
        top = (4, 18, 42) if not self.night_mode else (2, 4, 18)
        bottom = (0, 92, 118) if not self.night_mode else (4, 28, 55)
        for y in range(0, WIN_H, 3):
            k = y / WIN_H
            color = tuple(int(top[i] * (1 - k) + bottom[i] * k) for i in range(3))
            pygame.draw.rect(self.screen, color, (0, y, WIN_W, 3))

        for i in range(34):
            x = (i * 96 + math.sin(t * 0.7 + i) * 30 - self.camera.pos[0] * 0.08) % (WIN_W + 140) - 70
            wave = []
            for y in range(-40, WIN_H + 40, 24):
                sx = x + math.sin(y * 0.026 + t * 1.5 + i) * 18
                wave.append((sx, y))
            pygame.draw.lines(self.screen, (28, 175, 210), False, wave, 1)

        for i in range(16):
            x = (i * 140 - self.camera.pos[0] * 0.18 + math.sin(t * 0.4 + i) * 30) % (WIN_W + 180) - 90
            pygame.draw.line(self.screen, (35, 120, 150), (x, 0), (x + 95, WIN_H), 1)

    def draw_world(self, t):
        sand_y = int((WORLD_H - self.camera.pos[1]) * self.camera.zoom)
        pygame.draw.rect(self.screen, (38, 70, 76), (0, sand_y, WIN_W, WIN_H - sand_y))
        for i in range(90):
            x = (i * 53 - self.camera.pos[0] * self.camera.zoom) % WIN_W
            y = sand_y + math.sin(i * 2.1) * 9
            pygame.draw.circle(self.screen, (55, 100, 98), (int(x), int(y)), 2)

        for coral in self.corals:
            coral.draw(self.screen, self.camera, t)
        for bubble in self.bubbles:
            bubble.draw(self.screen, self.camera)
        for fish in sorted(self.fish, key=lambda f: f.size):
            fish.draw(self.screen, self.camera, t)

        if self.vortex_active:
            p = self.camera.world_to_screen(self.vortex_pos)
            for r in range(34, 190, 26):
                color = (90, 220, 255) if not self.night_mode else (180, 90, 255)
                pygame.draw.circle(self.screen, color, p.astype(int), int(r * self.camera.zoom), 1)
            for i in range(14):
                a = t * 3.5 + i * 0.45
                end = p + np.array([math.cos(a), math.sin(a)], dtype=np.float32) * 150 * self.camera.zoom
                pygame.draw.line(self.screen, (160, 245, 255), p, end, 1)

    def draw_hud(self):
        fps = np.mean(self.fps_history) if self.fps_history else 0
        panel = pygame.Surface((372, 158), pygame.SRCALPHA)
        panel.fill((5, 12, 22, 178))
        self.screen.blit(panel, (14, 14))
        draw_text(self.screen, self.font, "ANTI-GRAVITY AQUARIUM ENGINE", (190, 245, 255), (26, 24))
        draw_text(self.screen, self.small, f"FPS: {fps:5.1f}   Fish: {len(self.fish)}   Bubbles: {len(self.bubbles)}", (220, 235, 235), (26, 52))
        draw_text(self.screen, self.small, f"Camera: ({self.camera.pos[0]:.0f}, {self.camera.pos[1]:.0f})  Zoom: {self.camera.zoom:.2f}x", (180, 215, 235), (26, 74))
        draw_text(self.screen, self.small, f"Vortex: {'ON' if self.vortex_active else 'OFF'}   Mode: {'Night' if self.night_mode else 'Day'}", (255, 215, 145), (26, 96))
        if self.show_help:
            draw_text(self.screen, self.small, "WASD move | Wheel zoom | V vortex | N night | B bubbles | R reset | H help", (165, 225, 190), (26, 124))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_v:
                    self.vortex_active = not self.vortex_active
                elif event.key == pygame.K_n:
                    self.night_mode = not self.night_mode
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_b:
                    for _ in range(80):
                        self.emit_bubbles(1)
                elif event.key == pygame.K_r:
                    self._build_scene()
                    self.bubbles.clear()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.camera.zoom = min(1.65, self.camera.zoom * 1.08)
                elif event.button == 5:
                    self.camera.zoom = max(0.62, self.camera.zoom / 1.08)
                elif event.button == 1:
                    self.vortex_pos = self.camera.screen_to_world(event.pos)
                    self.vortex_active = True
            elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                self.vortex_pos = self.camera.screen_to_world(event.pos)

    def run(self):
        print("Anti-Gravity Aquarium Engine started")
        print("Controls: WASD, mouse wheel, click/drag vortex, V, N, B, R, H, SPACE, ESC")
        while self.running:
            dt = min(0.033, self.clock.tick(TARGET_FPS) / 1000.0)
            self.fps_history.append(self.clock.get_fps())
            self.handle_events()
            self.camera.update(pygame.key.get_pressed(), dt)
            self.update(dt)
            t = time.time() + self.caustic_phase
            self.draw_background(t)
            self.draw_world(t)
            self.draw_hud()
            pygame.display.flip()
        pygame.quit()
        print("Anti-Gravity Aquarium Engine closed")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    AquariumApp().run()
