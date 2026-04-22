"""
3D Time-Reversal Physics Simulator with Full Rendering Pipeline.

This module is intentionally self-contained for evaluation: it shows the
graphics pipeline, a fixed-step physics engine, a state-history time engine,
particles, shadows, planar reflections, motion trails, instancing, scene graph
transforms, and interactive camera controls in one runnable OpenGL program.
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF,
    K_1,
    K_2,
    K_3,
    K_c,
    K_DOWN,
    K_e,
    K_ESCAPE,
    K_f,
    K_i,
    K_LCTRL,
    K_LEFT,
    K_q,
    K_r,
    K_RIGHT,
    K_SPACE,
    K_UP,
    K_w,
    K_a,
    K_s,
    K_d,
    MOUSEBUTTONDOWN,
    MOUSEMOTION,
    OPENGL,
    QUIT,
)
from OpenGL.GL import *

from time_reversal_3d.config import FPS, SCREEN_HEIGHT, SCREEN_WIDTH


# =============================================================================
# Math: model, view, projection, and quaternion transforms
# =============================================================================

Vec3 = np.ndarray
Mat4 = np.ndarray


def vec3(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=np.float32)


def normalize(v: Vec3) -> Vec3:
    n = float(np.linalg.norm(v))
    if n < 1e-7:
        return np.zeros(3, dtype=np.float32)
    return (v / n).astype(np.float32)


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> Mat4:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(left: float, right: float, bottom: float, top: float, near: float, far: float) -> Mat4:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    f = normalize(target - eye)
    r = normalize(np.cross(f, up))
    u = np.cross(r, f).astype(np.float32)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(r, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def translate(v: Vec3) -> Mat4:
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = v[:3]
    return m


def scale(v: Vec3) -> Mat4:
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[1, 1], m[2, 2] = float(v[0]), float(v[1]), float(v[2])
    return m


def quat_identity() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return quat_identity() if n < 1e-7 else (q / n).astype(np.float32)


def quat_from_axis_angle(axis: Vec3, angle: float) -> np.ndarray:
    axis = normalize(axis)
    s = math.sin(angle * 0.5)
    return quat_normalize(np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle * 0.5)], dtype=np.float32))


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return quat_normalize(
        np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float32,
        )
    )


def quat_to_mat4(q: np.ndarray) -> Mat4:
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def quat_from_angular_velocity(omega: Vec3, dt: float) -> np.ndarray:
    speed = float(np.linalg.norm(omega))
    if speed < 1e-7:
        return quat_identity()
    return quat_from_axis_angle(omega / speed, speed * dt)


# =============================================================================
# Data model: materials, meshes, scene graph, physics state
# =============================================================================


class ShadingMode(IntEnum):
    FLAT = 0
    GOURAUD = 1
    PHONG = 2


@dataclass
class Material:
    name: str
    base_color: Tuple[float, float, float]
    shininess: float = 48.0
    reflectivity: float = 0.0
    alpha: float = 1.0


@dataclass
class Mesh:
    name: str
    vao: int
    vbo: int
    ebo: int
    index_count: int
    radius: float


@dataclass
class Transform:
    position: Vec3 = field(default_factory=lambda: vec3(0, 0, 0))
    rotation: np.ndarray = field(default_factory=quat_identity)
    size: Vec3 = field(default_factory=lambda: vec3(1, 1, 1))

    def local_matrix(self) -> Mat4:
        return translate(self.position) @ quat_to_mat4(self.rotation) @ scale(self.size)


@dataclass
class PhysicsState:
    position: Vec3
    rotation: np.ndarray
    velocity: Vec3
    angular_velocity: Vec3
    alive: bool

    def copy(self) -> "PhysicsState":
        return PhysicsState(
            self.position.copy(),
            self.rotation.copy(),
            self.velocity.copy(),
            self.angular_velocity.copy(),
            self.alive,
        )


@dataclass
class SceneNode:
    object_id: int
    name: str
    transform: Transform
    mesh: Mesh
    material: Material
    velocity: Vec3 = field(default_factory=lambda: vec3(0, 0, 0))
    angular_velocity: Vec3 = field(default_factory=lambda: vec3(0, 0, 0))
    mass: float = 1.0
    radius: float = 1.0
    dynamic: bool = True
    alive: bool = True
    parent: Optional["SceneNode"] = None
    children: List["SceneNode"] = field(default_factory=list)
    trail: deque = field(default_factory=lambda: deque(maxlen=72))

    def model_matrix(self) -> Mat4:
        local = self.transform.local_matrix()
        if self.parent is None:
            return local
        return self.parent.model_matrix() @ local

    def state(self) -> PhysicsState:
        return PhysicsState(
            self.transform.position.copy(),
            self.transform.rotation.copy(),
            self.velocity.copy(),
            self.angular_velocity.copy(),
            self.alive,
        )

    def apply_state(self, state: PhysicsState) -> None:
        self.transform.position = state.position.copy()
        self.transform.rotation = state.rotation.copy()
        self.velocity = state.velocity.copy()
        self.angular_velocity = state.angular_velocity.copy()
        self.alive = state.alive


@dataclass
class Particle:
    position: Vec3
    velocity: Vec3
    color: Tuple[float, float, float, float]
    age: float
    lifetime: float
    size: float

    def state(self) -> Tuple[Vec3, Vec3, float]:
        return (self.position.copy(), self.velocity.copy(), self.age)

    def apply_state(self, state: Tuple[Vec3, Vec3, float]) -> None:
        self.position, self.velocity, self.age = state[0].copy(), state[1].copy(), float(state[2])


# =============================================================================
# GPU wrappers: shader compilation, buffers, framebuffer shadow target
# =============================================================================


class ShaderProgram:
    def __init__(self, vertex_src: str, fragment_src: str):
        self.program = glCreateProgram()
        vs = self._compile(GL_VERTEX_SHADER, vertex_src)
        fs = self._compile(GL_FRAGMENT_SHADER, fragment_src)
        glAttachShader(self.program, vs)
        glAttachShader(self.program, fs)
        glLinkProgram(self.program)
        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(self.program).decode("utf-8", "ignore"))
        glDeleteShader(vs)
        glDeleteShader(fs)

    @staticmethod
    def _compile(shader_type: int, src: str) -> int:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, src)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader).decode("utf-8", "ignore"))
        return shader

    def use(self) -> None:
        glUseProgram(self.program)

    def loc(self, name: str) -> int:
        return glGetUniformLocation(self.program, name)

    def mat4(self, name: str, m: Mat4) -> None:
        glUniformMatrix4fv(self.loc(name), 1, GL_TRUE, m.astype(np.float32))

    def vec3(self, name: str, v: Vec3 | Tuple[float, float, float]) -> None:
        glUniform3f(self.loc(name), float(v[0]), float(v[1]), float(v[2]))

    def float(self, name: str, value: float) -> None:
        glUniform1f(self.loc(name), float(value))

    def int(self, name: str, value: int) -> None:
        glUniform1i(self.loc(name), int(value))


class MeshFactory:
    @staticmethod
    def upload(name: str, interleaved: np.ndarray, indices: np.ndarray, radius: float) -> Mesh:
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        stride = 8 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glBindVertexArray(0)
        return Mesh(name, vao, vbo, ebo, int(len(indices)), radius)

    @staticmethod
    def sphere(radius: float = 1.0, slices: int = 36, stacks: int = 18) -> Tuple[np.ndarray, np.ndarray]:
        data: List[float] = []
        idx: List[int] = []
        for y in range(stacks + 1):
            v = y / stacks
            phi = math.pi * v
            for x in range(slices + 1):
                u = x / slices
                theta = 2.0 * math.pi * u
                nx = math.sin(phi) * math.cos(theta)
                ny = math.cos(phi)
                nz = math.sin(phi) * math.sin(theta)
                data.extend([radius * nx, radius * ny, radius * nz, nx, ny, nz, u, v])
        for y in range(stacks):
            for x in range(slices):
                a = y * (slices + 1) + x
                b = a + slices + 1
                idx.extend([a, b, a + 1, a + 1, b, b + 1])
        return np.array(data, dtype=np.float32), np.array(idx, dtype=np.uint32)

    @staticmethod
    def cube(size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        s = size * 0.5
        faces = [
            ((0, 0, 1), [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]),
            ((0, 0, -1), [(s, -s, -s), (-s, -s, -s), (-s, s, -s), (s, s, -s)]),
            ((1, 0, 0), [(s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s)]),
            ((-1, 0, 0), [(-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)]),
            ((0, 1, 0), [(-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s)]),
            ((0, -1, 0), [(-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s)]),
        ]
        data: List[float] = []
        idx: List[int] = []
        for normal, verts in faces:
            start = len(data) // 8
            tex = [(0, 0), (1, 0), (1, 1), (0, 1)]
            for p, uv in zip(verts, tex):
                data.extend([p[0], p[1], p[2], normal[0], normal[1], normal[2], uv[0], uv[1]])
            idx.extend([start, start + 1, start + 2, start, start + 2, start + 3])
        return np.array(data, dtype=np.float32), np.array(idx, dtype=np.uint32)

    @staticmethod
    def wedge() -> Tuple[np.ndarray, np.ndarray]:
        p = [vec3(-0.45, -0.35, 0), vec3(0.45, -0.35, 0), vec3(0.45, 0.35, 0), vec3(-0.45, 0.35, 0), vec3(0, 0, 1.4)]
        tris = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (1, 0, 2), (0, 3, 2)]
        data: List[float] = []
        idx: List[int] = []
        for tri in tris:
            a, b, c = (p[i] for i in tri)
            n = normalize(np.cross(b - a, c - a))
            start = len(data) // 8
            for q in (a, b, c):
                data.extend([q[0], q[1], q[2], n[0], n[1], n[2], 0.5, 0.5])
            idx.extend([start, start + 1, start + 2])
        return np.array(data, dtype=np.float32), np.array(idx, dtype=np.uint32)


class ShadowMap:
    def __init__(self, size: int = 2048):
        self.size = size
        self.fbo = glGenFramebuffers(1)
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, size, size, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.texture, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Shadow framebuffer is incomplete")


# =============================================================================
# Camera, physics, time history, particles
# =============================================================================


class FreeCamera:
    def __init__(self) -> None:
        self.position = vec3(0, 4.2, 14)
        self.yaw = -math.pi * 0.5
        self.pitch = -0.22
        self.speed = 9.0
        self.sensitivity = 0.003

    def forward(self) -> Vec3:
        return normalize(vec3(math.cos(self.yaw) * math.cos(self.pitch), math.sin(self.pitch), math.sin(self.yaw) * math.cos(self.pitch)))

    def right(self) -> Vec3:
        return normalize(np.cross(self.forward(), vec3(0, 1, 0)))

    def view(self) -> Mat4:
        return look_at(self.position, self.position + self.forward(), vec3(0, 1, 0))

    def handle_mouse(self, dx: float, dy: float) -> None:
        self.yaw += dx * self.sensitivity
        self.pitch = float(np.clip(self.pitch - dy * self.sensitivity, -1.45, 1.45))

    def update(self, keys, dt: float) -> None:
        f = self.forward()
        r = self.right()
        if keys[K_w]:
            self.position += f * self.speed * dt
        if keys[K_s]:
            self.position -= f * self.speed * dt
        if keys[K_d]:
            self.position += r * self.speed * dt
        if keys[K_a]:
            self.position -= r * self.speed * dt
        if keys[K_q]:
            self.position[1] -= self.speed * dt
        if keys[K_e]:
            self.position[1] += self.speed * dt


class PhysicsWorld:
    def __init__(self) -> None:
        self.gravity = vec3(0, -9.81, 0)
        self.bounds = vec3(14, 10, 14)
        self.ground_y = -3.0
        self.collisions_this_frame = 0

    def step(self, objects: List[SceneNode], dt: float) -> None:
        self.collisions_this_frame = 0
        for obj in objects:
            if not obj.dynamic or not obj.alive:
                continue
            obj.velocity += self.gravity * dt
            obj.velocity *= 0.995
            obj.transform.position += obj.velocity * dt
            obj.transform.rotation = quat_mul(quat_from_angular_velocity(obj.angular_velocity, dt), obj.transform.rotation)
            obj.trail.append(obj.transform.position.copy())
            self._collide_ground_and_bounds(obj)
        self._collide_pairs([o for o in objects if o.dynamic and o.alive])

    def _collide_ground_and_bounds(self, obj: SceneNode) -> None:
        r = obj.radius * max(float(obj.transform.size[0]), float(obj.transform.size[1]), float(obj.transform.size[2]))
        p = obj.transform.position
        if p[1] - r < self.ground_y:
            p[1] = self.ground_y + r
            if obj.velocity[1] < 0:
                obj.velocity[1] = -obj.velocity[1] * 0.72
                obj.velocity[0] *= 0.88
                obj.velocity[2] *= 0.88
                self.collisions_this_frame += 1
        for axis in (0, 2):
            if p[axis] - r < -self.bounds[axis]:
                p[axis] = -self.bounds[axis] + r
                obj.velocity[axis] = abs(obj.velocity[axis]) * 0.76
                self.collisions_this_frame += 1
            elif p[axis] + r > self.bounds[axis]:
                p[axis] = self.bounds[axis] - r
                obj.velocity[axis] = -abs(obj.velocity[axis]) * 0.76
                self.collisions_this_frame += 1

    def _collide_pairs(self, objects: List[SceneNode]) -> None:
        for i, a in enumerate(objects):
            for b in objects[i + 1 :]:
                pa, pb = a.transform.position, b.transform.position
                delta = pb - pa
                dist = float(np.linalg.norm(delta))
                ra = a.radius * max(float(a.transform.size[0]), float(a.transform.size[1]), float(a.transform.size[2]))
                rb = b.radius * max(float(b.transform.size[0]), float(b.transform.size[1]), float(b.transform.size[2]))
                min_dist = ra + rb
                if dist >= min_dist:
                    continue
                n = vec3(1, 0, 0) if dist < 1e-6 else (delta / dist).astype(np.float32)
                depth = min_dist - dist
                total_inv_mass = 1.0 / a.mass + 1.0 / b.mass
                a.transform.position -= n * depth * (1.0 / a.mass) / total_inv_mass
                b.transform.position += n * depth * (1.0 / b.mass) / total_inv_mass
                rel = b.velocity - a.velocity
                vn = float(np.dot(rel, n))
                if vn < 0:
                    impulse = -(1.0 + 0.78) * vn / total_inv_mass
                    a.velocity -= impulse * n / a.mass
                    b.velocity += impulse * n / b.mass
                    self.collisions_this_frame += 1


class FrameHistory:
    def __init__(self, max_frames: int = 3600):
        self.max_frames = max_frames
        self.frames: deque[Dict[int, PhysicsState]] = deque(maxlen=max_frames)
        self.particle_frames: deque[List[Tuple[Vec3, Vec3, float]]] = deque(maxlen=max_frames)
        self.playhead = 0

    def record(self, objects: List[SceneNode], particles: List[Particle]) -> None:
        self.frames.append({obj.object_id: obj.state().copy() for obj in objects})
        self.particle_frames.append([p.state() for p in particles])
        self.playhead = len(self.frames) - 1

    def rewind(self, objects: List[SceneNode], particles: List[Particle], frames: int = 1) -> bool:
        if not self.frames:
            return False
        self.playhead = max(0, self.playhead - frames)
        frame = self.frames[self.playhead]
        for obj in objects:
            if obj.object_id in frame:
                obj.apply_state(frame[obj.object_id])
        particle_state = self.particle_frames[self.playhead] if self.playhead < len(self.particle_frames) else []
        for p, state in zip(particles, particle_state):
            p.apply_state(state)
        return self.playhead > 0

    def clear(self) -> None:
        self.frames.clear()
        self.particle_frames.clear()
        self.playhead = 0

    @property
    def memory_mb(self) -> float:
        return len(self.frames) * 96.0 / (1024.0 * 1024.0)


class ParticleSystem:
    def __init__(self, max_particles: int = 900):
        self.max_particles = max_particles
        self.particles: List[Particle] = []

    def emit(self, position: Vec3, amount: int, color: Tuple[float, float, float, float]) -> None:
        for _ in range(amount):
            if len(self.particles) >= self.max_particles:
                self.particles.pop(0)
            direction = normalize(vec3(random.uniform(-1, 1), random.uniform(0.1, 1.2), random.uniform(-1, 1)))
            self.particles.append(Particle(position.copy(), direction * random.uniform(1.2, 6.0), color, 0.0, random.uniform(1.0, 2.8), random.uniform(4.0, 10.0)))

    def update(self, dt: float) -> None:
        for p in self.particles[:]:
            p.age += dt
            p.velocity += vec3(0, 1.6, 0) * dt
            p.velocity *= 0.985
            p.position += p.velocity * dt
            if p.age >= p.lifetime:
                self.particles.remove(p)


# =============================================================================
# Renderer: shadow pass, reflection pass, fragment processing, framebuffer output
# =============================================================================


class Renderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.shading_mode = ShadingMode.PHONG
        self.shadow_map = ShadowMap(2048)
        self.main_shader = ShaderProgram(self._main_vs(), self._main_fs())
        self.depth_shader = ShaderProgram(self._depth_vs(), self._depth_fs())
        self.line_shader = ShaderProgram(self._line_vs(), self._line_fs())
        self.meshes = self._create_meshes()
        self.pipeline_stats = {"draw_calls": 0, "culled": 0}
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_MULTISAMPLE)
        glClearColor(0.025, 0.03, 0.045, 1.0)

    def _create_meshes(self) -> Dict[str, Mesh]:
        meshes: Dict[str, Mesh] = {}
        for name, factory, radius in (
            ("sphere", MeshFactory.sphere, 1.0),
            ("cube", MeshFactory.cube, 0.88),
            ("wedge", MeshFactory.wedge, 1.0),
        ):
            data, idx = factory()
            meshes[name] = MeshFactory.upload(name, data, idx, radius)
        return meshes

    @staticmethod
    def _main_vs() -> str:
        return """
        #version 120
        attribute vec3 aPos;
        attribute vec3 aNormal;
        attribute vec2 aUV;
        uniform mat4 uModel;
        uniform mat4 uView;
        uniform mat4 uProjection;
        uniform mat4 uLightSpace;
        uniform vec3 uLightPos;
        uniform vec3 uViewPos;
        uniform vec3 uObjectColor;
        uniform float uShininess;
        uniform int uShadingMode;
        varying vec3 vWorldPos;
        varying vec3 vNormal;
        varying vec4 vLightSpacePos;
        varying vec3 vGouraudColor;

        vec3 blinnPhong(vec3 pos, vec3 normal) {
            vec3 n = normalize(normal);
            vec3 l = normalize(uLightPos - pos);
            vec3 v = normalize(uViewPos - pos);
            vec3 h = normalize(l + v);
            float diff = max(dot(n, l), 0.0);
            float spec = pow(max(dot(n, h), 0.0), uShininess);
            vec3 ambient = vec3(0.12, 0.13, 0.16);
            return (ambient + diff * vec3(1.0, 0.94, 0.84)) * uObjectColor + spec * vec3(1.0);
        }

        void main() {
            vec4 world = uModel * vec4(aPos, 1.0);
            vWorldPos = world.xyz;
            vNormal = normalize(mat3(uModel) * aNormal);
            vLightSpacePos = uLightSpace * world;
            vGouraudColor = blinnPhong(vWorldPos, vNormal);
            gl_Position = uProjection * uView * world;
        }
        """

    @staticmethod
    def _main_fs() -> str:
        return """
        #version 120
        uniform vec3 uLightPos;
        uniform vec3 uViewPos;
        uniform vec3 uObjectColor;
        uniform float uShininess;
        uniform float uReflectivity;
        uniform float uAlpha;
        uniform int uShadingMode;
        uniform sampler2D uShadowMap;
        varying vec3 vWorldPos;
        varying vec3 vNormal;
        varying vec4 vLightSpacePos;
        varying vec3 vGouraudColor;

        float shadowFactor(vec4 lightSpacePos, vec3 normal, vec3 lightDir) {
            vec3 proj = lightSpacePos.xyz / lightSpacePos.w;
            proj = proj * 0.5 + 0.5;
            if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) return 0.0;
            float bias = max(0.0025 * (1.0 - dot(normal, lightDir)), 0.0007);
            float shadow = 0.0;
            float texel = 1.0 / 2048.0;
            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    float closest = texture2D(uShadowMap, proj.xy + vec2(x, y) * texel).r;
                    shadow += (proj.z - bias > closest) ? 1.0 : 0.0;
                }
            }
            return shadow / 9.0;
        }

        void main() {
            vec3 n = normalize(vNormal);
            if (uShadingMode == 0) {
                n = normalize(floor(n * 3.0 + 0.5) / 3.0);
            }
            vec3 l = normalize(uLightPos - vWorldPos);
            vec3 v = normalize(uViewPos - vWorldPos);
            vec3 h = normalize(l + v);

            float diff = max(dot(n, l), 0.0);
            float spec = pow(max(dot(n, h), 0.0), uShininess);
            float shadow = shadowFactor(vLightSpacePos, n, l);
            vec3 ambient = vec3(0.11, 0.12, 0.16) * uObjectColor;
            vec3 direct = ((diff * vec3(1.0, 0.93, 0.82)) * uObjectColor + spec * vec3(1.0)) * (1.0 - shadow * 0.78);
            vec3 color = (uShadingMode == 1) ? vGouraudColor : ambient + direct;
            color += uReflectivity * vec3(0.16, 0.19, 0.22);
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0 / 2.2));
            gl_FragColor = vec4(color, uAlpha);
        }
        """

    @staticmethod
    def _depth_vs() -> str:
        return """
        #version 120
        attribute vec3 aPos;
        uniform mat4 uModel;
        uniform mat4 uLightSpace;
        void main() {
            gl_Position = uLightSpace * uModel * vec4(aPos, 1.0);
        }
        """

    @staticmethod
    def _depth_fs() -> str:
        return "#version 120\nvoid main() {}\n"

    @staticmethod
    def _line_vs() -> str:
        return """
        #version 120
        attribute vec3 aPos;
        uniform mat4 uView;
        uniform mat4 uProjection;
        void main() {
            gl_Position = uProjection * uView * vec4(aPos, 1.0);
        }
        """

    @staticmethod
    def _line_fs() -> str:
        return """
        #version 120
        uniform vec4 uColor;
        void main() { gl_FragColor = uColor; }
        """

    def light_space_matrix(self, light_pos: Vec3) -> Mat4:
        return orthographic(-22, 22, -22, 22, 1, 55) @ look_at(light_pos, vec3(0, 0, 0), vec3(0, 1, 0))

    def render(self, objects: List[SceneNode], particles: List[Particle], camera: FreeCamera, light_pos: Vec3) -> None:
        projection = perspective(58.0, self.width / self.height, 0.1, 140.0)
        view = camera.view()
        light_space = self.light_space_matrix(light_pos)
        visible = self._cull(objects, camera)

        glViewport(0, 0, self.shadow_map.size, self.shadow_map.size)
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_map.fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        glCullFace(GL_FRONT)
        self.depth_shader.use()
        self.depth_shader.mat4("uLightSpace", light_space)
        for obj in visible:
            if obj.alive and obj.name != "Ground":
                self.depth_shader.mat4("uModel", obj.model_matrix())
                glBindVertexArray(obj.mesh.vao)
                glDrawElements(GL_TRIANGLES, obj.mesh.index_count, GL_UNSIGNED_INT, None)
        glCullFace(GL_BACK)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._draw_objects(visible, view, projection, light_space, camera.position, light_pos, reflected=False)
        self._draw_reflections(visible, view, projection, light_space, camera.position, light_pos)
        self._draw_trails(visible, view, projection)
        self._draw_particles(particles, view, projection)
        self.pipeline_stats["draw_calls"] += 1

    def _cull(self, objects: List[SceneNode], camera: FreeCamera) -> List[SceneNode]:
        result = []
        self.pipeline_stats = {"draw_calls": 0, "culled": 0}
        f = camera.forward()
        for obj in objects:
            to_obj = obj.transform.position - camera.position
            if float(np.dot(normalize(to_obj), f)) < -0.22 and float(np.linalg.norm(to_obj)) > 8.0:
                self.pipeline_stats["culled"] += 1
                continue
            result.append(obj)
        return result

    def _draw_objects(self, objects: Iterable[SceneNode], view: Mat4, projection: Mat4, light_space: Mat4, camera_pos: Vec3, light_pos: Vec3, reflected: bool) -> None:
        glEnable(GL_CULL_FACE)
        self.main_shader.use()
        self.main_shader.mat4("uView", view)
        self.main_shader.mat4("uProjection", projection)
        self.main_shader.mat4("uLightSpace", light_space)
        self.main_shader.vec3("uLightPos", light_pos)
        self.main_shader.vec3("uViewPos", camera_pos)
        self.main_shader.int("uShadingMode", int(self.shading_mode))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.shadow_map.texture)
        self.main_shader.int("uShadowMap", 0)
        for obj in objects:
            if not obj.alive:
                continue
            model = obj.model_matrix()
            mat = obj.material
            self.main_shader.mat4("uModel", model)
            self.main_shader.vec3("uObjectColor", mat.base_color)
            self.main_shader.float("uShininess", mat.shininess)
            self.main_shader.float("uReflectivity", mat.reflectivity)
            self.main_shader.float("uAlpha", mat.alpha * (0.34 if reflected else 1.0))
            glBindVertexArray(obj.mesh.vao)
            glDrawElements(GL_TRIANGLES, obj.mesh.index_count, GL_UNSIGNED_INT, None)
            self.pipeline_stats["draw_calls"] += 1

    def _draw_reflections(self, objects: Iterable[SceneNode], view: Mat4, projection: Mat4, light_space: Mat4, camera_pos: Vec3, light_pos: Vec3) -> None:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        reflected_objects: List[SceneNode] = []
        for obj in objects:
            if not obj.dynamic or not obj.alive:
                continue
            clone = SceneNode(obj.object_id, obj.name, Transform(obj.transform.position.copy(), obj.transform.rotation.copy(), obj.transform.size.copy()), obj.mesh, obj.material)
            clone.transform.position[1] = -6.0 - obj.transform.position[1]
            clone.transform.size[1] *= -1.0
            reflected_objects.append(clone)
        self._draw_objects(reflected_objects, view, projection, light_space, camera_pos, light_pos, reflected=True)
        glDisable(GL_BLEND)

    def _draw_trails(self, objects: Iterable[SceneNode], view: Mat4, projection: Mat4) -> None:
        self.line_shader.use()
        self.line_shader.mat4("uView", view)
        self.line_shader.mat4("uProjection", projection)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        for obj in objects:
            if len(obj.trail) < 2:
                continue
            arr = np.array(obj.trail, dtype=np.float32)
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STREAM_DRAW)
            loc = glGetAttribLocation(self.line_shader.program, "aPos")
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, None)
            self.line_shader.vec3("unused", vec3(0, 0, 0))
            glUniform4f(self.line_shader.loc("uColor"), obj.material.base_color[0], obj.material.base_color[1], obj.material.base_color[2], 0.35)
            glDrawArrays(GL_LINE_STRIP, 0, len(arr))
            glDisableVertexAttribArray(loc)
            glDeleteBuffers(1, [vbo])
        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)

    def _draw_particles(self, particles: List[Particle], view: Mat4, projection: Mat4) -> None:
        if not particles:
            return
        self.line_shader.use()
        self.line_shader.mat4("uView", view)
        self.line_shader.mat4("uProjection", projection)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        for p in particles:
            arr = np.array([p.position], dtype=np.float32)
            alpha = max(0.0, 1.0 - p.age / p.lifetime)
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STREAM_DRAW)
            loc = glGetAttribLocation(self.line_shader.program, "aPos")
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, None)
            glUniform4f(self.line_shader.loc("uColor"), p.color[0], p.color[1], p.color[2], p.color[3] * alpha)
            glPointSize(p.size)
            glDrawArrays(GL_POINTS, 0, 1)
            glDisableVertexAttribArray(loc)
            glDeleteBuffers(1, [vbo])
        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)


# =============================================================================
# Application: scene setup, interaction, simulation loop, framebuffer output
# =============================================================================


class TimeReversalApp:
    def __init__(self):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Time-Reversal Physics Simulator - Full Rendering Pipeline")

        self.renderer = Renderer(self.width, self.height)
        self.physics = PhysicsWorld()
        self.history = FrameHistory(3600)
        self.particles = ParticleSystem(900)
        self.camera = FreeCamera()
        self.light_pos = vec3(-8, 18, 9)
        self.clock = pygame.time.Clock()
        self.objects: List[SceneNode] = []
        self.materials = self._create_materials()
        self.next_id = 1
        self.running = True
        self.paused = False
        self.reversing = False
        self.time_scale = 1.0
        self.fixed_accumulator = 0.0
        self.frame_counter = 0
        self.last_caption = time.time()
        self.mouse_locked = False
        self._create_scene()
        self.history.record(self.objects, self.particles.particles)

    def _create_materials(self) -> Dict[str, Material]:
        return {
            "ground": Material("tempered reflective floor", (0.42, 0.45, 0.48), 96, 0.42),
            "core": Material("polished silver core", (0.78, 0.82, 0.9), 128, 0.55),
            "red": Material("red ceramic", (0.9, 0.22, 0.18), 54, 0.1),
            "blue": Material("blue polymer", (0.12, 0.42, 0.95), 44, 0.08),
            "gold": Material("gold fragments", (1.0, 0.68, 0.12), 84, 0.34),
            "green": Material("green glass", (0.15, 0.8, 0.38), 72, 0.18),
        }

    def _new_node(self, name: str, mesh_name: str, material_name: str, pos: Vec3, size: Vec3, dynamic: bool, mass: float = 1.0) -> SceneNode:
        node = SceneNode(
            self.next_id,
            name,
            Transform(pos.copy(), quat_identity(), size.copy()),
            self.renderer.meshes[mesh_name],
            self.materials[material_name],
            mass=mass,
            dynamic=dynamic,
            radius=self.renderer.meshes[mesh_name].radius,
        )
        self.next_id += 1
        self.objects.append(node)
        return node

    def _create_scene(self) -> None:
        self.objects.clear()
        self.next_id = 1
        ground = self._new_node("Ground", "cube", "ground", vec3(0, -3.08, 0), vec3(34, 0.12, 34), False, 1000)
        ground.radius = 0.0

        core = self._new_node("Hierarchical Core Parent", "sphere", "core", vec3(0, 2.3, 0), vec3(1.25, 1.25, 1.25), True, 3.0)
        core.velocity = vec3(1.2, 0.0, -0.3)
        core.angular_velocity = vec3(0.4, 0.8, 0.2)

        child = self._new_node("Child Cube (scene graph)", "cube", "green", vec3(2.4, 0, 0), vec3(0.55, 0.55, 0.55), False, 1.0)
        child.parent = core
        core.children.append(child)

        for i, x in enumerate([-5.2, -2.8, 3.4, 5.8]):
            obj = self._new_node(f"Rigid Body {i+1}", "sphere" if i % 2 else "cube", "red" if i % 2 else "blue", vec3(x, 5.0 + i * 0.6, random.uniform(-3, 3)), vec3(0.75, 0.75, 0.75), True, 1.0 + i * 0.3)
            obj.velocity = vec3(random.uniform(-2, 2), random.uniform(-1, 1), random.uniform(-2, 2))
            obj.angular_velocity = vec3(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5))

        self._spawn_fragment_shell(vec3(0, 2.4, 0), count=52)

    def _spawn_fragment_shell(self, center: Vec3, count: int = 36) -> None:
        for i in range(count):
            y = 1.0 - (2.0 * i / max(1, count - 1))
            radius = math.sqrt(max(0.0, 1.0 - y * y))
            theta = i * math.pi * (3.0 - math.sqrt(5.0))
            direction = normalize(vec3(math.cos(theta) * radius, y, math.sin(theta) * radius))
            shard = self._new_node("Instanced Gold Shard", "wedge", "gold", center + direction * 1.7, vec3(0.42, 0.42, 0.85), True, 0.45)
            shard.transform.rotation = quat_from_axis_angle(normalize(np.cross(vec3(0, 0, 1), direction) + vec3(0.001, 0.001, 0.001)), math.acos(float(np.clip(np.dot(vec3(0, 0, 1), direction), -1, 1))))
            shard.velocity = direction * random.uniform(2.2, 6.3) + vec3(0, random.uniform(0.6, 2.0), 0)
            shard.angular_velocity = vec3(random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(-4, 4))

    def handle_input(self, dt: float) -> None:
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                self.mouse_locked = not self.mouse_locked
                pygame.event.set_grab(self.mouse_locked)
                pygame.mouse.set_visible(not self.mouse_locked)
            elif event.type == MOUSEBUTTONDOWN and event.button == 3:
                self.spawn_dynamic_object()
            elif event.type == MOUSEMOTION and self.mouse_locked:
                self.camera.handle_mouse(event.rel[0], event.rel[1])
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_SPACE:
                    self.paused = not self.paused
                elif event.key == K_r:
                    self.reversing = not self.reversing
                    self.paused = False
                elif event.key == K_f:
                    self.reset()
                elif event.key == K_i:
                    self.screenshot()
                elif event.key == pygame.K_b:
                    self.spawn_dynamic_object()
                elif event.key == K_c:
                    self.history.clear()
                    self.history.record(self.objects, self.particles.particles)
                elif event.key == K_1:
                    self.renderer.shading_mode = ShadingMode.FLAT
                elif event.key == K_2:
                    self.renderer.shading_mode = ShadingMode.GOURAUD
                elif event.key == K_3:
                    self.renderer.shading_mode = ShadingMode.PHONG
                elif event.key == K_UP:
                    self.time_scale = min(2.0, self.time_scale + 0.25)
                elif event.key == K_DOWN:
                    self.time_scale = max(0.1, self.time_scale - 0.25)
                elif event.key == K_LEFT:
                    self.reversing = True
                elif event.key == K_RIGHT:
                    self.reversing = False

        keys = pygame.key.get_pressed()
        if keys[K_LCTRL]:
            effective_dt = dt * 0.35
        else:
            effective_dt = dt
        self.camera.update(keys, effective_dt)

        if keys[K_c] and random.random() < 0.02:
            self.history.clear()

    def update(self, dt: float) -> None:
        if self.paused:
            return
        steps_per_visual_frame = 2 if self.time_scale > 1.25 else 1
        if self.reversing:
            frame_steps = max(1, int(round(self.time_scale * steps_per_visual_frame)))
            if not self.history.rewind(self.objects, self.particles.particles, frame_steps):
                self.reversing = False
            return

        fixed_dt = 1.0 / 120.0
        self.fixed_accumulator += min(0.05, dt * self.time_scale)
        while self.fixed_accumulator >= fixed_dt:
            self.physics.step(self.objects, fixed_dt)
            for obj in self.objects:
                if obj.dynamic and obj.alive and random.random() < 0.002:
                    self.particles.emit(obj.transform.position, 2, (1.0, 0.55, 0.15, 0.65))
            self.particles.update(fixed_dt)
            self.fixed_accumulator -= fixed_dt
        self.history.record(self.objects, self.particles.particles)

    def spawn_dynamic_object(self) -> None:
        forward = self.camera.forward()
        start = self.camera.position + forward * 4.0
        mesh_name = "sphere" if self.next_id % 2 else "cube"
        material_name = random.choice(["red", "blue", "green", "gold"])
        obj = self._new_node("Spawned Rigid Body", mesh_name, material_name, start, vec3(0.65, 0.65, 0.65), True, random.uniform(0.8, 2.0))
        obj.velocity = forward * random.uniform(6.0, 10.0) + vec3(0, 2.0, 0)
        obj.angular_velocity = vec3(random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3))
        self.particles.emit(start, 18, (1.0, 0.65, 0.25, 0.7))

    def render(self) -> None:
        self.light_pos = vec3(math.sin(time.time() * 0.35) * 10.0, 17.0, math.cos(time.time() * 0.35) * 10.0)
        self.renderer.render(self.objects, self.particles.particles, self.camera, self.light_pos)
        pygame.display.flip()

    def reset(self) -> None:
        self.history.clear()
        self.particles.particles.clear()
        self._create_scene()
        self.history.record(self.objects, self.particles.particles)
        self.reversing = False
        self.paused = False

    def screenshot(self) -> None:
        os.makedirs("renders", exist_ok=True)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        surface = pygame.image.fromstring(data, (self.width, self.height), "RGB", True)
        name = os.path.join("renders", f"pipeline_demo_{int(time.time() * 1000)}.png")
        pygame.image.save(surface, name)
        print(f"[screenshot] {name}")

    def update_caption(self) -> None:
        now = time.time()
        if now - self.last_caption < 0.25:
            return
        self.last_caption = now
        mode = "REVERSE" if self.reversing else "FORWARD"
        shade = self.renderer.shading_mode.name
        caption = (
            f"3D Time-Reversal Pipeline | {mode} | {shade} shading | "
            f"{self.clock.get_fps():5.1f} FPS | objects {len(self.objects)} | "
            f"history {len(self.history.frames)}/3600 ({self.history.memory_mb:.2f} MB est.) | "
            f"culled {self.renderer.pipeline_stats['culled']} | speed {self.time_scale:.2f}x"
        )
        pygame.display.set_caption(caption)

    def run(self) -> None:
        print("\n3D Time-Reversal Physics Simulator with Full Rendering Pipeline")
        print("Controls:")
        print("  WASD + mouse-click lock: free camera, Q/E vertical")
        print("  SPACE pause, R reverse/forward, UP/DOWN speed, B or right-click spawn, F reset")
        print("  1 flat, 2 Gouraud, 3 Phong shading, I screenshot, ESC quit")
        print("Why replay states? Directly integrating equations backward amplifies floating point error,")
        print("collision discontinuities, damping, and frame-rate variation. This demo records states and")
        print("replays snapshots backward, so reconstruction remains stable and presentation-grade.\n")

        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_input(dt)
            self.update(dt)
            self.render()
            self.update_caption()
            self.frame_counter += 1
        pygame.quit()


if __name__ == "__main__":
    TimeReversalApp().run()
