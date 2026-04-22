"""Procedural meshes uploaded through modern OpenGL VAO/VBO/EBO objects."""

import ctypes
import math
from dataclasses import dataclass

import numpy as np
import glm
from OpenGL.GL import *


@dataclass
class Mesh:
    vao: int
    vbo: int
    ebo: int
    index_count: int

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)


@dataclass
class StarMesh:
    vao: int
    vbo: int
    vertex_count: int

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.vertex_count)
        glBindVertexArray(0)


@dataclass
class LineMesh:
    vao: int
    vbo: int
    vertex_count: int

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINE_LOOP, 0, self.vertex_count)
        glBindVertexArray(0)


@dataclass
class QuadMesh:
    vao: int
    vbo: int
    vertex_count: int = 6

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)


@dataclass
class AsteroidBeltMesh:
    vao: int
    vbo: int
    ebo: int
    instance_vbo: int
    color_vbo: int
    index_count: int
    instance_count: int

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElementsInstanced(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None, self.instance_count)
        glBindVertexArray(0)


def upload_indexed_mesh(vertices, indices):
    """Upload interleaved position/normal/uv vertices to GPU buffers."""
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    vertices = np.asarray(vertices, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.uint32)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    stride = 8 * ctypes.sizeof(ctypes.c_float)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

    glBindVertexArray(0)
    return Mesh(vao, vbo, ebo, len(indices))


def create_uv_sphere(stacks=48, slices=96):
    """Create a unit UV sphere with position, normal, and texture coordinate."""
    vertices = []
    indices = []

    for stack in range(stacks + 1):
        v = stack / stacks
        phi = math.pi * v
        y = math.cos(phi)
        ring_radius = math.sin(phi)

        for slc in range(slices + 1):
            u = slc / slices
            theta = 2.0 * math.pi * u
            x = ring_radius * math.cos(theta)
            z = ring_radius * math.sin(theta)
            vertices.extend([x, y, z, x, y, z, u, 1.0 - v])

    for stack in range(stacks):
        for slc in range(slices):
            first = stack * (slices + 1) + slc
            second = first + slices + 1
            indices.extend([first, second, first + 1])
            indices.extend([first + 1, second, second + 1])

    return upload_indexed_mesh(vertices, indices)


def create_ring(inner_radius=1.25, outer_radius=2.15, segments=192):
    """Create Saturn's flat transparent ring mesh as indexed triangles."""
    vertices = []
    indices = []
    normal = (0.0, 1.0, 0.0)

    for i in range(segments + 1):
        u = i / segments
        theta = 2.0 * math.pi * u
        c, s = math.cos(theta), math.sin(theta)
        for radius, v in ((inner_radius, 0.0), (outer_radius, 1.0)):
            vertices.extend([c * radius, 0.0, s * radius, normal[0], normal[1], normal[2], u * 6.0, v])

    for i in range(segments):
        a = i * 2
        indices.extend([a, a + 1, a + 2])
        indices.extend([a + 1, a + 3, a + 2])

    return upload_indexed_mesh(vertices, indices)


def create_starfield(count=2400, radius=1200.0):
    """Create random point-sprite stars on a large sphere around the camera."""
    rng = np.random.default_rng(1234)
    data = []
    for _ in range(count):
        z = rng.uniform(-1.0, 1.0)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        pos = (r * math.cos(theta) * radius, z * radius, r * math.sin(theta) * radius)
        brightness = rng.uniform(0.35, 1.0)
        data.extend([pos[0], pos[1], pos[2], brightness])

    arr = np.asarray(data, dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)

    stride = 4 * ctypes.sizeof(ctypes.c_float)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glBindVertexArray(0)
    return StarMesh(vao, vbo, count)


def create_orbit_line(radius, inclination=0.0, segments=256):
    """Create an orbit guide line loop in the planet's inclined orbital plane."""
    vertices = []
    inc = math.radians(inclination)
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        y = math.sin(inc) * z
        z2 = math.cos(inc) * z
        vertices.extend([x, y, z2])

    arr = np.asarray(vertices, dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
    glBindVertexArray(0)
    return LineMesh(vao, vbo, segments)


def create_screen_quad():
    """Create a reusable billboard/2D quad with position and uv attributes."""
    vertices = np.asarray(
        [
            -1, -1, 0, 0,
            1, -1, 1, 0,
            1, 1, 1, 1,
            -1, -1, 0, 0,
            1, 1, 1, 1,
            -1, 1, 0, 1,
        ],
        dtype=np.float32,
    )
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    stride = 4 * ctypes.sizeof(ctypes.c_float)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
    glBindVertexArray(0)
    return QuadMesh(vao, vbo)


def create_asteroid_belt(count=1200, inner_radius=38.0, outer_radius=46.0):
    """Create low-poly instanced asteroid rocks between Mars and Jupiter."""
    vertices, indices = _low_poly_rock()
    rng = np.random.default_rng(935)
    matrices = []
    colors = []
    for _ in range(count):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(inner_radius, outer_radius)
        y = rng.normal(0.0, 1.15)
        pos = glm.vec3(math.cos(angle) * radius, y, math.sin(angle) * radius)
        model = glm.mat4(1.0)
        model = glm.translate(model, pos)
        model = glm.rotate(model, rng.uniform(0, 6.28), glm.normalize(glm.vec3(rng.random(), rng.random(), rng.random())))
        s = rng.uniform(0.08, 0.34)
        model = glm.scale(model, glm.vec3(s, s * rng.uniform(0.65, 1.45), s))
        matrices.append(np.asarray(model, dtype=np.float32).T)
        tint = rng.uniform(0.45, 0.82)
        colors.append([tint, tint * rng.uniform(0.82, 0.96), tint * rng.uniform(0.70, 0.86)])

    vertex_arr = np.asarray(vertices, dtype=np.float32)
    index_arr = np.asarray(indices, dtype=np.uint32)
    matrix_arr = np.asarray(matrices, dtype=np.float32)
    color_arr = np.asarray(colors, dtype=np.float32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)
    instance_vbo = glGenBuffers(1)
    color_vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_arr.nbytes, vertex_arr, GL_STATIC_DRAW)
    stride = 6 * ctypes.sizeof(ctypes.c_float)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_arr.nbytes, index_arr, GL_STATIC_DRAW)

    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
    glBufferData(GL_ARRAY_BUFFER, matrix_arr.nbytes, matrix_arr, GL_STATIC_DRAW)
    vec4_size = 4 * ctypes.sizeof(ctypes.c_float)
    for i in range(4):
        glEnableVertexAttribArray(3 + i)
        glVertexAttribPointer(3 + i, 4, GL_FLOAT, GL_FALSE, 4 * vec4_size, ctypes.c_void_p(i * vec4_size))
        glVertexAttribDivisor(3 + i, 1)

    glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
    glBufferData(GL_ARRAY_BUFFER, color_arr.nbytes, color_arr, GL_STATIC_DRAW)
    glEnableVertexAttribArray(7)
    glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, 3 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
    glVertexAttribDivisor(7, 1)

    glBindVertexArray(0)
    return AsteroidBeltMesh(vao, vbo, ebo, instance_vbo, color_vbo, len(indices), count)


def _low_poly_rock():
    points = [
        (-1, 0, 0), (1, 0, 0), (0, -0.75, 0), (0, 0.85, 0), (0, 0, -0.9), (0, 0, 1.1)
    ]
    faces = [
        (0, 3, 5), (5, 3, 1), (1, 3, 4), (4, 3, 0),
        (0, 5, 2), (5, 1, 2), (1, 4, 2), (4, 0, 2)
    ]
    vertices = []
    indices = []
    for face in faces:
        a = np.array(points[face[0]], dtype=np.float32)
        b = np.array(points[face[1]], dtype=np.float32)
        c = np.array(points[face[2]], dtype=np.float32)
        n = np.cross(b - a, c - a)
        n /= max(np.linalg.norm(n), 1e-6)
        start = len(vertices) // 6
        for p in (a, b, c):
            vertices.extend([p[0], p[1], p[2], n[0], n[1], n[2]])
        indices.extend([start, start + 1, start + 2])
    return vertices, indices
