"""Higher-level simulation systems that sit beside the planet scene graph."""

import math

import glm

from geometry import create_dynamic_line_strip
from shaders import set_vec4


class CometSystem:
    """Elliptical comet orbit with a live tail that always points away from the Sun."""

    def __init__(self, stream_count=9, tail_segments=18):
        self.phase = 0.35
        self.position = glm.vec3(0.0)
        self.stream_count = stream_count
        self.tail_segments = tail_segments
        self.streams = [create_dynamic_line_strip(max_points=tail_segments) for _ in range(stream_count)]

    def update(self, dt, simulation_speed):
        self.phase = (self.phase + dt * simulation_speed * 0.42) % (math.pi * 2.0)
        self.position = self._elliptical_position(self.phase)

    def _elliptical_position(self, theta):
        semi_major = 96.0
        eccentricity = 0.64
        radius = semi_major * (1.0 - eccentricity * eccentricity) / (1.0 + eccentricity * math.cos(theta))
        return glm.vec3(
            math.cos(theta) * radius - 18.0,
            8.0 + math.sin(theta * 1.7) * 13.0,
            math.sin(theta) * radius * 0.58,
        )

    def tail_paths(self, now):
        away = glm.normalize(self.position) if glm.length(self.position) > 0.001 else glm.vec3(1.0, 0.0, 0.0)
        side = glm.normalize(glm.cross(away, glm.vec3(0.0, 1.0, 0.0)))
        if glm.length(side) < 0.001:
            side = glm.vec3(1.0, 0.0, 0.0)
        up = glm.normalize(glm.cross(side, away))
        sun_distance = max(18.0, glm.length(self.position))
        tail_length = 16.0 + max(0.0, 90.0 - sun_distance) * 0.22
        paths = []
        center = (self.stream_count - 1) * 0.5
        for stream_index in range(self.stream_count):
            spread_sign = stream_index - center
            spread = spread_sign / max(center, 1.0)
            points = []
            for i in range(self.tail_segments):
                t = i / (self.tail_segments - 1)
                wave = math.sin(now * 2.2 + stream_index * 1.7 + t * 6.5) * 0.22
                width = (t * t) * (1.2 + abs(spread) * 2.7)
                offset = side * spread * width + up * wave * t * 1.8
                taper = tail_length * (t ** 1.12)
                points.append(tuple(self.position + away * taper + offset))
            paths.append(points)
        return paths

    def draw_tail(self, line_program, now):
        for i, (stream, points) in enumerate(zip(self.streams, self.tail_paths(now))):
            alpha = 0.42 * (1.0 - i / max(1, self.stream_count) * 0.42)
            set_vec4(line_program, "uColor", (0.38, 0.92, 1.0, alpha))
            stream.update_and_draw(points)
