"""Free-fly camera with mouse-look and lookAt view matrix."""

import math

import glm
import pygame


class Camera:
    def __init__(self, position=None):
        self.position = position if position is not None else glm.vec3(0.0, 80.0, 185.0)
        self.front = glm.vec3(0.0, -0.26, -1.0)
        self.world_up = glm.vec3(0.0, 1.0, 0.0)
        self.right = glm.vec3(1.0, 0.0, 0.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.yaw = -90.0
        self.pitch = -23.4
        self.speed = 95.0
        self.mouse_sensitivity = 0.11
        self.fov = 45.0
        self.update_vectors()

    def view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)

    def look_at(self, target):
        direction = glm.normalize(target - self.position)
        self.pitch = math.degrees(math.asin(max(-1.0, min(1.0, direction.y))))
        self.yaw = math.degrees(math.atan2(direction.z, direction.x))
        self.update_vectors()

    def process_keyboard(self, dt):
        keys = pygame.key.get_pressed()
        velocity = self.speed * dt
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            velocity *= 3.0
        if keys[pygame.K_w]:
            self.position += self.front * velocity
        if keys[pygame.K_s]:
            self.position -= self.front * velocity
        if keys[pygame.K_a]:
            self.position -= self.right * velocity
        if keys[pygame.K_d]:
            self.position += self.right * velocity
        if keys[pygame.K_q]:
            self.position -= self.world_up * velocity
        if keys[pygame.K_e]:
            self.position += self.world_up * velocity

    def process_mouse(self, dx, dy):
        self.yaw += dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self.update_vectors()

    def process_scroll(self, y_offset):
        self.fov -= y_offset * 2.0
        self.fov = max(20.0, min(75.0, self.fov))

    def update_vectors(self):
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        front = glm.vec3(
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
            math.sin(yaw) * math.cos(pitch),
        )
        self.front = glm.normalize(front)
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))
