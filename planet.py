"""Planet scene-graph node with orbit, self-rotation, texture, and draw logic."""

import math

import glm
from OpenGL.GL import *

from shaders import set_float, set_int, set_mat4


class Planet:
    def __init__(
        self,
        name,
        radius,
        orbit_radius,
        orbit_speed,
        rotation_speed,
        inclination,
        texture_id,
        shininess=32.0,
        specular=0.35,
        emissive=False,
        parent=None,
        alpha=1.0,
    ):
        self.name = name
        self.radius = radius
        self.orbit_radius = orbit_radius
        self.orbit_speed = orbit_speed
        self.rotation_speed = rotation_speed
        self.inclination = inclination
        self.texture_id = texture_id
        self.shininess = shininess
        self.specular = specular
        self.emissive = emissive
        self.parent = parent
        self.alpha = alpha
        self.children = []
        self.orbit_angle = 0.0
        self.rotation_angle = 0.0
        self.world_position = glm.vec3(0.0)
        if parent is not None:
            parent.children.append(self)

    def update(self, dt, simulation_speed):
        self.orbit_angle += self.orbit_speed * simulation_speed * dt
        self.rotation_angle += self.rotation_speed * simulation_speed * dt
        for child in self.children:
            child.update(dt, simulation_speed)

    def model_matrix(self, world_offset=None, scale_multiplier=1.0):
        orbital = self.orbital_matrix()
        if world_offset is not None:
            orbital = glm.translate(glm.mat4(1.0), glm.vec3(world_offset)) * orbital
        model = glm.rotate(orbital, self.rotation_angle, glm.vec3(0, 1, 0))
        model = glm.scale(model, glm.vec3(self.radius * scale_multiplier))
        return model

    def orbital_matrix(self):
        local = glm.mat4(1.0)
        local = glm.rotate(local, glm.radians(self.inclination), glm.vec3(0, 0, 1))
        local = glm.rotate(local, self.orbit_angle, glm.vec3(0, 1, 0))
        local = glm.translate(local, glm.vec3(self.orbit_radius, 0.0, 0.0))
        if self.parent is not None:
            orbital = self.parent.orbital_matrix() * local
        else:
            orbital = local
        self.world_position = glm.vec3(orbital * glm.vec4(0, 0, 0, 1))
        return orbital

    def current_position(self):
        self.orbital_matrix()
        return glm.vec3(self.world_position)

    def draw(self, mesh, shader_program, world_offset=None, scale_multiplier=1.0):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        set_int(shader_program, "uTexture", 0)
        set_float(shader_program, "uSpecularStrength", self.specular)
        set_float(shader_program, "uShininess", self.shininess)
        set_int(shader_program, "uEmissive", 1 if self.emissive else 0)
        set_float(shader_program, "uAlpha", self.alpha)
        set_mat4(shader_program, "uModel", self.model_matrix(world_offset, scale_multiplier))
        mesh.draw()

    def draw_with_model(self, mesh, shader_program, model):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        set_int(shader_program, "uTexture", 0)
        set_float(shader_program, "uSpecularStrength", self.specular)
        set_float(shader_program, "uShininess", self.shininess)
        set_int(shader_program, "uEmissive", 1 if self.emissive else 0)
        set_float(shader_program, "uAlpha", self.alpha)
        set_mat4(shader_program, "uModel", model)
        mesh.draw()


class SaturnRing:
    def __init__(self, saturn, texture_id):
        self.saturn = saturn
        self.texture_id = texture_id

    def model_matrix(self, world_offset=None):
        model = self.saturn.orbital_matrix()
        if world_offset is not None:
            model = glm.translate(glm.mat4(1.0), glm.vec3(world_offset)) * model
        model = glm.rotate(model, glm.radians(26.7), glm.vec3(0, 0, 1))
        model = glm.scale(model, glm.vec3(self.saturn.radius))
        return model

    def draw(self, mesh, shader_program, world_offset=None):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        set_int(shader_program, "uTexture", 0)
        set_float(shader_program, "uSpecularStrength", 0.18)
        set_float(shader_program, "uShininess", 12.0)
        set_int(shader_program, "uEmissive", 0)
        set_float(shader_program, "uAlpha", 0.62)
        set_mat4(shader_program, "uModel", self.model_matrix(world_offset))
        mesh.draw()
