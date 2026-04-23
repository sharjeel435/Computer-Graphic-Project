"""OpenGL text overlay renderer backed by Pygame font surfaces."""

import ctypes

import glm
import numpy as np
import pygame
from OpenGL.GL import *

from shaders import set_int, set_mat4, set_vec4


class TextRenderer:
    def __init__(self, shader_program, width, height):
        self.program = shader_program
        self.width = width
        self.height = height
        self.font = pygame.font.SysFont("consolas", 16, bold=True)
        self.small = pygame.font.SysFont("consolas", 13)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 4 * ctypes.sizeof(ctypes.c_float), None, GL_DYNAMIC_DRAW)
        stride = 4 * ctypes.sizeof(ctypes.c_float)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glBindVertexArray(0)
        self.projection = glm.ortho(0.0, float(width), float(height), 0.0, -1.0, 1.0)
        
        # LRU Cache for textures
        self.cache = {}
        self.cache_order = []
        self.max_cache_size = 256

    def _get_texture(self, text, small):
        cache_key = (text, small)
        if cache_key in self.cache:
            self.cache_order.remove(cache_key)
            self.cache_order.append(cache_key)
            return self.cache[cache_key]

        font = self.small if small else self.font
        surface = font.render(str(text), True, (255, 255, 255)).convert_alpha()
        w, h = surface.get_size()
        data = pygame.image.tostring(surface, "RGBA", True)

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        tex_data = (texture, w, h)
        self.cache[cache_key] = tex_data
        self.cache_order.append(cache_key)
        
        if len(self.cache_order) > self.max_cache_size:
            oldest_key = self.cache_order.pop(0)
            old_tex, _, _ = self.cache.pop(oldest_key)
            glDeleteTextures([old_tex])
            
        return tex_data

    def draw(self, text, x, y, color=(1.0, 1.0, 1.0, 1.0), small=False):
        texture, w, h = self._get_texture(text, small)

        vertices = np.asarray(
            [
                x, y + h, 0.0, 0.0,
                x, y, 0.0, 1.0,
                x + w, y, 1.0, 1.0,
                x, y + h, 0.0, 0.0,
                x + w, y, 1.0, 1.0,
                x + w, y + h, 1.0, 0.0,
            ],
            dtype=np.float32,
        )

        glUseProgram(self.program)
        set_mat4(self.program, "uProjection", self.projection)
        set_vec4(self.program, "uColor", color)
        set_int(self.program, "uTexture", 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
