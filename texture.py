"""Texture loading utilities using Pillow and OpenGL texture objects."""

import os

from OpenGL.GL import *
from PIL import Image


def load_texture(path):
    """Load an image file into a mipmapped OpenGL 2D texture."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing texture: {path}")

    image = Image.open(path).convert("RGBA")
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    width, height = image.size
    data = image.tobytes()

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture_id
