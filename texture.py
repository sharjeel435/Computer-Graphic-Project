"""Texture loading utilities using Pillow and OpenGL texture objects."""

import os
import random

from OpenGL.GL import *
from PIL import Image, ImageDraw, ImageFilter


def load_texture(path):
    """Load an image file into a mipmapped OpenGL 2D texture."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing texture: {path}")

    with Image.open(path) as source:
        image = source.convert("RGBA")
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


def ensure_city_lights_texture(path, width=1024, height=512):
    """Create a procedural Earth night-lights texture when no asset is provided."""
    if os.path.exists(path):
        return

    rng = random.Random(404)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    # Approximate continental city bands in equirectangular UV space.
    clusters = [
        (0.52, 0.43, 0.16, 0.09, 420),  # Europe
        (0.62, 0.50, 0.20, 0.13, 520),  # India / east Asia
        (0.73, 0.43, 0.16, 0.12, 430),  # China / Japan
        (0.24, 0.43, 0.15, 0.12, 380),  # North America
        (0.33, 0.66, 0.10, 0.16, 160),  # South America
        (0.54, 0.61, 0.16, 0.16, 170),  # Africa / Middle East
        (0.78, 0.70, 0.10, 0.06, 70),   # Australia
    ]
    for cx, cy, sx, sy, count in clusters:
        for _ in range(count):
            x = int((rng.gauss(cx, sx * 0.38) % 1.0) * width)
            y = int(max(0.05, min(0.95, rng.gauss(cy, sy * 0.38))) * height)
            size = rng.choice((1, 1, 1, 2, 2, 3))
            intensity = rng.randint(95, 235)
            color = (255, rng.randint(176, 226), rng.randint(92, 150), intensity)
            draw.ellipse((x - size, y - size, x + size, y + size), fill=color)

    glow = image.filter(ImageFilter.GaussianBlur(radius=1.4))
    image = Image.alpha_composite(glow, image)
    image.save(path)
