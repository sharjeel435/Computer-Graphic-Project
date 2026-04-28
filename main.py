"""
3D Solar System Simulation - Modern OpenGL Pipeline Report
==========================================================

This project demonstrates the full programmable OpenGL rendering pipeline with
Python, Pygame, PyOpenGL, Pillow, and PyGLM. Pygame creates an OpenGL 3.3
core-profile context; each frame clears the color and depth buffers, updates
delta-time animation, and submits GPU draw calls through VAO/VBO/EBO objects.
Planet geometry is generated as indexed UV spheres containing position, normal,
and texture coordinate attributes. The vertex shader receives those attributes,
applies the model, view, and projection matrices, and passes world-space
fragment position, normal, and texture coordinates to the fragment shader.

The MVP pipeline is explicit: the model matrix places each body in its orbit,
applies axial self-rotation, and scales it to planet size; the view matrix comes
from a free-fly lookAt camera; the projection matrix is a 45 degree perspective
projection with a large far plane for the whole solar system. Planet hierarchy
is represented as parent-child transforms, so the Moon is transformed relative
to Earth while Earth orbits the Sun.

The fragment shader implements Phong lighting from a point light at the Sun:
ambient, diffuse, and specular terms are multiplied with sampled planet
textures. The Sun uses the same shader program but enables an emissive path so
it is drawn as a light source without receiving lighting. Depth testing gives
correct occlusion, blending draws Saturn's transparent rings last, and a
separate point-sprite shader renders a 2400-star background.
"""

import os
import sys
import time
import math

import glm
import pygame
from OpenGL.GL import *
from PIL import Image

from camera import Camera
from geometry import create_asteroid_belt, create_dynamic_line_strip, create_orbit_line, create_ring, create_screen_quad, create_starfield, create_uv_sphere
from planet import Planet, SaturnRing
from shaders import (
    ASTEROID_FRAGMENT_SHADER,
    ASTEROID_VERTEX_SHADER,
    ATMOSPHERE_FRAGMENT_SHADER,
    ATMOSPHERE_VERTEX_SHADER,
    BLUR_FRAG,
    BRIGHTNESS_EXTRACT_FRAG,
    COMPOSITE_FRAG,
    GLOW_FRAGMENT_SHADER,
    GLOW_VERTEX_SHADER,
    LINE_FRAGMENT_SHADER,
    LINE_VERTEX_SHADER,
    PLANET_FRAGMENT_SHADER,
    PLANET_VERTEX_SHADER,
    POST_VERTEX_SHADER,
    STAR_FRAGMENT_SHADER,
    STAR_VERTEX_SHADER,
    TEXT_FRAGMENT_SHADER,
    TEXT_VERTEX_SHADER,
    create_program,
    set_float,
    set_int,
    set_mat4,
    set_vec3,
    set_vec4,
)
from texture import load_texture
from ui_text import TextRenderer


WIDTH, HEIGHT = 1400, 900
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")
APP_TITLE = "Computer Graphics Solar System"


PLANET_DATA = [
    ("Mercury", 1.3, 12.0, 4.15, 2.5, 7.0, "mercury.jpg", 18.0, 0.18),
    ("Venus", 2.2, 18.0, 1.62, 1.1, 3.4, "venus.jpg", 24.0, 0.12),
    ("Earth", 2.35, 25.0, 1.00, 3.2, 0.0, "earth.jpg", 64.0, 0.55),
    ("Mars", 1.75, 33.0, 0.53, 2.8, 1.85, "mars.jpg", 28.0, 0.28),
    ("Jupiter", 6.2, 50.0, 0.084, 5.4, 1.3, "jupiter.jpg", 42.0, 0.42),
    ("Saturn", 5.25, 70.0, 0.034, 4.8, 2.5, "saturn.jpg", 36.0, 0.38),
    ("Uranus", 3.9, 90.0, 0.012, 3.4, 0.8, "uranus.jpg", 32.0, 0.30),
    ("Neptune", 3.8, 108.0, 0.006, 3.5, 1.8, "neptune.jpg", 34.0, 0.34),
]

TRAIL_COLORS = {
    "Mercury": (0.92, 0.78, 0.56, 0.34),
    "Venus": (0.95, 0.82, 0.45, 0.34),
    "Earth": (0.38, 0.80, 1.0, 0.38),
    "Moon": (0.82, 0.88, 1.0, 0.30),
    "Mars": (1.0, 0.46, 0.28, 0.34),
    "Jupiter": (0.98, 0.72, 0.48, 0.34),
    "Saturn": (0.96, 0.84, 0.58, 0.34),
    "Uranus": (0.56, 0.95, 0.98, 0.34),
    "Neptune": (0.40, 0.62, 1.0, 0.34),
    "Comet": (0.35, 0.90, 1.00, 0.50),
}


def require_assets():
    required = [
        "sun.jpg",
        "mercury.jpg",
        "venus.jpg",
        "earth.jpg",
        "moon.jpg",
        "mars.jpg",
        "jupiter.jpg",
        "saturn.jpg",
        "saturn_ring.png",
        "uranus.jpg",
        "neptune.jpg",
        "earth_clouds.png",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(ASSET_DIR, name))]
    if missing:
        raise FileNotFoundError(
            "Missing texture assets: "
            + ", ".join(missing)
            + ". Run the asset download step or place files in assets/."
        )


def init_window():
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.SHOWN)
    pygame.display.set_caption(f"{APP_TITLE} - HDR Bloom")
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)

    glViewport(0, 0, WIDTH, HEIGHT)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_PROGRAM_POINT_SIZE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.0, 0.0, 0.015, 1.0)

    version = glGetString(GL_VERSION).decode("ascii", "ignore")
    if not version.startswith("3.") and not version.startswith("4."):
        print(f"[warning] OpenGL version is {version}; requested 3.3 core profile.")
    return version


def create_hdr_fbo(width, height):
    """Floating-point HDR framebuffer the scene is rendered into."""
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("[warning] HDR FBO not complete — bloom disabled.")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return fbo, tex


def create_ping_pong_fbos(width, height):
    """Two half-res FBOs used for separable Gaussian blur."""
    fbos, texs = [], []
    for _ in range(2):
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width // 2, height // 2, 0, GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        fbos.append(fbo)
        texs.append(tex)
    return fbos, texs


def create_scene(textures):
    sun = Planet("Sun", 7.5, 0.0, 0.0, 0.55, 0.0, textures["sun.jpg"], 1.0, 0.0, True)
    planets = []
    earth = None
    saturn = None
    for name, radius, orbit_radius, orbit_speed, rotation_speed, inclination, tex, shininess, specular in PLANET_DATA:
        planet = Planet(
            name,
            radius,
            orbit_radius,
            orbit_speed,
            rotation_speed,
            inclination,
            textures[tex],
            shininess,
            specular,
            False,
        )
        planets.append(planet)
        if name == "Earth":
            earth = planet
        if name == "Saturn":
            saturn = planet

    moon = Planet("Moon", 0.62, 4.2, 5.7, 1.4, 5.1, textures["moon.jpg"], 16.0, 0.14, False, earth)
    ring = SaturnRing(saturn, textures["saturn_ring.png"])
    clouds = Planet("Earth Clouds", 2.48, 0.0, 0.0, 1.15, 0.0, textures["earth_clouds.png"], 8.0, 0.06, False, earth, alpha=0.46)
    return sun, planets, moon, ring, clouds


def update_orbits(sun, planets, dt, simulation_speed):
    sun.update(dt, simulation_speed)
    for planet in planets:
        planet.update(dt, simulation_speed)


def save_screenshot(path, width, height):
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    glReadBuffer(GL_BACK)
    pixels = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGBA", (width, height), pixels)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(path)


def update_trail(history, position, max_points):
    history.append((position.x, position.y, position.z))
    if len(history) > max_points:
        history.pop(0)


def selected_camera_pose(planet, now, mode):
    target = planet.current_position()
    radius = max(22.0, planet.radius * 12.0 + 10.0)
    if mode == "presentation":
        band = int(now * 0.42) % 3
        if band == 0:
            angle = now * 0.45
            offset = glm.vec3(math.cos(angle) * radius, planet.radius * 4.0 + 8.0, math.sin(angle) * radius)
        elif band == 1:
            angle = now * 0.36
            offset = glm.vec3(math.cos(angle) * radius * 0.52, planet.radius * 2.0 + 20.0, math.sin(angle) * radius * 1.15)
        else:
            drift = math.sin(now * 0.55) * 0.65
            offset = glm.vec3(radius * 0.20, planet.radius * 1.7 + 4.0, radius * (1.4 + drift))
        look_target = target + glm.vec3(0.0, planet.radius * 0.32, 0.0)
        return target + offset, look_target

    angle = now * 0.55
    offset = glm.vec3(math.cos(angle) * radius * 0.72, planet.radius * 2.8 + 8.0, math.sin(angle) * radius)
    look_target = target + glm.vec3(0.0, planet.radius * 0.18, 0.0)
    return target + offset, look_target


def smoothstep01(value):
    t = max(0.0, min(1.0, value))
    return t * t * (3.0 - 2.0 * t)


def lerp_float(start, end, t):
    return start + (end - start) * t


def cinematic_camera_pose(now):
    orbit_t = now * 0.08
    desired_position = glm.vec3(
        math.cos(orbit_t) * 155.0,
        104.0 + math.sin(orbit_t * 0.7) * 18.0,
        math.sin(orbit_t) * 155.0,
    )
    return desired_position, glm.vec3(0.0, -4.0, 0.0)


def intro_camera_pose(elapsed, now, planets):
    earth = next(planet for planet in planets if planet.name == "Earth")
    saturn = next(planet for planet in planets if planet.name == "Saturn")
    earth_pos = earth.current_position()
    saturn_pos = saturn.current_position()

    if elapsed < 4.2:
        t = smoothstep01(elapsed / 4.2)
        position = glm.mix(glm.vec3(-270.0, 240.0, 460.0), glm.vec3(158.0, 154.0, 260.0), t)
        target = glm.mix(glm.vec3(0.0, 4.0, 0.0), glm.vec3(0.0, -5.0, 0.0), t)
        fov = lerp_float(62.0, 50.0, t)
        return position, target, fov

    if elapsed < 8.2:
        t = smoothstep01((elapsed - 4.2) / 4.0)
        angle = now * 0.28 + math.pi * 0.15
        start = glm.vec3(158.0, 154.0, 260.0)
        end = glm.vec3(math.cos(angle) * 78.0, 54.0 + math.sin(angle * 1.3) * 8.0, math.sin(angle) * 78.0)
        position = glm.mix(start, end, t)
        target = glm.mix(glm.vec3(0.0, -5.0, 0.0), earth_pos * 0.22 + glm.vec3(0.0, -2.0, 0.0), t)
        fov = lerp_float(50.0, 45.0, t)
        return position, target, fov

    if elapsed < 11.8:
        t = smoothstep01((elapsed - 8.2) / 3.6)
        orbit = now * 0.42
        offset = glm.vec3(math.cos(orbit) * 30.0, 20.0 + math.sin(orbit * 0.8) * 3.5, math.sin(orbit) * 34.0)
        position = glm.mix(glm.vec3(math.cos(now * 0.28) * 78.0, 54.0, math.sin(now * 0.28) * 78.0), earth_pos + offset, t)
        target = glm.mix(earth_pos * 0.22 + glm.vec3(0.0, -2.0, 0.0), earth_pos + glm.vec3(0.0, 0.2, 0.0), t)
        fov = lerp_float(45.0, 38.0, t)
        return position, target, fov

    t = smoothstep01((elapsed - 11.8) / 4.2)
    angle = now * 0.32
    earth_offset = glm.vec3(math.cos(now * 0.42) * 30.0, 20.0, math.sin(now * 0.42) * 34.0)
    saturn_offset = glm.vec3(math.cos(angle) * 92.0, 64.0 + math.sin(angle * 0.9) * 6.0, math.sin(angle) * 104.0)
    position = glm.mix(earth_pos + earth_offset, saturn_pos + saturn_offset, t)
    target = glm.mix(earth_pos + glm.vec3(0.0, 0.2, 0.0), saturn_pos + glm.vec3(0.0, -1.0, 0.0), t)
    fov = lerp_float(38.0, 53.0, t)
    return position, target, fov


def intro_caption(elapsed):
    if elapsed < 4.2:
        return "SOLAR SYSTEM REVEAL"
    if elapsed < 8.2:
        return "SUN FLYBY"
    if elapsed < 11.8:
        return "EARTH AND MOON"
    return "SATURN RING PASS"


def destruction_amount(current, target, dt):
    blend = 1.0 - math.exp(-2.4 * dt)
    return lerp_float(current, target, blend)


def shockwave_values(now, start_time, strength):
    age = now - start_time
    if age < 0.0 or age > 1.35:
        return 0.0, 0.0
    amount = smoothstep01(1.0 - age / 1.35) * strength
    radius = 0.05 + age * 0.78
    return amount, radius


def camera_shake_offset(now, amount):
    if amount <= 0.001:
        return glm.vec3(0.0)
    return (
        camera_shake_axis(now * 39.0, 1.0) * math.sin(now * 62.0)
        + camera_shake_axis(now * 31.0, 2.0) * math.cos(now * 47.0)
    ) * (amount * 2.1)


def camera_shake_axis(value, phase):
    return glm.normalize(glm.vec3(math.sin(value + phase), math.cos(value * 0.77 + phase), math.sin(value * 0.43 + phase)))


def body_destroy_offset(body, amount, now):
    if amount <= 0.001 or body.name == "Sun":
        return glm.vec3(0.0)
    eased = smoothstep01(amount)
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(body.name))
    angle = seed * 0.173
    distance = (body.orbit_radius * 2.6 + body.radius * 18.0 + 44.0) * eased
    lift = ((seed % 5) - 2) * 10.0 * eased + math.sin(now * 1.8 + seed) * 7.0 * eased
    return glm.vec3(math.cos(angle) * distance, lift, math.sin(angle) * distance)


def body_destroy_scale(body, amount, now):
    if body.name == "Sun":
        return 1.0 + smoothstep01(amount) * (0.28 + 0.08 * math.sin(now * 8.0))
    return max(0.08, 1.0 - smoothstep01(amount) * 0.82)


def fragment_direction(seed):
    angle = seed * 1.618
    y = ((seed * 37) % 100) / 50.0 - 1.0
    radius = math.sqrt(max(0.0, 1.0 - y * y))
    return glm.normalize(glm.vec3(math.cos(angle) * radius, y, math.sin(angle) * radius))


def draw_body_fragments(body, mesh, shader_program, amount, now):
    if amount <= 0.02 or body.name == "Sun":
        return
    eased = smoothstep01(amount)
    base = body.current_position()
    name_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(body.name))
    fragment_count = 12 if body.radius < 3.0 else 18

    for i in range(fragment_count):
        seed = name_seed + i * 19
        direction = fragment_direction(seed)
        surface = direction * body.radius * (0.35 + (i % 4) * 0.18)
        burst = direction * (body.radius * 8.0 + body.orbit_radius * 0.34 + (i % 5) * 3.2) * eased
        spin_axis = fragment_direction(seed + 11)
        wobble = glm.vec3(
            math.sin(now * 2.1 + seed) * 0.6,
            math.cos(now * 1.7 + seed) * 0.45,
            math.sin(now * 1.4 + seed * 0.7) * 0.6,
        ) * eased
        position = base + surface * (1.0 - eased * 0.35) + burst + wobble
        size = body.radius * (0.13 + (seed % 7) * 0.018) * (0.65 + eased * 0.35)
        model = glm.mat4(1.0)
        model = glm.translate(model, position)
        model = glm.rotate(model, now * (1.7 + (seed % 9) * 0.22) + seed, spin_axis)
        model = glm.scale(model, glm.vec3(size, size * (0.65 + (seed % 5) * 0.12), size * (0.72 + (seed % 3) * 0.16)))
        body.draw_with_model(mesh, shader_program, model)


def main():
    require_assets()
    version = init_window()

    planet_program = create_program(PLANET_VERTEX_SHADER, PLANET_FRAGMENT_SHADER)
    star_program = create_program(STAR_VERTEX_SHADER, STAR_FRAGMENT_SHADER)
    line_program = create_program(LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER)
    glow_program = create_program(GLOW_VERTEX_SHADER, GLOW_FRAGMENT_SHADER)
    asteroid_program = create_program(ASTEROID_VERTEX_SHADER, ASTEROID_FRAGMENT_SHADER)
    atmosphere_program = create_program(ATMOSPHERE_VERTEX_SHADER, ATMOSPHERE_FRAGMENT_SHADER)
    text_program = create_program(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)
    extract_program = create_program(POST_VERTEX_SHADER, BRIGHTNESS_EXTRACT_FRAG)
    blur_program = create_program(POST_VERTEX_SHADER, BLUR_FRAG)
    composite_program = create_program(POST_VERTEX_SHADER, COMPOSITE_FRAG)
    sphere_mesh = create_uv_sphere(stacks=48, slices=96)
    fragment_mesh = create_uv_sphere(stacks=8, slices=14)
    ring_mesh = create_ring()
    glow_quad = create_screen_quad()
    post_quad = create_screen_quad()
    stars = create_starfield(count=2400, radius=1200.0)
    asteroid_belt = create_asteroid_belt(count=1400)
    comet_trail = create_dynamic_line_strip(max_points=220)
    orbit_lines = []
    for name, _radius, orbit_radius, _orbit_speed, _rotation_speed, inclination, *_rest in PLANET_DATA:
        orbit_lines.append((name, create_orbit_line(orbit_radius, inclination), glm.mat4(1.0)))

    texture_names = [
        "sun.jpg",
        "mercury.jpg",
        "venus.jpg",
        "earth.jpg",
        "moon.jpg",
        "mars.jpg",
        "jupiter.jpg",
        "saturn.jpg",
        "saturn_ring.png",
        "uranus.jpg",
        "neptune.jpg",
        "earth_clouds.png",
    ]
    textures = {name: load_texture(os.path.join(ASSET_DIR, name)) for name in texture_names}
    sun, planets, moon, saturn_ring, earth_clouds = create_scene(textures)
    earth = next(planet for planet in planets if planet.name == "Earth")
    moon_orbit_line = create_orbit_line(moon.orbit_radius, moon.inclination, segments=96)
    text_renderer = TextRenderer(text_program, WIDTH, HEIGHT)

    # HDR + Bloom FBOs
    hdr_fbo, hdr_tex = create_hdr_fbo(WIDTH, HEIGHT)
    ping_pong_fbos, ping_pong_texs = create_ping_pong_fbos(WIDTH, HEIGHT)

    camera = Camera()
    clock = pygame.time.Clock()
    running = True
    paused = False
    simulation_speed = 0.62
    selected_index = 2
    follow_mode = False
    cinematic_mode = True
    presentation_mode = False
    intro_cinematic_duration = 16.0
    intro_cinematic_active = True
    show_labels = True
    show_orbits = True
    show_trails = True
    show_principles = True
    destruction_level = 0.0
    destruction_target = 0.0
    shockwave_start = -10.0
    shockwave_strength = 0.0
    mouse_captured = False
    comet_points = []
    trail_length = 160
    body_trail_meshes = {}
    trail_history = {}
    for body in planets + [moon]:
        body_trail_meshes[body.name] = create_dynamic_line_strip(max_points=trail_length)
        trail_history[body.name] = []
    last_time = time.perf_counter()
    intro_cinematic_end = last_time + intro_cinematic_duration
    start_position, start_target, start_fov = intro_camera_pose(0.0, last_time, planets)
    camera.snap_to(start_position, start_target)
    camera.fov = start_fov
    frame_counter = 0
    fps_timer = time.perf_counter()
    current_fps = 60.0
    screenshot_counter = 0
    pending_screenshot = None

    print(f"OpenGL: {version}", flush=True)
    print("Startup: 16-second cinematic intro, then free exploration.", flush=True)
    print("Controls: WASD move, Q/E up/down, M mouse look, wheel FOV, +/- speed, SPACE pause", flush=True)
    print("Keys 1-8 jump | F follow | C cinematic | P presentation | B shatter/restore | T add/remove trails | O orbits | L labels | F12 screenshot | ESC quit", flush=True)

    while running:
        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        dt = min(dt, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_TAB:
                    intro_cinematic_active = False
                    cinematic_mode = False
                    selected_index = (selected_index + 1) % len(planets)
                elif event.key == pygame.K_f:
                    intro_cinematic_active = False
                    follow_mode = not follow_mode
                    cinematic_mode = False
                elif event.key == pygame.K_c:
                    was_intro = intro_cinematic_active
                    intro_cinematic_active = False
                    cinematic_mode = True if was_intro else not cinematic_mode
                    follow_mode = False
                    presentation_mode = False
                elif event.key == pygame.K_p:
                    intro_cinematic_active = False
                    presentation_mode = not presentation_mode
                    cinematic_mode = False
                    follow_mode = False
                elif event.key == pygame.K_l:
                    show_labels = not show_labels
                elif event.key == pygame.K_o:
                    show_orbits = not show_orbits
                elif event.key == pygame.K_t:
                    show_trails = not show_trails
                elif event.key == pygame.K_b:
                    intro_cinematic_active = False
                    cinematic_mode = False
                    presentation_mode = False
                    follow_mode = False
                    destruction_target = 0.0 if destruction_target > 0.5 else 1.0
                    shockwave_start = now
                    shockwave_strength = 0.62 if destruction_target < 0.5 else 1.0
                elif event.key == pygame.K_g:
                    show_principles = not show_principles
                elif event.key == pygame.K_m:
                    intro_cinematic_active = False
                    cinematic_mode = False
                    mouse_captured = not mouse_captured
                    pygame.event.set_grab(mouse_captured)
                    pygame.mouse.set_visible(not mouse_captured)
                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    simulation_speed = min(5.0, simulation_speed * 1.25)
                elif event.key == pygame.K_MINUS:
                    simulation_speed = max(0.05, simulation_speed / 1.25)
                # Keys 1-8: jump directly to a planet
                elif event.key in (
                    pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                    pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
                ):
                    idx = event.key - pygame.K_1
                    if 0 <= idx < len(planets):
                        intro_cinematic_active = False
                        selected_index = idx
                        follow_mode = True
                        cinematic_mode = False
                        presentation_mode = False
                # F12: screenshot
                elif event.key == pygame.K_F12:
                    screenshot_counter += 1
                    pending_screenshot = os.path.join(BASE_DIR, f"screenshot_{screenshot_counter:04d}.png")
                elif event.key in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_q, pygame.K_e):
                    intro_cinematic_active = False
                    cinematic_mode = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_captured:
                    intro_cinematic_active = False
                    cinematic_mode = False
                    camera.process_mouse(event.rel[0], event.rel[1])
            elif event.type == pygame.MOUSEWHEEL:
                intro_cinematic_active = False
                cinematic_mode = False
                camera.process_scroll(event.y)

        if intro_cinematic_active and now >= intro_cinematic_end:
            intro_cinematic_active = False
            cinematic_mode = False
            camera.fov = 45.0

        if not paused:
            update_orbits(sun, planets, dt, simulation_speed)
        destruction_level = destruction_amount(destruction_level, destruction_target, dt)

        if presentation_mode:
            selected_index = int((now * 0.14) % len(planets))
        selected_planet = planets[selected_index]

        tracked_bodies = planets + [moon]
        for body in tracked_bodies:
            update_trail(trail_history[body.name], body.current_position(), trail_length)

        if intro_cinematic_active:
            elapsed = intro_cinematic_duration - max(0.0, intro_cinematic_end - now)
            desired_position, desired_target, desired_fov = intro_camera_pose(elapsed, now, planets)
            camera.snap_to(desired_position, desired_target)
            camera.fov = desired_fov
        elif destruction_level > 0.02 or destruction_target > 0.5:
            camera.blend_to(glm.vec3(0.0, 215.0, 285.0), glm.vec3(0.0, -4.0, 0.0), dt, position_sharpness=2.6, target_sharpness=3.2)
            camera.fov = lerp_float(camera.fov, 62.0, 1.0 - math.exp(-2.2 * dt))
        elif presentation_mode:
            desired_position, desired_target = selected_camera_pose(selected_planet, now, "presentation")
            camera.blend_to(desired_position, desired_target, dt, position_sharpness=2.8, target_sharpness=3.6)
        elif cinematic_mode:
            desired_position, desired_target = cinematic_camera_pose(now)
            camera.blend_to(desired_position, desired_target, dt, position_sharpness=2.1, target_sharpness=2.8)
        elif follow_mode:
            desired_position, desired_target = selected_camera_pose(selected_planet, now, "follow")
            camera.blend_to(desired_position, desired_target, dt, position_sharpness=4.8, target_sharpness=6.2)
        else:
            camera.process_keyboard(dt)

        comet_pos = comet_position(now)
        update_trail(comet_points, comet_pos, 220)

        shock_amount, shock_radius = shockwave_values(now, shockwave_start, shockwave_strength)
        projection = glm.perspective(glm.radians(camera.fov), WIDTH / HEIGHT, 0.1, 2000.0)
        if shock_amount > 0.001:
            shake = camera_shake_offset(now, shock_amount)
            view = glm.lookAt(camera.position + shake, camera.position + shake + camera.front, camera.up)
        else:
            view = camera.view_matrix()

        # ── 1. Render scene into HDR FBO ─────────────────────────────────────
        glBindFramebuffer(GL_FRAMEBUFFER, hdr_fbo)
        glViewport(0, 0, WIDTH, HEIGHT)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glDepthMask(GL_FALSE)
        glUseProgram(star_program)
        set_mat4(star_program, "uView", view)
        set_mat4(star_program, "uProjection", projection)
        set_float(star_program, "uTime", now)
        stars.draw()
        glDepthMask(GL_TRUE)

        if show_orbits:
            glUseProgram(line_program)
            set_mat4(line_program, "uView", view)
            set_mat4(line_program, "uProjection", projection)
            set_vec4(line_program, "uColor", (0.30, 0.55, 0.90, 0.22))
            set_mat4(line_program, "uModel", glm.mat4(1.0))
            glEnable(GL_BLEND)
            for _name, line, model in orbit_lines:
                set_mat4(line_program, "uModel", model)
                line.draw()
            set_vec4(line_program, "uColor", (0.70, 0.80, 1.00, 0.36))
            set_mat4(line_program, "uModel", earth.orbital_matrix())
            moon_orbit_line.draw()

        if show_trails:
            glUseProgram(line_program)
            set_mat4(line_program, "uView", view)
            set_mat4(line_program, "uProjection", projection)
            set_mat4(line_program, "uModel", glm.mat4(1.0))
            for body in tracked_bodies:
                color = TRAIL_COLORS.get(body.name, (0.72, 0.88, 1.0, 0.28))
                if body.name == selected_planet.name:
                    color = (min(color[0] + 0.16, 1.0), min(color[1] + 0.10, 1.0), min(color[2] + 0.10, 1.0), 0.55)
                set_vec4(line_program, "uColor", color)
                body_trail_meshes[body.name].update_and_draw(trail_history[body.name])
            set_vec4(line_program, "uColor", TRAIL_COLORS["Comet"])
            comet_trail.update_and_draw(comet_points)

        glUseProgram(asteroid_program)
        set_mat4(asteroid_program, "uView", view)
        set_mat4(asteroid_program, "uProjection", projection)
        set_vec3(asteroid_program, "uLightPos", glm.vec3(0.0, 0.0, 0.0))
        set_vec3(asteroid_program, "uViewPos", camera.position)
        set_float(asteroid_program, "uTime", now)
        asteroid_belt.draw()

        glDepthMask(GL_FALSE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glUseProgram(glow_program)
        set_vec3(glow_program, "uCenter", glm.vec3(0.0, 0.0, 0.0))
        set_vec3(glow_program, "uCameraRight", camera.right)
        set_vec3(glow_program, "uCameraUp", camera.up)
        set_float(glow_program, "uSize", 42.0 * body_destroy_scale(sun, destruction_level, now))
        set_mat4(glow_program, "uView", view)
        set_mat4(glow_program, "uProjection", projection)
        glow_quad.draw()
        set_vec3(glow_program, "uCenter", comet_pos)
        set_float(glow_program, "uSize", 3.2)
        glow_quad.draw()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_TRUE)

        glUseProgram(planet_program)
        set_mat4(planet_program, "uView", view)
        set_mat4(planet_program, "uProjection", projection)
        set_vec3(planet_program, "uLightPos", glm.vec3(0.0, 0.0, 0.0))
        set_vec3(planet_program, "uViewPos", camera.position)
        set_float(planet_program, "uAmbientStrength", 0.08)
        set_float(planet_program, "uTime", now)   # animated Sun

        sun.draw(sphere_mesh, planet_program, scale_multiplier=body_destroy_scale(sun, destruction_level, now))
        for planet in planets:
            planet.draw(
                sphere_mesh,
                planet_program,
                body_destroy_offset(planet, destruction_level, now),
                body_destroy_scale(planet, destruction_level, now),
            )
            draw_body_fragments(planet, fragment_mesh, planet_program, destruction_level, now)
        moon.draw(
            sphere_mesh,
            planet_program,
            body_destroy_offset(moon, destruction_level, now),
            body_destroy_scale(moon, destruction_level, now),
        )
        draw_body_fragments(moon, fragment_mesh, planet_program, destruction_level, now)

        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)
        earth_clouds.draw(
            sphere_mesh,
            planet_program,
            body_destroy_offset(earth, destruction_level, now),
            max(0.02, body_destroy_scale(earth, destruction_level, now) * (1.0 - smoothstep01(destruction_level) * 0.45)),
        )
        glDepthMask(GL_TRUE)

        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)
        glDisable(GL_CULL_FACE)
        saturn_ring.draw(ring_mesh, planet_program, body_destroy_offset(saturn_ring.saturn, destruction_level, now))
        glEnable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)

        # ── Earth atmospheric glow (rim lighting) ────────────────────────────
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glDepthMask(GL_FALSE)
        glDisable(GL_CULL_FACE)
        glUseProgram(atmosphere_program)
        set_mat4(atmosphere_program, "uView", view)
        set_mat4(atmosphere_program, "uProjection", projection)
        set_vec3(atmosphere_program, "uViewPos", camera.position)
        # Slightly larger sphere around Earth
        atm_model = earth.model_matrix(
            body_destroy_offset(earth, destruction_level, now),
            body_destroy_scale(earth, destruction_level, now),
        )
        atm_scale = glm.scale(glm.mat4(1.0), glm.vec3(1.12))
        set_mat4(atmosphere_program, "uModel", atm_model * atm_scale)
        set_vec3(atmosphere_program, "uAtmosphereColor", glm.vec3(0.2, 0.5, 1.0))
        set_float(atmosphere_program, "uAtmosphereStrength", 0.85)
        sphere_mesh.draw()
        glEnable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # ── 2. Extract bright pixels (bloom source) ─────────────────────────
        glBindFramebuffer(GL_FRAMEBUFFER, ping_pong_fbos[0])
        glViewport(0, 0, WIDTH // 2, HEIGHT // 2)
        glDisable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(extract_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, hdr_tex)
        set_int(extract_program, "uScene", 0)
        set_float(extract_program, "uThreshold", 0.72)
        post_quad.draw()

        # ── 3. Separable Gaussian blur (5 ping-pong passes) ─────────────────
        glUseProgram(blur_program)
        for i in range(5):
            src, dst = i % 2, (i + 1) % 2
            glBindFramebuffer(GL_FRAMEBUFFER, ping_pong_fbos[dst])
            glClear(GL_COLOR_BUFFER_BIT)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, ping_pong_texs[src])
            set_int(blur_program, "uImage", 0)
            set_int(blur_program, "uHorizontal", 1 if i % 2 == 0 else 0)
            post_quad.draw()
        bloom_tex = ping_pong_texs[5 % 2]

        # ── 4. Composite: tone-map HDR + add bloom, render to screen ────────
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, WIDTH, HEIGHT)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(composite_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, hdr_tex)
        set_int(composite_program, "uScene", 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, bloom_tex)
        set_int(composite_program, "uBloom", 1)
        set_float(composite_program, "uExposure", 1.05)
        set_float(composite_program, "uBloomStrength", 1.2)
        set_float(composite_program, "uTime", now)
        set_float(composite_program, "uModePulse", 1.0 if (presentation_mode or cinematic_mode) else 0.35)
        set_float(composite_program, "uShockAmount", shock_amount)
        set_float(composite_program, "uShockRadius", shock_radius)
        post_quad.draw()

        # ── 5. UI overlay (after composite so it is not tone-mapped) ────────
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        if show_labels and not intro_cinematic_active:
            label_bodies = [sun] + planets + [moon]
            for body in label_bodies:
                screen = project_to_screen(body.world_position + body_destroy_offset(body, destruction_level, now), view, projection)
                if screen is None:
                    continue
                color = (1.0, 0.86, 0.45, 1.0) if body.name == selected_planet.name else (0.74, 0.88, 1.0, 0.92)
                text_renderer.draw(body.name, screen[0] + 7, screen[1] - 9, color, small=True)
                if body.name == selected_planet.name:
                    text_renderer.draw("Sharjeel & Suhaib", screen[0] + 7, screen[1] + 10, (1.0, 0.86, 0.45, 0.98), small=True)
        selected_screen = project_to_screen(
            selected_planet.world_position + body_destroy_offset(selected_planet, destruction_level, now),
            view,
            projection,
        )
        if selected_screen is not None and not intro_cinematic_active:
            text_renderer.draw("Sharjeel & Suhaib", selected_screen[0] + 10, selected_screen[1] + 28, (1.0, 0.86, 0.45, 1.0))
        mode = "INTRO" if intro_cinematic_active else ("PRESENTATION" if presentation_mode else ("CINEMATIC" if cinematic_mode else ("FOLLOW" if follow_mode else "FREE")))
        if destruction_level > 0.05:
            mode = "SYSTEM COLLAPSE" if destruction_target > 0.5 else "SYSTEM RESTORE"
        text_renderer.draw(APP_TITLE.upper(), 16, 16, (1.0, 0.86, 0.36, 1.0))
        text_renderer.draw("Sharjeel", 16, 40, (1.0, 0.86, 0.45, 1.0))
        text_renderer.draw("Suhaib", 16, 64, (1.0, 0.86, 0.45, 1.0))
        trail_status = "ON" if show_trails else "OFF"
        text_renderer.draw(f"Mode: {mode}  |  {selected_planet.name}  |  Speed: {simulation_speed:.2f}x  |  Trails: {trail_status}  |  FPS: {current_fps:.0f}", 16, 88, (0.78, 0.92, 1.0, 0.95), small=True)
        if intro_cinematic_active:
            seconds_left = max(0.0, intro_cinematic_end - now)
            intro_elapsed = intro_cinematic_duration - seconds_left
            text_renderer.draw(intro_caption(intro_elapsed), 16, HEIGHT - 86, (1.0, 0.86, 0.36, 1.0))
            text_renderer.draw(f"Free exploration begins in {seconds_left:04.1f}s  |  Press WASD to control  |  T add/remove trails", 16, HEIGHT - 56, (0.78, 0.92, 1.0, 0.92), small=True)
        elif destruction_level > 0.05:
            action = "Fragments restoring into orbit" if destruction_target < 0.5 else "Planets shattering into fragments"
            text_renderer.draw(f"{action}  |  Press B to {'shatter again' if destruction_target < 0.5 else 'restore'}", 16, 108, (1.0, 0.42, 0.30, 0.98), small=True)
        else:
            text_renderer.draw("Use the controls panel on the right for the full key guide.", 16, 108, (0.70, 0.82, 0.95, 0.86), small=True)
            draw_planet_panel(text_renderer, selected_planet, 16, HEIGHT - 124)
            draw_controls_panel(text_renderer, WIDTH - 390, HEIGHT - 384)
            if show_principles:
                draw_principles_panel(text_renderer, WIDTH - 430, 18)
        glEnable(GL_DEPTH_TEST)

        if pending_screenshot is not None:
            save_screenshot(pending_screenshot, WIDTH, HEIGHT)
            print(f"[screenshot] saved {pending_screenshot}", flush=True)
            pending_screenshot = None

        pygame.display.flip()
        clock.tick(60)

        frame_counter += 1
        if now - fps_timer >= 0.35:
            current_fps = frame_counter / (now - fps_timer)
            frame_counter = 0
            fps_timer = now
            pygame.display.set_caption(
                f"{APP_TITLE} - HDR Bloom | {current_fps:5.1f} FPS | {mode} | {selected_planet.name} | speed {simulation_speed:.2f}x"
            )

    pygame.quit()


def project_to_screen(world_pos, view, projection):
    clip = projection * view * glm.vec4(world_pos, 1.0)
    if clip.w <= 0.0:
        return None
    ndc = glm.vec3(clip) / clip.w
    if ndc.z < -1.0 or ndc.z > 1.0:
        return None
    x = int((ndc.x * 0.5 + 0.5) * WIDTH)
    y = int((1.0 - (ndc.y * 0.5 + 0.5)) * HEIGHT)
    if x < -80 or x > WIDTH + 80 or y < -80 or y > HEIGHT + 80:
        return None
    return x, y


def comet_position(t):
    angle = t * 0.34
    radius = 78.0 + math.sin(t * 0.17) * 12.0
    return glm.vec3(math.cos(angle) * radius, 10.0 + math.sin(angle * 1.7) * 18.0, math.sin(angle) * radius * 0.62)


def draw_planet_panel(text_renderer, planet, x, y):
    draw_ui_panel(x - 10, y - 10, 380, 132, (0.03, 0.06, 0.10, 0.68), (0.16, 0.22, 0.30, 0.92))
    text_renderer.draw("SELECTED BODY", x, y, (1.0, 0.82, 0.36, 1.0), small=True)
    text_renderer.draw(f"{planet.name}", x, y + 20, (0.85, 0.95, 1.0, 1.0))
    text_renderer.draw("Sharjeel & Suhaib", x, y + 46, (1.0, 0.86, 0.45, 1.0))
    text_renderer.draw(f"Orbit radius: {planet.orbit_radius:.1f} | Orbit speed: {planet.orbit_speed:.3f}", x, y + 74, (0.68, 0.84, 1.0, 0.95), small=True)
    text_renderer.draw(f"Self rotation: {planet.rotation_speed:.2f} | Shininess: {planet.shininess:.0f}", x, y + 94, (0.68, 0.84, 1.0, 0.95), small=True)
    text_renderer.draw("Scene graph + MVP + texture + Phong material", x, y + 114, (0.72, 1.0, 0.72, 0.95), small=True)


def draw_principles_panel(text_renderer, x, y):
    draw_ui_panel(x - 12, y - 12, 418, 246, (0.03, 0.06, 0.10, 0.68), (0.16, 0.22, 0.30, 0.92))
    lines = [
        "CG PRINCIPLES DEMONSTRATED",
        "1. VAO/VBO/EBO indexed UV spheres",
        "2. GLSL 330 vertex + fragment shaders",
        "3. Model-View-Projection matrix pipeline",
        "4. Phong lighting from emissive Sun",
        "5. Scene graph: Earth -> Moon -> Clouds",
        "6. Transparent blending: Saturn rings/clouds",
        "7. Instancing: 1400 asteroid draw calls -> 1",
        "8. Twinkling colored starfield + billboard glow",
        "9. Dynamic orbital trails for planets + comet",
        "10. HDR FBO + 2-pass Gaussian Bloom",
        "11. Atmospheric rim/Fresnel glow (Earth)",
        "12. Smooth cinematic camera direction",
    ]
    for i, line in enumerate(lines):
        color = (1.0, 0.86, 0.38, 1.0) if i == 0 else (0.72, 0.88, 1.0, 0.92)
        text_renderer.draw(line, x, y + i * 20, color, small=True)


def draw_controls_panel(text_renderer, x, y):
    lines = [
        ("CONTROLS", (1.0, 0.86, 0.38, 1.0), False),
        ("W A S D      move camera", (0.82, 0.93, 1.0, 0.96), True),
        ("Q / E        move down / up", (0.82, 0.93, 1.0, 0.96), True),
        ("M            toggle mouse-look", (0.82, 0.93, 1.0, 0.96), True),
        ("Mouse Wheel  zoom FOV", (0.82, 0.93, 1.0, 0.96), True),
        ("+ / -        change simulation speed", (0.82, 0.93, 1.0, 0.96), True),
        ("Space        pause / resume", (0.82, 0.93, 1.0, 0.96), True),
        ("Tab          cycle selected planet", (0.82, 0.93, 1.0, 0.96), True),
        ("1-8          jump to a specific planet", (0.82, 0.93, 1.0, 0.96), True),
        ("F            follow selected planet", (0.82, 0.93, 1.0, 0.96), True),
        ("C            cinematic mode", (0.82, 0.93, 1.0, 0.96), True),
        ("P            presentation mode", (0.82, 0.93, 1.0, 0.96), True),
        ("B            shockwave shatter / restore", (1.0, 0.72, 0.55, 0.98), True),
        ("T            add / remove trails", (0.82, 0.93, 1.0, 0.96), True),
        ("O            toggle orbits", (0.82, 0.93, 1.0, 0.96), True),
        ("L            toggle labels", (0.82, 0.93, 1.0, 0.96), True),
        ("G            toggle principles/info panel", (0.82, 0.93, 1.0, 0.96), True),
        ("F12          take screenshot", (0.82, 0.93, 1.0, 0.96), True),
        ("Esc          quit", (0.82, 0.93, 1.0, 0.96), True),
    ]
    draw_ui_panel(x - 12, y - 12, 350, 370, (0.03, 0.06, 0.10, 0.68), (0.16, 0.22, 0.30, 0.92))
    for i, (line, color, small) in enumerate(lines):
        text_renderer.draw(line, x, y + i * 19, color, small=small)


def draw_ui_panel(x, y, width, height, fill_color, border_color):
    return


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        pygame.quit()
        print(f"[fatal] {exc}", flush=True)
        sys.exit(1)
