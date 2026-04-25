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
    pygame.display.set_caption("Advanced OpenGL 3.3 Solar System — HDR Bloom")
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
    cinematic_mode = False
    presentation_mode = False
    show_labels = True
    show_orbits = True
    show_trails = True
    show_principles = True
    mouse_captured = False
    comet_points = []
    trail_length = 160
    body_trail_meshes = {}
    trail_history = {}
    for body in planets + [moon]:
        body_trail_meshes[body.name] = create_dynamic_line_strip(max_points=trail_length)
        trail_history[body.name] = []
    camera.snap_to(camera.position, glm.vec3(0.0, 0.0, 0.0))
    last_time = time.perf_counter()
    frame_counter = 0
    fps_timer = time.perf_counter()
    current_fps = 60.0
    screenshot_counter = 0
    pending_screenshot = None

    print(f"OpenGL: {version}", flush=True)
    print("Controls: WASD move, Q/E up/down, M mouse look, wheel FOV, +/- speed, SPACE pause", flush=True)
    print("Keys 1-8 jump | F follow | C cinematic | P presentation | T trails | O orbits | L labels | F12 screenshot | ESC quit", flush=True)

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
                    selected_index = (selected_index + 1) % len(planets)
                elif event.key == pygame.K_f:
                    follow_mode = not follow_mode
                    cinematic_mode = False
                elif event.key == pygame.K_c:
                    cinematic_mode = not cinematic_mode
                    follow_mode = False
                    presentation_mode = False
                elif event.key == pygame.K_p:
                    presentation_mode = not presentation_mode
                    cinematic_mode = False
                    follow_mode = False
                elif event.key == pygame.K_l:
                    show_labels = not show_labels
                elif event.key == pygame.K_o:
                    show_orbits = not show_orbits
                elif event.key == pygame.K_t:
                    show_trails = not show_trails
                elif event.key == pygame.K_g:
                    show_principles = not show_principles
                elif event.key == pygame.K_m:
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
                        selected_index = idx
                        follow_mode = True
                        cinematic_mode = False
                        presentation_mode = False
                # F12: screenshot
                elif event.key == pygame.K_F12:
                    screenshot_counter += 1
                    pending_screenshot = os.path.join(BASE_DIR, f"screenshot_{screenshot_counter:04d}.png")
            elif event.type == pygame.MOUSEMOTION:
                if mouse_captured:
                    camera.process_mouse(event.rel[0], event.rel[1])
            elif event.type == pygame.MOUSEWHEEL:
                camera.process_scroll(event.y)

        if not paused:
            update_orbits(sun, planets, dt, simulation_speed)

        if presentation_mode:
            selected_index = int((now * 0.14) % len(planets))
        selected_planet = planets[selected_index]

        tracked_bodies = planets + [moon]
        for body in tracked_bodies:
            update_trail(trail_history[body.name], body.current_position(), trail_length)

        if presentation_mode:
            desired_position, desired_target = selected_camera_pose(selected_planet, now, "presentation")
            camera.blend_to(desired_position, desired_target, dt, position_sharpness=2.8, target_sharpness=3.6)
        elif cinematic_mode:
            orbit_t = now * 0.08
            desired_position = glm.vec3(
                math.cos(orbit_t) * 155.0,
                72.0 + math.sin(orbit_t * 0.7) * 20.0,
                math.sin(orbit_t) * 155.0,
            )
            camera.blend_to(desired_position, glm.vec3(0.0, 0.0, 0.0), dt, position_sharpness=2.1, target_sharpness=2.8)
        elif follow_mode:
            desired_position, desired_target = selected_camera_pose(selected_planet, now, "follow")
            camera.blend_to(desired_position, desired_target, dt, position_sharpness=4.8, target_sharpness=6.2)
        else:
            camera.process_keyboard(dt)

        comet_pos = comet_position(now)
        update_trail(comet_points, comet_pos, 220)

        projection = glm.perspective(glm.radians(camera.fov), WIDTH / HEIGHT, 0.1, 2000.0)
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
        set_float(glow_program, "uSize", 42.0)
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

        sun.draw(sphere_mesh, planet_program)
        for planet in planets:
            planet.draw(sphere_mesh, planet_program)
        moon.draw(sphere_mesh, planet_program)

        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)
        earth_clouds.draw(sphere_mesh, planet_program)
        glDepthMask(GL_TRUE)

        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)
        glDisable(GL_CULL_FACE)
        saturn_ring.draw(ring_mesh, planet_program)
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
        atm_model = earth.model_matrix()
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
        post_quad.draw()

        # ── 5. UI overlay (after composite so it is not tone-mapped) ────────
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        if show_labels:
            label_bodies = [sun] + planets + [moon]
            for body in label_bodies:
                screen = project_to_screen(body.world_position, view, projection)
                if screen is None:
                    continue
                color = (1.0, 0.86, 0.45, 1.0) if body.name == selected_planet.name else (0.74, 0.88, 1.0, 0.92)
                text_renderer.draw(body.name, screen[0] + 7, screen[1] - 9, color, small=True)
        mode = "PRESENTATION" if presentation_mode else ("CINEMATIC" if cinematic_mode else ("FOLLOW" if follow_mode else "FREE"))
        text_renderer.draw("NEXT-LEVEL SOLAR SYSTEM SHOWCASE", 16, 16, (1.0, 0.86, 0.36, 1.0))
        text_renderer.draw(f"Mode: {mode}  |  {selected_planet.name}  |  Speed: {simulation_speed:.2f}x  |  FPS: {current_fps:.0f}", 16, 40, (0.78, 0.92, 1.0, 0.95), small=True)
        text_renderer.draw("Use the controls panel on the right for the full key guide.", 16, 60, (0.70, 0.82, 0.95, 0.86), small=True)
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
                f"Advanced OpenGL Solar System — HDR Bloom | {current_fps:5.1f} FPS | {mode} | {selected_planet.name} | speed {simulation_speed:.2f}x"
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
    draw_ui_panel(x - 10, y - 10, 380, 112, (0.03, 0.06, 0.10, 0.68), (0.16, 0.22, 0.30, 0.92))
    text_renderer.draw("SELECTED BODY", x, y, (1.0, 0.82, 0.36, 1.0), small=True)
    text_renderer.draw(f"{planet.name}", x, y + 20, (0.85, 0.95, 1.0, 1.0))
    text_renderer.draw(f"Orbit radius: {planet.orbit_radius:.1f} | Orbit speed: {planet.orbit_speed:.3f}", x, y + 46, (0.68, 0.84, 1.0, 0.95), small=True)
    text_renderer.draw(f"Self rotation: {planet.rotation_speed:.2f} | Shininess: {planet.shininess:.0f}", x, y + 66, (0.68, 0.84, 1.0, 0.95), small=True)
    text_renderer.draw("Scene graph + MVP + texture + Phong material", x, y + 86, (0.72, 1.0, 0.72, 0.95), small=True)


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
        ("T            toggle trails", (0.82, 0.93, 1.0, 0.96), True),
        ("O            toggle orbits", (0.82, 0.93, 1.0, 0.96), True),
        ("L            toggle labels", (0.82, 0.93, 1.0, 0.96), True),
        ("G            toggle principles/info panel", (0.82, 0.93, 1.0, 0.96), True),
        ("F12          take screenshot", (0.82, 0.93, 1.0, 0.96), True),
        ("Esc          quit", (0.82, 0.93, 1.0, 0.96), True),
    ]
    draw_ui_panel(x - 12, y - 12, 350, 350, (0.03, 0.06, 0.10, 0.68), (0.16, 0.22, 0.30, 0.92))
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
