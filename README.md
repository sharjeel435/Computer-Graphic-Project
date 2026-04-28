# 🌌 Advanced OpenGL 3.3 Solar System — All-Time Best Edition

A real-time, physically-inspired 3D Solar System simulation built from scratch with Python and modern OpenGL 3.3 Core Profile. Demonstrates 11 major Computer Graphics principles in a single, self-contained application.

---

## ✨ Features

### Rendering Pipeline
| Feature | Detail |
|---|---|
| **HDR Framebuffer + Bloom** | Scene rendered to a floating-point FBO; bright pixels extracted, blurred with separable Gaussian, and composited with Reinhard tone-mapping + gamma correction |
| **Phong Lighting** | Per-fragment ambient + diffuse + specular from a point light at the Sun |
| **Animated Sun** | Pulsing emissive surface with warm corona tint driven by `sin(uTime)` |
| **Earth Atmosphere** | Rim/Fresnel glow shader pass renders a blue atmospheric limb around Earth |
| **Instanced Asteroid Belt** | 1 400 asteroids rendered in a single draw call; each tumbles with a unique per-instance rotation derived from a position hash — zero CPU cost |
| **Saturn Rings** | Transparent indexed ring mesh with alpha-blended texture |
| **Earth Cloud Layer** | Semi-transparent sphere independently rotating above the surface |
| **Starfield** | 2 400 point-sprite stars on a large enclosing sphere |
| **Billboard Glow** | Additive-blend quad at the Sun and comet for lens-glow effect |
| **Comet + Dynamic Trail** | Parametric comet orbit with a GPU-streamed `GL_LINE_STRIP` history trail |
| **Orbit Guide Lines** | Inclined elliptical orbit loops per planet + Moon |

### OpenGL Concepts Demonstrated
1. VAO / VBO / EBO — indexed UV sphere geometry
2. GLSL 330 Core vertex & fragment shaders
3. Model – View – Projection matrix pipeline
4. Scene graph: Sun → Planets → Moon → Clouds (parent-child transforms)
5. Transparent alpha blending (Saturn rings, cloud layer)
6. Hardware instancing (`glDrawElementsInstanced`)
7. Point sprites (`GL_PROGRAM_POINT_SIZE`)
8. Framebuffer Objects (HDR FBO + ping-pong blur FBOs)
9. Two-pass separable Gaussian Bloom
10. Atmospheric rim / Fresnel lighting
11. Texture mipmapping (`GL_LINEAR_MIPMAP_LINEAR`)

---

## 🎮 Controls

The app starts with a 16-second cinematic intro: a wide solar-system reveal, a Sun pass, an Earth/Moon shot, and a Saturn flyby. It then automatically switches to free exploration. Press any movement/view mode control to start exploring sooner.

| Key | Action |
|---|---|
| `W A S D` | Move camera (hold `Shift` for 3× speed) |
| `Q / E` | Move camera down / up |
| `M` | Toggle mouse-look (captured / free) |
| Mouse wheel | Zoom (adjust FoV) |
| `+` / `-` | Increase / decrease simulation speed |
| `Space` | Pause / Resume |
| `1` – `8` | Jump directly to Mercury … Neptune (activates Follow mode) |
| `Tab` | Cycle through planets |
| `F` | Toggle **Follow** mode (camera locks onto selected planet) |
| `C` | Toggle **Cinematic** mode (slow orbit around the whole system) |
| `P` | Toggle **Presentation** mode (auto-tours every planet) |
| `B` | Shockwave shatter planets into fragments / restore the solar system |
| `T` | Add / remove orbital trails |
| `L` | Toggle planet name labels |
| `O` | Toggle orbit guide lines |
| `G` | Toggle CG-principles overlay |
| `F12` | Save a PNG screenshot to the project folder |
| `Esc` | Quit |

---

## 🚀 Setup & Running

### Requirements
- Python 3.9+
- A GPU supporting OpenGL 3.3 Core Profile

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run
```bash
python main.py
```
Or double-click **`RUN.bat`** on Windows (installs dependencies automatically).

### Dependencies (`requirements.txt`)
```
numpy>=1.21.0
pygame-ce>=2.5.0
PyOpenGL>=3.1.5
PyOpenGL-accelerate>=3.1.5
Pillow>=10.0.0
PyGLM>=2.7.0
```

---

## 📁 Project Structure

```
Computer-Graphic-Project/
├── main.py          # Main loop, HDR/Bloom pipeline, event handling
├── shaders.py       # All GLSL shader sources + compile/link helpers
├── geometry.py      # Procedural mesh generation (sphere, ring, asteroids…)
├── planet.py        # Planet scene-graph node (orbit, rotation, draw)
├── camera.py        # Free-fly camera with mouse-look
├── texture.py       # Pillow-based mipmapped texture loader
├── ui_text.py       # OpenGL text renderer with LRU texture cache
├── assets/          # Planet texture images (JPG/PNG)
├── requirements.txt
└── RUN.bat          # One-click launcher (Windows)
```

---

## 🏗 Architecture Overview

```
main()
 ├─ init_window()           → Pygame + OpenGL 3.3 Core context
 ├─ create_hdr_fbo()        → Floating-point HDR render target
 ├─ create_ping_pong_fbos() → Two half-res FBOs for Gaussian blur
 ├─ Event loop
 │   ├─ Keyboard / mouse input
 │   └─ Simulation update (orbits, comet)
 └─ Render pass
     ├─ Pass 1 → HDR FBO   : stars, orbits, asteroids, glow, planets, atmosphere
     ├─ Pass 2 → PingPong 0: brightness extract (threshold)
     ├─ Pass 3 → PingPong * : 5× separable Gaussian blur
     ├─ Pass 4 → Screen    : Reinhard tone-map + bloom composite
     └─ Pass 5 → Screen    : HUD text overlay (not tone-mapped)
```

---

## 📝 Notes
- Textures are not included in this repository. Place standard solar system texture images in the `assets/` folder (named as listed in `main.py`).
- Screenshots are saved as `screenshot_XXXX.png` in the project root when you press `F12`.
