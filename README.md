<div align="center">
  <img src="Showcase.png" alt="Anti-Gravity Aquarium Engine Cinematic Render" width="100%">
  
  # 🌌 Anti-Gravity Virtual Aquarium Engine
  **Version 2.0 — All-Time Best Edition**
  
  <p align="center">
    <a href="#about">About</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#controls">Controls</a> •
    <a href="#technical-architecture">Architecture</a>
  </p>
</div>

---

## 🧠 About The Project

This repository houses an advanced, real-time computer graphics simulation of a **zero-gravity aquatic ecosystem**, written entirely from scratch in Python and OpenGL. 

By removing traditional gravitational constraints and relying heavily on GPU-accelerated **GLSL Shader Pipelines** and emergent **Swarm AI (Boids Model)**, the engine creates a physically unconventional, deeply immersive, and visually stunning underwater world. This is not a standard aquarium—it's a hybrid of procedural simulation, high-performance visual computing, and artistic rendering.

If you are looking for an institution-grade, massive high-density particle and flocking simulation, **this is the All-Time Best Edition.**

---

## 🚀 Key Features

*   🔥 **Massive Boid Swarm AI:** Hundreds of silver schooling fish following Craig Reynolds’ separation, alignment, and cohesion rules. It features predator-prey fear response!
*   🌊 **GPU Water Shader Pipeline:** Multi-octave wave distortion, Fresnel reflections, seabed caustics, and real-time volumetric light shafts (god-rays).
*   🦈 **Multi-Species Ecosystem:** 5 meticulously crafted species including bioluminescent Neon fish, glowing Angels, and predatory Jellyfish.
*   🎥 **6-DOF Interactive Camera:** Smooth momentum-based free-fly camera to explore the deep ocean up close. 
*   🎨 **Day & Night Cycles:** Toggle between a vibrant deep-blue ocean floor with bright magenta coral, and an eerie glowing bioluminescent night mode!
*   🌌 **Anti-Gravity Physics:** Spawn massive vortexes and manipulate forces in real-time, defying standard Navier-Stokes fluid limitations.

---

## 🧩 Technical Architecture

The engine is built around a custom highly optimized loop avoiding Python bottlenecks by keeping heavy computation on the GPU:

1.  **Simulation Core (CPU):** Vectorized Numpy computations handle the Boids algorithms, collision logic, and anti-gravity physics matrix. 
2.  **Rendering Pipeline (GPU):** Instanced rendering dispatches thousands of procedurally generated geometry nodes (Fish, Corals, Kelp, Seabed) through custom GLSL Vertex & Fragment shaders (Blinn-Phong lighting, Caustics, Chromatic Aberration).
3.  **Particle System Architecture:** Dual particle system rendering lightweight bubles and magical bioluminescent sparkles across the space.

---

## 🎮 Controls

*   **WASD + QE** → Free-fly camera (LShift for Boost)
*   **Mouse** → Look around the environment
*   **G** → Toggle anti-gravity physics 
*   **N** → Switch Day / Night Cycle
*   **B** → Toggle volumetric bubbles
*   **V** → Spawn chaotic vortex at camera position
*   **Z** → Focus / Zoom lens
*   **ESC** → Exit safely

---

## ⚙️ Installation & Usage

It only takes a few seconds to run the simulation natively.

**Requirements:**
Python 3.9+ (Works up to 3.14!)

```bash
# 1. Clone the repository
git clone https://github.com/sharjeel435/Computer-Graphic-Project.git
cd Computer-Graphic-Project/aquarium

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Launch Engine
python main.py
```
*(Alternatively, you can just double-click the included `RUN_AQUARIUM.bat` script on Windows!)*

---

### 👨‍💻 Author
**Sharjeel Safdar** — *Computer Graphics Final Project* 🏆
