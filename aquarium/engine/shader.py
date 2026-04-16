"""
shader.py — GLSL Shader Manager  (UPGRADED — All-Time Best Edition)
════════════════════════════════════════════════════════════════════
Compiles and links vertex/fragment shader programs for all render passes.

Shader programs:
  fish       — Blinn-Phong + bioluminescent glow + chromatic ripple
  water      — Fresnel / multi-octave distortion / caustics / foam
  particle   — Additive glow point sprites (bubbles + embers)
  background — Volumetric god-rays / animated nebula night sky
  seabed     — Sand ripple + caustic projection
"""

from OpenGL.GL import *
import numpy as np


# ═══════════════════════════════════════  FISH / CORAL  ═══════════════════════

FISH_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;
uniform float waveMag;
uniform float fishScale;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out float Depth;

void main() {
    vec3 pos = aPos * fishScale;

    // Multi-harmonic body undulation
    pos.y += sin(pos.x * 5.0 + time * 3.5) * waveMag;
    pos.y += sin(pos.x * 9.0 + time * 5.0) * waveMag * 0.3;
    pos.z += cos(pos.x * 4.0 + time * 2.8) * waveMag * 0.5;

    // Tail flap (rear quarter only)
    float tailMask = clamp((pos.x - 0.5) * 3.0, 0.0, 1.0);
    pos.z += sin(time * 6.0) * tailMask * waveMag * 1.5;

    FragPos  = vec3(model * vec4(pos, 1.0));
    Normal   = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = vec2(pos.x * 0.5 + 0.5, pos.y * 0.5 + 0.5);
    Depth    = FragPos.y;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

FISH_FRAG = """
#version 330 core
in vec3  FragPos;
in vec3  Normal;
in vec2  TexCoord;
in float Depth;

uniform vec3  lightPos;
uniform vec3  viewPos;
uniform vec3  fishColor;
uniform float time;
uniform bool  bioluminescent;
uniform bool  nightMode;
uniform bool  isPredator;
uniform float fearLevel;   // 0..1, set when fish is frightened

out vec4 FragColor;

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

void main() {
    // ── Ambient ───────────────────────────────────────────────────────────────
    float ambStr = nightMode ? 0.04 : 0.28;
    vec3  ambient = ambStr * fishColor;

    // ── Diffuse ───────────────────────────────────────────────────────────────
    vec3  N        = normalize(Normal);
    vec3  L        = normalize(lightPos - FragPos);
    float diff     = max(dot(N, L), 0.0);
    vec3  diffuse  = diff * fishColor;

    // ── Specular (Blinn-Phong) ─────────────────────────────────────────────────
    vec3  V        = normalize(viewPos - FragPos);
    vec3  H        = normalize(L + V);
    float spec     = pow(max(dot(N, H), 0.0), 96.0);
    vec3  specular = spec * vec3(0.9, 0.95, 1.2);

    // ── Scale rainbow / iridescent stripe pattern ──────────────────────────────
    float stripe  = sin(TexCoord.x * 18.0 + TexCoord.y * 8.0) * 0.5 + 0.5;
    vec3  irid    = mix(fishColor, fishColor.zxy, stripe * 0.25);

    // ── Bioluminescent pulse ──────────────────────────────────────────────────
    vec3 glow = vec3(0.0);
    if (bioluminescent) {
        float pulse  = 0.5 + 0.5 * sin(time * 2.2 + FragPos.x * 4.0 + FragPos.z * 2.0);
        float pulse2 = 0.5 + 0.5 * sin(time * 3.7 + FragPos.y * 3.0);
        glow = irid * pulse * pulse2 * (nightMode ? 2.5 : 0.8);
    }

    // ── Predator flash (red danger) ──────────────────────────────────────────
    vec3 dangerFlash = vec3(0.0);
    if (isPredator) {
        float flash = 0.5 + 0.5 * sin(time * 4.0);
        dangerFlash = vec3(1.0, 0.1, 0.0) * flash * 0.6;
    }

    // ── Fear response (brightness spike when near predator) ──────────────────
    vec3 fearGlow = irid * fearLevel * 1.8;

    // ── Underwater depth tint ─────────────────────────────────────────────────
    vec3 waterTint = nightMode ? vec3(0.01, 0.03, 0.12) : vec3(0.04, 0.14, 0.32);
    float depthF   = clamp((FragPos.y + 15.0) / 30.0, 0.0, 1.0);

    vec3 result = ambient + diffuse + specular + glow + dangerFlash + fearGlow;
    result = mix(result, waterTint, depthF * 0.45);

    // Chromatic aberration rim
    float rim = 1.0 - max(dot(V, N), 0.0);
    rim = pow(rim, 3.0);
    result += fishColor.brg * rim * 0.15;

    FragColor = vec4(result, 1.0);
}
"""


# ═══════════════════════════════════════  WATER SURFACE  ══════════════════════

WATER_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

uniform mat4  projection;
uniform mat4  view;
uniform float time;

out vec2  TexCoord;
out vec3  FragPos;
out vec3  Normal;
out float WaveHeight;

void main() {
    vec3 pos = aPos;

    // ── Multi-octave Gerstner-ish surface waves ─────────────────────────────
    float w0 = sin(pos.x * 1.1 + time * 1.6) * 0.38
             + sin(pos.z * 0.9 + time * 1.2) * 0.28
             + sin((pos.x + pos.z) * 0.65 + time * 2.1) * 0.14
             + sin(pos.x * 3.2 + time * 3.0) * 0.05
             + sin(pos.z * 2.7 + time * 2.5) * 0.04;

    pos.y += w0;
    WaveHeight = w0;

    // Approximate wave normal via finite differences
    float eps = 0.15;
    float hx = sin((pos.x+eps)*1.1 + time*1.6)*0.38
             + sin(pos.z*0.9 + time*1.2)*0.28
             + sin(((pos.x+eps)+pos.z)*0.65+time*2.1)*0.14;
    float hz = sin(pos.x*1.1 + time*1.6)*0.38
             + sin((pos.z+eps)*0.9+time*1.2)*0.28
             + sin((pos.x+(pos.z+eps))*0.65+time*2.1)*0.14;

    Normal = normalize(vec3(-(hx - w0)/eps, 1.0, -(hz - w0)/eps));

    FragPos  = pos;
    TexCoord = aTexCoord + vec2(time * 0.025, time * 0.018);
    gl_Position = projection * view * vec4(pos, 1.0);
}
"""

WATER_FRAG = """
#version 330 core
in vec2  TexCoord;
in vec3  FragPos;
in vec3  Normal;
in float WaveHeight;

uniform float time;
uniform bool  nightMode;
uniform vec3  lightPos;
uniform vec3  viewPos;

out vec4 FragColor;

// Multi-layer caustic pattern
float causticLayer(vec2 uv, float t, float freq, float speed) {
    vec2 p = fract(uv * freq) - 0.5;
    float r = length(p);
    return pow(0.5 + 0.5 * sin(r * 14.0 - t * speed), 4.0);
}

float caustic(vec2 uv, float t) {
    return causticLayer(uv, t, 5.0, 3.6)  * 0.50
         + causticLayer(uv, t, 8.5, 5.1)  * 0.30
         + causticLayer(uv, t + 1.3, 12.0, 6.8) * 0.20;
}

void main() {
    vec2 uv = TexCoord;

    // ── UV distortion (normal-map driven) ─────────────────────────────────────
    uv.x += sin(uv.y * 9.0 + time * 2.2) * 0.018;
    uv.y += cos(uv.x * 7.0 + time * 1.8) * 0.014;

    // ── Base water colour ─────────────────────────────────────────────────────
    vec3 dayCol   = vec3(0.03, 0.25, 0.55);
    vec3 nightCol = vec3(0.01, 0.04, 0.18);
    vec3 water    = nightMode ? nightCol : dayCol;

    // ── Caustics ──────────────────────────────────────────────────────────────
    float c = caustic(uv, time);
    vec3 causticTint = nightMode ? vec3(0.1, 0.3, 0.9) : vec3(0.55, 0.88, 1.0);
    water += causticTint * c * (nightMode ? 0.18 : 0.32);

    // ── Fresnel reflection ─────────────────────────────────────────────────────
    vec3 V      = normalize(viewPos - FragPos);
    vec3 N      = normalize(Normal);
    float fres  = pow(1.0 - max(dot(V, N), 0.0), 4.0);
    vec3 refCol = nightMode ? vec3(0.05, 0.05, 0.2) : vec3(0.7, 0.9, 1.0);
    water       = mix(water, refCol, fres * 0.55);

    // ── Specular highlight ─────────────────────────────────────────────────────
    vec3 L      = normalize(lightPos - FragPos);
    vec3 H      = normalize(L + V);
    float spec  = pow(max(dot(N, H), 0.0), 200.0);
    water      += vec3(1.0) * spec * (nightMode ? 0.3 : 0.7);

    // ── Foam at crests ─────────────────────────────────────────────────────────
    float foam = smoothstep(0.25, 0.65, WaveHeight);
    water = mix(water, vec3(0.88, 0.96, 1.0), foam * 0.30);

    // ── Alpha: more opaque at steep angles ────────────────────────────────────
    float alpha = mix(0.70, 0.90, fres);

    FragColor = vec4(water, alpha);
}
"""


# ═══════════════════════════════════════  PARTICLES  ══════════════════════════

PARTICLE_VERT = """
#version 330 core
layout(location = 0) in vec3  aPos;
layout(location = 1) in float aSize;
layout(location = 2) in float aAlpha;
layout(location = 3) in float aType;   // 0=bubble, 1=ember/sparkle

uniform mat4  projection;
uniform mat4  view;

out float Alpha;
out float Type;

void main() {
    Alpha = aAlpha;
    Type  = aType;
    vec4 viewPos = view * vec4(aPos, 1.0);
    gl_PointSize = aSize / (-viewPos.z + 0.001);
    gl_Position  = projection * viewPos;
}
"""

PARTICLE_FRAG = """
#version 330 core
in  float Alpha;
in  float Type;
out vec4  FragColor;

uniform vec3  particleColor;
uniform bool  nightMode;

void main() {
    vec2  coord = gl_PointCoord - vec2(0.5);
    float r     = length(coord);
    if (r > 0.5) discard;

    if (Type < 0.5) {
        // ── Bubble: ring outline + faint interior ──────────────────────────
        float ring   = smoothstep(0.42, 0.48, r) * (1.0 - smoothstep(0.48, 0.50, r));
        float fill   = (1.0 - smoothstep(0.0, 0.45, r)) * 0.25;
        float mask   = ring + fill;
        vec3  col    = nightMode ? particleColor * 1.6 : particleColor;
        FragColor    = vec4(col, mask * Alpha);
    } else {
        // ── Ember / sparkle: soft bright star ─────────────────────────────
        float glow   = 1.0 - smoothstep(0.0, 0.5, r);
        glow  = pow(glow, 1.5);
        vec3  col    = mix(vec3(1.0, 0.8, 0.3), particleColor, 0.4);
        FragColor    = vec4(col, glow * Alpha);
    }
}
"""


# ═══════════════════════════════════════  BACKGROUND  ═════════════════════════

BACKGROUND_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
out vec3 vPos;
void main() {
    vPos = aPos;
    gl_Position = vec4(aPos.xy, 0.999, 1.0);
}
"""

BACKGROUND_FRAG = """
#version 330 core
in vec3 vPos;

uniform float time;
uniform bool  nightMode;
uniform vec2  resolution;

out vec4 FragColor;

// Hash / noise
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float smoothNoise(vec2 p){
    vec2  i = floor(p);
    vec2  f = fract(p);
    vec2  u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i),            hash(i+vec2(1,0)), u.x),
               mix(hash(i+vec2(0,1)),  hash(i+vec2(1,1)), u.x), u.y);
}
float fbm(vec2 p){
    float v = 0.0; float a = 0.5;
    for(int i=0;i<5;i++){ v += a * smoothNoise(p); p *= 2.1; a *= 0.5; }
    return v;
}

void main() {
    vec2 uv = vPos.xy * 0.5 + 0.5;

    // ── Gradient ──────────────────────────────────────────────────────────────
    vec3 top    = nightMode ? vec3(0.01, 0.02, 0.09) : vec3(0.04, 0.20, 0.50);
    vec3 bottom = nightMode ? vec3(0.00, 0.01, 0.05) : vec3(0.01, 0.08, 0.28);
    vec3 bg     = mix(bottom, top, uv.y);

    // ── Mid-water depth haze ──────────────────────────────────────────────────
    float haze  = fbm(uv * 3.5 + vec2(time * 0.05, 0.0));
    vec3 hazeCol = nightMode ? vec3(0.01, 0.03, 0.12) : vec3(0.03, 0.18, 0.42);
    bg = mix(bg, hazeCol, haze * 0.18);

    if (!nightMode) {
        // ── Volumetric god-rays ───────────────────────────────────────────────
        float ray = 0.0;
        for (int i = 0; i < 8; i++) {
            float fi  = float(i);
            float xc  = 0.15 + 0.10 * fi + sin(time * 0.25 + fi * 1.3) * 0.08;
            float w   = 0.015 + 0.008 * fi;
            float att = exp(-abs(uv.x - xc) / w);
            ray += att * pow(1.0 - uv.y, 1.5) * 0.18;
        }
        bg += vec3(0.65, 0.88, 1.0) * ray;

        // ── Surface shimmer at top ────────────────────────────────────────────
        float shim  = sin(uv.x * 40.0 + time * 3.0) * sin(uv.x * 27.0 - time * 2.0);
        float shimM = smoothstep(0.85, 1.0, uv.y);
        bg         += vec3(0.8, 0.95, 1.0) * shim * shimM * 0.08;

    } else {
        // ── Nebula cloud ──────────────────────────────────────────────────────
        float neb   = fbm(uv * 2.5 + vec2(time * 0.01, 0.0));
        vec3 nebCol = mix(vec3(0.05, 0.0, 0.18), vec3(0.0, 0.05, 0.25), neb);
        bg = mix(bg, nebCol, neb * 0.45);

        // ── Stars ─────────────────────────────────────────────────────────────
        // Fine stars
        float star1 = step(0.975, hash(floor(uv * 120.0)));
        float tw1   = 0.5 + 0.5 * sin(time * 1.8 + hash(floor(uv * 120.0)) * 40.0);
        bg += vec3(0.9, 0.95, 1.0) * star1 * tw1 * 0.9;

        // Coarse bright stars
        float star2 = step(0.992, hash(floor(uv * 60.0)));
        float tw2   = 0.4 + 0.6 * sin(time * 2.6 + hash(floor(uv * 60.0)) * 70.0);
        bg += vec3(1.0, 0.9, 0.7) * star2 * tw2 * 1.5;

        // ── Moon glow ─────────────────────────────────────────────────────────
        vec2  moonC = vec2(0.82, 0.88);
        float moonD = length(uv - moonC);
        float moon  = smoothstep(0.07, 0.04, moonD);
        float halo  = exp(-moonD * 6.0) * 0.25;
        bg += vec3(0.95, 0.95, 0.75) * (moon + halo);

        // Moonlight column reflection
        float mref = exp(-abs(uv.x - 0.82) * 14.0) * (1.0 - uv.y) * 0.15;
        bg += vec3(0.7, 0.7, 0.5) * mref;
    }

    FragColor = vec4(bg, 1.0);
}
"""


# ═══════════════════════════════════════  SEABED  ═════════════════════════════

SEABED_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

uniform mat4  model;
uniform mat4  view;
uniform mat4  projection;
uniform float time;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

void main() {
    // Subtle sand ripple via vertex displacement
    vec3 pos = aPos;
    pos.y += sin(pos.x * 2.5 + time * 0.4) * 0.05
           + cos(pos.z * 3.1 + time * 0.3) * 0.05;

    FragPos  = vec3(model * vec4(pos, 1.0));
    Normal   = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

SEABED_FRAG = """
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform vec3  lightPos;
uniform vec3  viewPos;
uniform float time;
uniform bool  nightMode;

out vec4 FragColor;

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p){
    vec2 i=floor(p); vec2 f=fract(p)*fract(p)*(3.0-2.0*fract(p));
    return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),
               mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);
}

// Caustic projected on the seabed
float caustic(vec2 uv, float t){
    vec2 p = fract(uv*5.0)-0.5;
    float r=length(p);
    return pow(0.5+0.5*sin(r*12.0-t*4.0),4.0)*0.6
         + pow(0.5+0.5*sin(r*19.0-t*6.0),4.0)*0.3;
}

void main() {
    // ── Procedural sand colour ─────────────────────────────────────────────────
    float n1 = noise(TexCoord * 12.0);
    float n2 = noise(TexCoord * 30.0 + vec2(5.3, 2.1));
    vec3  sandDay   = mix(vec3(0.72, 0.62, 0.40), vec3(0.82, 0.74, 0.55), n1*0.7+n2*0.3);
    vec3  sandNight = mix(vec3(0.18, 0.14, 0.08), vec3(0.25, 0.20, 0.12), n1*0.7+n2*0.3);
    vec3  sand      = nightMode ? sandNight : sandDay;

    // ── Lighting ──────────────────────────────────────────────────────────────
    vec3  N   = normalize(Normal);
    vec3  L   = normalize(lightPos - FragPos);
    float amb = nightMode ? 0.06 : 0.22;
    float dif = max(dot(N, L), 0.0) * (nightMode ? 0.4 : 0.9);

    // ── Caustics on floor ─────────────────────────────────────────────────────
    float c   = caustic(TexCoord * 4.0, time);
    vec3  cTint = nightMode ? vec3(0.1, 0.2, 0.6) : vec3(0.6, 0.85, 1.0);
    vec3 result = sand * (amb + dif) + cTint * c * (nightMode ? 0.06 : 0.18);

    // ── Ripple shadow lines ───────────────────────────────────────────────────
    float ripple = sin(TexCoord.x * 40.0 + time * 0.35) * 0.5 + 0.5;
    result *= 1.0 - ripple * 0.07;

    // ── Depth tint ────────────────────────────────────────────────────────────
    vec3 deepTint = nightMode ? vec3(0.0, 0.01, 0.06) : vec3(0.02, 0.10, 0.26);
    result = mix(result, deepTint, 0.35);

    FragColor = vec4(result, 1.0);
}
"""


# ═══════════════════════════════════════  MANAGER  ════════════════════════════

class ShaderProgram:
    """Compiles, links, and provides uniform helpers for a GLSL program."""

    def __init__(self, vert_src: str, frag_src: str):
        self.id = self._link(self._compile(GL_VERTEX_SHADER,   vert_src),
                             self._compile(GL_FRAGMENT_SHADER, frag_src))
        self._uloc_cache: dict = {}

    # ── Compilation ──────────────────────────────────────────────────────────
    @staticmethod
    def _compile(shader_type: int, src: str) -> int:
        sh = glCreateShader(shader_type)
        glShaderSource(sh, src)
        glCompileShader(sh)
        if not glGetShaderiv(sh, GL_COMPILE_STATUS):
            raise RuntimeError(
                f"Shader compile error:\n{glGetShaderInfoLog(sh).decode()}\n"
                f"Source snippet: {src[:200]}")
        return sh

    @staticmethod
    def _link(vert: int, frag: int) -> int:
        prog = glCreateProgram()
        glAttachShader(prog, vert)
        glAttachShader(prog, frag)
        glLinkProgram(prog)
        glDeleteShader(vert)
        glDeleteShader(frag)
        if not glGetProgramiv(prog, GL_LINK_STATUS):
            raise RuntimeError(
                f"Program link error:\n{glGetProgramInfoLog(prog).decode()}")
        return prog

    # ── Uniform helpers (with location caching) ───────────────────────────────
    def _loc(self, name: str) -> int:
        if name not in self._uloc_cache:
            self._uloc_cache[name] = glGetUniformLocation(self.id, name)
        return self._uloc_cache[name]

    def use(self):                              glUseProgram(self.id)
    def set_bool (self, n, v):                  glUniform1i (self._loc(n), int(v))
    def set_int  (self, n, v):                  glUniform1i (self._loc(n), v)
    def set_float(self, n, v):                  glUniform1f (self._loc(n), v)
    def set_vec2 (self, n, x, y):               glUniform2f (self._loc(n), x, y)
    def set_vec3 (self, n, v):                  glUniform3f (self._loc(n), *v)
    def set_mat4 (self, n, m):
        glUniformMatrix4fv(self._loc(n), 1, GL_FALSE, m.astype(np.float32))


class ShaderLibrary:
    """Factory that builds and caches all shader programs at startup."""
    def __init__(self):
        self.fish       = ShaderProgram(FISH_VERT,       FISH_FRAG)
        self.water      = ShaderProgram(WATER_VERT,      WATER_FRAG)
        self.particle   = ShaderProgram(PARTICLE_VERT,   PARTICLE_FRAG)
        self.background = ShaderProgram(BACKGROUND_VERT, BACKGROUND_FRAG)
        self.seabed     = ShaderProgram(SEABED_VERT,     SEABED_FRAG)
