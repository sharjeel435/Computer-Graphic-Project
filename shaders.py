"""GLSL 330 core shader sources and compile/link helpers."""

import numpy as np
from OpenGL.GL import *


PLANET_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vFragPos;
out vec3 vNormal;
out vec2 vTexCoord;

void main()
{
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
    vTexCoord = aTexCoord;
    gl_Position = uProjection * uView * worldPos;
}
"""


PLANET_FRAGMENT_SHADER = """
#version 330 core
in vec3 vFragPos;
in vec3 vNormal;
in vec2 vTexCoord;

uniform sampler2D uTexture;
uniform sampler2D uNightTexture;
uniform vec3 uLightPos;
uniform vec3 uViewPos;
uniform float uAmbientStrength;
uniform float uSpecularStrength;
uniform float uShininess;
uniform int uEmissive;
uniform int uHasNightTexture;
uniform float uAlpha;
uniform float uTime;

out vec4 FragColor;

void main()
{
    vec4 texel = texture(uTexture, vTexCoord);
    vec3 texColor = texel.rgb;

    if (uEmissive == 1) {
        // Animated sun surface: pulsing brightness + warm corona tint
        float pulse = 1.0 + 0.06 * sin(uTime * 1.8) + 0.03 * sin(uTime * 3.7 + 1.2);
        vec3 sunColor = mix(texColor * vec3(1.4, 1.1, 0.7), texColor * vec3(1.0, 0.6, 0.2), 0.15);
        FragColor = vec4(sunColor * pulse, uAlpha * texel.a);
        return;
    }

    vec3 ambient = uAmbientStrength * texColor;

    vec3 norm = normalize(vNormal);
    vec3 lightDir = normalize(uLightPos - vFragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * texColor;
    if (uHasNightTexture == 1) {
        vec3 cityLights = texture(uNightTexture, vTexCoord).rgb;
        float nightMask = pow(1.0 - smoothstep(0.02, 0.42, diff), 1.35);
        diffuse += cityLights * nightMask * 2.6;
        ambient *= 0.45;
    }

    vec3 viewDir = normalize(uViewPos - vFragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), uShininess);
    vec3 specular = uSpecularStrength * spec * vec3(1.0);

    FragColor = vec4(ambient + diffuse + specular, uAlpha * texel.a);
}"""


STAR_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aBrightness;
layout (location = 2) in float aPhase;
layout (location = 3) in vec3 aColor;

uniform mat4 uView;
uniform mat4 uProjection;
uniform float uTime;

out float vBrightness;
out vec3 vColor;

void main()
{
    mat4 viewNoTranslation = mat4(mat3(uView));
    gl_Position = uProjection * viewNoTranslation * vec4(aPos, 1.0);
    float twinkle = 0.78 + 0.22 * sin(uTime * (0.65 + aBrightness * 1.8) + aPhase);
    gl_PointSize = 1.1 + aBrightness * 2.8 * twinkle;
    vBrightness = aBrightness * twinkle;
    vColor = aColor;
}
"""


STAR_FRAGMENT_SHADER = """
#version 330 core
in float vBrightness;
in vec3 vColor;
out vec4 FragColor;

void main()
{
    vec2 uv = gl_PointCoord - vec2(0.5);
    float d = length(uv);
    if (d > 0.5) discard;
    float glow = smoothstep(0.5, 0.0, d);
    float core = smoothstep(0.22, 0.0, d);
    vec3 color = vColor * (0.65 + core * 0.55);
    FragColor = vec4(color * vBrightness, glow);
}
"""


LINE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uModel;

void main()
{
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
}
"""


LINE_FRAGMENT_SHADER = """
#version 330 core
uniform vec4 uColor;
out vec4 FragColor;

void main()
{
    FragColor = uColor;
}
"""


GLOW_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform vec3 uCenter;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform float uSize;
uniform mat4 uView;
uniform mat4 uProjection;

out vec2 vTexCoord;

void main()
{
    vec3 worldPos = uCenter + uCameraRight * aPos.x * uSize + uCameraUp * aPos.y * uSize;
    gl_Position = uProjection * uView * vec4(worldPos, 1.0);
    vTexCoord = aTexCoord;
}
"""


GLOW_FRAGMENT_SHADER = """
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;

void main()
{
    vec2 uv = vTexCoord - vec2(0.5);
    float d = length(uv) * 2.0;
    float core = smoothstep(1.0, 0.0, d);
    float halo = smoothstep(1.0, 0.18, d) * 0.55;
    vec3 color = vec3(1.0, 0.55, 0.13) * halo + vec3(1.0, 0.90, 0.45) * core;
    FragColor = vec4(color, max(core, halo) * 0.72);
}
"""


TEXT_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 uProjection;

out vec2 vTexCoord;

void main()
{
    gl_Position = uProjection * vec4(aPos.xy, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"""


TEXT_FRAGMENT_SHADER = """
#version 330 core
in vec2 vTexCoord;

uniform sampler2D uTexture;
uniform vec4 uColor;

out vec4 FragColor;

void main()
{
    vec4 texel = texture(uTexture, vTexCoord);
    FragColor = vec4(uColor.rgb, uColor.a * texel.a);
}
"""


ASTEROID_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 3) in mat4 aInstanceModel;
layout (location = 7) in vec3 aInstanceColor;

uniform mat4 uView;
uniform mat4 uProjection;
uniform vec3 uLightPos;
uniform float uTime;

out vec3 vFragPos;
out vec3 vNormal;
out vec3 vColor;

void main()
{
    // Per-instance tumble: hash position to get a unique random spin speed
    vec3 tr = vec3(aInstanceModel[3]);
    float h = fract(sin(dot(tr.xz, vec2(12.9898, 78.233))) * 43758.5453);
    float angle = uTime * (0.12 + h * 0.50);
    float c = cos(angle); float s = sin(angle);
    // Rotate around world-Y axis per instance
    vec3 rotPos = vec3(c * aPos.x + s * aPos.z, aPos.y, -s * aPos.x + c * aPos.z);

    // Secondary tilt axis from second hash
    float h2 = fract(sin(dot(tr.xz, vec2(93.9898, 17.233))) * 23758.5453);
    float angle2 = uTime * (0.07 + h2 * 0.25);
    float c2 = cos(angle2); float s2 = sin(angle2);
    rotPos = vec3(rotPos.x, c2 * rotPos.y - s2 * rotPos.z, s2 * rotPos.y + c2 * rotPos.z);

    vec4 worldPos = aInstanceModel * vec4(rotPos, 1.0);
    vFragPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(aInstanceModel))) * aNormal;
    vColor = aInstanceColor;
    gl_Position = uProjection * uView * worldPos;
}"""


ASTEROID_FRAGMENT_SHADER = """
#version 330 core
in vec3 vFragPos;
in vec3 vNormal;
in vec3 vColor;

uniform vec3 uLightPos;
uniform vec3 uViewPos;

out vec4 FragColor;

void main()
{
    vec3 n = normalize(vNormal);
    vec3 l = normalize(uLightPos - vFragPos);
    vec3 v = normalize(uViewPos - vFragPos);
    vec3 r = reflect(-l, n);
    float diff = max(dot(n, l), 0.0);
    float spec = pow(max(dot(v, r), 0.0), 18.0) * 0.18;
    vec3 color = vColor * (0.12 + diff * 0.82) + vec3(spec);
    FragColor = vec4(color, 1.0);
}
"""


ATMOSPHERE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform vec3 uViewPos;

out vec3 vNormal;
out vec3 vViewDir;

void main()
{
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vNormal = normalize(mat3(transpose(inverse(uModel))) * aNormal);
    vViewDir = normalize(uViewPos - worldPos.xyz);
    gl_Position = uProjection * uView * worldPos;
}
"""

ATMOSPHERE_FRAGMENT_SHADER = """
#version 330 core
in vec3 vNormal;
in vec3 vViewDir;

uniform vec3 uAtmosphereColor;
uniform float uAtmosphereStrength;

out vec4 FragColor;

void main()
{
    // Rim / Fresnel effect: glow on edges facing away from camera
    float rim = 1.0 - max(dot(normalize(vViewDir), normalize(vNormal)), 0.0);
    rim = pow(rim, 3.5);
    FragColor = vec4(uAtmosphereColor, rim * uAtmosphereStrength);
}
"""


POST_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"""

BRIGHTNESS_EXTRACT_FRAG = """
#version 330 core
in vec2 vTexCoord;
uniform sampler2D uScene;
uniform float uThreshold;
out vec4 FragColor;
void main() {
    vec3 color = texture(uScene, vTexCoord).rgb;
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    FragColor = (brightness > uThreshold) ? vec4(color, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
}
"""

BLUR_FRAG = """
#version 330 core
in vec2 vTexCoord;
uniform sampler2D uImage;
uniform int uHorizontal;
out vec4 FragColor;
void main() {
    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec2 offset = 1.0 / textureSize(uImage, 0);
    vec3 result = texture(uImage, vTexCoord).rgb * weights[0];
    for (int i = 1; i < 5; ++i) {
        vec2 d = (uHorizontal == 1) ? vec2(offset.x * float(i), 0.0) : vec2(0.0, offset.y * float(i));
        result += texture(uImage, vTexCoord + d).rgb * weights[i];
        result += texture(uImage, vTexCoord - d).rgb * weights[i];
    }
    FragColor = vec4(result, 1.0);
}
"""

COMPOSITE_FRAG = """
#version 330 core
in vec2 vTexCoord;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uExposure;
uniform float uBloomStrength;
uniform float uTime;
uniform float uModePulse;
uniform float uShockAmount;
uniform float uShockRadius;
out vec4 FragColor;
void main() {
    const float gamma = 2.2;
    vec2 center = vec2(0.5);
    vec2 fromCenter = vTexCoord - center;
    float dist = length(fromCenter);
    vec2 dir = fromCenter / max(dist, 0.0001);
    float ring = exp(-pow((dist - uShockRadius) * 18.0, 2.0)) * uShockAmount;
    vec2 uv = clamp(vTexCoord + dir * ring * 0.035, vec2(0.001), vec2(0.999));

    vec3 color = texture(uScene, uv).rgb;
    vec3 bloom  = texture(uBloom,  uv).rgb;
    color += bloom * (uBloomStrength + uShockAmount * 1.65);
    float vignette = smoothstep(1.2, 0.2, length(vTexCoord - vec2(0.5)));
    float pulse = 1.0 + 0.03 * sin(uTime * 0.45) * uModePulse;
    color *= mix(0.92, 1.04, vignette) * pulse;
    color += vec3(1.0, 0.42, 0.18) * ring * 1.8;
    color *= 1.0 + uShockAmount * 0.22;
    color = mix(color, color * vec3(1.02, 0.99, 0.96), 0.18);
    // Reinhard tone mapping
    vec3 mapped = vec3(1.0) - exp(-color * uExposure);
    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / gamma));
    FragColor = vec4(mapped, 1.0);
}
"""


def compile_shader(shader_type, source):
    """Compile one GLSL shader and raise a readable error on failure."""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode("utf-8", "ignore")
        glDeleteShader(shader)
        raise RuntimeError(f"Shader compilation failed:\n{log}")
    return shader


def create_program(vertex_source, fragment_source):
    """Compile, link, and validate a shader program."""
    vertex = compile_shader(GL_VERTEX_SHADER, vertex_source)
    fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source)
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    glDeleteShader(vertex)
    glDeleteShader(fragment)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program).decode("utf-8", "ignore")
        glDeleteProgram(program)
        raise RuntimeError(f"Program link failed:\n{log}")

    glValidateProgram(program)
    if not glGetProgramiv(program, GL_VALIDATE_STATUS):
        log = glGetProgramInfoLog(program).decode("utf-8", "ignore")
        glDeleteProgram(program)
        raise RuntimeError(f"Program validation failed:\n{log}")

    return program


def set_mat4(program, name, matrix):
    loc = glGetUniformLocation(program, name)
    glUniformMatrix4fv(loc, 1, GL_TRUE, np.asarray(matrix, dtype=np.float32))


def set_vec3(program, name, value):
    loc = glGetUniformLocation(program, name)
    glUniform3f(loc, float(value.x), float(value.y), float(value.z))


def set_float(program, name, value):
    glUniform1f(glGetUniformLocation(program, name), float(value))


def set_int(program, name, value):
    glUniform1i(glGetUniformLocation(program, name), int(value))


def set_vec4(program, name, value):
    glUniform4f(glGetUniformLocation(program, name), float(value[0]), float(value[1]), float(value[2]), float(value[3]))
