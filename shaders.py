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
uniform vec3 uLightPos;
uniform vec3 uViewPos;
uniform float uAmbientStrength;
uniform float uSpecularStrength;
uniform float uShininess;
uniform int uEmissive;
uniform float uAlpha;

out vec4 FragColor;

void main()
{
    vec4 texel = texture(uTexture, vTexCoord);
    vec3 texColor = texel.rgb;

    if (uEmissive == 1) {
        FragColor = vec4(texColor * 1.45, uAlpha * texel.a);
        return;
    }

    vec3 ambient = uAmbientStrength * texColor;

    vec3 norm = normalize(vNormal);
    vec3 lightDir = normalize(uLightPos - vFragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * texColor;

    vec3 viewDir = normalize(uViewPos - vFragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), uShininess);
    vec3 specular = uSpecularStrength * spec * vec3(1.0);

    FragColor = vec4(ambient + diffuse + specular, uAlpha * texel.a);
}
"""


STAR_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aBrightness;

uniform mat4 uView;
uniform mat4 uProjection;

out float vBrightness;

void main()
{
    mat4 viewNoTranslation = mat4(mat3(uView));
    gl_Position = uProjection * viewNoTranslation * vec4(aPos, 1.0);
    gl_PointSize = 1.0 + aBrightness * 2.4;
    vBrightness = aBrightness;
}
"""


STAR_FRAGMENT_SHADER = """
#version 330 core
in float vBrightness;
out vec4 FragColor;

void main()
{
    vec2 uv = gl_PointCoord - vec2(0.5);
    float d = length(uv);
    if (d > 0.5) discard;
    float glow = smoothstep(0.5, 0.0, d);
    FragColor = vec4(vec3(0.72, 0.82, 1.0) * vBrightness, glow);
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

out vec3 vFragPos;
out vec3 vNormal;
out vec3 vColor;

void main()
{
    vec4 worldPos = aInstanceModel * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(aInstanceModel))) * aNormal;
    vColor = aInstanceColor;
    gl_Position = uProjection * uView * worldPos;
}
"""


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
