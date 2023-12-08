#version 460 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 translation;

uniform mat4 u_mvp;

void main()
{
    vec2 translatedPosition = position + translation;
    gl_Position = u_mvp * vec4(translatedPosition, 0.0, 1.0);
}
