#version 460 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 translation;
layout (location = 2) in vec3 circleColor;

uniform mat4 u_mvp;

out vec3 fragmentColor;

void main()
{
    vec2 translatedPosition = position + translation;
    gl_Position = u_mvp * vec4(translatedPosition, 0.0, 1.0);
    fragmentColor = circleColor;
}
