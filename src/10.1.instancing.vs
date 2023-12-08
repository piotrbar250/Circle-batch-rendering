#version 460 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 transformation;

uniform mat4 u_mvp;

void main()
{
    // gl_Position = (u_mvp * vec4(position, 0.0, 1.0)) + (transformation, 0.0, 1.0);
    gl_Position = u_mvp * (vec4(position, 0.0, 1.0) + vec4(transformation, 0.0, 1.0));
    // gl_Position = u_mvp * (vec4(position, 0.0, 1.0) + vec4(800.0, 800.0, 0.0, 1.0));
}
