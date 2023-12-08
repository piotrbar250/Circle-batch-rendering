#version 460 

out vec4 screenColor;

uniform vec3 u_Color;

void main()
{
    screenColor = vec4(u_Color, 1.0);
}
