#version 460 

in vec3 fragmentColor;
out vec4 screenColor;

uniform vec3 u_Color;

void main()
{
    // screenColor = vec4(u_Color, 1.0);
    screenColor = vec4(fragmentColor, 1.0);
}
