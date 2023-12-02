cmake_minimum_required(VERSION 3.0.0)
project(base-opengl-project VERSION 0.1.0)
cmake_policy(SET CMP0072 NEW)
find_package(OpenGL REQUIRED)

add_executable(base-opengl-project 
    src/main.cpp
    src/glad.c
)

target_link_libraries(base-opengl-project
    glfw
    OpenGL::GL
)
