cmake_minimum_required(VERSION 3.0.0)
project(app VERSION 0.1.0)
cmake_policy(SET CMP0072 NEW)
find_package(OpenGL REQUIRED)

add_executable(app
    src/main.cpp
    src/glad.c
)

target_link_libraries(app
    glfw
    OpenGL::GL
)
