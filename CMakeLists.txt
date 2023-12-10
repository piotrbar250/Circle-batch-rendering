cmake_minimum_required(VERSION 3.0.0)
project(app VERSION 0.1.0)
cmake_policy(SET CMP0072 NEW)
find_package(OpenGL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(app
    src/main.cpp
    src/glad.c
    src/Boid.cpp
    src/global.hpp
)

target_link_libraries(app
    glfw
    OpenGL::GL
)
