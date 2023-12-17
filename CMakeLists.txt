# cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
# project(app LANGUAGES CXX CUDA)

# enable_language(CUDA)  

# # cmake_policy(SET CMP0072 NEW)
# find_package(OpenGL REQUIRED)
# find_package(CUDA REQUIRED)

# # set(CMAKE_CXX_STANDARD 17)
# # set(CMAKE_CXX_STANDARD_REQUIRED True)

# add_executable(app
#     src/main.cpp
#     src/mainGPU.cpp
#     src/glad.c
#     src/CPU/Boid.cpp  
#     src/GPU/BoidKernel.cu
#     src/global.hpp
# )

# target_link_libraries(app
#     glfw
#     OpenGL::GL
# )

# set_target_properties(app PROPERTIES
#     CUDA_ARCHITECTURES "50"  # Set CUDA architectures
#     CUDA_SEPARABLE_COMPILATION ON)  


cmake_minimum_required(VERSION 3.8.0) #deleted
# cmake_minimum_required(VERSION 3.8 FATAL_ERROR) #added
# project(app VERSION 0.1.0)
project(app VERSION 0.1.0)
enable_language(CUDA)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -diag-suppress=20012")

cmake_policy(SET CMP0072 NEW)
find_package(OpenGL REQUIRED)
find_package(CUDA 12.3 REQUIRED)

# find_package(CUDA REQUIRED)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options("$<$<OR:$<CXX_COMPILER_ID:GNU>,$<C_COMPILER_ID:GNU>>:-w>")


add_executable(app
    src/main.cpp
    src/mainGPU.cpp
    src/glad.c
    src/CPU/Boid.cpp  
    src/GPU/BoidKernel.cu
    src/GPU/BoidsMemoryManagement.cu
    src/global.hpp
)

# target_compile_options(app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-diag-suppress=20012-D>)


target_link_libraries(app
    glfw
    OpenGL::GL
)
set_target_properties(app PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(app PROPERTIES
                      CUDA_ARCHITECTURES "50") # Replace with your GPU's compute capability

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -diag-suppress=20012")
