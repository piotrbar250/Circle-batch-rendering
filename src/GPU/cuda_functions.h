#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <glm/glm.hpp>

namespace cuda_functions
{
    // Memory functions
    void allocateDataOnGPU(int boidsCount, glm::vec2 *&device_positions, glm::vec2 *&device_velocities, glm::vec2 *&device_newPositions, glm::vec2 *&device_newVelocities, glm::vec2 *&device_accelerations, glm::vec2 *&device_translations);
    void freeDataOnGPU(glm::vec2 *device_positions, glm::vec2 *device_velocities, glm::vec2 *device_newPositions, glm::vec2 *device_newVelocities, glm::vec2 *device_accelerations, glm::vec2 *device_translations);

    void sendDataToGPU(int boidsCount, glm::vec2 *device_positions, glm::vec2 *device_velocities, glm::vec2 *host_positions, glm::vec2 *host_velocities);
    void getDataFromGPU(int boidsCount, glm::vec2 *device_translations, glm::vec2 *host_translations);

    // Boid kernel
    void computeNextFrame(int boidsCount, glm::vec2 *device_positions, glm::vec2 *device_velocities, glm::vec2 *device_newPositions, glm::vec2 *device_newVelocities, glm::vec2 *device_accelerations, glm::vec2 *device_translations);
    void swapFrames(int boidsCount, glm::vec2 *positions, glm::vec2 *velocities, glm::vec2 *newPositions, glm::vec2 *newVelocities);

    // CUDA-GL-interop
    void registerVBO(void** cuda_vbo_resource, unsigned int instanceVBO);

    glm::vec2* getMappedPointer(void **cuda_vbo_resource);
    void unmapResource(void **cuda_vbo_resource);
    glm::vec2 *getMappedPointerV2(unsigned int instanceVBO);
    glm::vec2 *getMappedPointerV3(void*& cuda_vbo_resource);
}
#endif