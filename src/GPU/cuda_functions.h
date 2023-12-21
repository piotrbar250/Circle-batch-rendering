#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <glm/glm.hpp>

struct Params
{
    int numberOfSegments;
    float radius;
    float start_x;
    float start_y;
    float perception;
    float borderForce;
    float minSpeed;
    float maxSpeed;
    float maxForce;
    float alignmentForce;
    float cohesionForce;
    float separationForce;

    int width;
    int height;
    int cellSize;
    int widthCount;
    int heightCount;
    int cellCount;

    float cursorX;
    float cursorY;
};

struct BoidData
{
    glm::vec2 *device_positions;
    glm::vec2 *device_velocities;
    glm::vec2 *device_newPositions;
    glm::vec2 *device_newVelocities;
    glm::vec2 *device_accelerations; 
    glm::vec2 *device_translations; 

    int* device_gridCellIndex;
    int* device_gridCellStart;
    int* device_gridCellEnd;
    int* boidSequence;

    glm::vec2* device_positionsSorted;
    glm::vec2* device_velocitiesSorted;

    int* colorIndex;
    int* colorSorted;

    Params params;
};

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
    void registerVBO(void **cuda_vbo_resource, unsigned int instanceVBO);

    glm::vec2 *getMappedPointer(void **cuda_vbo_resource);
    void unmapResource(void **cuda_vbo_resource);   

    void sendColorsToGPU(int boidsCount, int* device_colorIndex, int* host_colorIndex);
}

namespace cuda_functions_grid
{
    void allocateGrid(int boidsCount, int cellCount, int *&device_gridCellIndex, int *&device_boidSequence, int *&device_gridCellStart, int *&device_gridCellEnd, glm::vec2*& device_positionsSorted, glm::vec2*& device_velocitiesSorted, int*& colorIndex, int*& colorSorted);
    void computeGridCellIndex(int boidsCount, Params params, glm::vec2* device_positions, glm::vec2* device_velocities, int* device_gridCellIndex, int* device_gridCellStart, int* device_gridCellEnd, int* boidSequence, glm::vec2* device_positionsSorted, glm::vec2* device_velocitiesSorted, int* colorIndex, int* colorSorted);
    void sendColorsToGPU(int boidsCount, int* device_colorIndex, int* host_colorIndex);
    void computeNextFrame(int boidsCount, BoidData boidData);
}

#endif