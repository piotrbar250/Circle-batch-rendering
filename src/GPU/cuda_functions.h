#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <glm/glm.hpp>

struct GridParams
{
    int width;
    int height;
    int cellSize;
    int widthCount;
    int heightCount;
    int cellCount;
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

    GridParams params;
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
}

namespace cuda_functions_grid
{
    // void allocateGrid(int boidsCount, int cellCount, int *&device_gridCellIndex, int *&device_boidSequence, int *&device_gridCellStart, int *&device_gridCellEnd, glm::vec2*& device_positionsSorted, glm::vec2*& device_velocitiesSorted);
    // void computeGridCellIndex(int boidsCount, GridParams params, glm::vec2* device_positions, glm::vec2* device_velocities, int* device_gridCellIndex, int* device_gridCellStart, int* device_gridCellEnd, int* boidSequence, glm::vec2* device_positionsSorted, glm::vec2* device_velocitiesSorted);
    // void computeNextFrame(int boidsCount, glm::vec2 *device_positions, glm::vec2 *device_velocities, glm::vec2 *device_newPositions, glm::vec2 *device_newVelocities, glm::vec2 *device_accelerations, glm::vec2 *device_translations, int* device_gridCellIndex, int* device_gridCellStart, int* device_gridCellEnd, int* boidSequence, glm::vec2* device_positionsSorted, glm::vec2* device_velocitiesSorted);
    
    void allocateGrid(int boidsCount, int cellCount, int *&device_gridCellIndex, int *&device_boidSequence, int *&device_gridCellStart, int *&device_gridCellEnd, glm::vec2*& device_positionsSorted, glm::vec2*& device_velocitiesSorted);
    void computeGridCellIndex(int boidsCount, GridParams params, glm::vec2* device_positions, glm::vec2* device_velocities, int* device_gridCellIndex, int* device_gridCellStart, int* device_gridCellEnd, int* boidSequence, glm::vec2* device_positionsSorted, glm::vec2* device_velocitiesSorted);
    void computeNextFrame(int boidsCount, BoidData boidData);
}

#endif