#include "cuda_functions.h"
#include <cuda_runtime.h>
#include<cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <stdio.h>
#include <glm/glm.hpp>
#include "Grid.cuh"

namespace cuda_functions
{
    void allocateDataOnGPU(int boidsCount, glm::vec2 *&device_positions, glm::vec2 *&device_velocities, glm::vec2 *&device_newPositions, glm::vec2 *&device_newVelocities, glm::vec2 *&device_accelerations, glm::vec2 *&device_translations)
    {
        cudaMalloc((void **)&device_positions, boidsCount * sizeof(glm::vec2));
        cudaMalloc((void **)&device_velocities, boidsCount * sizeof(glm::vec2));
        cudaMalloc((void **)&device_newPositions, boidsCount * sizeof(glm::vec2));
        cudaMalloc((void **)&device_newVelocities, boidsCount * sizeof(glm::vec2));
        cudaMalloc((void **)&device_accelerations, boidsCount * sizeof(glm::vec2));
        cudaMalloc((void **)&device_translations, boidsCount * sizeof(glm::vec2));
       
    }

    void freeDataOnGPU(glm::vec2 *device_positions, glm::vec2 *device_velocities, glm::vec2 *device_newPositions, glm::vec2 *device_newVelocities, glm::vec2 *device_accelerations, glm::vec2 *device_translations)
    {
        cudaFree(device_positions);
        cudaFree(device_velocities);
        cudaFree(device_newPositions);
        cudaFree(device_newVelocities);
        cudaFree(device_accelerations);
        cudaFree(device_translations);
    }

    void sendDataToGPU(int boidsCount, glm::vec2 *device_positions, glm::vec2 *device_velocities, glm::vec2 *host_positions, glm::vec2 *host_velocities)
    {
        cudaMemcpy(device_positions, host_positions, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);
        cudaMemcpy(device_velocities, host_velocities, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    }

    void getDataFromGPU(int boidsCount, glm::vec2 *device_translations, glm::vec2 *host_translations)
    {
        cudaMemcpy(host_translations, device_translations, boidsCount * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
    }

    void registerVBO(void** cuda_vbo_resource, unsigned int instanceVBO)
    {
        cudaGraphicsGLRegisterBuffer((cudaGraphicsResource**)cuda_vbo_resource, instanceVBO, cudaGraphicsMapFlagsWriteDiscard);
    }

    glm::vec2* getMappedPointer(void **cuda_vbo_resource)
    {

        cudaGraphicsMapResources(1, (cudaGraphicsResource**)cuda_vbo_resource, 0);
        size_t num_bytes;
        glm::vec2* devPtr;
        cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &num_bytes, (cudaGraphicsResource*)*cuda_vbo_resource);
        return devPtr;
    }

    void unmapResource(void **cuda_vbo_resource)
    {
        cudaGraphicsUnmapResources(1, (cudaGraphicsResource**)cuda_vbo_resource,0);
    }

    void sendColorsToGPU(int boidsCount, int* device_colorIndex, int* host_colorIndex)
    {
        cudaMemcpy(device_colorIndex, host_colorIndex, boidsCount * sizeof(int), cudaMemcpyHostToDevice);
    }

}

namespace cuda_functions_grid
{
    void allocateGrid(int boidsCount, int cellCount, int *&device_gridCellIndex, int *&device_boidSequence, int *&device_gridCellStart, int *&device_gridCellEnd, glm::vec2*& device_positionsSorted, glm::vec2*& device_velocitiesSorted,int*& colorIndex ,int*& colorSorted)
    {
        cudaMalloc((void**)&device_gridCellIndex, boidsCount*sizeof(int));
        cudaMalloc((void**)&device_boidSequence, boidsCount*sizeof(int));
        cudaMalloc((void**)&device_gridCellStart, cellCount*sizeof(int));
        cudaMalloc((void**)&device_gridCellEnd, cellCount*sizeof(int));
        cudaMalloc((void**)&device_positionsSorted, boidsCount*sizeof(glm::vec2));
        cudaMalloc((void**)&device_velocitiesSorted, boidsCount*sizeof(glm::vec2));
        cudaMalloc((void **)&colorIndex, boidsCount * sizeof(int));
        cudaMalloc((void**)&colorSorted, boidsCount*sizeof(int));
    } 
}