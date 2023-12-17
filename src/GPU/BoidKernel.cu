#include "cuda_functions.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <glm/glm.hpp>

__global__ void swapFramesKernel(int boidsCount, glm::vec2 *positions,  glm::vec2 *velocities, glm::vec2 *newPositions, glm::vec2* newVelocities)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < boidsCount)
    {
        positions[gid] = newPositions[gid];
        velocities[gid] = newVelocities[gid];
    }
}

void swapFrames(int boidsCount, glm::vec2 *positions, glm::vec2 *velocities, glm::vec2* newPositions, glm::vec2* newVelocities)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = boidsCount / threadsPerBlock + 1;

    swapFramesKernel<<<blocksPerGrid, threadsPerBlock>>>(boidsCount, positions, velocities, newPositions, newVelocities);
    cudaDeviceSynchronize();
}


void sendData(int boidsCount, glm::vec2* host_positions, glm::vec2* host_velocities, glm::vec2* host_newPositions, glm::vec2* host_newVelocities)
{


    // cudaMemcpy(device_newPositions, host_newPositions, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    // cudaMemcpy(device_newVelocities, host_newVelocities, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    // swapFrames(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities);

    // cudaMemcpy(host_positions, device_positions, boidsCount * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_velocities, device_velocities, boidsCount * sizeof(glm::vec2), cudaMemcpyDeviceToHost);

    // cudaFree(device_positions);
    // cudaFree(device_velocities);
    // cudaFree(device_newPositions);
    // cudaFree(device_newVelocities);
}


void allocateDataOnGPU(int boidsCount, glm::vec2*& device_positions, glm::vec2*& device_velocities, glm::vec2*& device_newPositions, glm::vec2*& device_newVelocities)
{
    // error handling
    cudaMalloc((void**)&device_positions, boidsCount * sizeof(glm::vec2));
    cudaMalloc((void**)&device_velocities, boidsCount * sizeof(glm::vec2));
    cudaMalloc((void**)&device_newPositions, boidsCount * sizeof(glm::vec2));
    cudaMalloc((void**)&device_newVelocities, boidsCount * sizeof(glm::vec2));
}

void freeDataOnGPU(glm::vec2* device_positions, glm::vec2* device_velocities, glm::vec2* device_newPositions, glm::vec2* device_newVelocities)
{
    cudaFree(device_positions);
    cudaFree(device_velocities);
    cudaFree(device_newPositions);
    cudaFree(device_newVelocities);
}

void sendDataToGPU(int boidsCount, glm::vec2* device_newPositions, glm::vec2* device_newVelocities, glm::vec2* host_newPositions, glm::vec2* host_newVelocities)
{
    cudaMemcpy(device_newPositions, host_newPositions, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    cudaMemcpy(device_newVelocities, host_newVelocities, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);
}

void getDataFromGPU(int boidsCount, glm::vec2* device_positions, glm::vec2* device_velocities, glm::vec2* host_positions, glm::vec2* host_velocities)
{
    cudaMemcpy(host_positions, device_positions, boidsCount * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_velocities, device_velocities, boidsCount * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
}

// void sendDataToGPUv2(int boidsCount, glm::vec2* device_newPositions, glm::vec2* device_newVelocities, glm::vec2* host_newPositions, glm::vec2* host_newVelocities)
// {
//     cudaMemcpy(device_newPositions, host_newPositions, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);
//     cudaMemcpy(device_newVelocities, host_newVelocities, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);
// }


void sendData2(int boidsCount, glm::vec2* host_positions, glm::vec2* host_velocities, glm::vec2* host_newPositions, glm::vec2* host_newVelocities)
{
    glm::vec2* device_positions;
    glm::vec2* device_velocities;

    glm::vec2* device_newPositions;
    glm::vec2* device_newVelocities;

    cudaMalloc((void**)&device_positions, boidsCount * sizeof(glm::vec2));
    cudaMalloc((void**)&device_velocities, boidsCount * sizeof(glm::vec2));

    cudaMalloc((void**)&device_newPositions, boidsCount * sizeof(glm::vec2));
    cudaMalloc((void**)&device_newVelocities, boidsCount * sizeof(glm::vec2));

    cudaMemcpy(device_newPositions, host_newPositions, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    cudaMemcpy(device_newVelocities, host_newVelocities, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    swapFrames(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities);

    cudaMemcpy(host_positions, device_positions, boidsCount * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_velocities, device_velocities, boidsCount * sizeof(glm::vec2), cudaMemcpyDeviceToHost);

    cudaFree(device_positions);
    cudaFree(device_velocities);
    cudaFree(device_newPositions);
    cudaFree(device_newVelocities);
}
