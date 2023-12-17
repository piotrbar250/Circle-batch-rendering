#include "cuda_functions.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <glm/glm.hpp>


void allocateDataOnGPU(int boidsCount, glm::vec2*& device_positions, glm::vec2*& device_velocities, glm::vec2*& device_newPositions, glm::vec2*& device_newVelocities, glm::vec2*& device_accelerations, glm::vec2*& device_translations)
{
    // error handling
    cudaMalloc((void**)&device_positions, boidsCount * sizeof(glm::vec2));
    cudaMalloc((void**)&device_velocities, boidsCount * sizeof(glm::vec2));
    cudaMalloc((void**)&device_newPositions, boidsCount * sizeof(glm::vec2));
    cudaMalloc((void**)&device_newVelocities, boidsCount * sizeof(glm::vec2));
    cudaMalloc((void**)&device_accelerations, boidsCount * sizeof(glm::vec2));
    cudaMalloc((void**)&device_translations, boidsCount * sizeof(glm::vec2));
}

void freeDataOnGPU(glm::vec2* device_positions, glm::vec2* device_velocities, glm::vec2* device_newPositions, glm::vec2* device_newVelocities, glm::vec2* device_accelerations, glm::vec2* device_translations)
{
    cudaFree(device_positions);
    cudaFree(device_velocities);
    cudaFree(device_newPositions);
    cudaFree(device_newVelocities);
    cudaFree(device_accelerations);
    cudaFree(device_translations);
}

void sendDataToGPUinit(int boidsCount, glm::vec2* device_positions, glm::vec2* device_velocities, glm::vec2* host_positions, glm::vec2* host_velocities)
{
    cudaMemcpy(device_positions, host_positions, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    cudaMemcpy(device_velocities, host_velocities, boidsCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);
}

void getDataFromGPUinit(int boidsCount, glm::vec2* device_translations, glm::vec2* host_translations)
{
    cudaMemcpy(host_translations, device_translations, boidsCount * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
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
