#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <glm/glm.hpp>

void allocateDataOnGPU(int boidsCount, glm::vec2*& device_positions, glm::vec2*& device_velocities, glm::vec2*& device_newPositions, glm::vec2*& device_newVelocities);
void freeDataOnGPU(glm::vec2* device_positions, glm::vec2* device_velocities, glm::vec2* device_newPositions, glm::vec2* device_newVelocities);

void sendDataToGPU(int boidsCount, glm::vec2* device_newPositions, glm::vec2* device_newVelocities, glm::vec2* host_newPositions, glm::vec2* host_newVelocities);
void getDataFromGPU(int boidsCount, glm::vec2* device_positions, glm::vec2* device_velocities, glm::vec2* host_positions, glm::vec2* host_velocities);

void swapFrames(int boidsCount, glm::vec2 *positions, glm::vec2 *velocities, glm::vec2* newPositions, glm::vec2* newVelocities);
void sendData(int boidsCount, glm::vec2* host_positions, glm::vec2* host_velocities, glm::vec2* host_newPositions, glm::vec2* host_newVelocities);

void sendData2(int boidsCount, glm::vec2* host_positions, glm::vec2* host_velocities, glm::vec2* host_newPositions, glm::vec2* host_newVelocities);
#endif