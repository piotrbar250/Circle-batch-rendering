#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <glm/glm.hpp>

void swapFrames(int boidsCount, glm::vec2 *positions, glm::vec2 *velocities, glm::vec2* newPositions, glm::vec2* newVelocities);
void sendData(int boidsCount, glm::vec2* host_positions, glm::vec2* host_velocities, glm::vec2* host_newPositions, glm::vec2* host_newVelocities);

#endif