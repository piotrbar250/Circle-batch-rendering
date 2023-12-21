#pragma once
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include "../global.hpp"
#include "BoidGPU.h"
#include "cuda_functions.h"

using namespace std;
using namespace glm;

class FlockGPU
{
public:
    int boidsCount;
    vector<vec2> positions;
    vector<vec2> accelerations;
    vector<vec2> velocities;
    vector<vec2> translations;

    vector<vec2> newPositions;
    vector<vec2> newVelocities;


    glm::vec2* device_positions;
    glm::vec2* device_velocities;
    glm::vec2* device_newPositions;
    glm::vec2* device_newVelocities;
    glm::vec2* device_accelerations;
    glm::vec2* device_translations;


    FlockGPU(int boidsCount) : boidsCount(boidsCount)
    {
        positions.resize(boidsCount);
        accelerations.resize(boidsCount);
        velocities.resize(boidsCount);
        translations.resize(boidsCount);
        newPositions.resize(boidsCount);
        newVelocities.resize(boidsCount);

        for (int i = 0; i < boidsCount; i++)
        {
            positions[i] = {randomFloat(100, 700), randomFloat(100, 700)};

            accelerations[i] = vec2(0, 0);

            velocities[i] = {randomFloat(-3, 3), randomFloat(-3, 3)};
            if (length(velocities[i]) == 0)
                velocities[i] = vec2(1, 1);

            velocities[i] = setMagnitude(velocities[i], MAX_SPEED);   
        }

        cuda_functions::allocateDataOnGPU(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities, device_accelerations, device_translations);
        cuda_functions::sendDataToGPU(boidsCount, device_positions, device_velocities, positions.data(), velocities.data());
    }

    ~FlockGPU()
    {
        cuda_functions::freeDataOnGPU(device_positions, device_velocities, device_newPositions, device_newVelocities, device_accelerations, device_translations);
    }

    void computeNextFrame(void** cuda_vbo_resource)
    {
        glm::vec2* devPtr = cuda_functions::getMappedPointer(cuda_vbo_resource);

        cuda_functions::computeNextFrame(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities, device_accelerations, devPtr);
        cuda_functions::swapFrames(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities);
        cuda_functions::unmapResource(cuda_vbo_resource);
    }
};