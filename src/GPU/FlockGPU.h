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

        allocateDataOnGPU(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities, device_accelerations, device_translations);
        sendDataToGPUinit(boidsCount, device_positions, device_velocities, positions.data(), velocities.data());
    }

    // ~FlockGPU()
    // {
    //     freeDataOnGPU(device_positions, device_velocities, device_newPositions, device_newVelocities);
    // }

    void computeNextFrame()
    {
        computeNextFrameInit(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities, device_accelerations, device_translations);
        swapFrames(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities);
        getDataFromGPUinit(boidsCount, device_translations, translations.data());

        // for (int i = 0; i < boidsCount; i++)
        // {
        //     BoidGPU::computeNextFrame(i, boidsCount, positions.data(), velocities.data(), accelerations.data(), newPositions.data(), newVelocities.data(), translations.data());
        //     // BoidGPU::swapFrames(i, positions.data(), velocities.data(), newPositions.data(), newVelocities.data());
        // }
        // sendData2(boidsCount, positions.data(), velocities.data(), newPositions.data(), newVelocities.data());

        // sendDataToGPU(boidsCount, device_newPositions, device_newVelocities, newPositions.data(), newVelocities.data());
        // swapFrames(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities);
        // getDataFromGPU(boidsCount, device_positions, device_velocities, positions.data(), velocities.data());
    }
};