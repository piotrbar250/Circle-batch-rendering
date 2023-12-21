#pragma once
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include "../global.hpp"
#include "BoidGPU.h"
#include "cuda_functions.h"
#include <chrono>
#include <thread>
using namespace std;
using namespace glm;

class FlockGridGPU
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

    int* device_gridCellIndex;
    int* device_gridCellStart;
    int* device_gridCellEnd;
    int* device_boidSequence;

    glm::vec2* device_positionsSorted;
    glm::vec2* device_velocitiesSorted;

    BoidData boidData;

    FlockGridGPU(int boidsCount, GridParams params) : boidsCount(boidsCount)
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
        cuda_functions_grid::allocateGrid(boidsCount, params.cellCount, device_gridCellIndex, device_boidSequence, device_gridCellStart, device_gridCellEnd, device_positionsSorted, device_velocitiesSorted);

            boidData.device_positions = device_positions;
            boidData.device_velocities = device_velocities;
            boidData.device_newPositions = device_newPositions;
            boidData.device_newVelocities = device_newVelocities;
            boidData.device_accelerations =  device_accelerations;
            boidData.device_translations = device_translations;
            boidData.device_gridCellIndex = device_gridCellIndex;
            boidData.device_gridCellStart = device_gridCellStart;
            boidData.device_gridCellEnd = device_gridCellEnd;
            boidData.boidSequence = device_boidSequence;
            boidData.device_positionsSorted = device_positionsSorted;
            boidData.device_velocitiesSorted = device_velocitiesSorted;
    }

    ~FlockGridGPU()
    {
        cuda_functions::freeDataOnGPU(device_positions, device_velocities, device_newPositions, device_newVelocities, device_accelerations, device_translations);
    }

    void computeNextFrame(void** cuda_vbo_resource, GridParams params)
    {
        // allocate positionsSorted !!!!!!!!
        // printf("nawet tu\n");
        cuda_functions_grid::computeGridCellIndex(boidsCount, params, device_positions, device_velocities, device_gridCellIndex ,device_gridCellStart, device_gridCellEnd, device_boidSequence, device_positionsSorted, device_velocitiesSorted);
        //   std::this_thread::sleep_for(std::chrono::seconds(200));
        // printf("hello\n");
        glm::vec2* devPtr = cuda_functions::getMappedPointer(cuda_vbo_resource);
        boidData.device_translations = devPtr;
        boidData.params = params;
        // cuda_functions_grid::computeNextFrame(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities, device_accelerations, devPtr, device_positionsSorted, device_velocitiesSorted);
        cuda_functions_grid::computeNextFrame(boidsCount, boidData);
        cuda_functions::swapFrames(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities);
        // cuda_functions::getDataFromGPU(boidsCount, device_translations, translations.data());
        // printf("hello here\n");
        cuda_functions::unmapResource(cuda_vbo_resource);
    }
};