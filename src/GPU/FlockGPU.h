#pragma once
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include "../global.hpp"
#include "BoidGPU.h"

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
    vector<vec2> initTranslations;

    static inline float minSpeed = MIN_SPEED;
    static inline float maxSpeed = MAX_SPEED;
    static inline float maxForce = MAX_FORCE;
    static inline glm::vec2 start = glm::vec2(START_X, START_Y);

    FlockGPU(int boidsCount) : boidsCount(boidsCount)
    {
        positions.resize(boidsCount);
        velocities.resize(boidsCount);
        accelerations.resize(boidsCount);
        velocities.resize(boidsCount);
        translations.resize(boidsCount);
        initTranslations.resize(boidsCount);
        
        for(int i = 0; i < boidsCount; i++)
        {
            positions[i] = {randomFloat(100, 700), randomFloat(100, 700)};
            initTranslations[i] = positions[i] - start;
            
            accelerations[i] = vec2(0, 0);
            
            velocities[i] = {randomFloat(-3, 3), randomFloat(-3, 3)};
            if (length(velocities[i]) == 0)
                velocities[i] = vec2(1, 1);

            velocities[i] = setMagnitude(velocities[i], maxSpeed);
        }
    }    

    void computeNextFrame()
    {
        for(int i = 0; i < boidsCount; i++)
        {
            vector<int> neighs = getNeighs(i);
            BoidGPU::computeNextFrame(i, neighs.size(), neighs.data(), positions.data(), velocities.data(), accelerations.data(), translations.data());
        }
    }
    
    vector<int> getNeighs(int index)
    {
        vector<int> neighs;

        for(int i = 0; i < boidsCount; i++)
        {
            if(i == index)
                continue;
            if(fabs(glm::length(positions[i] - positions[index])) <= PERCEPTION)
                neighs.push_back(i);
        }
        return move(neighs);
    }
};