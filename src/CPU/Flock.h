#pragma once
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include "../global.hpp"
#include "Boid.h"
using namespace std;
using namespace glm;

class Flock
{
public:
    int boidsCount;
    vector<Boid> boids;
    vector<glm::vec2> translations;

    Flock(int boidsCount) : boidsCount(boidsCount)
    {
        boids.resize(boidsCount);
        translations.resize(boidsCount);
    }

    void computeNextFrame()
    {   
        for(int i = 0; i < boidsCount; i++)
        {
            Boid& boid = boids[i];
            vector<Boid> neighs = getNeighs(i);
            boid.computeNextFrame(i, neighs);
            translations[i] = boid.translation;
        }
    }

    vector<Boid> getNeighs(int index)
    {
        vector<Boid> neighs;

        for(int i = 0; i < boidsCount; i++)
        {
            if(i == index)
                continue;
            if(fabs(glm::length(boids[i].position - boids[index].position)) <= PERCEPTION)
                neighs.push_back(boids[i]);
        }
        return move(neighs);
    }
};
