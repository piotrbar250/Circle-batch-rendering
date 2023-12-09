#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include "global.hpp"
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
        int index = 0;
        for(auto& boid: boids)
        {
            boid.computeNextFrame();
            translations[index++] = boid.getTranslation();;
        }
    }
};