#pragma once
#include <random>
#include <iostream>
#include <stdexcept>
#include <glm/glm.hpp>


const float screenWidth = 1000;
const float screenHeight = 1000;

const float NUMBER_OF_SEGMENTS = 50;
const float RADIUS = 3;
const float START_X = 500;
const float START_Y = 500;
const glm::vec2 START = {START_X, START_Y};

const float BORDER_FORCE = 10;
const float PERCEPTION = 50;
const float SLOWING_DISTANCE = 100;

const float MIN_SPEED = 0.0f;
const float MAX_SPEED = 4.0f;
const float MAX_FORCE = 1.0f;


static std::random_device rd;
static std::mt19937 eng(rd());

inline float randomFloat(float l, float r) {
    std::uniform_real_distribution<> distr(l, r);

    return distr(eng);
}

inline glm::vec2 limit(glm::vec2 v, float l)
{
    if(glm::length(v) > l)
    {
        if(length(v) > 0)
            v = glm::normalize(v) * l;
    }
    return v;
}

inline glm::vec2 setMagnitude(glm::vec2 v, float l)
{
    if(length(v) == 0)
    {
        throw std::runtime_error("sm_length(v) == 0");
        // return v;
    }
    v = glm::normalize(v) * l;
    return v;
}