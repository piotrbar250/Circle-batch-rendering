#pragma once
#include <random>
#include <iostream>
#include <stdexcept>
#include <glm/glm.hpp>


const float screenWidth = 1000;
const float screenHeight = 1000;

const float NUMBER_OF_SEGMENTS = 50;
const float RADIUS = 10;
const float START_X = 500;
const float START_Y = 500;

const float BORDER_FORCE = 10;
const float PERCEPTION = 50;
const float SLOWING_DISTANCE = 100;

std::random_device rd;
std::mt19937 eng(rd());

float randomFloat(float l, float r) {
    std::uniform_real_distribution<> distr(l, r);

    return distr(eng);
}

glm::vec2 limit(glm::vec2 v, float l)
{
    if(glm::length(v) > l)
    {
        if(length(v) > 0)
            v = glm::normalize(v) * l;
    }
    return v;
}

glm::vec2 setMagnitude(glm::vec2 v, float l)
{
    if(length(v) == 0)
    {
        throw std::runtime_error("sm_length(v) == 0");
        // return v;
    }
    v = glm::normalize(v) * l;
    return v;
}