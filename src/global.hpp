#pragma once
#include <random>
#include <iostream>
#include <glm/glm.hpp>


const float screenWidth = 1000;
const float screenHeight = 1000;

const float NUMBER_OF_SEGMENTS = 50;
const float RADIUS = 10;
const float START_X = 500;
const float START_Y = 500;

const float BORDER_FORCE = 10;
const float PERCEPTION = 40;

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
        v = glm::normalize(v) * l;
    }
    return v;
}
