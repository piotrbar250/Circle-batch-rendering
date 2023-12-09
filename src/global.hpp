#pragma once
#include <random>
#include <iostream>

const float screenWidth = 1000;
const float screenHeight = 1000;

const float NUMBER_OF_SEGMENTS = 50;
const float RADIUS = 10;
const float START_X = 500;
const float START_Y = 500;

const float BORDER_FORCE = 10;
const float PERCEPTION = 100;

std::random_device rd;
std::mt19937 eng(rd());

float randomFloat() {
    std::uniform_real_distribution<> distr(-3, 3);

    return distr(eng);
}