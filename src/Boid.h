#pragma once
#include <iostream>
#include <stdexcept>
#include <glm/glm.hpp>
#include "global.hpp"
using namespace std;
using namespace glm;

class Boid
{
private:
    glm::vec2 position;
    glm::vec2 acceleration;
    glm::vec2 velocity;
    glm::vec2 translation;

    static vec2 start;
    static float minSpeed;
    static float maxSpeed;

public:
    Boid()
    {
        position = start;
        velocity = {randomFloat(), randomFloat()};

        if(glm::length(velocity) < 0.01)
            velocity = vec2(1, 1);
    }

    void adjustVelocity()
    {
        if(glm::length(velocity) == 0)
        {
            throw runtime_error("velocity == 0");
        }

        if(glm::length(velocity) > maxSpeed)
        {
            velocity = glm::normalize(velocity);
            velocity *= maxSpeed;
            return;
        }

        if(glm::length(velocity) < minSpeed)
        {
            velocity = glm::normalize(velocity);
            velocity *= minSpeed;
            return;
        }
    }

    void computeAcceleration()
    {
        float forceValue = BORDER_FORCE;

        if(position.x < RADIUS)
        {
            acceleration += vec2(forceValue, 0);
        }
        if(position.x + RADIUS > screenWidth)
        {
            acceleration += vec2(-forceValue, 0);
        }
        if(position.y < RADIUS)
        {
            acceleration += vec2(0, forceValue);
        }
        if(position.y + RADIUS > screenHeight)
        {
            acceleration += vec2(0, -forceValue);
        }
    }   

    void computeVelocity()
    {
        velocity += acceleration;
    } 

    void computePosition()
    {
        position += velocity;
    }

    void computeNextFrame()
    {
        acceleration *= 0;
        computeAcceleration();
        computeVelocity();
        adjustVelocity();
        computePosition();
        translation = position - start;
    }

    glm::vec2 getTranslation()
    {
        return translation;
    }
};

float Boid::maxSpeed = 5;
float Boid::minSpeed = 1;
vec2 Boid::start = vec2(START_X, START_Y);