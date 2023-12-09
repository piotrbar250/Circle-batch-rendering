#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>
#include <glm/glm.hpp>
#include "global.hpp"
using namespace std;
using namespace glm;

class Boid
{
public:
    glm::vec2 position;
    glm::vec2 acceleration;
    glm::vec2 velocity;
    glm::vec2 translation;
    
    glm::vec2 initTranslation;

    static vec2 start;
    static float minSpeed;
    static float maxSpeed;

public:
    Boid()
    {
        vec2 tmp = {randomFloat(100, 700), randomFloat(100, 700)};

        initTranslation = tmp - start;
        // cout << start.x << " " << start.y << endl; 
        position = tmp;

        acceleration = vec2(0, 0);
        velocity = {randomFloat(-3, 3), randomFloat(-3, 3)};

        if(glm::length(velocity) == 0)
            velocity = vec2(1, 1);

        velocity = glm::normalize(velocity);
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

    vec2 bordersForce()
    {
        float forceValue = BORDER_FORCE;

        if(position.x < RADIUS)
        {
            return vec2(forceValue, 0);
            // acceleration += vec2(forceValue, 0);
        }
        if(position.x + RADIUS > screenWidth)
        {
            return vec2(-forceValue, 0);
            // acceleration += vec2(-forceValue, 0);
        }
        if(position.y < RADIUS)
        {
            return vec2(0, forceValue);
            // acceleration += vec2(0, forceValue);
        }
        if(position.y + RADIUS > screenHeight)
        {
            return vec2(0, -forceValue);;
            // acceleration += vec2(0, forceValue);
        }
        return vec2(0,0);
    }  

    vec2 alignmentForce(const vector<Boid>& neighs)
    {
        vec2 steering = vec2(0,0);
        for(auto& boid : neighs)
            steering += boid.velocity;
    
        if(neighs.size() > 0)
        {
            steering /= neighs.size();
            steering -= velocity;
        }
        return steering;
    }

    void applyForces(const vector<Boid>& neighs)
    {
        acceleration *= 0;
        // acceleration = vec2
        acceleration += alignmentForce(neighs);
        acceleration += bordersForce();
    }
 
    void computeNextFrame(const vector<Boid>& neighs)
    {
        applyForces(neighs);

        velocity += acceleration;
        position += velocity;
        
        adjustVelocity();

        translation = position - start;
    }


    glm::vec2 getTranslation()
    {
        return translation;
    }
};

float Boid::maxSpeed = 2;
float Boid::minSpeed = 1;
vec2 Boid::start = vec2(START_X, START_Y);