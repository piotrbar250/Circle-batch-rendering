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
    static float maxForce;

public:
    Boid()
    {
        // vec2 tmp = {randomFloat(100, 700), randomFloat(100, 700)};
        vec2 tmp = {200.0f, 800.0f};

        initTranslation = tmp - start;
        // cout << start.x << " " << start.y << endl; 
        position = tmp;

        acceleration = vec2(0, 0);
        // velocity = {randomFloat(-3, 3), randomFloat(-3, 3)};
        velocity = {0.0f, -3.0f};

        // if(glm::length(velocity) == 0)
        //     velocity = vec2(1, 1);

        // velocity = glm::normalize(velocity);
    }

    void adjustVelocity()
    {
        if(glm::length(velocity) == 0)
        {
            // throw runtime_error("velocity == 0");
            velocity *= 0;
            return;
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

    vec2 steeringForce()
    {
        vec2 target = {200.0f, 200.f};
        vec2 desiredVelocity = {0, 0};
        vec2 steeringForce = {0, 0};

        if(length(target - position) > 0)
        {
            desiredVelocity = normalize(target - position) * maxSpeed;
            vec2 steeringVelocity = desiredVelocity - velocity;
            steeringForce = limit(steeringVelocity, maxForce);
        }
        return steeringForce;
    }

    vec2 arrivesteeringForce()
    {
        vec2 target = {200.0f, 200.f};
        
        vec2 targetOffset = target - position;

        float distance = length(targetOffset);

        float rampSpeed = maxSpeed * (distance / SLOWING_DISTANCE);
        float clippedSpeed = std::min(maxSpeed, rampSpeed);

        vec2 desiredVelocity = (clippedSpeed / distance) * targetOffset;

        vec2 steeringVelocity = desiredVelocity - velocity;
        vec2 steeringForce = limit(steeringVelocity, maxForce);

        cout << length(steeringForce) << endl;
        return steeringForce;
    }   

    void applyForces(const vector<Boid>& neighs)
    {
        acceleration *= 0;
        // acceleration = vec2
        // acceleration += alignmentForce(neighs);
        // acceleration += bordersForce();
        // acceleration += steeringForce();
        acceleration += arrivesteeringForce();
    }
 
    void computeNextFrame(const vector<Boid>& neighs)
    {
        applyForces(neighs);

        velocity += acceleration;

        position += velocity;
        
        // adjustVelocity();

        translation = position - start;

        cout << velocity.x << " " << velocity.y << " " << position.x << " " << position.y << " " << length(velocity) << " " << length(acceleration) << endl;
    }


    glm::vec2 getTranslation()
    {
        return translation;
    }
};

float Boid::maxSpeed = 3;
float Boid::minSpeed = 0;
float Boid::maxForce = 0.01;
vec2 Boid::start = vec2(START_X, START_Y);