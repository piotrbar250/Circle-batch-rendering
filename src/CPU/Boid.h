#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>
#include <glm/glm.hpp>
#include "../global.hpp"

using namespace std;
using namespace glm;

class Boid
{
public:
    int index;
    glm::vec2 position;
    glm::vec2 acceleration;
    glm::vec2 velocity;
    glm::vec2 translation;

    glm::vec2 initTranslation;

    static inline float minSpeed = MIN_SPEED;
    static inline float maxSpeed = MAX_SPEED;
    static inline float maxForce = MAX_FORCE;
    static inline glm::vec2 start = glm::vec2(START_X, START_Y);

public:
    Boid()
    {
        position = {randomFloat(100, 700), randomFloat(100, 700)};
        initTranslation = position - start;

        acceleration = vec2(0, 0);

        velocity = {randomFloat(-3, 3), randomFloat(-3, 3)};
        if (length(velocity) == 0)
            velocity = vec2(1, 1);

        // velocity = vec2(1, 1);

        velocity = setMagnitude(velocity, maxSpeed);
        // velocity = setMagnitude(velocity, randomFloat(0.2, 2.5));
    }

    void antiBorderCollisionThrough()
    {
        if (position.x < RADIUS)
            position.x = screenWidth - RADIUS;

        if (position.x + RADIUS > screenWidth)
            position.x = RADIUS;

        if (position.y < RADIUS)
            position.y = screenHeight - RADIUS;

        if (position.y + RADIUS > screenHeight)
            position.y = RADIUS;
    }

    vec2 alignmentForce(const vector<Boid> &neighs)
    {
        vec2 target = vec2(0, 0);
        for (auto &boid : neighs)
            target += boid.velocity;

        if (neighs.size() > 0)
            target /= neighs.size();
        else
            target = velocity;

        return steeringForce(target);
    }

    vec2 cohesionForce(const vector<Boid> &neighs)
    {
        vec2 target = vec2(0, 0);
        for (auto &boid : neighs)
            target += boid.position;

        if (neighs.size() > 0)
            target /= neighs.size();
        else
            target = position;

        return steeringForce(target - position); 
    }

    vec2 separationForce(const vector<Boid> &neighs)
    {
        vec2 target = vec2(0, 0);
        for (auto &boid : neighs)
        {
            vec2 offset = position - boid.position;
            if(length(offset) == 0)
                continue;

            // vec2 value = normalize(offset) * (1 / length(offset));
            vec2 value = offset * (1 / length(offset));
            target += value;
        }    

        if (neighs.size() > 0)
            target /= neighs.size();
        else
            return vec2(0,0);

        return steeringForce(target); 
    }

    vec2 steeringForce(vec2 target)
    {
        vec2 targetOffset = target;

        vec2 desiredVelocity = {0, 0};
        vec2 steeringForce = {0, 0};

        if (length(targetOffset) > 0)
        {
            desiredVelocity = normalize(targetOffset) * maxSpeed;
            vec2 steeringVelocity = desiredVelocity - velocity;
            steeringForce = limit(steeringVelocity, maxForce);
        }

        return steeringForce;
    }

    void applyForces(const vector<Boid> &neighs)
    {
        acceleration *= 0;
        acceleration += alignmentForce(neighs);
        acceleration += (cohesionForce(neighs));
        acceleration += (separationForce(neighs));
    }

    void computeNextFrame(int index, const vector<Boid> &neighs)
    {
        this->index = index;

        applyForces(neighs);
        velocity += acceleration;

        position += velocity;

        antiBorderCollisionThrough();
        antiBorderCollision();

        translation = position - start;

        static float mx = 0;
        if(length(velocity) > mx)
        {
            mx = length(velocity);
            cout << mx << endl;
        }
    }

    void adjustVelocity();
    void antiBorderCollision();
    vec2 alignmentForce2(const vector<Boid> &neighs);
    vec2 arrivesteeringForce(vec2 target);
    vec2 bordersForce();
};