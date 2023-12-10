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
    int index;
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
        vec2 realPosition = {randomFloat(100, 700), randomFloat(100, 700)};
        // vec2 realPosition = {500, 500};
        // vec2 tmp = {200.0f, 800.0f};

        initTranslation = realPosition - start;
        position = realPosition;

        acceleration = vec2(0, 0);

        velocity = {randomFloat(-3, 3), randomFloat(-3, 3)};
        if (length(velocity) == 0)
            velocity = vec2(1, 1);
        // velocity = vec2(1, 1);

        // velocity = vec2(1, 1);

        velocity = setMagnitude(velocity, maxSpeed);

        // velocity = setMagnitude(velocity, randomFloat(0.2, 2.5));
        // velocity = {0.0f, -3.0f};

        // velocity = glm::normalize(velocity);
    }

    void adjustVelocity()
    {
        if (glm::length(velocity) == 0)
        {
            return;
        }

        if (glm::length(velocity) > maxSpeed)
        {
            velocity = glm::normalize(velocity);
            velocity *= maxSpeed;
            return;
        }

        if (glm::length(velocity) < minSpeed)
        {
            velocity = glm::normalize(velocity);
            velocity *= minSpeed;
            return;
        }
    }

    void antiBorderCollision()
    {
        if (position.x < RADIUS)
            velocity.x *= (-1);

        if (position.x + RADIUS > screenWidth)
            velocity.x *= (-1);

        if (position.y < RADIUS)
            velocity.y *= (-1);

        if (position.y + RADIUS > screenHeight)
            velocity.y *= (-1);
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

    vec2 alignmentForce2(const vector<Boid> &neighs)
    {
        vec2 steering = vec2(0, 0);
        for (auto &boid : neighs)
            steering += boid.velocity;

        if (neighs.size() > 0)
        {
            steering /= neighs.size();
            steering = setMagnitude(steering, maxSpeed);
            steering -= velocity;
            steering = limit(steering, maxForce);
        }
            
        return steering;
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

    vec2 steeringForce(vec2 target)
    {
        vec2 targetOffset = target;

        vec2 desiredVelocity = {0, 0};
        vec2 steeringForce = {0, 0};

        if(length(targetOffset) > 0)
        {
            desiredVelocity = normalize(targetOffset) * maxSpeed;
            vec2 steeringVelocity = desiredVelocity - velocity;
            steeringForce = limit(steeringVelocity, maxForce);
        }

        return steeringForce;
    }

    vec2 arrivesteeringForce(vec2 target)
    {
        vec2 targetOffset = target - position;

        float distance = length(targetOffset);

        float rampSpeed = maxSpeed * (distance / SLOWING_DISTANCE);
        float clippedSpeed = std::min(maxSpeed, rampSpeed);

        vec2 desiredVelocity = (clippedSpeed / distance) * targetOffset;

        vec2 steeringVelocity = desiredVelocity - velocity;
        vec2 steeringForce = limit(steeringVelocity, maxForce);

        // cout << length(steeringForce) << endl;
        return steeringForce;
    }

    void applyForces(const vector<Boid> &neighs)
    {
        acceleration *= 0;

        float diff = fabs(length(alignmentForce(neighs)) - length(alignmentForce2(neighs)));

        if(diff != 0)
        {
            cout << diff << " " <<length(alignmentForce(neighs)) << " " << length(alignmentForce2(neighs)) << endl;
            // exit(1);
        }
        else
        {
            cout << "ok" << endl;
        }

        acceleration += alignmentForce(neighs);

        // acceleration = vec2
        // acceleration += alignmentForce2(neighs);
        // acceleration += bordersForce();
        // acceleration += steeringForce();
        // acceleration += arrivesteeringForce();
    }

    void computeNextFrame(int index, const vector<Boid> &neighs)
    {
        this->index = index;
        applyForces(neighs);

        velocity += acceleration;

        // antiBorderCollision();
        position += velocity;

        antiBorderCollisionThrough();

        translation = position - start;

        // cout << velocity.x << " " << velocity.y << " " << position.x << " " << position.y << " " << length(velocity) << " " << length(acceleration) << endl;
    }

    glm::vec2 getTranslation()
    {
        return translation;
    }

    vec2 bordersForce()
    {
        float forceValue = BORDER_FORCE;

        if (position.x < RADIUS)
        {
            return vec2(forceValue, 0);
            // acceleration += vec2(forceValue, 0);
        }
        if (position.x + RADIUS > screenWidth)
        {
            return vec2(-forceValue, 0);
            // acceleration += vec2(-forceValue, 0);
        }
        if (position.y < RADIUS)
        {
            return vec2(0, forceValue);
            // acceleration += vec2(0, forceValue);
        }
        if (position.y + RADIUS > screenHeight)
        {
            return vec2(0, -forceValue);
            ;
            // acceleration += vec2(0, forceValue);
        }
        return vec2(0, 0);
    }
};

float Boid::maxSpeed = 4;
float Boid::minSpeed = 0;
float Boid::maxForce = 0.2;
vec2 Boid::start = vec2(START_X, START_Y);