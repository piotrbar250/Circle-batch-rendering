#include "../global.hpp" 
#include "Boid.h"

void Boid::adjustVelocity()
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

void Boid::antiBorderCollision()
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

vec2 Boid::alignmentForce2(const vector<Boid> &neighs)
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

vec2 Boid::arrivesteeringForce(vec2 target)
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

vec2 Boid::bordersForce()
{
    float forceValue = BORDER_FORCE;

    if (position.x < RADIUS)
    {
        return vec2(forceValue, 0);
    }
    if (position.x + RADIUS > screenWidth)
    {
        return vec2(-forceValue, 0);
    }
    if (position.y < RADIUS)
    {
        return vec2(0, forceValue);
    }
    if (position.y + RADIUS > screenHeight)
    {
        return vec2(0, -forceValue);
    }
    return vec2(0, 0);
}

