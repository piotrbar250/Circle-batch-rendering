#pragma once
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include "../global.hpp"

using namespace std;
using namespace glm;

namespace BoidGPU
{
    /*
        TRANSFER CONSTANTS TO GPU!!!
        consider changing to int** neighsIndices
        consider saving result in alignmentForce array
        limit, normalize should prepared for the device !!!!!!
        rearrange kernel parameters
    */

    bool checkNeighbour(int gid, int neighIndex, vec2 *positions)
    {
        if (gid == neighIndex)
            return false;
        if (fabs(glm::length(positions[gid] - positions[neighIndex])) <= PERCEPTION)
            return true;
        return false;
    }

    void antiBorderCollisionThrough(int gid, vec2 *positions)
    {
        vec2 &position = positions[gid];

        if (position.x < RADIUS)
            position.x = screenWidth - RADIUS;

        if (position.x + RADIUS > screenWidth)
            position.x = RADIUS;

        if (position.y < RADIUS)
            position.y = screenHeight - RADIUS;

        if (position.y + RADIUS > screenHeight)
            position.y = RADIUS;
    }

    vec2 steeringForce(vec2 target, vec2 velocity)
    {
        // limit, normalize should prepared for the device !!!!!!
        vec2 targetOffset = target;

        vec2 desiredVelocity = {0, 0};
        vec2 steeringForce = {0, 0};

        if (length(targetOffset) > 0)
        {
            desiredVelocity = normalize(targetOffset) * MAX_SPEED;
            vec2 steeringVelocity = desiredVelocity - velocity;
            steeringForce = limit(steeringVelocity, MAX_FORCE);
        }

        return steeringForce;
    }

    vec2 alignmentForce(int gid, int boidsCount, vec2 *positions, vec2 *velocities)
    {
        // consider saving result in alignmentForce
        vec2 target = vec2(0, 0);
        int neighsCount = 0;

        for (int i = 0; i < boidsCount; i++)
        {
            if (checkNeighbour(gid, i, positions))
            {
                target += velocities[i];
                neighsCount++;
            }
        }
        if (neighsCount > 0)
            target /= neighsCount;
        else
            target = velocities[gid];

        return steeringForce(target, velocities[gid]);
    }

    vec2 cohesionForce(int gid, int boidsCount, vec2 *positions, vec2 *velocities)
    {
        vec2 target = vec2(0, 0);
        int neighsCount = 0;

        for (int i = 0; i < boidsCount; i++)
        {
            if (checkNeighbour(gid, i, positions))
            {
                target += positions[i];
                neighsCount++;
            }
        }
        if (neighsCount > 0)
            target /= neighsCount;
        else
            target = positions[gid];

        return steeringForce(target - positions[gid], velocities[gid]);
    }

    vec2 separationForce(int gid, int boidsCount, vec2 *positions, vec2 *velocities)
    {
        // review force computation
        vec2 target = vec2(0, 0);
        int neighsCount = 0;

        for (int i = 0; i < boidsCount; i++)
        {
            if (checkNeighbour(gid, i, positions))
            {
                vec2 offset = positions[gid] - positions[i];
                if (length(offset) == 0)
                    continue;

                // value = normalize(offset) * (1 / length(offset));
                vec2 value = offset * (1 / length(offset));
                target += value;
                neighsCount++;
            }
        }

        if (neighsCount > 0)
            target /= neighsCount;
        else
            return vec2(0, 0);

        return steeringForce(target, velocities[gid]);
    }

    void applyForces(int gid, int boidsCount, vec2 *positions, vec2 *velocities, vec2 *accelerations)
    {
        accelerations[gid] *= 0;
        accelerations[gid] += alignmentForce(gid, boidsCount, positions, velocities);
        accelerations[gid] += (cohesionForce(gid, boidsCount, positions, velocities));
        accelerations[gid] += (separationForce(gid, boidsCount, positions, velocities));
    }

    void computeNextFrame(int gid, int boidsCount, vec2 *positions, vec2 *velocities, vec2 *accelerations, vec2 *translations)
    {
        applyForces(gid, boidsCount, positions, velocities, accelerations);
        velocities[gid] += accelerations[gid];

        positions[gid] += velocities[gid];

        antiBorderCollisionThrough(gid, positions);

        translations[gid] = positions[gid] - START;
    }
}