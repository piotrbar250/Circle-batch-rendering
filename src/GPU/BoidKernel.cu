#define screenWidth 1000
#define screenHeight 1000

#define NUMBER_OF_SEGMENTS 50
#define RADIUS 10
#define START_X 500
#define START_Y 500

#define BORDER_FORCE 10
#define PERCEPTION 50
#define SLOWING_DISTANCE 100

#define MIN_SPEED 0.0f
#define MAX_SPEED 4.0f
#define MAX_FORCE 1.0f

#include "cuda_functions.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <glm/glm.hpp>

namespace cuda_functions
{
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
    {
        if (code != cudaSuccess)
        {
            std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
            if (abort)
                exit(code);
        }
    }
    __device__ glm::vec2 limit(glm::vec2 v, float l)
    {
        if (glm::length(v) > l)
        {
            if (length(v) > 0)
                v = glm::normalize(v) * l;
        }
        return v;
    }

    __device__ bool checkNeighbour(int gid, int neighIndex, const glm::vec2 *positions)
    {
        if (gid == neighIndex)
            return false;
        if (fabs(glm::length(positions[gid] - positions[neighIndex])) <= PERCEPTION)
            return true;
        return false;
    }

    __device__ void antiBorderCollisionThrough(int gid, glm::vec2 *newPositions)
    {
        glm::vec2 &position = newPositions[gid];

        if (position.x < RADIUS)
            position.x = screenWidth - RADIUS;

        if (position.x + RADIUS > screenWidth)
            position.x = RADIUS;

        if (position.y < RADIUS)
            position.y = screenHeight - RADIUS;

        if (position.y + RADIUS > screenHeight)
            position.y = RADIUS;
    }

    __device__ glm::vec2 steeringForce(glm::vec2 target, glm::vec2 velocity)
    {
        // limit, normalize should prepared for the device !!!!!!
        glm::vec2 targetOffset = target;

        glm::vec2 desiredVelocity = {0, 0};
        glm::vec2 steeringForce = {0, 0};

        if (length(targetOffset) > 0)
        {
            desiredVelocity = normalize(targetOffset) * MAX_SPEED;
            glm::vec2 steeringVelocity = desiredVelocity - velocity;
            steeringForce = limit(steeringVelocity, MAX_FORCE);
        }
        return steeringForce;
    }

    __device__ glm::vec2 alignmentForce(int gid, int boidsCount, const glm::vec2 *positions, const glm::vec2 *velocities)
    {
        // consider saving result in alignmentForce
        glm::vec2 target = glm::vec2(0, 0);
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

    __device__ glm::vec2 cohesionForce(int gid, int boidsCount, const glm::vec2 *positions, const glm::vec2 *velocities)
    {
        glm::vec2 target = glm::vec2(0, 0);
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

    __device__ glm::vec2 separationForce(int gid, int boidsCount, const glm::vec2 *positions, const glm::vec2 *velocities)
    {
        // review force computation
        glm::vec2 target = glm::vec2(0, 0);
        int neighsCount = 0;

        for (int i = 0; i < boidsCount; i++)
        {
            if (checkNeighbour(gid, i, positions))
            {
                glm::vec2 offset = positions[gid] - positions[i];
                if (length(offset) == 0)
                    continue;

                // value = normalize(offset) * (1 / length(offset));
                glm::vec2 value = offset * (1 / length(offset));
                target += value;
                neighsCount++;
            }
        }

        if (neighsCount > 0)
            target /= neighsCount;
        else
            return glm::vec2(0, 0);

        return steeringForce(target, velocities[gid]);
    }

    __device__ void applyForces(int gid, int boidsCount, const glm::vec2 *positions, const glm::vec2 *velocities, glm::vec2 *accelerations)
    {
        accelerations[gid] *= 0;
        accelerations[gid] += alignmentForce(gid, boidsCount, positions, velocities);
        accelerations[gid] += (cohesionForce(gid, boidsCount, positions, velocities));
        accelerations[gid] += (separationForce(gid, boidsCount, positions, velocities));
    }

    __global__ void computeNextFrameKernel(int boidsCount, glm::vec2 *positions, glm::vec2 *velocities, glm::vec2 *newPositions, glm::vec2 *newVelocities, glm::vec2 *accelerations, glm::vec2 *translations)
    {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < boidsCount)
        {
            applyForces(gid, boidsCount, positions, velocities, accelerations);
            newVelocities[gid] = velocities[gid] + accelerations[gid];

            newPositions[gid] = positions[gid] + velocities[gid];

            antiBorderCollisionThrough(gid, newPositions);

            // translations[gid] = newPositions[gid] - START;
            translations[gid] = newPositions[gid] - glm::vec2(START_X, START_Y);
        }
    }

    void computeNextFrame(int boidsCount, glm::vec2 *device_positions, glm::vec2 *device_velocities, glm::vec2 *device_newPositions, glm::vec2 *device_newVelocities, glm::vec2 *device_accelerations, glm::vec2 *device_translations)
    {
        // parameters rearranged!!!!!
        // int threadsPerBlock = 10;
        // int blocksPerGrid = boidsCount / threadsPerBlock + 1;
        // blocksPerGrid*=2;
        //  blocksPerGrid = 60;
        int threadsPerBlock = 128;
        int blocksPerGrid = (boidsCount + threadsPerBlock - 1) / threadsPerBlock;

        computeNextFrameKernel<<<blocksPerGrid, threadsPerBlock>>>(boidsCount, device_positions, device_velocities, device_newPositions, device_newVelocities, device_accelerations, device_translations);
        cudaDeviceSynchronize();
    }

    __global__ void swapFramesKernel(int boidsCount, glm::vec2 *positions, glm::vec2 *velocities, glm::vec2 *newPositions, glm::vec2 *newVelocities)
    {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < boidsCount)
        {
            positions[gid] = newPositions[gid];
            velocities[gid] = newVelocities[gid];
        }
    }

    void swapFrames(int boidsCount, glm::vec2 *positions, glm::vec2 *velocities, glm::vec2 *newPositions, glm::vec2 *newVelocities)
    {
        // int threadsPerBlock = 10;
        // int blocksPerGrid = boidsCount / threadsPerBlock + 1;

        // blocksPerGrid*=2;
        int threadsPerBlock = 128;
        int blocksPerGrid = (boidsCount + threadsPerBlock - 1) / threadsPerBlock;

        swapFramesKernel<<<blocksPerGrid, threadsPerBlock>>>(boidsCount, positions, velocities, newPositions, newVelocities);
        // cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Check for errors on the CUDA device side after kernel execution
        gpuErrchk(cudaDeviceSynchronize());
    }

}