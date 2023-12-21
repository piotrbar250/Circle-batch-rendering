#include "cuda_functions.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <glm/glm.hpp>

#include "Grid.cuh"

namespace cuda_functions_grid
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

    __device__ bool checkNeighbour(int gid, int neighIndex, BoidData &boidData)
    {
        if (gid == neighIndex)
            return false;
        if (fabs(glm::length(boidData.device_positions[gid] - boidData.device_positions[neighIndex])) <= boidData.params.perception)
            return true;
        return false;
    }

    __device__ bool checkSpecies(int gid, int neighIndex, BoidData &boidData)
    {
        if (boidData.colorIndex[gid] == boidData.colorSorted[neighIndex])
            return true;
        return false;
    }

    __device__ bool checkNeighbourGrid(glm::vec2 boidPosition, glm::vec2 neighPosition, BoidData &boidData)
    {
        if (boidPosition.x == neighPosition.x && boidPosition.y == neighPosition.y)
            return false;

        if (fabs(glm::length(boidPosition - neighPosition)) <= boidData.params.perception)
            return true;
        return false;
    }

    __device__ void antiBorderCollisionThrough(int gid, BoidData &boidData)
    {
        glm::vec2 &position = boidData.device_newPositions[gid];

        if (position.x < boidData.params.radius)
            position.x = boidData.params.width - boidData.params.radius;

        if (position.x + boidData.params.radius > boidData.params.width)
            position.x = boidData.params.radius;

        if (position.y < boidData.params.radius)
            position.y = boidData.params.height - boidData.params.radius;

        if (position.y + boidData.params.radius > boidData.params.height)
            position.y = boidData.params.radius;
    }

    __device__ glm::vec2 steeringForce(glm::vec2 target, glm::vec2 velocity, BoidData &boidData)
    {
        glm::vec2 targetOffset = target;

        glm::vec2 desiredVelocity = {0, 0};
        glm::vec2 steeringForce = {0, 0};

        if (length(targetOffset) > 0)
        {
            desiredVelocity = normalize(targetOffset) * boidData.params.maxSpeed;
            glm::vec2 steeringVelocity = desiredVelocity - velocity;
            steeringForce = limit(steeringVelocity, boidData.params.maxForce);
        }
        return steeringForce;
    }

    __device__ glm::vec2 alignmentForceGrid(int gid, int boidsCount, BoidData &boidData)
    {
        glm::vec2 target = glm::vec2(0, 0);
        int neighsCount = 0;

        int cell = grid::pixels2Cell(boidData.device_positions[gid].x, boidData.device_positions[gid].y, boidData.params);

        for (int i = boidData.device_gridCellStart[cell]; i <= boidData.device_gridCellEnd[cell]; i++)
        {
            if (!checkNeighbourGrid(boidData.device_positions[gid], boidData.device_positionsSorted[i], boidData) || !checkSpecies(gid, i, boidData))
                continue;
            target += boidData.device_velocitiesSorted[i];
            neighsCount++;
        }

        int neighCells[9];
        grid::getAdjacentCells(cell, neighCells, boidData.params);

        for (int neighCell : neighCells)
        {
            if (neighCell == -1)
                continue;
            for (int i = boidData.device_gridCellStart[neighCell]; i <= boidData.device_gridCellEnd[neighCell]; i++)
            {
                if (!checkNeighbourGrid(boidData.device_positions[gid], boidData.device_positionsSorted[i], boidData) || !checkSpecies(gid, i, boidData))
                    continue;
                target += boidData.device_velocitiesSorted[i];
                neighsCount++;
            }
        }
        if (neighsCount > 0)
            target /= neighsCount;
        else
            target = boidData.device_velocities[gid];

        return steeringForce(target, boidData.device_velocities[gid], boidData);
    }

    __device__ glm::vec2 alignmentForce(int gid, int boidsCount, BoidData &boidData)
    {
        glm::vec2 target = glm::vec2(0, 0);
        int neighsCount = 0;

        for (int i = 0; i < boidsCount; i++)
        {
            if (checkNeighbour(gid, i, boidData))
            {
                target += boidData.device_velocities[i];
                neighsCount++;
            }
        }
        if (neighsCount > 0)
            target /= neighsCount;
        else
            target = boidData.device_velocities[gid];

        return steeringForce(target, boidData.device_velocities[gid], boidData);
    }

    __device__ glm::vec2 cohesionForceGrid(int gid, int boidsCount, BoidData &boidData)
    {
        glm::vec2 target = glm::vec2(0, 0);
        int neighsCount = 0;

        int cell = grid::pixels2Cell(boidData.device_positions[gid].x, boidData.device_positions[gid].y, boidData.params);

        for (int i = boidData.device_gridCellStart[cell]; i <= boidData.device_gridCellEnd[cell]; i++)
        {
            if (!checkNeighbourGrid(boidData.device_positions[gid], boidData.device_positionsSorted[i], boidData) || !checkSpecies(gid, i, boidData))
                continue;
            target += boidData.device_positionsSorted[i];
            neighsCount++;
        }

        int neighCells[9];
        grid::getAdjacentCells(cell, neighCells, boidData.params);

        for (int neighCell : neighCells)
        {
            if (neighCell == -1)
                continue;
            for (int i = boidData.device_gridCellStart[neighCell]; i <= boidData.device_gridCellEnd[neighCell]; i++)
            {
                if (!checkNeighbourGrid(boidData.device_positions[gid], boidData.device_positionsSorted[i], boidData) || !checkSpecies(gid, i, boidData))
                    continue;
                target += boidData.device_positionsSorted[i];
                neighsCount++;
            }
        }

        if (neighsCount > 0)
            target /= neighsCount;
        else
            target = boidData.device_positions[gid];

        return steeringForce(target - boidData.device_positions[gid], boidData.device_velocities[gid], boidData);
    }

    __device__ glm::vec2 cohesionForce(int gid, int boidsCount, BoidData &boidData)
    {
        glm::vec2 target = glm::vec2(0, 0);
        int neighsCount = 0;

        for (int i = 0; i < boidsCount; i++)
        {
            if (checkNeighbour(gid, i, boidData))
            {
                target += boidData.device_positions[i];
                neighsCount++;
            }
        }
        if (neighsCount > 0)
            target /= neighsCount;
        else
            target = boidData.device_positions[gid];

        return steeringForce(target - boidData.device_positions[gid], boidData.device_velocities[gid], boidData);
    }

    __device__ glm::vec2 separationForceGrid(int gid, int boidsCount, BoidData &boidData)
    {
        glm::vec2 target = glm::vec2(0, 0);
        int neighsCount = 0;

        int cell = grid::pixels2Cell(boidData.device_positions[gid].x, boidData.device_positions[gid].y, boidData.params);

        for (int i = boidData.device_gridCellStart[cell]; i <= boidData.device_gridCellEnd[cell]; i++)
        {
            if (!checkNeighbourGrid(boidData.device_positions[gid], boidData.device_positionsSorted[i], boidData))
                continue;
            glm::vec2 offset = boidData.device_positions[gid] - boidData.device_positionsSorted[i];
            if (length(offset) == 0)
                continue;

            glm::vec2 value = offset * (1 / length(offset));
            target += value;
            neighsCount++;
        }

        int neighCells[9];
        grid::getAdjacentCells(cell, neighCells, boidData.params);

        for (int neighCell : neighCells)
        {
            if (neighCell == -1)
                continue;

            for (int i = boidData.device_gridCellStart[neighCell]; i <= boidData.device_gridCellEnd[neighCell]; i++)
            {
                if (!checkNeighbourGrid(boidData.device_positions[gid], boidData.device_positionsSorted[i], boidData))
                    continue;
                glm::vec2 offset = boidData.device_positions[gid] - boidData.device_positionsSorted[i];
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

        return steeringForce(target, boidData.device_velocities[gid], boidData);
    }

    __device__ glm::vec2 separationForce(int gid, int boidsCount, BoidData &boidData)
    {
        glm::vec2 target = glm::vec2(0, 0);
        int neighsCount = 0;

        for (int i = 0; i < boidsCount; i++)
        {
            if (checkNeighbour(gid, i, boidData))
            {
                glm::vec2 offset = boidData.device_positions[gid] - boidData.device_positions[i];
                if (length(offset) == 0)
                    continue;

                glm::vec2 value = offset * (1 / length(offset));
                target += value;
                neighsCount++;
            }
        }

        if (neighsCount > 0)
            target /= neighsCount;
        else
            return glm::vec2(0, 0);

        return steeringForce(target, boidData.device_velocities[gid], boidData);
    }

    __device__ glm::vec2 cursorDodgeForce(int gid, int boidsCount, BoidData &boidData)
    {
        glm::vec2 target = glm::vec2(0, 0);
        glm::vec2 offset = boidData.device_positions[gid] - glm::vec2(boidData.params.cursorX, boidData.params.cursorY);

        if (length(offset) == 0 || length(offset) > 100.0)
            return glm::vec2(0, 0);

        glm::vec2 value = offset * (1 / length(offset));
        target += value;
        return steeringForce(target, boidData.device_velocities[gid], boidData);
    }

    __device__ void applyForces(int gid, int boidsCount, BoidData &boidData)
    {
        boidData.device_accelerations[gid] *= 0;

        boidData.device_accelerations[gid] += boidData.params.alignmentForce * alignmentForceGrid(gid, boidsCount, boidData);
        boidData.device_accelerations[gid] += boidData.params.cohesionForce * (cohesionForceGrid(gid, boidsCount, boidData));
        boidData.device_accelerations[gid] += boidData.params.separationForce * (separationForceGrid(gid, boidsCount, boidData));
        boidData.device_accelerations[gid] += 5.0f * (cursorDodgeForce(gid, boidsCount, boidData));
    }

    __global__ void computeNextFrameKernel(int boidsCount, BoidData boidData)
    {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < boidsCount)
        {
            applyForces(gid, boidsCount, boidData);
            boidData.device_newVelocities[gid] = boidData.device_velocities[gid] + boidData.device_accelerations[gid];

            boidData.device_newPositions[gid] = boidData.device_positions[gid] + boidData.device_velocities[gid];

            antiBorderCollisionThrough(gid, boidData);

            boidData.device_translations[gid] = boidData.device_newPositions[gid] - glm::vec2(boidData.params.start_x, boidData.params.start_y);
        }
    }

    void computeNextFrame(int boidsCount, BoidData boidData)
    {
        int threadsPerBlock = 128;
        int blocksPerGrid = (boidsCount + threadsPerBlock - 1) / threadsPerBlock;

        computeNextFrameKernel<<<blocksPerGrid, threadsPerBlock>>>(boidsCount, boidData);
        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());
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
        int threadsPerBlock = 128;
        int blocksPerGrid = (boidsCount + threadsPerBlock - 1) / threadsPerBlock;

        swapFramesKernel<<<blocksPerGrid, threadsPerBlock>>>(boidsCount, positions, velocities, newPositions, newVelocities);
        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());
    }
}