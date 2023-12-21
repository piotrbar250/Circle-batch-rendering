#ifndef MY_CUDA_HEADER_H
#define MY_CUDA_HEADER_H

#include <glm/glm.hpp>
#include "cuda_functions.h"

namespace grid
{
    // __device__ void prepGrid(gridParams params)
    // {
    //     params.widthCount = (width + params.cellSize - 1) / params.cellSize;
    //     params.heightCount = (height + params.cellSize - 1) / params.cellSize;
    //     params.cellCount = params.widthCount * params.heightCount;
    // }

    __always_inline __device__ int pixels2Cell(float px, float py, GridParams params)
    {
        int xCell = px / params.cellSize;
        int yCell = py / params.cellSize;

        return yCell * params.widthCount + xCell;
    }

    __always_inline __device__ int leftCell(int index, GridParams params)
    {
        int row = index % params.widthCount;
        if(row == 0)
            return -1;
        return index - 1;
    }

    __always_inline __device__ int rightCell(int index, GridParams params)
    {
        int row = (index + 1) % params.widthCount;
        if(row == 0)
            return -1;
        return index + 1;
    }

    __always_inline __device__ int topCell(int index, GridParams params)
    {
        int cell = index + params.widthCount;
        if(cell >= params.cellCount)
            return -1;
        return cell;
    }

    __always_inline __device__ int  bottomCell(int index, GridParams params)
    {
        int cell = index - params.widthCount;
        if(cell < 0)
            return -1;
        return cell;
    }

    __always_inline __device__ void getAdjacentCells(int index, int*neighs, GridParams params)
    {
        for(int i = 0; i < 8; i++)
            neighs[i] = -1;

        neighs[0] = leftCell(index, params);
        neighs[1] = rightCell(index, params);
        neighs[2] = topCell(index, params);
        neighs[3] = bottomCell(index, params);

        if(topCell(index, params) != -1)
        {
            neighs[4] = leftCell(topCell(index, params), params);
            neighs[5] = rightCell(topCell(index, params), params);
        }

        if(bottomCell(index, params) != -1)
        {
            neighs[6] = leftCell(bottomCell(index, params), params);
            neighs[7] = rightCell(bottomCell(index, params), params);
        }
    }

    __always_inline void run(GridParams params)
    {
        glm::vec2 positions[] = {
            {340, 150},
            {100, 100},
            {200, 200},
            {379, 180},
            {160, 115}
        };
        int boidCount = sizeof(positions) / sizeof(typeof(*positions));
    }
}
#endif // MY_CUDA_HEADER_H