#include "cuda_dependencies.cu"

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

    __global__ void initStartEndKernel(int cellCount, int* device_gridCellStart, int* device_gridCellEnd)
    {
         int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if(gid < cellCount)
        {
            device_gridCellStart[gid] = -1;
            device_gridCellEnd[gid] = -2;
        }
    }

    __global__ void boidCellKernel(int boidsCount, Params params, glm::vec2* positions, int* gridCellIndex)
    {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if(gid < boidsCount)
        {
            gridCellIndex[gid] = grid::pixels2Cell(positions[gid].x, positions[gid].y, params);
        }
    }

    __global__ void cellStartEndKernel(int boidCount,  int* device_gridCellIndex, int* device_gridCellStart, int* device_gridCellEnd)
    {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if(gid < boidCount)
        {   
            if(gid == 0 || device_gridCellIndex[gid] != device_gridCellIndex[gid-1])
            {
                device_gridCellStart[device_gridCellIndex[gid]] = gid;
            }
            if(gid == boidCount-1 || device_gridCellIndex[gid] != device_gridCellIndex[gid+1])
            {
                device_gridCellEnd[device_gridCellIndex[gid]] = gid;
            }
        }
    }

    void computeGridCellIndex(int boidsCount, Params params, glm::vec2* device_positions, glm::vec2* device_velocities, int* device_gridCellIndex, int* device_gridCellStart, int* device_gridCellEnd, int* boidSequence, glm::vec2* device_positionsSorted, glm::vec2* device_velocitiesSorted, int* colorIndex, int* colorSorted)
    {
        int threadsPerBlock = 128;
        int blocksPerGrid = (boidsCount + threadsPerBlock - 1) / threadsPerBlock;
        boidCellKernel<<<blocksPerGrid, threadsPerBlock>>>(boidsCount, params, device_positions, device_gridCellIndex);
        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());

        thrust::device_vector<int> sequence(boidsCount);
        thrust::sequence(thrust::device, sequence.begin(), sequence.end());
        thrust::sort_by_key(thrust::device, device_gridCellIndex, device_gridCellIndex+boidsCount, sequence.begin());


        thrust::device_vector<glm::vec2> device_positionsSortedVector(boidsCount);
        thrust::device_vector<glm::vec2> device_velocitiesSortedVector(boidsCount);

        thrust::device_ptr<glm::vec2> dev_ptr_positions(device_positions);
        thrust::gather(sequence.begin(), sequence.end(), dev_ptr_positions, device_positionsSortedVector.begin());

        thrust::device_ptr<glm::vec2> dev_ptr_velocities(device_velocities);
        thrust::gather(sequence.begin(), sequence.end(), dev_ptr_velocities, device_velocitiesSortedVector.begin());

        thrust::device_ptr<int> dev_ptr_colorIndex(colorIndex);
        thrust::device_ptr<int> dev_ptr_colorSorted(colorSorted);
        thrust::gather(sequence.begin(), sequence.end(), dev_ptr_colorIndex, dev_ptr_colorSorted);

        thrust::copy(device_positionsSortedVector.begin(), device_positionsSortedVector.end(), device_positionsSorted);
        thrust::copy(device_velocitiesSortedVector.begin(), device_velocitiesSortedVector.end(), device_velocitiesSorted);

        blocksPerGrid = (params.cellCount + threadsPerBlock - 1) / threadsPerBlock;
        initStartEndKernel<<<blocksPerGrid, threadsPerBlock>>>(params.cellCount, device_gridCellStart, device_gridCellEnd);
        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());

        blocksPerGrid = (boidsCount + threadsPerBlock - 1) / threadsPerBlock;

        cellStartEndKernel<<<blocksPerGrid, threadsPerBlock>>>(boidsCount, device_gridCellIndex, device_gridCellStart, device_gridCellEnd);
        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());
    }
}