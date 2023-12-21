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
        // printf("cell %d, device_gridCellStart[cell] %d\n", gid, device_gridCellStart[gid]);
    }

    __global__ void boidCellKernel(int boidsCount, GridParams params, glm::vec2* positions, int* gridCellIndex)
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
        // printf("jakie gidy: %d\n", gid);
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
            // printf("gid %d, cell %d, device_gridCellStart[cell] %d\n", gid, device_gridCellIndex[gid], device_gridCellStart[device_gridCellIndex[gid]]);
        }
    }

    void computeGridCellIndex(int boidsCount, GridParams params, glm::vec2* device_positions, glm::vec2* device_velocities, int* device_gridCellIndex, int* device_gridCellStart, int* device_gridCellEnd, int* boidSequence, glm::vec2* device_positionsSorted, glm::vec2* device_velocitiesSorted)
    {
        int threadsPerBlock = 128;
        int blocksPerGrid = (boidsCount + threadsPerBlock - 1) / threadsPerBlock;
        boidCellKernel<<<blocksPerGrid, threadsPerBlock>>>(boidsCount, params, device_positions, device_gridCellIndex);
        // cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Check for errors on the CUDA device side after kernel execution
        gpuErrchk(cudaDeviceSynchronize());

        thrust::device_vector<int> sequence(boidsCount);
        thrust::sequence(thrust::device, sequence.begin(), sequence.end());
        thrust::sort_by_key(thrust::device, device_gridCellIndex, device_gridCellIndex+boidsCount, sequence.begin());

        // for(int i = 0; i < boidsCount; i++)
        // {
        //     printf("i: %d, device_gridCellIndex: %d\n", i, device_gridCellIndex[i]);
        // }

        // for(int i = 0; i < boidsCount; i++)
        // {
        //     printf("i: %d, sequence: %d\n", i, sequence[i]);
        // }

        thrust::device_vector<glm::vec2> device_positionsSortedVector(boidsCount);
        thrust::device_vector<glm::vec2> device_velocitiesSortedVector(boidsCount);

        thrust::device_ptr<glm::vec2> dev_ptr_positions(device_positions);
        thrust::gather(sequence.begin(), sequence.end(), dev_ptr_positions, device_positionsSortedVector.begin());

        thrust::device_ptr<glm::vec2> dev_ptr_velocities(device_velocities);
        thrust::gather(sequence.begin(), sequence.end(), dev_ptr_velocities, device_velocitiesSortedVector.begin());
// printf("hello\n");
        // thrust::gather(sequence.begin(), sequence.end(), device_positions, device_positionsSorted);
        // thrust::gather(sequence.begin(), sequence.end(), device_positionsSortedVector.begin(), device_velocitiesSortedVector.begin());
        // thrust::gather(device_velocities,device_velocities, device_positionsSortedVector.begin(), device_velocitiesSortedVector.begin());
        
        // thrust::gather(sequence.begin(), sequence.end(), device_velocities, device_velocitiesSorted);
        // cudaMalloc(&device_positionsSorted, boidsCount * sizeof(glm::vec2));

        thrust::copy(device_positionsSortedVector.begin(), device_positionsSortedVector.end(), device_positionsSorted);
        thrust::copy(device_velocitiesSortedVector.begin(), device_velocitiesSortedVector.end(), device_velocitiesSorted);

        blocksPerGrid = (params.cellCount + threadsPerBlock - 1) / threadsPerBlock;

        // printf("cellCount: %d\n", params.cellCount);
        initStartEndKernel<<<blocksPerGrid, threadsPerBlock>>>(params.cellCount, device_gridCellStart, device_gridCellEnd);
        gpuErrchk(cudaGetLastError());

        // Check for errors on the CUDA device side after kernel execution
        gpuErrchk(cudaDeviceSynchronize());

        blocksPerGrid = (boidsCount + threadsPerBlock - 1) / threadsPerBlock;

        cellStartEndKernel<<<blocksPerGrid, threadsPerBlock>>>(boidsCount, device_gridCellIndex, device_gridCellStart, device_gridCellEnd);


        // for(int i = 0; i < params.cellCount; i++)
        // {
        //     printf("i: %d, device_gridCellStart: %d\n", i, device_gridCellStart[i]);
        // }

        // cudaDeviceSynchronize();

        gpuErrchk(cudaGetLastError());

        // Check for errors on the CUDA device side after kernel execution
        gpuErrchk(cudaDeviceSynchronize());

        // printf("passed\n");
        // int N = boidsCount;

        // thrust::device_vector<glm::vec2> dev_pos(N);             // Positions of boids
        // thrust::device_vector<glm::vec2> dev_vel1(N);
        // thrust::device_vector<glm::vec2> sorted_pos(N);
        // thrust::device_vector<glm::vec2> sorted_vel1(N);


        // thrust::device_ptr<glm::vec2> dev_ptr_positions(device_positions);

        // thrust::gather(sequence.begin(), sequence.end(), dev_ptr_positions, device_positionsSortedVector.begin());
        // thrust::gather(sequence.begin(), sequence.end(), dev_vel1.begin(), sorted_vel1.begin());
    
    }
}