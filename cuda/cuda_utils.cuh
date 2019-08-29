#pragma once

#define FREE_ARRAY(arr)                                                        \
  {                                                                            \
    (arr)->free_memory();                                                      \
    delete (arr);                                                              \
  }

#define CUDACHECK(ans)                                                         \
  {                                                                            \
    gpu_assert((ans), __FILE__, __LINE__);                                     \
  }

inline void
gpu_assert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess) {
    fprintf(
      stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      throw code;
    }
  }
}

template<typename T>
inline void
get_kernel_dims(const int max_x,
                const int max_y,
                T kernel,
                dim3& out_blocksize,
                dim3& out_gridsize)
{

    // Use the occupancy calculator to find the 1D numbr of threads per block which maximises occupancy. Assumes a square number. 
    int minGridSize = 0; // Minimum grid size to achieve max occupancy
    int totalThreadsPerBlock = 0; // Number of threads per block
    // Query the occupancy calculator.!
    CUDACHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &totalThreadsPerBlock, kernel, 0, 0));

    // Assume we alwasy want square kernels. This may be sub-optimal.
    int blocksize_xy = (int)floor(sqrt(totalThreadsPerBlock));


    // Suggest block dimensions. Threads per block must not exceed 1024 on most
    // hardware, registers will probably be a limiting factor.
    dim3 blocksize(blocksize_xy, blocksize_xy);

    // Shrink either if larger than the actual dimensions to minimise work
    // @note this might reduce the work below ideal occupancy, for very wide/narrow problems
    if (blocksize.x > max_x) {
    blocksize.x = max_y;
    }
    if (blocksize.y > max_x) {
    blocksize.y = max_y;
    }

    // Calculate the gridsize. 
    dim3 gridsize;
    gridsize.x = (max_x + blocksize.x - 1) / blocksize.x;
    gridsize.y = (max_y + blocksize.y - 1) / blocksize.y;

    //  Set for the outside ones. 
    out_blocksize = blocksize;
    out_gridsize = gridsize;
}
