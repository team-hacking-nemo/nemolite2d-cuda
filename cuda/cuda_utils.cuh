#pragma once

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

inline void
get_kernel_dims(const int max_x,
                const int max_y,
                dim3& out_threads_per_block,
                dim3& out_num_blocks)
{

  // Suggest block dimensions. Threads per block must not exceed 1024 on most
  // hardware, registers will probably be a limiting factor.
  dim3 threads_per_block(16, 16);

  // Shrink either if larger than the actual dimensions to minimise work
  if (threads_per_block.x > max_x) {
    threads_per_block = max_y;
  }
  if (threads_per_block.y > max_x) {
    threads_per_block.y = max_y;
  }

  dim3 num_blocks;
  num_blocks.x = (max_x + threads_per_block.x - 1) / threads_per_block.x;
  num_blocks.y = (max_y + threads_per_block.y - 1) / threads_per_block.y;

  out_threads_per_block = threads_per_block;
  out_num_blocks = num_blocks;
}
