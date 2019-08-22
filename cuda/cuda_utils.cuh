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
      exit(code);
    }
  }
}
