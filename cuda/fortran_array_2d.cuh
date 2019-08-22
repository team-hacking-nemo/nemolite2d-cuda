#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>

template<typename type, int row_start_idx, int col_start_idx>
class FortranArray2D
{
public:
  __host__ FortranArray2D(const int row_end_idx, const int col_end_idx)
    // const int initial_value)
    : num_rows(row_end_idx - row_start_idx + 1)
    , num_cols(col_end_idx - col_start_idx + 1)
    , data_size(num_rows * num_cols * sizeof(type))
  {
    // data_size = num_rows * num_cols * sizeof(type);

    printf("Row end index: %d, column end index: %d, type size: %d",
           row_end_idx,
           col_end_idx,
           sizeof(type));

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&device_data, data_size);
    assert(cudaStatus == cudaSuccess);

    host_data =
      reinterpret_cast<type*>(std::calloc(num_rows * num_cols, data_size));

    // Prepare the device object
    cudaStatus =
      cudaMemcpy(device_data, host_data, data_size, cudaMemcpyHostToDevice);
    assert(cudaStatus == cudaSuccess);
  }

  __host__ ~FortranArray2D()
  {
    free(this->host_data);

    cudaError_t cudaResult = cudaFree(this->device_data);

    if (cudaResult != cudaSuccess) {
      printf("Failed to free 2D array.");
      exit(EXIT_FAILURE);
    }
  }

  __host__ type* retrieve_data_from_device(type* const out_data)
  {
    cudaMemcpy(out_data, device_data, data_size, cudaMemcpyDeviceToHost);
  }

  __device__ inline type& operator()(const int i, const int j) const
  {
    return this->device_data[(i - row_start_idx) +
                             (j - col_start_idx) * (this->num_rows)];
  }

private:
  const int num_rows;
  const int num_cols;
  const size_t data_size;
  type* device_data;
  type* host_data;
};

/*
int
testFortranArray2D()
{
  const int M = 3, N = 2;

  std::printf("\n(0:,0:) Indexing\n");
  FortranArray2D<int, 0, 0> A(N, M);

  for (int i = 0; i <= M; ++i)
    for (int j = 0; j <= N; ++j) {
      std::printf("%d\n", A(j, i));
    }

  std::printf("\n(1:,0:) Indexing\n");
  FortranArray2D<int, 1, 0> B(N, M);

  for (int i = 0; i <= M; ++i) {
    for (int j = 1; j <= N; ++j) {
      std::printf("%d\n", B(j, i));
    }
  }

  std::printf("\n(0:,1:) Indexing\n");
  FortranArray2D<int, 0, 1> C(N, M);

  for (int i = 1; i <= M; ++i) {
    for (int j = 0; j <= N; ++j) {
      std::printf("%d\n", C(j, i));
    }
  }

  std::printf("\n(1:,1:) Indexing\n");
  FortranArray2D<int, 1, 1> D(N, M);

  for (int i = 1; i <= M; ++i) {
    for (int j = 1; j <= N; ++j) {
      std::printf("%d\n", D(j, i));
    }
  }

  return EXIT_SUCCESS;
}
*/
