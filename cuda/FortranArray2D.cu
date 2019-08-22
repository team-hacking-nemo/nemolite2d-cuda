#include <cstring>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

template<typename type, int row_start_idx, int col_start_idx>
class FortranArray2D
{
public:
  __host__ FortranArray2D(const int row_end_idx,
                          const int col_end_idx)
                          //const int initial_value)
    : num_rows(row_end_idx - row_start_idx + 1)
    , num_cols(col_end_idx - col_start_idx + 1)
    , data_size(num_rows * num_cols * sizeof(type))
  {
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&device_data, data_size);
    if (cudaStatus != cudaSuccess) {
      printf("Failed to allocate 2D array on device.\n");
      exit(EXIT_FAILURE);
    }

    host_data = reinterpret_cast<type*>(std::malloc(data_size));
    std::memset(host_data, 0, data_size);

    // Prepare the device object
    cudaStatus = cudaMemcpy(device_data, host_data, data_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      printf("Failed to copy 2D array to device.\n");
      exit(EXIT_FAILURE);
    }
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

  __host__ type* retrieve_data_from_device(type* const out_data) {
    cudaMemcpy(out_data, device_data, data_size, cudaMemcpyDeviceToHost);
  }

  __device__ void set_value(type input_value)
  {
    // TODO:
  }

  __device__ inline type& operator()(int i, int j)
  {
    return this
      ->device_data[(i - row_start_idx) + (j - col_start_idx) * (this->num_rows)];
  }

private:
  const int num_rows;
  const int num_cols;
  const int data_size;
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
