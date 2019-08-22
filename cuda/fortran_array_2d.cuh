#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>

#include "cuda_utils.cuh"

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

    CUDACHECK(cudaMalloc((void**)&device_data, data_size));

    type* zero_data =
      reinterpret_cast<type*>(std::calloc(num_rows * num_cols, data_size));

    // Prepare the device object
    CUDACHECK(cudaMemcpy(device_data, zero_data, data_size, cudaMemcpyHostToDevice));

    free(zero_data);
  }

  __host__ ~FortranArray2D()
  {
		CUDACHECK(cudaFree(this->device_data));
  }

  __host__ void retrieve_data_from_device(type* const out_data)
  {
    CUDACHECK(cudaMemcpy(out_data, device_data, data_size, cudaMemcpyDeviceToHost));
  }

  __host__ type* get_device_data_ptr() { return this->device_data; }

  __host__ void set_device_data_ptr(type* new_device_data_ptr)
  {
    this->device_data = new_device_data_ptr;
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
};

#if defined(TEST_CODE)
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
#endif
