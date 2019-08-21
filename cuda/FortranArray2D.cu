#include <iostream>
#include <cstdlib>

template < typename type, int row_start_idx, int col_start_idx >
class FortranArray2D
{
  public:
    FortranArray2D(int row_end_idx, int col_end_idx)
      :
        n_rows(row_end_idx - row_start_idx + 1),
        n_cols(col_end_idx - col_start_idx + 1)
    {
      const size_t array_size = n_rows * n_cols * sizeof(type);

      cudaError_t cudaResult = cudaMalloc(reinterpret_cast<void**>(&data), array_size);

      if (cudaResult != cudaSuccess) {
        printf("Failed to allocate 2D array.");
        exit(EXIT_FAILURE);
      }
    }

    ~FortranArray2D()
    {
      cudaError_t cudaResult = cudaFree(this->data);

      if (cudaResult != cudaSuccess) {
        printf("Failed to free 2D array.");
        exit(EXIT_FAILURE);
      }
    }

    void set_value(type input_value) {
      // TODO:
    }

    inline type& operator()( int i, int j ) 
    {
      return this->data[ (i - row_start_idx) + (j - col_start_idx)*(this->n_rows) ];
    }

  private:
    const int n_rows;
    const int n_cols;
    type * data;
};

int testFortranArray2D()
{
  const int M = 3, N = 2;

  std::printf("\n(0:,0:) Indexing\n");
  FortranArray2D<int,0,0> A(N,M);

  for ( int i = 0; i <= M; ++i )
    for ( int j = 0; j <= N; ++j )
    {
      std::printf( "%d\n", A(j,i) ); 
    }

  std::printf("\n(1:,0:) Indexing\n");
  FortranArray2D<int,1,0> B(N,M);

  for ( int i = 0; i <= M; ++i )
    for ( int j = 1; j <= N; ++j )
    {
      std::printf( "%d\n", B(j,i) ); 
    }

  std::printf("\n(0:,1:) Indexing\n");
  FortranArray2D<int,0,1> C(N,M);

  for ( int i = 1; i <= M; ++i )
    for ( int j = 0; j <= N; ++j )
    {
      std::printf( "%d\n", C(j,i) ); 
    }

  std::printf("\n(1:,1:) Indexing\n");
  FortranArray2D<int,1,1> D(N,M);

  for ( int i = 1; i <= M; ++i )
    for ( int j = 1; j <= N; ++j )
    {
      std::printf( "%d\n", D(j,i) ); 
    }
  return EXIT_SUCCESS;
}
