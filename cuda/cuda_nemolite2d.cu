#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "FortranArray2D.cu"

// Working precision
typedef double wp_t;

// <<<number_of_blocks, size_of_block>>>

struct GridConstants
{
  FortranArray2D<wp_t, 1, 1>* e1t = nullptr;
  FortranArray2D<wp_t, 1, 1>* e2t = nullptr;
  FortranArray2D<wp_t, 0, 1>* e1u = nullptr;
  FortranArray2D<wp_t, 0, 1>* e2u = nullptr;

  FortranArray2D<wp_t, 0, 0>* e1f = nullptr;
  FortranArray2D<wp_t, 0, 0>* e2f = nullptr;
  FortranArray2D<wp_t, 1, 0>* e1v = nullptr;
  FortranArray2D<wp_t, 1, 0>* e2v = nullptr;

  FortranArray2D<wp_t, 1, 1>* e12t = nullptr;
  FortranArray2D<wp_t, 0, 1>* e12u = nullptr;
  FortranArray2D<wp_t, 1, 0>* e12v = nullptr;

  FortranArray2D<wp_t, 0, 1>* gphiu = nullptr;
  FortranArray2D<wp_t, 1, 0>* gphiv = nullptr;
  FortranArray2D<wp_t, 0, 0>* gphif = nullptr;

  FortranArray2D<wp_t, 1, 1>* xt = nullptr;
  FortranArray2D<wp_t, 1, 1>* yt = nullptr;

  FortranArray2D<wp_t, 1, 1>* ht = nullptr;
  FortranArray2D<wp_t, 0, 1>* hu = nullptr;
  FortranArray2D<wp_t, 0, 1>* hv = nullptr;

  // -1 = Water cell outside computational domain
  //  0 = Land cell
  //  1 = Water cell inside computational domain
  FortranArray2D<int, 0, 0>* pt = nullptr;
};

struct SimulationVariables
{
  // Sea surface height - current values.
  FortranArray2D<wp_t, 1, 1>* sshn = nullptr;
  FortranArray2D<wp_t, 0, 1>* sshn_u = nullptr;
  FortranArray2D<wp_t, 1, 0>* sshn_v = nullptr;

  // Sea surface height - next step's values
  FortranArray2D<wp_t, 1, 1>* ssha = nullptr;
  FortranArray2D<wp_t, 0, 1>* ssha_u = nullptr;
  FortranArray2D<wp_t, 1, 0>* ssha_v = nullptr;

  // Velocities - current values
  FortranArray2D<wp_t, 0, 1>* un = nullptr;
  FortranArray2D<wp_t, 1, 0>* vn = nullptr;

  // Velocities - next step's values
  FortranArray2D<wp_t, 0, 1>* ua = nullptr;
  FortranArray2D<wp_t, 1, 0>* va = nullptr;
};

struct ModelParameters
{
  // Number of columns in modle grid
  int jpi = 0;

  // Number of rows in model grid
  int jpj = 0;

  // Grid size in x and y directions (m)
  wp_t dx = 0;
  wp_t dy = 0;

  // Constant depth (m)
  wp_t dep_const = 0.0;

  // First time step
  int nit000 = 0;

  // Final time step
  int nitend = 0;

  // Interval on which to save results
  int irecord = 0;

  // Size of time step (s)
  wp_t rdt = 0.0;

  // Bottom friction coefficient
  wp_t cbfr = 0.0;

  // Horizontal kinematic viscosity coefficient
  wp_t visc = 0.0;
};

__global__ void
k_setup_model_params(const int jpi,
                     const int jpj,
                     const wp_t dx,
                     const wp_t dy,
                     const wp_t dep_const,
                     const int nit000,
                     const int nitend,
                     const int irecord,
                     const wp_t rdt,
                     const wp_t cbfr,
                     const wp_t visc);

__global__ void
k_continuity();

__global__ void
k_boundary_conditions();

__global__ void
k_momentum();

__global__ void
k_next();

void
finalise();

FortranArray2D<wp_t, 0, 1>* global_array = nullptr;

extern "C"
{
  void cuda_setup_model_params_(int jpi,
                                int jpj,
                                wp_t dx,
                                wp_t dy,
                                wp_t dep_const,
                                int nit000,
                                int nitend,
                                int irecord,
                                wp_t rdt,
                                wp_t cbfr,
                                wp_t visc);

  void cuda_initialise_grid_();

  void cuda_continuity_() { k_continuity<<<1, 10>>>(); }

  void cuda_boundary_conditions_() { k_boundary_conditions<<<1, 10>>>(); }

  void cuda_momentum_()
  {
    k_momentum<<<1, 10>>>();
    cudaDeviceSynchronize();
  }

  void cuda_next_() { k_next<<<1, 10>>>(); }

  void cuda_finalise_() { finalise(); }
};

__device__ GridConstants grid_constants;
__device__ SimulationVariables simulation_vars;
__device__ ModelParameters model_params;

void
cuda_initialise_grid_()
{
  const int jpi = model_params.jpi;
  const int jpj = model_params.jpj;

  if (jpi == 0 || jpj == 0) {
    fprintf(stderr,
            "Invalid grid size: (%d, %d); have you setup model params?",
            model_params.jpi,
            model_params.jpj);
  }

  // Create and allocate the grid constants
  grid_constants.e1t = new FortranArray2D<wp_t, 1, 1>(jpi, jpj);
  grid_constants.e2t = new FortranArray2D<wp_t, 1, 1>(jpi, jpj);
  grid_constants.e1u = new FortranArray2D<wp_t, 0, 1>(jpi, jpj);
  grid_constants.e2u = new FortranArray2D<wp_t, 0, 1>(jpi, jpj);

  grid_constants.e1f = new FortranArray2D<wp_t, 0, 0>(jpi, jpj);
  grid_constants.e2f = new FortranArray2D<wp_t, 0, 0>(jpi, jpj);
  grid_constants.e1v = new FortranArray2D<wp_t, 1, 0>(jpi, jpj);
  grid_constants.e2v = new FortranArray2D<wp_t, 1, 0>(jpi, jpj);

  grid_constants.e12t = new FortranArray2D<wp_t, 1, 1>(jpi, jpj);
  grid_constants.e12u = new FortranArray2D<wp_t, 0, 1>(jpi, jpj);
  grid_constants.e12v = new FortranArray2D<wp_t, 1, 0>(jpi, jpj);

  grid_constants.gphiu = new FortranArray2D<wp_t, 0, 1>(jpi, jpj);
  grid_constants.gphiv = new FortranArray2D<wp_t, 1, 0>(jpi, jpj);
  grid_constants.gphif = new FortranArray2D<wp_t, 0, 0>(jpi, jpj);

  grid_constants.xt = new FortranArray2D<wp_t, 1, 1>(jpi, jpj);
  grid_constants.yt = new FortranArray2D<wp_t, 1, 1>(jpi, jpj);

  grid_constants.ht = new FortranArray2D<wp_t, 1, 1>(jpi, jpj);
  grid_constants.hu = new FortranArray2D<wp_t, 0, 1>(jpi, jpj);
  grid_constants.hv = new FortranArray2D<wp_t, 0, 1>(jpi, jpj);

  grid_constants.pt = new FortranArray2D<int, 0, 0>(jpi + 1, jpj + 1);

  // TODO: Initialise on host and memcpy over to device.
}

void
cuda_setup_model_params_(int jpi,
                         int jpj,
                         wp_t dx,
                         wp_t dy,
                         wp_t dep_const,
                         int nit000,
                         int nitend,
                         int irecord,
                         wp_t rdt,
                         wp_t cbfr,
                         wp_t visc)
{
  k_setup_model_params<<<1, 1>>>(
    jpi, jpj, dx, dy, dep_const, nit000, nitend, irecord, rdt, cbfr, visc);
}

__global__ void
k_setup_model_params(const int jpi,
                     const int jpj,
                     const wp_t dx,
                     const wp_t dy,
                     const wp_t dep_const,
                     const int nit000,
                     const int nitend,
                     const int irecord,
                     const wp_t rdt,
                     const wp_t cbfr,
                     const wp_t visc)
{
  printf("Initialising model parameters on device.\n");
  model_params = {
    .jpi = jpi,
    .jpj = jpj,
    .dx = dx,
    .dy = dy,
    .dep_const = dep_const,
    .nit000 = nit000,
    .nitend = nitend,
    .irecord = irecord,
    .rdt = rdt,
    .cbfr = cbfr,
    .visc = visc,
  };
}

__global__ void
k_continuity()
{
  // TODO:
}

__global__ void
k_momentum()
{
  // TODO:
}

__global__ void
k_boundary_conditions()
{
  // TODO:
}

__global__ void
k_next()
{
  // TODO:
}

void
finalise()
{
  cudaError_t cudaStatus;

  delete grid_constants.e1t;
  delete grid_constants.e2t;
  delete grid_constants.e1u;
  delete grid_constants.e2u;

  delete grid_constants.e1f;
  delete grid_constants.e2f;
  delete grid_constants.e1v;
  delete grid_constants.e2v;

  delete grid_constants.e12t;
  delete grid_constants.e12u;
  delete grid_constants.e12v;

  delete grid_constants.gphiu;
  delete grid_constants.gphiv;
  delete grid_constants.gphif;

  delete grid_constants.xt;
  delete grid_constants.yt;

  delete grid_constants.ht;
  delete grid_constants.hu;
  delete grid_constants.hv;

  delete grid_constants.pt;

  // TODO: Delete simulation params arrays.

  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "CUDA device reset failed.");
  }
}
