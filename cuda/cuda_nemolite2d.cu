#include <cassert>
#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "cuda_utils.cuh"
#include "fortran_array_2d.cuh"

// Working precision
typedef double wp_t;

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
  FortranArray2D<wp_t, 1, 0>* hv = nullptr;

  // -1 = Water cell outside computational domain
  //  0 = Land cell
  //  1 = Water cell inside computational domain
  FortranArray2D<int, 0, 0>* pt = nullptr;

  GridConstants() {}
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

  // We need to double buffer the ua and va due to possible race conditions in
  // the Flather boundary conditions.
  FortranArray2D<wp_t, 0, 1>* ua_buffer = nullptr;
  FortranArray2D<wp_t, 1, 0>* va_buffer = nullptr;

  SimulationVariables() {}
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
k_initialise_grid(const FortranArray2D<wp_t, 1, 1> sshn,
                  const FortranArray2D<wp_t, 0, 1> sshn_u,
                  const FortranArray2D<wp_t, 1, 0> sshn_v,

                  const FortranArray2D<wp_t, 1, 1> e1t,
                  const FortranArray2D<wp_t, 1, 1> e2t,

                  const FortranArray2D<wp_t, 0, 1> e1u,
                  const FortranArray2D<wp_t, 0, 1> e2u,

                  const FortranArray2D<wp_t, 0, 0> e1f,
                  const FortranArray2D<wp_t, 0, 0> e2f,

                  const FortranArray2D<wp_t, 1, 0> e1v,
                  const FortranArray2D<wp_t, 1, 0> e2v,

                  const FortranArray2D<wp_t, 1, 1> e12t,
                  const FortranArray2D<wp_t, 0, 1> e12u,
                  const FortranArray2D<wp_t, 1, 0> e12v,

                  const FortranArray2D<wp_t, 0, 1> gphiu,
                  const FortranArray2D<wp_t, 1, 0> gphiv,
                  const FortranArray2D<wp_t, 0, 0> gphif,

                  const FortranArray2D<wp_t, 1, 1> xt,
                  const FortranArray2D<wp_t, 1, 1> yt,

                  const FortranArray2D<wp_t, 1, 1> ht,
                  const FortranArray2D<wp_t, 0, 1> hu,
                  const FortranArray2D<wp_t, 1, 0> hv,

                  const FortranArray2D<int, 0, 0> pt,

                  const int jpi,
                  const int jpj,

                  const wp_t dx,
                  const wp_t dy,

                  const wp_t dep_const);

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
k_continuity(const FortranArray2D<wp_t, 1, 1> sshn,
             const FortranArray2D<wp_t, 0, 1> sshn_u,
             const FortranArray2D<wp_t, 1, 0> sshn_v,

             const FortranArray2D<wp_t, 1, 1> ssha,

             const FortranArray2D<wp_t, 0, 1> un,
             const FortranArray2D<wp_t, 1, 0> vn,

             const FortranArray2D<wp_t, 0, 1> hu,
             const FortranArray2D<wp_t, 1, 0> hv,

             const FortranArray2D<wp_t, 1, 1> e12t,

             const int jpi,
             const int jpj,

             const int rdt);

__global__ void
k_boundary_conditions(wp_t rtime,

                      const FortranArray2D<wp_t, 0, 1> sshn_u,
                      const FortranArray2D<wp_t, 1, 0> sshn_v,

                      const FortranArray2D<wp_t, 1, 1> ssha,

                      const FortranArray2D<wp_t, 0, 1> ua,
                      const FortranArray2D<wp_t, 1, 0> va,

                      const FortranArray2D<wp_t, 0, 1> ua_buffer,
                      const FortranArray2D<wp_t, 1, 0> va_buffer,

                      const FortranArray2D<wp_t, 0, 1> hu,
                      const FortranArray2D<wp_t, 1, 0> hv,

                      const FortranArray2D<int, 0, 0> pt,

                      const int jpi,
                      const int jpj);

__global__ void
k_momentum();

__global__ void
k_next(const FortranArray2D<wp_t, 1, 1> sshn,
       const FortranArray2D<wp_t, 0, 1> sshn_u,
       const FortranArray2D<wp_t, 1, 0> sshn_v,

       const FortranArray2D<wp_t, 1, 1> ssha,

       const FortranArray2D<wp_t, 0, 1> un,
       const FortranArray2D<wp_t, 1, 0> vn,

       const FortranArray2D<wp_t, 0, 1> ua,
       const FortranArray2D<wp_t, 1, 0> va,

       const FortranArray2D<wp_t, 1, 1> e12t,
       const FortranArray2D<wp_t, 0, 1> e12u,
       const FortranArray2D<wp_t, 1, 0> e12v,

       const FortranArray2D<int, 0, 0> pt,

       const int jpi,
       const int jpj);

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

  void cuda_continuity_();
  void cuda_momentum_();
  void cuda_boundary_conditions_(wp_t rtime);
  void cuda_next_();

  void cuda_retrieve_grid_constants_(wp_t* const out_xt,
                                     wp_t* const out_yt,
                                     wp_t* const out_ht,
                                     const int jpi,
                                     const int jpj);

  void cuda_retrieve_results_(wp_t* const out_sshn,
                              wp_t* const out_un,
                              wp_t* const out_vn,
                              wp_t* const out_ua,
                              wp_t* const out_va,
                              const int jpi,
                              const int jpj);

  void cuda_finalise_();
};

GridConstants grid_constants;
SimulationVariables simulation_vars;

ModelParameters model_params;

__constant__ wp_t pi;
__constant__ wp_t g;     // Gravity
__constant__ wp_t omega; // Earth rotation speed (s^(-1))
__constant__ wp_t d2r;   // Degrees to radians

void
cuda_initialise_grid_()
{
  const wp_t host_pi = 3.1415926535897932;
  const wp_t host_g = 9.80665;
  const wp_t host_omega = 7.292116e-05;
  const wp_t host_d2r = host_pi / 180.0;

  CUDACHECK(cudaMemcpyToSymbol(pi, &host_pi, sizeof(pi)));
  CUDACHECK(cudaMemcpyToSymbol(g, &host_g, sizeof(g)));
  CUDACHECK(cudaMemcpyToSymbol(omega, &host_omega, sizeof(omega)));
  CUDACHECK(cudaMemcpyToSymbol(d2r, &host_d2r, sizeof(d2r)));

  const int jpi = model_params.jpi;
  const int jpj = model_params.jpj;

  if (jpi == 0 || jpj == 0) {
    fprintf(stderr,
            "Invalid grid size: (%d, %d); have you setup model params?",
            jpi,
            jpj);
  }

  printf(
    "[CUDA](Host) Initialising grid constants and simulation variables.\n");

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
  grid_constants.hv = new FortranArray2D<wp_t, 1, 0>(jpi, jpj);

  grid_constants.pt = new FortranArray2D<int, 0, 0>(jpi + 1, jpj + 1);

  // Create and allocate simulation variables
  simulation_vars.sshn = new FortranArray2D<wp_t, 1, 1>(jpi, jpj);
  simulation_vars.sshn_u = new FortranArray2D<wp_t, 0, 1>(jpi, jpj);
  simulation_vars.sshn_v = new FortranArray2D<wp_t, 1, 0>(jpi, jpj);

  simulation_vars.ssha = new FortranArray2D<wp_t, 1, 1>(jpi, jpj);
  simulation_vars.ssha_u = new FortranArray2D<wp_t, 0, 1>(jpi, jpj);
  simulation_vars.ssha_v = new FortranArray2D<wp_t, 1, 0>(jpi, jpj);

  simulation_vars.un = new FortranArray2D<wp_t, 0, 1>(jpi, jpj);
  simulation_vars.vn = new FortranArray2D<wp_t, 1, 0>(jpi, jpj);

  simulation_vars.ua = new FortranArray2D<wp_t, 0, 1>(jpi, jpj);
  simulation_vars.va = new FortranArray2D<wp_t, 1, 0>(jpi, jpj);

  simulation_vars.ua_buffer = new FortranArray2D<wp_t, 0, 1>(jpi, jpj);
  simulation_vars.va_buffer = new FortranArray2D<wp_t, 1, 0>(jpi, jpj);

  CUDACHECK(cudaDeviceSynchronize());

  // printf("Launching test kernel...\n");
  // k_test_kernel<<<numBlocks, threadsPerBlock>>>(*simulation_vars.ua);
  // CUDACHECK(cudaDeviceSynchronize());
  // printf("Synchronised test kernel.\n");

  dim3 threads_per_block;
  dim3 num_blocks;
  get_kernel_dims(jpi + 2, jpj + 2, k_initialise_grid, threads_per_block, num_blocks);

  printf("Kernel dimensions: ((%d, %d), (%d, %d).\n",
         threads_per_block.x,
         threads_per_block.y,
         num_blocks.x,
         num_blocks.y);

  // Initialise simulation parameters
  k_initialise_grid<<<num_blocks, threads_per_block>>>(*simulation_vars.sshn,
                                                       *simulation_vars.sshn_u,
                                                       *simulation_vars.sshn_v,

                                                       *grid_constants.e1t,
                                                       *grid_constants.e2t,

                                                       *grid_constants.e1u,
                                                       *grid_constants.e2u,

                                                       *grid_constants.e1f,
                                                       *grid_constants.e2f,

                                                       *grid_constants.e1v,
                                                       *grid_constants.e2v,

                                                       *grid_constants.e12t,
                                                       *grid_constants.e12u,
                                                       *grid_constants.e12v,

                                                       *grid_constants.gphiu,
                                                       *grid_constants.gphiv,
                                                       *grid_constants.gphif,

                                                       *grid_constants.xt,
                                                       *grid_constants.yt,

                                                       *grid_constants.ht,
                                                       *grid_constants.hu,
                                                       *grid_constants.hv,

                                                       *grid_constants.pt,

                                                       jpi,
                                                       jpj,

                                                       model_params.dx,
                                                       model_params.dy,

                                                       model_params.dep_const);

  CUDACHECK(cudaDeviceSynchronize());
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
  printf("[CUDA](Host) Initialising model params.\n");

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

void
cuda_continuity_()
{
  const int jpi = model_params.jpi;
  const int jpj = model_params.jpj;

  dim3 threads_per_block;
  dim3 num_blocks;
  get_kernel_dims(jpi + 1, jpj + 1, k_continuity, threads_per_block, num_blocks);

  k_continuity<<<num_blocks, threads_per_block>>>(*simulation_vars.sshn,
                                                  *simulation_vars.sshn_u,
                                                  *simulation_vars.sshn_v,

                                                  *simulation_vars.ssha,

                                                  *simulation_vars.un,
                                                  *simulation_vars.vn,

                                                  *grid_constants.hu,
                                                  *grid_constants.hv,

                                                  *grid_constants.e12t,

                                                  jpi,
                                                  jpj,

                                                  model_params.rdt);
}

void
cuda_momentum_()
{
  // TODO:
}

void
cuda_boundary_conditions_(wp_t rtime)
{
  const int jpi = model_params.jpi;
  const int jpj = model_params.jpj;

  dim3 threads_per_block;
  dim3 num_blocks;
  get_kernel_dims(jpi + 1, jpj + 1, k_boundary_conditions, threads_per_block, num_blocks);

  k_boundary_conditions<<<num_blocks, threads_per_block>>>(
    rtime,

    *simulation_vars.sshn_u,
    *simulation_vars.sshn_v,

    *simulation_vars.ssha,

    *simulation_vars.ua,
    *simulation_vars.va,

    *simulation_vars.ua_buffer,
    *simulation_vars.va_buffer,

    *grid_constants.hu,
    *grid_constants.hv,

    *grid_constants.pt,

    jpi,
    jpj);

  CUDACHECK(cudaDeviceSynchronize());

  // Now swap the double buffered arrays.
  wp_t* const ua_buffered_data =
    simulation_vars.ua_buffer->get_device_data_ptr();
  wp_t* const va_buffered_data =
    simulation_vars.va_buffer->get_device_data_ptr();

  simulation_vars.ua_buffer->set_device_data_ptr(
    simulation_vars.ua->get_device_data_ptr());
  simulation_vars.va_buffer->set_device_data_ptr(
    simulation_vars.va->get_device_data_ptr());

  simulation_vars.ua->set_device_data_ptr(ua_buffered_data);
  simulation_vars.va->set_device_data_ptr(va_buffered_data);
}

void
cuda_next_()
{
  const int jpi = model_params.jpi;
  const int jpj = model_params.jpj;

  dim3 threads_per_block;
  dim3 num_blocks;
  get_kernel_dims(jpi + 1, jpj + 1, k_next, threads_per_block, num_blocks);

  k_next<<<num_blocks, threads_per_block>>>(*simulation_vars.sshn,
                                            *simulation_vars.sshn_u,
                                            *simulation_vars.sshn_v,

                                            *simulation_vars.ssha,

                                            *simulation_vars.un,
                                            *simulation_vars.vn,

                                            *simulation_vars.ua,
                                            *simulation_vars.va,

                                            *grid_constants.e12t,
                                            *grid_constants.e12u,
                                            *grid_constants.e12v,

                                            *grid_constants.pt,

                                            jpi,
                                            jpj);
}

void
cuda_retrieve_grid_constants_(wp_t* const out_xt,
                              wp_t* const out_yt,
                              wp_t* const out_ht,
                              const int jpi,
                              const int jpj)
{
  CUDACHECK(cudaDeviceSynchronize());
  grid_constants.xt->retrieve_data_from_device(out_xt);
  grid_constants.yt->retrieve_data_from_device(out_yt);
  grid_constants.ht->retrieve_data_from_device(out_ht);
}

void
cuda_retrieve_results_(wp_t* const out_sshn,
                       wp_t* const out_un,
                       wp_t* const out_vn,
                       wp_t* const out_ua,
                       wp_t* const out_va,
                       const int jpi,
                       const int jpj)
{
  CUDACHECK(cudaDeviceSynchronize());
  simulation_vars.sshn->retrieve_data_from_device(out_sshn);
  simulation_vars.un->retrieve_data_from_device(out_un);
  simulation_vars.vn->retrieve_data_from_device(out_vn);
  simulation_vars.ua->retrieve_data_from_device(out_ua);
  simulation_vars.va->retrieve_data_from_device(out_va);
}

void
cuda_finalise_()
{
  // Clean up grid constants arrays.
  FREE_ARRAY(grid_constants.e1t);
  FREE_ARRAY(grid_constants.e2t);
  FREE_ARRAY(grid_constants.e1u);
  FREE_ARRAY(grid_constants.e2u);

  FREE_ARRAY(grid_constants.e1f);
  FREE_ARRAY(grid_constants.e2f);
  FREE_ARRAY(grid_constants.e1v);
  FREE_ARRAY(grid_constants.e2v);

  FREE_ARRAY(grid_constants.e12t);
  FREE_ARRAY(grid_constants.e12u);
  FREE_ARRAY(grid_constants.e12v);

  FREE_ARRAY(grid_constants.gphiu);
  FREE_ARRAY(grid_constants.gphiv);
  FREE_ARRAY(grid_constants.gphif);

  FREE_ARRAY(grid_constants.xt);
  FREE_ARRAY(grid_constants.yt);

  FREE_ARRAY(grid_constants.ht);
  FREE_ARRAY(grid_constants.hu);
  FREE_ARRAY(grid_constants.hv);

  FREE_ARRAY(grid_constants.pt);

  // Clean up simulation params arrays.
  FREE_ARRAY(simulation_vars.sshn);
  FREE_ARRAY(simulation_vars.sshn_u);
  FREE_ARRAY(simulation_vars.sshn_v);

  FREE_ARRAY(simulation_vars.ssha);
  FREE_ARRAY(simulation_vars.ssha_u);
  FREE_ARRAY(simulation_vars.ssha_v);

  FREE_ARRAY(simulation_vars.un);
  FREE_ARRAY(simulation_vars.vn);

  FREE_ARRAY(simulation_vars.ua);
  FREE_ARRAY(simulation_vars.va);

  FREE_ARRAY(simulation_vars.ua_buffer);
  FREE_ARRAY(simulation_vars.va_buffer);

  CUDACHECK(cudaDeviceReset());
}

__global__ void
k_initialise_grid(const FortranArray2D<wp_t, 1, 1> sshn,
                  const FortranArray2D<wp_t, 0, 1> sshn_u,
                  const FortranArray2D<wp_t, 1, 0> sshn_v,

                  const FortranArray2D<wp_t, 1, 1> e1t,
                  const FortranArray2D<wp_t, 1, 1> e2t,

                  const FortranArray2D<wp_t, 0, 1> e1u,
                  const FortranArray2D<wp_t, 0, 1> e2u,

                  const FortranArray2D<wp_t, 0, 0> e1f,
                  const FortranArray2D<wp_t, 0, 0> e2f,

                  const FortranArray2D<wp_t, 1, 0> e1v,
                  const FortranArray2D<wp_t, 1, 0> e2v,

                  const FortranArray2D<wp_t, 1, 1> e12t,
                  const FortranArray2D<wp_t, 0, 1> e12u,
                  const FortranArray2D<wp_t, 1, 0> e12v,

                  const FortranArray2D<wp_t, 0, 1> gphiu,
                  const FortranArray2D<wp_t, 1, 0> gphiv,
                  const FortranArray2D<wp_t, 0, 0> gphif,

                  const FortranArray2D<wp_t, 1, 1> xt,
                  const FortranArray2D<wp_t, 1, 1> yt,

                  const FortranArray2D<wp_t, 1, 1> ht,
                  const FortranArray2D<wp_t, 0, 1> hu,
                  const FortranArray2D<wp_t, 1, 0> hv,

                  const FortranArray2D<int, 0, 0> pt,

                  const int jpi,
                  const int jpj,

                  const wp_t dx,
                  const wp_t dy,

                  const wp_t dep_const)
{
  int ji = threadIdx.x + blockIdx.x * blockDim.x;
  int jj = threadIdx.y + blockIdx.y * blockDim.y;

  // Setup the grid constants values.

  // Define model solid/open boundaries via the properties of t-cells.
  if (jj <= jpj + 1 && ji <= jpi + 1) {
    // All inner cells
    pt(ji, jj) = 1;

    // West, East and North have solid boundaries
    if (ji == 0 || ji == jpi + 1 || jj == jpj + 1) {
      pt(ji, jj) = 0;
    }

    // South open boundary
    if (jj == 0) {
      pt(ji, jj) = -1;
    }
  }

  if (ji <= jpi && jj <= jpj) {
    // 1:N, 1:M
    if (ji > 0 && jj > 0) {
      e1t(ji, jj) = dx;
      e2t(ji, jj) = dy;
      e12t(ji, jj) = e1t(ji, jj) * e2t(ji, jj);

      // NOTE: The NEMOLite2D Fortran code was designed to handle a dx that
      // varies, indicating a non-linear physical grid size (different cells
      // have different sizes). Here we assume that the dx and dy are fixed and
      // not variant on the grid cell. This makes the calculation much easier
      // and makes parallelising the below xt, yt initilisation possible.
      xt(ji, jj) = e1t(ji, jj) * (static_cast<wp_t>(ji) - 0.5);
      yt(ji, jj) = e2t(ji, jj) * (static_cast<wp_t>(jj) - 0.5);

      ht(ji, jj) = dep_const;
    }

    // 0:N, 1:M
    if (jj > 0) {
      e1u(ji, jj) = dx;
      e2u(ji, jj) = dy;
      e12u(ji, jj) = e1u(ji, jj) * e2u(ji, jj);

      gphiu(ji, jj) = 50.0;
      hu(ji, jj) = dep_const;
    }

    // 1:N, 0:M
    if (ji > 0) {
      e1v(ji, jj) = dx;
      e2v(ji, jj) = dy;
      e12v(ji, jj) = e1v(ji, jj) * e2v(ji, jj);

      gphiv(ji, jj) = 50.0;
      hv(ji, jj) = dep_const;
    }

    // 0:N, 0:M
    e1f(ji, jj) = dx;
    e2f(ji, jj) = dy;
    gphif(ji, jj) = 50.0;
  }

  // Setup the simulation variables initial values.

  if (ji <= jpi && jj > 0 && jj <= jpj) {
    int itmp1 = min(ji + 1, jpi);
    int itmp2 = max(ji, 1);
    wp_t rtmp1 =
      e12t(itmp1, jj) * sshn(itmp1, jj) + e12t(itmp2, jj) * sshn(itmp2, jj);
    sshn_u(ji, jj) = 0.5 * rtmp1 / e12u(ji, jj);
  }

  if (ji > 0 && ji <= jpi && jj <= jpj) {
    int itmp1 = min(jj + 1, jpj);
    int itmp2 = max(jj, 1);
    wp_t rtmp1 =
      e12t(ji, itmp1) * sshn(ji, itmp1) + e12t(ji, itmp2) * sshn(ji, itmp2);
    sshn_v(ji, jj) = 0.5 * rtmp1 / e12v(ji, jj);
  }
}

__global__ void
k_continuity(const FortranArray2D<wp_t, 1, 1> sshn,
             const FortranArray2D<wp_t, 0, 1> sshn_u,
             const FortranArray2D<wp_t, 1, 0> sshn_v,

             const FortranArray2D<wp_t, 1, 1> ssha,

             const FortranArray2D<wp_t, 0, 1> un,
             const FortranArray2D<wp_t, 1, 0> vn,

             const FortranArray2D<wp_t, 0, 1> hu,
             const FortranArray2D<wp_t, 1, 0> hv,

             const FortranArray2D<wp_t, 1, 1> e12t,

             const int jpi,
             const int jpj,

             const int rdt)
{
  const int ji = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = threadIdx.y + blockIdx.y * blockDim.y;

  if (ji == 0 || ji > jpi || jj == 0 || jj > jpj) {
    return;
  }

  const wp_t rtmp1 = (sshn_u(ji, jj) + hu(ji, jj)) * un(ji, jj);
  const wp_t rtmp2 = (sshn_u(ji - 1, jj) + hu(ji - 1, jj)) * un(ji - 1, jj);
  const wp_t rtmp3 = (sshn_v(ji, jj) + hv(ji, jj)) * vn(ji, jj);
  const wp_t rtmp4 = (sshn_v(ji, jj - 1) + hv(ji, jj - 1)) * vn(ji, jj - 1);

  ssha(ji, jj) =
    sshn(ji, jj) + (rtmp2 - rtmp1 + rtmp4 - rtmp3) * rdt / e12t(ji, jj);
}

__global__ void
k_momentum()
{
  // TODO:
}

__global__ void
k_boundary_conditions(wp_t rtime,
                      const FortranArray2D<wp_t, 0, 1> sshn_u,
                      const FortranArray2D<wp_t, 1, 0> sshn_v,

                      const FortranArray2D<wp_t, 1, 1> ssha,

                      const FortranArray2D<wp_t, 0, 1> ua,
                      const FortranArray2D<wp_t, 1, 0> va,

                      const FortranArray2D<wp_t, 0, 1> ua_buffer,
                      const FortranArray2D<wp_t, 1, 0> va_buffer,

                      const FortranArray2D<wp_t, 0, 1> hu,
                      const FortranArray2D<wp_t, 1, 0> hv,

                      const FortranArray2D<int, 0, 0> pt,

                      const int jpi,
                      const int jpj)
{
  const wp_t amp_tide = 0.2;
  const wp_t omega_tide = (2.0 * 3.14159) / (12.42 * 3600.0);

  int ji = threadIdx.x + blockIdx.x * blockDim.x;
  int jj = threadIdx.y + blockIdx.y * blockDim.y;

  if (ji > jpi || jj > jpj) {
    return;
  }

  // Sea surface height clamping
  if (ji > 0 && jj > 0) {
    bool is_near_boundary = pt(ji, jj - 1) < 0 || pt(ji, jj + 1) < 0 ||
                            pt(ji - 1, jj) < 0 || pt(ji + 1, jj) < 0;
    if (pt(ji, jj) > 0 && is_near_boundary) {
      ssha(ji, jj) = amp_tide * sin(omega_tide * rtime);
    }
  }

  // Solid boundary conditions for u-velocity
  if (jj > 0) {
    if (pt(ji, jj) * pt(ji + 1, jj) == 0) {
      ua_buffer(ji, jj) = 0.0;
    }
  }

  // Solid boundary conditions for v-velocity
  if (ji > 0) {
    if (pt(ji, jj) * pt(ji, jj + 1) == 0) {
      va_buffer(ji, jj) = 0.0;
    }
  }

  // Flather boundary conditions conditions for u
  if (jj > 0) {
    if (pt(ji, jj) + pt(ji + 1, jj) <= -1) {
      ua_buffer(ji, jj) = ua(ji, jj);
    } else if (pt(ji, jj) < 0) {
      const int jiu = ji + 1;
      ua_buffer(ji, jj) =
        ua(jiu, jj) + sqrt(g / hu(ji, jj)) * (sshn_u(ji, jj) - sshn_u(jiu, jj));
    } else if (pt(ji + 1, jj) < 0) {
      const int jiu = ji - 1;
      ua_buffer(ji, jj) =
        ua(jiu, jj) + sqrt(g / hu(ji, jj)) * (sshn_u(ji, jj) - sshn_u(jiu, jj));
    } else {
      ua_buffer(ji, jj) = ua(ji, jj);
    }
  }

  // Flather boundary conditions for v
  if (ji > 0) {
    if (pt(ji, jj) + pt(ji, jj + 1) <= -1) {
      va_buffer(ji, jj) = va(ji, jj);
    } else if (pt(ji, jj) < 0) {
      const int jiv = jj + 1;
      va_buffer(ji, jj) =
        va(ji, jiv) + sqrt(g / hv(ji, jj)) * (sshn_v(ji, jj) - sshn_v(ji, jiv));
    } else if (pt(ji, jj + 1) < 0) {
      const int jiv = jj - 1;
      va_buffer(ji, jj) =
        va(ji, jiv) + sqrt(g / hv(ji, jj)) * (sshn_v(ji, jj) - sshn_v(ji, jiv));
    } else {
      va_buffer(ji, jj) = va(ji, jj);
    }
  }
}

__global__ void
k_next(const FortranArray2D<wp_t, 1, 1> sshn,
       const FortranArray2D<wp_t, 0, 1> sshn_u,
       const FortranArray2D<wp_t, 1, 0> sshn_v,

       const FortranArray2D<wp_t, 1, 1> ssha,

       const FortranArray2D<wp_t, 0, 1> un,
       const FortranArray2D<wp_t, 1, 0> vn,

       const FortranArray2D<wp_t, 0, 1> ua,
       const FortranArray2D<wp_t, 1, 0> va,

       const FortranArray2D<wp_t, 1, 1> e12t,
       const FortranArray2D<wp_t, 0, 1> e12u,
       const FortranArray2D<wp_t, 1, 0> e12v,

       const FortranArray2D<int, 0, 0> pt,

       const int jpi,
       const int jpj)
{
  int ji = threadIdx.x + blockIdx.x * blockDim.x;
  int jj = threadIdx.y + blockIdx.y * blockDim.y;

  if (ji > jpi || jj > jpj) {
    return;
  }

  if (jj > 0) {
    un(ji, jj) = ua(ji, jj);
  }

  if (ji > 0) {
    vn(ji, jj) = va(ji, jj);
  }

  if (ji > 0 && jj > 0) {
    sshn(ji, jj) = ssha(ji, jj);
  }

  if (jj > 0) {
    if (pt(ji, jj) + pt(ji + 1, jj) <= 0) {
      // This cell or it's +ve i-side neighbour is outside the computational
      // domain.
    } else if (pt(ji, jj) * pt(ji + 1, jj) > 0) {
      const wp_t rtmp1 =
        e12t(ji, jj) * sshn(ji, jj) + e12t(ji + 1, jj) * sshn(ji + 1, jj);
      sshn_u(ji, jj) = 0.5 * rtmp1 / e12u(ji, jj);
    } else if (pt(ji, jj) <= 0) {
      sshn_u(ji, jj) = sshn(ji + 1, jj);
    } else if (pt(ji + 1, jj) <= 0) {
      sshn_u(ji, jj) = sshn(ji, jj);
    }
  }

  if (ji > 0) {
    if (pt(ji, jj) + pt(ji, jj + 1) <= 0) {
      // This cell or it's +ve j-side neighbour is outside th computational
      // domain.
    } else if (pt(ji, jj) * pt(ji, jj + 1) > 0) {
      const wp_t rtmp1 =
        e12t(ji, jj) * sshn(ji, jj) + e12t(ji, jj + 1) * sshn(ji, jj + 1);
      sshn_v(ji, jj) = 0.5 * rtmp1 / e12v(ji, jj);
    } else if (pt(ji, jj) <= 0) {
      sshn_v(ji, jj) = sshn(ji, jj + 1);
    } else if (pt(ji, jj + 1) <= 0) {
      sshn_v(ji, jj) = sshn(ji, jj);
    }
  }
}
