#include <cassert>
#include <stdint.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "fortran_array_2d.cuh"

// Working precision
typedef double wp_t;

// FORTRAN SIGN FUNCTION ON DEVICE
__device__ inline wp_t SIGN( wp_t A, wp_t B )
{
  return A*((B > 0.) - (B < 0.));
}

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
k_initialise_grid(const FortranArray2D<wp_t, 1, 1>& sshn,
                  const FortranArray2D<wp_t, 0, 1>& sshn_u,
                  const FortranArray2D<wp_t, 1, 0>& sshn_v,

                  const FortranArray2D<wp_t, 1, 1>& e1t,
                  const FortranArray2D<wp_t, 1, 1>& e2t,

                  const FortranArray2D<wp_t, 0, 1>& e1u,
                  const FortranArray2D<wp_t, 0, 1>& e2u,

                  const FortranArray2D<wp_t, 0, 0>& e1f,
                  const FortranArray2D<wp_t, 0, 0>& e2f,

                  const FortranArray2D<wp_t, 1, 0>& e1v,
                  const FortranArray2D<wp_t, 1, 0>& e2v,

                  const FortranArray2D<wp_t, 1, 1>& e12t,
                  const FortranArray2D<wp_t, 0, 1>& e12u,
                  const FortranArray2D<wp_t, 1, 0>& e12v,

                  const FortranArray2D<wp_t, 0, 1>& gphiu,
                  const FortranArray2D<wp_t, 1, 0>& gphiv,
                  const FortranArray2D<wp_t, 0, 0>& gphif,

                  const FortranArray2D<wp_t, 1, 1>& xt,
                  const FortranArray2D<wp_t, 1, 1>& yt,

                  const FortranArray2D<wp_t, 1, 1>& ht,
                  const FortranArray2D<wp_t, 0, 1>& hu,
                  const FortranArray2D<wp_t, 1, 0>& hv,

                  const FortranArray2D<int, 0, 0>& pt,

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
k_continuity();

__global__ void
k_boundary_conditions();

__global__ void
k_momentum();

__global__ void kernel_momentum(  
                      const int jpj, 
                      const int jpi,
                      const wp_t visc,
                      const wp_t rdt,
                      const wp_t cbfr,
                      FortranArray2D<wp_t, 1, 1>& e1t, 
                      FortranArray2D<wp_t, 1, 1>& e2t, 
                      FortranArray2D<wp_t, 0, 1>& e1u, 
                      FortranArray2D<wp_t, 0, 1>& e2u, 
                      FortranArray2D<wp_t, 1, 0>& e1v, 
                      FortranArray2D<wp_t, 1, 0>& e2v, 
                      FortranArray2D<wp_t, 0, 1>& e12u, 
                      FortranArray2D<wp_t, 1, 0>& e12v, 
                      FortranArray2D<wp_t, 0, 1>& gphiu,
                      FortranArray2D<wp_t, 1, 0>& gphiv,
                      FortranArray2D<wp_t, 1, 1>& ht,
                      FortranArray2D<wp_t, 0, 1>& hu,
                      FortranArray2D<wp_t, 0, 1>& hv,
                      FortranArray2D<int, 0, 0>& pt, 
                      FortranArray2D<wp_t, 1, 1>& sshn,
                      FortranArray2D<wp_t, 0, 1>& sshn_u,
                      FortranArray2D<wp_t, 1, 0>& sshn_v,
                      FortranArray2D<wp_t, 0, 1>& ssha_u,
                      FortranArray2D<wp_t, 1, 0>& ssha_v,
                      FortranArray2D<wp_t, 0, 1>& un,
                      FortranArray2D<wp_t, 1, 0>& vn,
                      FortranArray2D<wp_t, 0, 1>& ua,
                      FortranArray2D<wp_t, 1, 0>& va
                    )
{
  int jj = 1 + threadIdx.x + blockIdx.x * blockDim.x;
  int ji = 1 + threadIdx.y + blockIdx.y * blockDim.y;

  wp_t u_e;
  wp_t u_w;
  wp_t v_s;
  wp_t v_n;
  wp_t v_sc;
  wp_t v_nc; 
  wp_t u_ec;
  wp_t u_wc;
  wp_t uu_e; 
  wp_t uu_w; 
  wp_t uu_s; 
  wp_t uu_n;
  wp_t vv_e;
  wp_t vv_w; 
  wp_t vv_s; 
  wp_t vv_n;
  wp_t depe;
  wp_t depw; 
  wp_t deps; 
  wp_t depn;
  wp_t dudx_e; 
  wp_t dudy_n; 
  wp_t dvdx_e; 
  wp_t dvdy_n;
  wp_t dudx_w; 
  wp_t dudy_s;
  wp_t dvdx_w; 
  wp_t dvdy_s;
  wp_t adv; 
  wp_t vis; 
  wp_t hpg;
  wp_t cor;
  static const wp_t pi = 3.1415926535897932;
  static const wp_t g = 9.80665;
  static const wp_t omega = 7.292116e-05;
  static const wp_t d2r = pi/180.;

  if ( (jj <= jpj) && (ji < jpi) )
  {
    if (pt(ji,jj) + pt(ji+1,jj) <= 0) {} // jump over non-computatinal domain
    else if (pt(ji,jj) <= 0 || pt(ji+1,jj) <= 0) {} //jump over boundary u
    else 
    {
      u_e  = 0.5 * (un(ji,jj) + un(ji+1,jj)) * e2t(ji+1,jj);      // add length scale.
      depe = ht(ji+1,jj) + sshn(ji+1,jj);
    
      u_w  = 0.5 * (un(ji,jj) + un(ji-1,jj)) * e2t(ji,jj);        // add length scale
      depw = ht(ji,jj) + sshn(ji,jj);
    
      v_sc = 0.5 * (vn(ji,jj-1) + vn(ji+1,jj-1));
      v_s  = 0.5 * v_sc * (e1v(ji,jj-1) + e1v(ji+1,jj-1));
      deps = 0.5 * (hv(ji,jj-1) + sshn_v(ji,jj-1) + hv(ji+1,jj-1) + sshn_v(ji+1,jj-1));
    
      v_nc = 0.5 * (vn(ji,jj) + vn(ji+1,jj));
      v_n  = 0.5 * v_nc * (e1v(ji,jj) + e1v(ji+1,jj));
      depn = 0.5 * (hv(ji,jj) + sshn_v(ji,jj) + hv(ji+1,jj) + sshn_v(ji+1,jj));
    
      // -advection (currently first order upwind)
      uu_w = (0.5 - SIGN(0.5, u_w)) * un(ji,jj) + (0.5 + SIGN(0.5, u_w)) * un(ji-1,jj); 
      uu_e = (0.5 + SIGN(0.5, u_e)) * un(ji,jj) + (0.5 - SIGN(0.5, u_e)) * un(ji+1,jj); 
    
      if (pt(ji,jj-1) <=0 || pt(ji+1,jj-1) <= 0)    
         uu_s = (0.5 - SIGN(0.5, v_s)) * un(ji,jj); 
      else
         uu_s = (0.5 - SIGN(0.5, v_s)) * un(ji,jj) + (0.5 + SIGN(0.5, v_s)) * un(ji,jj-1); 
    
      if (pt(ji,jj+1) <= 0 || pt(ji+1,jj+1) <= 0)   
         uu_n = (0.5 + SIGN(0.5, v_n)) * un(ji,jj);
      else
         uu_n = (0.5 + SIGN(0.5, v_n)) * un(ji,jj) + (0.5 - SIGN(0.5, v_n)) * un(ji,jj+1);
    
      adv = uu_w * u_w * depw - uu_e * u_e * depe + uu_s * v_s * deps - uu_n * v_n * depn;
          
      // -viscosity
    
      dudx_e = (un(ji+1,jj) - un(ji,  jj)) / e1t(ji+1,jj) * (ht(ji+1,jj) + sshn(ji+1,jj));
      dudx_w = (un(ji,  jj) - un(ji-1,jj)) / e1t(ji,  jj) * (ht(ji,  jj) + sshn(ji,  jj));
      if (pt(ji,jj-1) <=0 || pt(ji+1,jj-1) <= 0)    
        dudy_s = 0.0; //slip boundary
      else
        dudy_s = (un(ji,jj) - un(ji,jj-1)) / (e2u(ji,jj) + e2u(ji,jj-1)) 
               * (hu(ji,jj) + sshn_u(ji,jj) + hu(ji,jj-1) + sshn_u(ji,jj-1));
    
      if (pt(ji,jj+1) <= 0 || pt(ji+1,jj+1) <= 0)
        dudy_n = 0.0; // slip boundary
      else
        dudy_n = (un(ji,jj+1) - un(ji,jj)) / (e2u(ji,jj) + e2u(ji,jj+1)) 
               * (hu(ji,jj) + sshn_u(ji,jj) + hu(ji,jj+1) + sshn_u(ji,jj+1));
    
      vis = (dudx_e - dudx_w ) * e2u(ji,jj) + (dudy_n - dudy_s ) * e1u(ji,jj) * 0.5;
      vis = visc * vis ;  //visc will be an array visc(1:jpijglou) 
                                 //for variable viscosity, such as turbulent viscosity
    
              // -Coriolis' force (can be implemented implicitly)
      cor = 0.5 * (2. * omega * sin(gphiu(ji,jj) * d2r) * (v_sc + v_nc)) * e12u(ji,jj) * (hu(ji,jj) + sshn_u(ji,jj));
    
              // -pressure gradient
      hpg = -g * (hu(ji,jj) + sshn_u(ji,jj)) * e2u(ji,jj) * (sshn(ji+1,jj) - sshn(ji,jj));
              // -linear bottom friction (implemented implicitly.
      ua(ji,jj) = (un(ji,jj) * (hu(ji,jj) + sshn_u(ji,jj)) + rdt * (adv + vis + cor + hpg) / e12u(ji,jj)) 
                / (hu(ji,jj) + ssha_u(ji,jj)) / (1.0 + cbfr * rdt);
    }
  
  }

  __syncthreads();
  
  // v equation
  if ( (jj < jpj) && (ji <= jpi) )
  {
    if (pt(ji,jj) + pt(ji+1,jj) <= 0) {} //jump over non-computatinal domain
    else if (pt(ji,jj) <= 0 || pt(ji,jj+1) <= 0) {} //jump over v boundary cells
    else
    {
      v_n  = 0.5 * (vn(ji,jj) + vn(ji,jj+1)) * e1t(ji,jj+1);  //add length scale.
      depn = ht(ji,jj+1) + sshn(ji,jj+1);

      v_s  = 0.5 * (vn(ji,jj) + vn(ji,jj-1)) * e1t(ji,jj);    //add length scale
      deps = ht(ji,jj) + sshn(ji,jj);

      u_wc = 0.5 * (un(ji-1,jj) + un(ji-1,jj+1));
      u_w  = 0.5 * u_wc * (e2u(ji-1,jj) + e2u(ji-1,jj+1));
      depw = 0.50 * (hu(ji-1,jj) + sshn_u(ji-1,jj) + hu(ji-1,jj+1) + sshn_u(ji-1,jj+1));

      u_ec = 0.5 * (un(ji,jj) + un(ji,jj+1));
      u_e  = 0.5 * u_ec * (e2u(ji,jj) + e2u(ji,jj+1));
      depe = 0.50 * (hu(ji,jj) + sshn_u(ji,jj) + hu(ji,jj+1) + sshn_u(ji,jj+1));

      // -advection (currently first order upwind)
      vv_s = (0.5 - SIGN(0.5, v_s)) * vn(ji,jj) + (0.5 + SIGN(0.5, v_s)) * vn(ji,jj-1); 
      vv_n = (0.5 + SIGN(0.5, v_n)) * vn(ji,jj) + (0.5 - SIGN(0.5, v_n)) * vn(ji,jj+1); 

      if (pt(ji-1,jj) <= 0 || pt(ji-1,jj+1) <= 0)   
         vv_w = (0.5 - SIGN(0.5, u_w)) * vn(ji,jj); 
      else
         vv_w = (0.5 - SIGN(0.5, u_w)) * vn(ji,jj) + (0.5 + SIGN(0.5, u_w)) * vn(ji-1,jj); 

      if (pt(ji+1,jj) <= 0 || pt(ji+1,jj+1) <= 0)
         vv_e = (0.5 + SIGN(0.5, u_e)) * vn(ji,jj);
      else
         vv_e = (0.5 + SIGN(0.5, u_e)) * vn(ji,jj) + (0.5 - SIGN(0.5, u_e)) * vn(ji+1,jj);

      adv = vv_w * u_w * depw - vv_e * u_e * depe + vv_s * v_s * deps - vv_n * v_n * depn;

      dvdy_n = (vn(ji,jj+1) - vn(ji,  jj)) / e2t(ji,jj+1) * (ht(ji,jj+1) + sshn(ji,jj+1));
      dvdy_s = (vn(ji,  jj) - vn(ji,jj-1)) / e2t(ji,  jj) * (ht(ji,  jj) + sshn(ji,  jj));

      if (pt(ji-1,jj) <= 0 || pt(ji-1,jj+1) <= 0)
        dvdx_w = 0.0;
      else
        dvdx_w = (vn(ji,jj) - vn(ji-1,jj)) / (e1v(ji,jj) + e1v(ji-1,jj)) 
               * (hv(ji,jj) + sshn_v(ji,jj) + hv(ji-1,jj) + sshn_v(ji-1,jj));

      if (pt(ji+1,jj) <= 0 || pt(ji+1,jj+1) <= 0)
        dvdx_e = 0.0;
      else
        dvdx_e = (vn(ji+1,jj) - vn(ji,jj)) / (e1v(ji,jj) + e1v(ji+1,jj)) 
               * (hv(ji,jj) + sshn_v(ji,jj) + hv(ji+1,jj) + sshn_v(ji+1,jj));

      vis = (dvdy_n - dvdy_s ) * e1v(ji,jj) + (dvdx_e - dvdx_w ) * e2v(ji,jj) * 0.5;  

      vis = visc * vis; 
      cor = -0.5 * (2. * omega * sin(gphiv(ji,jj) * d2r) * (u_ec + u_wc)) 
          * e12v(ji,jj) * (hv(ji,jj) + sshn_v(ji,jj));
      hpg = -g * (hv(ji,jj) + sshn_v(ji,jj)) * e1v(ji,jj) * (sshn(ji,jj+1) - sshn(ji,jj));
      va(ji,jj) = (vn(ji,jj) * (hv(ji,jj) + sshn_v(ji,jj)) + rdt * (adv + vis + cor + hpg) / e12v(ji,jj) ) 
                / ((hv(ji,jj) + ssha_v(ji,jj))) / (1.0 + cbfr * rdt); 
        
    }
  }
}

__global__ void
k_next();

void
finalise();

FortranArray2D<wp_t, 0, 1>* global_array = nullptr;

extern "C"
{
  void cuda_setup_model_params_(
		                int jpi,
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

  void cuda_finalise_();
};

GridConstants grid_constants;
SimulationVariables simulation_vars;

ModelParameters model_params;

void
cuda_initialise_grid_()
{
  const int jpi = model_params.jpi;
  const int jpj = model_params.jpj;

  if (jpi == 0 || jpj == 0) {
    fprintf(stderr,
            "Invalid grid size: (%d, %d); have you setup model params?",
            jpi,
            jpj);
  }

  printf(
    "[CUDA](Host) Initialising grid constants and simluation variables.\n");

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

  // Initialise simulation parameters
  k_initialise_grid<<<jpi + 2, jpj + 2>>>(*simulation_vars.sshn,
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

  cudaDeviceSynchronize();
}

__global__ void
k_initialise_grid(const FortranArray2D<wp_t, 1, 1>& sshn,
                  const FortranArray2D<wp_t, 0, 1>& sshn_u,
                  const FortranArray2D<wp_t, 1, 0>& sshn_v,

                  const FortranArray2D<wp_t, 1, 1>& e1t,
                  const FortranArray2D<wp_t, 1, 1>& e2t,

                  const FortranArray2D<wp_t, 0, 1>& e1u,
                  const FortranArray2D<wp_t, 0, 1>& e2u,

                  const FortranArray2D<wp_t, 0, 0>& e1f,
                  const FortranArray2D<wp_t, 0, 0>& e2f,

                  const FortranArray2D<wp_t, 1, 0>& e1v,
                  const FortranArray2D<wp_t, 1, 0>& e2v,

                  const FortranArray2D<wp_t, 1, 1>& e12t,
                  const FortranArray2D<wp_t, 0, 1>& e12u,
                  const FortranArray2D<wp_t, 1, 0>& e12v,

                  const FortranArray2D<wp_t, 0, 1>& gphiu,
                  const FortranArray2D<wp_t, 1, 0>& gphiv,
                  const FortranArray2D<wp_t, 0, 0>& gphif,

                  const FortranArray2D<wp_t, 1, 1>& xt,
                  const FortranArray2D<wp_t, 1, 1>& yt,

                  const FortranArray2D<wp_t, 1, 1>& ht,
                  const FortranArray2D<wp_t, 0, 1>& hu,
                  const FortranArray2D<wp_t, 1, 0>& hv,

                  const FortranArray2D<int, 0, 0>& pt,

                  const int jpi,
                  const int jpj,

                  const wp_t dx,
                  const wp_t dy,

                  const wp_t dep_const)
{
  int ji = threadIdx.x * blockIdx.x + blockDim.x;
  int jj = threadIdx.y * blockIdx.y + blockDim.y;

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
cuda_finalise_()
{
  // Clean up grid constants arrays.
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

  // Clean up simulation params arrays.
  delete simulation_vars.sshn;
  delete simulation_vars.sshn_u;
  delete simulation_vars.sshn_v;

  delete simulation_vars.ssha;
  delete simulation_vars.ssha_u;
  delete simulation_vars.ssha_v;

  delete simulation_vars.un;
  delete simulation_vars.vn;

  delete simulation_vars.ua;
  delete simulation_vars.va;

  cudaError_t cudaStatus = cudaDeviceReset();
  assert(cudaStatus == cudaSuccess);
}
