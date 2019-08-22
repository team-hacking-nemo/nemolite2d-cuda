#include <stdio.h>
#include <math.h>
#define real float
// These macros are some of the worst ideas I've had...
#define e1t(i,j) e1t_[ (i-1) + (j-1)*(jpi) ]
#define e2t(i,j) e2t_[ (i-1) + (j-1)*(jpi) ]
#define e1u(i,j) e1u_[ (i) + (j-1)*(jpi+1) ] 
#define e2u(i,j) e2u_[ (i) + (j-1)*(jpi+1) ] 
#define e1f(i,j) e1f_[ (i) + (j)*(jpi+1)   ]
#define e2f(i,j) e2f_[ (i) + (j)*(jpi+1)   ]
#define e1v(i,j) e1v_[ (i-1) + (j)*(jpi)   ]
#define e2v(i,j) e2v_[ (i-1) + (j)*(jpi)   ]
#define e12t(i,j) e12t_[ (i-1) + (j-1)*(jpi) ]
#define e12u(i,j) e12u_[ (i) + (j-1)*(jpi+1) ]
#define e12v(i,j) e12v_[ (i-1) + (j)*(jpi) ]
#define gphiu(i,j) gphiu_[ (i) + (j-1)*(jpi+1) ]
#define gphiv(i,j) gphiv_[ (i-1) + (j)*(jpi) ]
#define gphif(i,j) gphif_[ (i) + (j)*(jpi+1) ] 
#define xt(i,j) xt_[ (i-1) + (j-1)*(jpi) ]
#define yt(i,j) yt_[ (i-1) + (j-1)*(jpi) ]
#define ht(i,j) ht_[ (i-1) + (j-1)*(jpi) ]
#define hu(i,j) hu_[ (i) + (j-1)*(jpi+1) ]
#define hv(i,j) hv_[ (i-1) + (j)*(jpi)   ]
#define hf(i,j) hf_[ (i) + (j)*(jpi+1)   ]
#define sshb(i,j) sshb_[ (i-1) + (j-1)*(jpi) ]
#define sshn(i,j) sshn_[ (i-1) + (j-1)*(jpi) ]
#define ssha(i,j) ssha_[ (i-1) + (j-1)*(jpi) ]
#define sshb_u(i,j) sshb_u_[ (i) + (j-1)*(jpi+1) ]
#define sshn_u(i,j) sshn_u_[ (i) + (j-1)*(jpi+1) ]
#define ssha_u(i,j) ssha_u_[ (i) + (j-1)*(jpi+1) ]
#define sshb_v(i,j) sshb_v_[ (i-1) + (j)*(jpi) ]
#define sshn_v(i,j) sshn_v_[ (i-1) + (j)*(jpi) ]
#define ssha_v(i,j) ssha_v_[ (i-1) + (j)*(jpi) ]
#define un(i,j) un_[ (i) + (j-1)*(jpi+1) ]
#define ua(i,j) ua_[ (i) + (j-1)*(jpi+1) ]
#define vn(i,j) vn_[ (i-1) + (j)*(jpi) ]
#define va(i,j) va_[ (i-1) + (j)*(jpi) ]
#define pt(i,j) pt_[ (i) + (j)*(jpi+2) ]

__device__ inline real SIGN( real A, real B )
{
  return A*((B > 0.) - (B < 0.));
}

void kernel_momentum( real &pi, real &g, real &omega, real &d2r,  
                int *pt_,
                real *e1t_, real *e2t_, real *e1u_, real *e2u_,
                real *e1f_, real *e2f_, real *e1v_, real *e2v_, 
                real *e12t_, real *e12u_, real *e12v_,
                real *gphiu_, real *gphiv_, real *gphif_,
                real *xt_, real *yt_,
                real *ht_, real *hu_, real *hv_, real *hf_,
                real *sshb_, real *sshb_u_, real *sshb_v_,
                real *sshn_, real *sshn_u_, real *sshn_v_,
                real *ssha_, real *ssha_u_, real *ssha_v_,
                real *un_,  real *vn_, real *ua_,  real *va_,
                int &jpiglo, int &jpjglo, int &jpi, int &jpj,
                int &jphgr_msh,
                int &nit000, int &nitend, int &irecord,         
                real &dx, real &dy, real &dep_const,               
                real &rdt,                             
                real &cbfr,                            
                real &visc,                            
                int &istp,                            
                int &ji, int &jj,                   
                int &itmp1, int &itmp2,                   
                real &rtmp1, real& rtmp2, real &rtmp3, real &rtmp4,      
                int &idxt )
{
 unsigned int jj = threadIdx.x + blockIdx.x * blockDim.x;
 unsigned int ji = threadIdx.y + blockIdx.y * blockDim.y;

  real u_e, u_w;
  real v_s, v_n;
  real v_sc, v_nc, u_ec, u_wc;
  real uu_e, uu_w, uu_s, uu_n;
  real vv_e, vv_w, vv_s, vv_n;
  real depe, depw, deps, depn;
  real dudx_e, dudy_n, dvdx_e, dvdy_n;
  real dudx_w, dudy_s, dvdx_w, dvdy_s;

  real adv, vis, hpg, cor;

  printf("START C\n");

  for ( jj = 1; jj <= jpj; ++jj )
  {
    for ( ji = 1; ji < jpi; ++ji )
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
  }

  // OUTSIDE IF
  // __syncThreads
  
  // v equation
  for ( jj = 1; jj < jpj; ++jj )
  {
    for ( ji = 1; ji <= jpi; ++ji )
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
  printf("END C\n");
}

void callsteps( real &pi, real &g, real &omega, real &d2r,  
                int *pt_,
                real *e1t_, real *e2t_, real *e1u_, real *e2u_,
                real *e1f_, real *e2f_, real *e1v_, real *e2v_, 
                real *e12t_, real *e12u_, real *e12v_,
                real *gphiu_, real *gphiv_, real *gphif_,
                real *xt_, real *yt_,
                real *ht_, real *hu_, real *hv_, real *hf_,
                real *sshb_, real *sshb_u_, real *sshb_v_,
                real *sshn_, real *sshn_u_, real *sshn_v_,
                real *ssha_, real *ssha_u_, real *ssha_v_,
                real *un_,  real *vn_, real *ua_,  real *va_,
                int &jpiglo, int &jpjglo, int &jpi, int &jpj,
                int &jphgr_msh,
                int &nit000, int &nitend, int &irecord,         
                real &dx, real &dy, real &dep_const,               
                real &rdt,                             
                real &cbfr,                            
                real &visc,                            
                int &istp,                            
                int &ji, int &jj,                   
                int &itmp1, int &itmp2,                   
                real &rtmp1, real& rtmp2, real &rtmp3, real &rtmp4,      
                int &idxt )
{
}
