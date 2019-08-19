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
#define gphiu(i,j) gphiu_[ (i) * (j-1)*(jpi+1) ]
__global__ void cuda_kernel_u
{
             if (pt(ji,jj) + pt(ji+1,jj) <= 0) {}; // jump over non-computatinal domain
             else if (pt(ji,jj) <= 0 || pt(ji+1,jj) <= 0) {}; //jump over boundary u
             else 
             {
               u_e  = 0.5 * (un(ji,jj) + un(ji+1,jj)) * e2t(ji+1,jj)      !add length scale.
               depe = ht(ji+1,jj) + sshn(ji+1,jj)

               u_w  = 0.5 * (un(ji,jj) + un(ji-1,jj)) * e2t(ji,jj)        !add length scale
               depw = ht(ji,jj) + sshn(ji,jj)

               v_sc = 0.5_wp * (vn(ji,jj-1) + vn(ji+1,jj-1))
               v_s  = 0.5_wp * v_sc * (e1v(ji,jj-1) + e1v(ji+1,jj-1))
               deps = 0.5_wp * (hv(ji,jj-1) + sshn_v(ji,jj-1) + hv(ji+1,jj-1) + sshn_v(ji+1,jj-1))

               v_nc = 0.5_wp * (vn(ji,jj) + vn(ji+1,jj))
               v_n  = 0.5_wp * v_nc * (e1v(ji,jj) + e1v(ji+1,jj))
               depn = 0.5_wp * (hv(ji,jj) + sshn_v(ji,jj) + hv(ji+1,jj) + sshn_v(ji+1,jj))

               // -advection (currently first order upwind)
               uu_w = (0.5_wp - SIGN(0.5_wp, u_w)) * un(ji,jj)              + & 
                    & (0.5_wp + SIGN(0.5_wp, u_w)) * un(ji-1,jj) 
               uu_e = (0.5_wp + SIGN(0.5_wp, u_e)) * un(ji,jj)              + & 
                    & (0.5_wp - SIGN(0.5_wp, u_e)) * un(ji+1,jj) 

               IF(pt(ji,jj-1) <=0 .OR. pt(ji+1,jj-1) <= 0) THEN   
                  uu_s = (0.5_wp - SIGN(0.5_wp, v_s)) * un(ji,jj)   
               ELSE
                  uu_s = (0.5_wp - SIGN(0.5_wp, v_s)) * un(ji,jj)              + & 
                       & (0.5_wp + SIGN(0.5_wp, v_s)) * un(ji,jj-1) 
               END If

               IF(pt(ji,jj+1) <=0 .OR. pt(ji+1,jj+1) <= 0) THEN   
                  uu_n = (0.5_wp + SIGN(0.5_wp, v_n)) * un(ji,jj)
               ELSE
                  uu_n = (0.5_wp + SIGN(0.5_wp, v_n)) * un(ji,jj)              + & 
                       & (0.5_wp - SIGN(0.5_wp, v_n)) * un(ji,jj+1)
               END IF

               adv = uu_w * u_w * depw - uu_e * u_e * depe + uu_s * v_s * deps - uu_n * v_n * depn
!end kernel u adv 

            ! -viscosity

!kernel  u vis 
            dudx_e = (un(ji+1,jj) - un(ji,  jj)) / e1t(ji+1,jj) * (ht(ji+1,jj) + sshn(ji+1,jj))
            dudx_w = (un(ji,  jj) - un(ji-1,jj)) / e1t(ji,  jj) * (ht(ji,  jj) + sshn(ji,  jj))
            IF(pt(ji,jj-1) <=0 .OR. pt(ji+1,jj-1) <= 0) THEN   
              dudy_s = 0.0_wp !slip boundary
            ELSE
              dudy_s = (un(ji,jj) - un(ji,jj-1)) / (e2u(ji,jj) + e2u(ji,jj-1)) * &
                     & (hu(ji,jj) + sshn_u(ji,jj) + hu(ji,jj-1) + sshn_u(ji,jj-1))
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
                real *ssha_, real *ssha_u_, real *sha_v_,
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
