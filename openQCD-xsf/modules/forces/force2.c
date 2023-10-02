
/*******************************************************************************
*
* File force2.c
*
* Copyright (C) 2011, 2012, 2013, 2014 Stefan Schaefer, Martin Luescher
*                                      Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Hasenbusch twisted_mass pseudo-fermion action and force
*
* The externally accessible functions are
*
*   double setpf2(double mu0,double mu1,int ipf,int isp,int icom,
*                 int *status)
*     Generates a pseudo-fermion field phi with probability proportional
*     to exp(-Spf) and returns the action Spf (see the notes).
*
*   void force2(double mu0,int mu1,int ipf,int isp,int icr,double c,
*               int *status)
*     Computes the force deriving from the action Spf (see the notes).
*     The calculated force is multiplied by c and added to the molecular-
*     dynamics force field.
*
*   double action2(double mu0,double mu1,int ipf,int isp,int icom,
*                  int *status)
*     Returns the action Spf (see the notes).
*
* Notes:
*
* The pseudo-fermion action Spf is given by
*
*   Spf=(phi,(Q+imu1)[(Q^dag-imu0)(Q+imu0)]^(-1)(Q^dag-imu1)*phi)
*
* with Q=\gamma5*Dw, where Dw denotes the (improved) Wilson-Dirac operator
* and phi the pseudo-fermion field.
*
* The common parameters of the programs in this module are:
*
*   mu0,mu1       Twisted mass parameters in Spf.
*
*   ipf           Index of the pseudo-fermion field phi in the
*                 structure returned by mdflds() [mdflds.c].
*
*   isp           Index of the solver parameter set that describes
*                 the solver to be used for the solution of the
*                 Dirac equation.
*
*   icom          The action returned by the programs setpf3() and
*                 action3() is summed over all MPI processes if icom=1.
*                 Otherwise the local part of the action is returned.
*
*   status        Status values returned by the solver used for the
*                 solution of the Dirac equation.
*
* The supported solver is the CGNE. The number of status variables is given by:
*
*                  CGNE         
*   setpf2()         1          
*   force2()         1          
*   action2()        1          
*
* The solver used in the case of setpf2() is for the Dirac equation with
* twisted mass mu1, while force2() and action2() use the solver for the
* equation with twisted mass mu0. Different solver parameters may be needed
* in the two cases if mu1>>mu0, for example.
*
* The bare quark mass m0 is the one last set by sw_parms() [flags/parms.c] 
* and it is taken for granted that the solver parameters have been set by 
* set_solver_parms() [flags/sparms.c].
*
* The program force2() attempts to propagate the solutions of the Dirac
* equation along the molecular-dynamics trajectories, using the field
* stack number icr (no fields are propagated if icr=0). If this feature
* is used, the program setup_chrono() [update/chrono.c] must be called
* before force2() is called for the first time.
*
* The required workspaces of double-precision spinor fields are
*
*                  CGNE         
*   setpf2()         2          
*   force2()     2+(1+icr>0)     
*   action2()        2          
*
* (these figures do not include the workspace required by the solver).
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define FORCE2_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "mdflds.h"
#include "sw_term.h"
#include "sflds.h"
#include "dirac.h"
#include "linalg.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "update.h"
#include "global.h"

static const int tau3=1;


static void erase_all(void)
{
   set_flags(ERASED_SW);
   set_flags(ERASED_SWD);
   set_grid_flags(SAP_BLOCKS,ERASED_SW);
   set_flags(ERASED_AW);
   set_flags(ERASED_AWHAT);
}


double setpf2(double mu0,double mu1,int ipf,int isp,int icom,int *status)
{
   double act;
   spinor_dble *phi,*chi,*psi;
   spinor_dble **wsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   tm_parms_t tm;

   erase_all();

   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   wsd=reserve_wsd(2);
   chi=wsd[0];
   psi=wsd[1];
   random_sd(VOLUME,chi,1.0);
   act=norm_square_dble(VOLUME,icom,chi);
   
   sw_term_xsf(NO_PTS);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];

   set_xsf_parms(-1);
   Dw_xsf_dble(-mu0,chi,psi);
   mulg5_dble(VOLUME,psi);

   sp=solver_parms(isp);
   
   if (sp.solver==CGNE)
   {
      tmcg_xsf(sp.nmx,sp.res,tau3,mu1,psi,chi,status);

      error_root(status[0]<0,1,"setpf2 [force2.c]","CGNE solver failed "
                 "(mu = %.4e, parameter set no %d, status = %d)",
                 mu1,isp,status[0]);

      set_xsf_parms(1);
      Dw_xsf_dble(mu1,chi,phi);
      mulg5_dble(VOLUME,phi);
   }
   else
      error_root(1,1,"setpf2 [force2.c]","Unknown solver only CG is supported");

   release_wsd();

   return act;
}

/*

   CHECK HERE: The time-interpolated solution might not
   work too well. Indeed the stored previous solution chi
   in fact is not simply D\dagD\phi but D\dagD\psi where 
   \psi depends on the previous gauge field.

*/


void force2(double mu0,double mu1,int ipf,int isp,int icr,
            double c,int *status)
{
   double res0,res1; 
   spinor_dble *phi,*chi,*psi,**wsd;
   spinor_dble *rho,*eta,**rsd; 
   mdflds_t *mdfs;
   solver_parms_t sp;
   tm_parms_t tm;

   erase_all();

   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);
   
   mdfs=mdflds();
   sp=solver_parms(isp);
   sw_term_xsf(NO_PTS);

   wsd=reserve_wsd(2);
   phi=(*mdfs).pf[ipf];   
   psi=wsd[0];
   chi=wsd[1];

   set_xsf_parms(-1);
   Dw_xsf_dble(-mu1,phi,psi);
   mulg5_dble(VOLUME,psi);

   if (sp.solver==CGNE)
   {
      if (get_chrono(icr,chi))
      {
         rsd=reserve_wsd(2);
         rho=rsd[0];
         eta=rsd[1];

         set_xsf_parms(1); 
         Dw_xsf_dble(mu0,chi,eta);
         mulg5_dble(VOLUME,eta);
         set_xsf_parms(-1);
         Dw_xsf_dble(-mu0,eta,rho);
         mulg5_dble(VOLUME,rho);
         mulr_spinor_add_dble(VOLUME,rho,psi,-1.0);

         res0=norm_square_dble(VOLUME,1,psi);
         res1=norm_square_dble(VOLUME,1,rho);
         res1=sqrt(res1/res0);

         if (res1<1.0)
         {
            if (res1>sp.res)
            {
               tmcg_xsf(sp.nmx,sp.res/res1,tau3,mu0,rho,eta,status);
               mulr_spinor_add_dble(VOLUME,chi,eta,-1.0);
            }
            else
               status[0]=0;
         }
         else
            tmcg_xsf(sp.nmx,sp.res,tau3,mu0,psi,chi,status);   
         
         release_wsd();
      }
      else
         tmcg_xsf(sp.nmx,sp.res,tau3,mu0,psi,chi,status);

      error_root(status[0]<0,1,"force2 [force2.c]",
                 "CGNE solver failed (mu = %.4e, parameter set no %d, "
                 "status = %d)",mu0,isp,status[0]);

      if (icr)
         add_chrono(icr,chi);

      set_xsf_parms(1);
      Dw_xsf_dble(mu0,chi,psi);
      mulg5_dble(VOLUME,psi);
      mulr_spinor_add_dble(VOLUME,psi,phi,-1.0);
   }
   else
      error_root(1,1,"force2 [force2.c]","Unknown solver only CG is supported");

   set_xv2zero();
   add_prod2xv_xsf(1.0,chi,psi);   
   hop_frc(c);

   set_xt2zero();
   add_prod2xt_xsf(1.0,chi,psi);
   sw_frc(c);

   release_wsd();
}


double action2(double mu0,double mu1,int ipf,int isp,int icom,int *status)
{
   double act;
   spinor_dble *phi,*psi,*chi;
   spinor_dble **wsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   tm_parms_t tm;
   
   erase_all();

   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   mdfs=mdflds();
   sp=solver_parms(isp);

   sw_term_xsf(NO_PTS);

   wsd=reserve_wsd(2);   
   psi=wsd[0];
   chi=wsd[1];
   phi=(*mdfs).pf[ipf];

   set_xsf_parms(-1);
   Dw_xsf_dble(-mu1,phi,psi);
   mulg5_dble(VOLUME,psi);

   if (sp.solver==CGNE)
   {
      tmcg_xsf(sp.nmx,sp.res,tau3,mu0,psi,chi,status);

      error_root(status[0]<0,1,"action2 [force2.c]","CGNE solver failed "
                 "(mu = %.4e, parameter set no %d, status = %d)",
                 mu0,isp,status[0]);

      set_xsf_parms(1);
      Dw_xsf_dble(mu0,chi,psi);
   }
   else
      error_root(1,1,"action2 [force2.c]","Unknown solver only CG is supported");

   act=norm_square_dble(VOLUME,icom,psi);

   release_wsd();

   return act;
}
