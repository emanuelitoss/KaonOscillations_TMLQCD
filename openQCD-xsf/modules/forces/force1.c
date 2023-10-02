
/*******************************************************************************
*
* File force1.c
*
* Copyright (C) 2011, 2012, 2013, 2014 Stefan Schaefer, Martin Luescher, 
*                                      John Bulava, Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Twisted mass pseudo-fermion action and force
*
* The externally accessible functions are
*
*   double setpf1(double mu,int ipf,int icom)
*     Generates a pseudo-fermion field phi with probability proportional
*     to exp(-Spf) and returns the action Spf (see the notes).
*
*   void force1(double mu,int ipf,int isp,int icr,double c,int *status)
*     Computes the force deriving from the action Spf (see the notes).
*     The calculated force is multiplied by c and added to the molecular-
*     dynamics force field.
*
*   double action1(double mu,int ipf,int isp,int icom,int *status)
*     Returns the action Spf (see the notes).
*
* Notes:
*
* The pseudo-fermion action Spf is given by
*
*   Spf=(phi,[(Q^dag-imu)(Q+imu)]^(-1)*phi),
*
* with Q=\gamma5*Dw, where Dw denotes the (improved) Wilson-Dirac operator
* and phi the pseudo-fermion field.
*
* The common parameters of the programs in this module are:
*
*   mu            Twisted mass parameter in Spf.
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
* The supported solver is CGNE. The number of status variable is given by:
*
*                  CGNE         
*   force1()         1          
*   action1()        1          
*
* The bare quark mass m0 is the one last set by sw_parms()[flags/parms.c] 
* and it is taken for granted that the solver parameters have been set by
* set_solver_parms() [flags/sparms.c].
*
* The program force1() attempts to propagate the solutions of the Dirac
* equation along the molecular-dynamics trajectories, using the field
* stack number icr (no fields are propagated if icr=0). If this feature
* is used, the program setup_chrono() [update/chrono.c] must be called
* before force1() is called for the first time.
*
* The required workspaces of double-precision spinor fields is
*
*                  CGNE      
*   setpf1()         1       
*   force1()     2+(icr>0)   
*   action1()        1       
*
* (these figures do not include the workspace required by the solver).
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define FORCE1_C

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
#include "update.h"
#include "forces.h"
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


double setpf1(double mu,int ipf,int icom)
{
   double act;
   spinor_dble **wsd,*phi;
   mdflds_t *mdfs;
   tm_parms_t tm;

   erase_all();

   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   wsd=reserve_wsd(1);
   random_sd(VOLUME,wsd[0],1.0);
   act=norm_square_dble(VOLUME,icom,wsd[0]);

   sw_term_xsf(NO_PTS);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   set_xsf_parms(-1);
   Dw_xsf_dble(-mu,wsd[0],phi);
   mulg5_dble(VOLUME,phi);
   release_wsd();

   return act;
}


void force1(double mu,int ipf,int isp,int icr,double c,int *status)
{
   double res0,res1;
   spinor_dble *phi,*chi,*psi,**wsd;
   spinor_dble *rho,**rsd;
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

   if (sp.solver==CGNE)
   {
      if (get_chrono(icr,chi))
      {
         rsd=reserve_wsd(1);
         rho=rsd[0];

         set_xsf_parms(1);
         Dw_xsf_dble(mu,chi,psi);
         mulg5_dble(VOLUME,psi);
         set_xsf_parms(-1);      
         Dw_xsf_dble(-mu,psi,rho);
         mulg5_dble(VOLUME,rho);
         mulr_spinor_add_dble(VOLUME,rho,phi,-1.0);

         res0=norm_square_dble(VOLUME,1,phi);
         res1=norm_square_dble(VOLUME,1,rho);
         res1=sqrt(res1/res0);

         if (res1<1.0)
         {
            if (res1>sp.res)
            {
               tmcg_xsf(sp.nmx,sp.res/res1,tau3,mu,rho,psi,status);
               mulr_spinor_add_dble(VOLUME,chi,psi,-1.0);
            }
            else
               status[0]=0;
         }
         else
            tmcg_xsf(sp.nmx,sp.res,tau3,mu,phi,chi,status);   
         
         release_wsd();
      }
      else
         tmcg_xsf(sp.nmx,sp.res,tau3,mu,phi,chi,status);

      error_root(status[0]<0,1,"force1 [force1.c]",
                 "CGNE solver failed (mu = %.4e, parameter set no %d, "
                 "status = %d)",mu,isp,status[0]);
      if (icr)
         add_chrono(icr,chi);
      set_xsf_parms(1);
      Dw_xsf_dble(mu,chi,psi);
      mulg5_dble(VOLUME,psi);
   }
   else
      error_root(1,1,"force1 [force1.c]","Unknown solver only CG is supported");

   set_xv2zero();
   add_prod2xv_xsf(1.0,chi,psi);   
   hop_frc(c);

   set_xt2zero();
   add_prod2xt_xsf(1.0,chi,psi);
   sw_frc(c);

   release_wsd();
}


double action1(double mu,int ipf,int isp,int icom,int *status)
{
   double act;
   spinor_dble *phi,*psi,**wsd,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   tm_parms_t tm;

   erase_all();

   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);
   
   mdfs=mdflds();
   sp=solver_parms(isp);

   wsd=reserve_wsd(1);   
   psi=wsd[0];
   phi=(*mdfs).pf[ipf];

   if (sp.solver==CGNE)
   {
      tmcg_xsf(sp.nmx,sp.res,tau3,mu,phi,psi,status);

      error_root(status[0]<0,1,"action1 [force1.c]",
                 "CGNE solver failed (mu = %.4e, parameter set no %d, "
                 "status = %d)",mu,isp,status[0]);

      rsd=reserve_wsd(1);
      set_xsf_parms(1);
      Dw_xsf_dble(mu,psi,rsd[0]);
      act=norm_square_dble(VOLUME,icom,rsd[0]);
      release_wsd();
   }
   else
   {
      error_root(1,1,"action1 [force1.c]","Unknown solver only CG is supported");
      act=0.0;
   }

   release_wsd();

   return act;
}
