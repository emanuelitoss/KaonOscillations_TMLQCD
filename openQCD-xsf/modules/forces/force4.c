
/*******************************************************************************
*
* File force4.c
*
* Copyright (C) 2012, 2013, 2014 Martin Luescher, Stefan Schaefer, 
                                 Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Twisted mass pseudo-fermion action and force with even-odd preconditioning
*
* The externally accessible functions are
*
*   double setpf4(double mu,int ipf,int isw,int icom)
*     Generates a pseudo-fermion field phi with probability proportional
*     to exp(-Spf) and returns the action Spf+Sdet if isw=1 or Spf if
*     isw!=1 (see the notes).
*
*   void force4(double mu,int ipf,int isw,int isp,int icr,double c,
*               int *status)
*     Computes the force deriving from the action Spf+Sdet if isw=1 or
*     Spf if isw!=1 (see the notes). The calculated force is multiplied 
*     by c and added to the molecular-dynamics force field.
*
*   double action4(double mu,int ipf,int isw,int isp,int icom,
*                  int *status)
*     Returns the action Spf+Sdet if isw=1 or Spf if isw!=1 (see the
*     notes).
*
* Notes:
*
* The pseudo-fermion action Spf is given by
*
*   Spf=(phi,[(Qhat^dag-imu)(Qhat+imu)]^(-1)*phi),
*
* with Qhat=\gamma5*Dwhat, where Dwhat denotes the even-odd preconditioned
* (improved) Wilson-Dirac operator and phi the pseudo-fermion field. The 
* latter vanishes on the odd lattice sites.
*
* The inclusion of the "small quark determinant" amounts to adding the
* action
*
*   Sdet=-ln{det(1e+Doo^dag*Doo)}+constant
*
* to the molecular-dynamics Hamilton function, where 1e is the projector
* to the quark fields that vanish on the odd lattice sites and Doo the
* odd-odd component of the Dirac operator (the constant is adjusted so
* as to reduce the significance losses when the action differences are
* computed at the end of the molecular-dynamics trajectories).
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
*   icom          The action returned by the programs setpf4() and
*                 action4() is summed over all MPI processes if icom=1.
*                 Otherwise the local part of the action is returned.
*
*   status        Status values returned by the solver used for the
*                 solution of the Dirac equation.
*
* The supported solvers is CGNE. The number of status variables is given by:
*
*                  CGNE        
*   force4()         1         
*   action4()        1         
*
* The bare quark mass m0 is the one last set by sw_parms() [flags/parms.c] 
* and it is taken for granted that the solver parameters have been set by 
* set_solver_parms() [flags/sparms.c].
*
* The program force4() attempts to propagate the solutions of the Dirac
* equation along the molecular-dynamics trajectories, using the field
* stack number icr (no fields are propagated if icr=0). If this feature
* is used, the program setup_chrono() [update/chrono.c] must be called
* before force4() is called for the first time.
*
* The required workspaces of double-precision spinor fields are
*
*                  CGNE      
*   setpf4()         1       
*   force4()     2+(icr>0)   
*   action4()        1       
*
* (these figures do not include the workspace required by the solver).
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define FORCE4_C

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

#define MAX_LEVELS 8
#define BLK_LENGTH 8

static int cnt[MAX_LEVELS];
static double smx[MAX_LEVELS];
static const int tau3=1;


static void erase_all(void)
{
   set_flags(ERASED_SW);
   set_flags(ERASED_SWD);
   set_grid_flags(SAP_BLOCKS,ERASED_SW);
   set_flags(ERASED_AW);
   set_flags(ERASED_AWHAT);
}


/*
   CHECK HERE:
   Do we have contributions to the actions coming 
   from the boundaries?
   For the action we do not really care, since 
   they are constant. They might enther though in 
   the calculation of the forces.
*/


static double sdet(void)
{
   int ix,iy,t,n,ie;
   double c,p;
   complex_dble z;
   pauli_dble *m;
   sw_parms_t swp;

   swp=sw_parms();

   if ((4.0+swp.m0)>1.0)
      c=pow(4.0+swp.m0,-6.0);
   else
      c=1.0;

   for (n=0;n<MAX_LEVELS;n++)
   {
      cnt[n]=0;
      smx[n]=0.0;
   }

   sw_term_xsf(NO_PTS);   
   m=swdfld()+VOLUME;
   ix=(VOLUME/2);
   ie=0;
   
   while (ix<VOLUME)
   {
      p=1.0;
      iy=ix+BLK_LENGTH;
      if (iy>VOLUME)
         iy=VOLUME;

      for (;ix<iy;ix++)
      {
         t=global_time(ix);

         if ((t>0)&&(t<(NPROC0*L0-1)))
         {
            z=det_pauli_dble(0.0,m);

            if (z.re>0.0)
               p*=(c*z.re);
            else
               ie=1;
            
            z=det_pauli_dble(0.0,m+1);

            if (z.re>0.0)
               p*=(c*z.re);
            else
               ie=1;
         }

         m+=2;
      }

      if (p!=0.0)
      {
         cnt[0]+=1;
         smx[0]-=2.0*log(p);

         for (n=1;(cnt[n-1]>=BLK_LENGTH)&&(n<MAX_LEVELS);n++)
         {
            cnt[n]+=1;
            smx[n]+=smx[n-1];

            cnt[n-1]=0;
            smx[n-1]=0.0;
         }
      }
      else
         ie=1;
   }

   error(ie!=0,1,"sdet [force4.c]",
         "SW term has vanishing determinant");
   
   for (n=1;n<MAX_LEVELS;n++)
      smx[0]+=smx[n];

   return smx[0];
}


double setpf4(double mu,int ipf,int isw,int icom)
{
   int ifail;
   double act,r;
   spinor_dble *phi,**wsd;
   mdflds_t *mdfs;
   
   erase_all();

   if (isw==1)
      act=sdet();
   else
      act=0.0;
   
   wsd=reserve_wsd(1);
   random_sd(VOLUME/2,wsd[0],1.0);
   act+=norm_square_dble(VOLUME/2,0,wsd[0]);            

   ifail=sw_term_xsf(ODD_PTS);
   error_root(ifail!=0,1,"set_pf4 [force4.c]",
              "Inversion of the SW term was not safe");

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   set_xsf_parms(-1);
   Dwhat_xsf_dble(-mu,wsd[0],phi);
   mulg5_dble(VOLUME/2,phi);
   set_sd2zero(VOLUME/2,phi+(VOLUME/2));
   release_wsd();  

   if (icom==1)
   {
      r=act;
      MPI_Reduce(&r,&act,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&act,1,MPI_DOUBLE,0,MPI_COMM_WORLD);       
   }

   return act;
}


void force4(double mu,int ipf,int isw,int isp,int icr,double c,int *status)
{
   int ifail;
   double res0,res1;
   spinor_dble *phi,*chi,*psi,**wsd;
   spinor_dble *rho,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   tm_parms_t tm;
   
   erase_all();

   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   sp=solver_parms(isp);
   
   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];   
   wsd=reserve_wsd(2);
   psi=wsd[0];
   chi=wsd[1];

   set_xt2zero();   
   set_xv2zero();

   if (isw==1)
   {
      ifail=add_det2xt_xsf(2.0,ODD_PTS);
      error_root(ifail!=0,1,"force4 [force4.c]",
                 "Inversion of the SW term was not safe");
   }
   
   if (sp.solver==CGNE)
   {
      if (get_chrono(icr,chi))
      {
         rsd=reserve_wsd(1);
         rho=rsd[0];

         ifail=sw_term_xsf(ODD_PTS);
         error_root(ifail!=0,1,"force4 [force4.c]",
                    "Inversion of the SW term was not safe");

         set_xsf_parms(1);
         Dwhat_xsf_dble(mu,chi,psi);
         mulg5_dble(VOLUME/2,psi);
         set_xsf_parms(-1);
         Dwhat_xsf_dble(-mu,psi,rho);
         mulg5_dble(VOLUME/2,rho);
         mulr_spinor_add_dble(VOLUME/2,rho,phi,-1.0);

         res0=norm_square_dble(VOLUME/2,1,phi);
         res1=norm_square_dble(VOLUME/2,1,rho);
         res1=sqrt(res1/res0);
         
         if (res1<1.0)
         {
            if (res1>sp.res)
            {
               tmcgeo_xsf(sp.nmx,sp.res/res1,tau3,mu,rho,psi,status);
               mulr_spinor_add_dble(VOLUME/2,chi,psi,-1.0);
            } 
            else 
               status[0]=0;
         } 
         else 
            tmcgeo_xsf(sp.nmx,sp.res,tau3,mu,phi,chi,status);   
         
         release_wsd();
      } 
      else 
         tmcgeo_xsf(sp.nmx,sp.res,tau3,mu,phi,chi,status);

      error_root(status[0]<0,1,"force4 [force4.c]",
                 "CGNE solver failed (mu = %.4e, parameter set no %d, "
                 "status = %d)",mu,isp,status[0]);

      set_xsf_parms(1);
      Dwoe_xsf_dble(chi,chi);
      Dwoo_xsf_dble(0.0,chi,chi);
      Dwhat_xsf_dble(mu,chi,psi);
      mulg5_dble(VOLUME/2,psi);
      set_xsf_parms(-1);
      Dwoe_xsf_dble(psi,psi);
      Dwoo_xsf_dble(0.0,psi,psi);

      if (icr)
         add_chrono(icr,chi);
           
      add_prod2xv_xsf(-1.0,chi,psi);
      add_prod2xt_xsf(1.0,chi,psi);
   }
   else
      error_root(1,1,"force4 [force4.c]","Unknown solver only CG is supported");

   sw_frc(c);
   hop_frc(c);

   release_wsd();
}


double action4(double mu,int ipf,int isw,int isp,int icom,int *status)
{
   double act,r;
   spinor_dble *phi,*chi,*psi;
   spinor_dble **rsd,**wsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   tm_parms_t tm;

   erase_all();

   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);
   
   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   sp=solver_parms(isp);

   if (isw==1)
      act=sdet();
   else
      act=0.0;

   if (sp.solver==CGNE)
   {
      rsd=reserve_wsd(1);
      chi=rsd[0];
      
      tmcgeo_xsf(sp.nmx,sp.res,tau3,mu,phi,chi,status);

      error_root(status[0]<0,1,"action4 [force4.c]",
                 "CGNE solver failed (mu = %.4e, parameter set no %d, "
                 "status = %d)",mu,isp,status[0]);

      wsd=reserve_wsd(1);
      psi=wsd[0];

      set_xsf_parms(1);
      Dwhat_xsf_dble(mu,chi,psi);
      act+=norm_square_dble(VOLUME/2,0,psi);

      release_wsd();
      release_wsd();
   }
   else
      error_root(1,1,"action4 [force4.c]","Unknown solver only CG is supported");

   if (icom==1)
   {
      r=act;
      MPI_Reduce(&r,&act,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&act,1,MPI_DOUBLE,0,MPI_COMM_WORLD);       
   }

   return act;
}
