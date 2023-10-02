
/*******************************************************************************
*
* File rwtm_xsf.c
*
* Copyright (C) 2014 Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Twisted-mass reweighting factors
*
* The externally accessible functions are
*
*   double rwtm1_xsf(double mu,int isp,double *sqn,int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqn and returns -ln(r1) (see the notes).
*     The twisted-mass Dirac equation is solved using the solver specified
*     by the parameter set number isp. Currently only the CG is supported.
*     The argument status must be pointing to an array of at least 1,1
*     and 4 elements, respectively, in the case of the CGNE, SAP_GCR and
*     DFL_SAP_GCR solver. On exit the array elements return the status
*     values reported by the solver program.
*
*   double rwtm1_act(double mu,int isp,double *sqn,int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqn and returns -ln(r1act) (see the notes).
*     The twisted-mass Dirac equation is solved using the solver specified
*     by the parameter set number isp. Currently only the CG is supported.
*     The argument status must be pointing to an array of at least 1,1
*     and 4 elements, respectively, in the case of the CGNE, SAP_GCR and
*     DFL_SAP_GCR solver. On exit the array elements return the status
*     values reported by the solver program.
* 
* Notes:
*
* Twisted-mass reweighting of the quark determinant was introduced by
*
*  M. Luescher, F. Palombi: "Fluctuations and reweighting of the quark
*  determinant on large lattices", PoS LATTICE2008 (2008) 049
*
* The stochastic reweighting factors computed here are analogous to the 
* ones defined in this paper.
*
* For a given random pseudo-fermion field eta with distribution proportional
* to exp{-(eta,eta)}, the factors r1 and r1act are given by
*
*  r1=exp{-(eta,[imu*dQ+mu^2](Q^dag*Q)^(-1)*eta)},
*
*  r1act=exp{-(eta,(Q+imu)(Q^dag*Q)^(-1)(Q^dag-imu)*eta)}
*
* with Q=\gamma5*Dw and dQ=Q^(down)-Q^(up), where Dw^(i) denotes the massive 
* O(a)-improved Wilson-Dirac operator with flavour i=up,down. In the programs 
* rwtm*(), the bare quark mass is taken to be the one last set by sw_parms() 
* [flags/parms.c] and it is assumed that the chosen solver parameters have 
* been set by set_solver_parms() [flags/sparms.c].
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define RWTM_XSF_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "sw_term.h"
#include "linalg.h"
#include "sap.h"
#include "dfl.h"
#include "dirac.h"
#include "forces.h"
#include "update.h"
#include "global.h"

static const int tau3=1;
static spinor_dble sd0={{{0.0}}};


static double set_eta(spinor_dble *eta)
{
   random_sd(VOLUME,eta,1.0);
   
   return norm_square_dble(VOLUME,1,eta);
}


static void apply_dQ_dble(double mu,spinor_dble *s,spinor_dble *r)
{
   int ix,t;
   spinor_dble rs,*sm;

   sm=s+VOLUME;

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=0;
 
      for (;s<sm;s++)
      {
         t=global_time(ix);

         if (t==0)
         {
            _vector_add(rs.c1,(*s).c1,(*s).c3);
            rs.c3=rs.c1;
            _vector_add(rs.c2,(*s).c2,(*s).c4);
            rs.c4=rs.c2;
         }
         else if (t==(NPROC0*L0-1))
         {
            _vector_sub(rs.c1,(*s).c1,(*s).c3);
            _vector_mul(rs.c3,-1.0,rs.c1);
            _vector_sub(rs.c2,(*s).c2,(*s).c4);
            _vector_mul(rs.c4,-1.0,rs.c2);
         }
         else
            rs=sd0;

         _vector_combine(rs.c1,(*s).c1,mu,mu*mu);
         _vector_combine(rs.c2,(*s).c2,mu,mu*mu);
         _vector_combine(rs.c3,(*s).c3,mu,mu*mu);
         _vector_combine(rs.c4,(*s).c4,mu,mu*mu);

         (*r)=rs;

         r+=1;
         ix+=1;
      }
   }
   else
   {
      for (;s<sm;s++)
      {
         _vector_mul((*r).c1,mu*mu,(*s).c1);
         _vector_mul((*r).c2,mu*mu,(*s).c2);
         _vector_mul((*r).c3,mu*mu,(*s).c3);
         _vector_mul((*r).c4,mu*mu,(*s).c4);
      
         r+=1;
      }
   }
}


double rwtm1_xsf(double mu,int isp,double *sqn,int *status)
{
   double lnr;
   spinor_dble *eta,*phi,**wsd;
   solver_parms_t sp;

   wsd=reserve_wsd(2);
   eta=wsd[0];
   phi=wsd[1];
   (*sqn)=set_eta(eta);
   sp=solver_parms(isp);   

   if (sp.solver==CGNE)
   {
      tmcg_xsf(sp.nmx,sp.res,tau3,0.0,eta,phi,status);

      error_root(status[0]<0,1,"rwtm1_xsf [rwtm_xsf.c]",
                 "CGNE solver failed (mu = 0.0, parameter set no %d, "
                 "status = %d)",isp,status[0]);

      apply_dQ_dble(mu,phi,phi);
      lnr=spinor_prod_re_dble(VOLUME,1,eta,phi);
   }
   else
   {
      lnr=0.0;
      error_root(1,1,"rwtm1_xsf [rwtm_xsf.c]","Unknown solver only CG is supported");
   }
   
   release_wsd();

   return lnr;
}


double rwtm1_act(double mu,int isp,double *sqn,int *status)
{
   double lnr;
   spinor_dble *eta,*phi,*psi,**wsd;
   solver_parms_t sp;
   tm_parms_t tm;
   
   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   sw_term_xsf(NO_PTS);

   wsd=reserve_wsd(3);   
   eta=wsd[0];
   phi=wsd[1];
   psi=wsd[2];
   (*sqn)=set_eta(eta);
   sp=solver_parms(isp);

   set_xsf_parms(-1);
   Dw_xsf_dble(-mu,eta,psi);
   mulg5_dble(VOLUME,psi);

   if (sp.solver==CGNE)
   {
      tmcg_xsf(sp.nmx,sp.res,tau3,0.0,psi,phi,status);

      error_root(status[0]<0,1,"rwtm1_act [rwtm_xsf.c]",
                 "CGNE solver failed (mu = 0.0, parameter set no %d, "
                 "status = %d)",isp,status[0]);

      set_xsf_parms(1);
      Dw_xsf_dble(mu,phi,psi);
      mulg5_dble(VOLUME,psi);
      lnr=spinor_prod_re_dble(VOLUME,1,eta,psi);
   }
   else
   {
      lnr=0.0;
      error_root(1,1,"rwtm1_act [rwtm_xsf.c]","Unknown solver only CG is supported");
   }
 
   release_wsd();

   return lnr;
}


