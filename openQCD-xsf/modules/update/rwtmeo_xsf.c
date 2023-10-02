
/*******************************************************************************
*
* File rwtmeo_xsf.c
*
* Copyright (C) 2014 Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Twisted-mass reweighting factors (even-odd preconditioned version)
*
* The externally accessible functions are
*
*   double rwtm1eo_xsf(double mu,int isp,double *sqn,int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqn and returns -ln(r1) (see the notes).
*     The twisted-mass Dirac equation is solved using the solver specified
*     by the parameter set number isp. Currently only the CG is supported.
*     The argument status must be pointing to an array of at least 1,1
*     and 4 elements, respectively, in the case of the CGNE, SAP_GCR and
*     DFL_SAP_GCR solver. On exit the array elements return the status
*     values reported by the solver program.
*
*   double rwtm1eo_act(double mu,int isp,double *sqn,int *status)
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
* ones defined in this paper except for the fact that the Wilson-Dirac 
* operator is replaced by the even-odd preconditioned operator.
*
* For a given random pseudo-fermion field eta with distribution proportional
* to exp{-(eta,eta)}, the factors r1 and r1act are given by
*
*  r1=exp{-(eta,[imu*dQhat+mu^2](Qhat^dag*Qhat)^(-1)*eta)},
*
*  r1act=exp{-(eta,(Qhat+imu)(Qhat^dag*Qhat)^(-1)(Qhat^dag-imu)*eta)}
*
* with Qhat=\gamma5*Dwhat and dQhat=Qhat^(down)-Qhat^(up), where Dwhat^(i)
* denotes the even-odd preconditioned, massive O(a)-improved Wilson-Dirac
* operator with flavour i=up,down. Note that the pseudo-fermion field 
* vanishes on the odd sites of the lattice. The bare quark mass is taken 
* to be the one last set by sw_parms() [flags/parms.c] and it is assumed
* that the chosen solver parameters have been set by set_solver_parms() 
* [flags/sparms.c].
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define RWTMEO_XSF_C

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
static spinor_dble rs,sd0={{{0.0}}};


static double set_eta(spinor_dble *eta)
{
   random_sd(VOLUME/2,eta,1.0);
   set_sd2zero(VOLUME/2,eta+(VOLUME/2));
   
   return norm_square_dble(VOLUME/2,1,eta);
}


static void apply_dQee_dble(double mu,spinor_dble *s,spinor_dble *r)
{
   int ix,t;
   spinor_dble *sm;

   sm=s+VOLUME/2;

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=VOLUME/2;
 
      for (;s<sm;s++)
      {
         t=global_time(ix);

         if (t==0)
         {
            _vector_add(rs.c1,(*s).c1,(*s).c3);
            _vector_mul(rs.c3,-1.0,rs.c1);
            _vector_add(rs.c2,(*s).c2,(*s).c4);
            _vector_mul(rs.c4,-1.0,rs.c2);
         }
         else if (t==(NPROC0*L0-1))
         {
            _vector_sub(rs.c1,(*s).c1,(*s).c3);
            rs.c3=rs.c1;
            _vector_sub(rs.c2,(*s).c2,(*s).c4);
            rs.c4=rs.c2;
         }
         else
            rs=sd0;

         _vector_combine(rs.c1,(*s).c1,mu, mu*mu);
         _vector_combine(rs.c2,(*s).c2,mu, mu*mu);
         _vector_combine(rs.c3,(*s).c3,mu,-mu*mu);
         _vector_combine(rs.c4,(*s).c4,mu,-mu*mu);

         (*r)=rs;

         r+=1;
         ix+=1;
      }
   }
   else
   {
      for (;s<sm;s++)
      {
         _vector_mul((*r).c1, mu*mu,(*s).c1);
         _vector_mul((*r).c2, mu*mu,(*s).c2);
         _vector_mul((*r).c3,-mu*mu,(*s).c3);
         _vector_mul((*r).c4,-mu*mu,(*s).c4);
      
         r+=1;
      }
   }
}


static void apply_dQoo_dble(double mu,spinor_dble *s, spinor_dble *r)
{
   int ix,t;
   double c;
   spinor_dble *ro,*so,*sm;
   sw_parms_t sw;
   lat_parms_t lat;

   sw=sw_parms();
   lat=lat_parms();

   c=lat.zF+3.0*lat.dF+sw.m0;
   c=-mu/(c*c);

   so=s+VOLUME/2;
   ro=r+VOLUME/2;
   sm=s+VOLUME;

   ix=VOLUME/2;

   for (;so<sm;so++)
   {
      t=global_time(ix);

      if (t==0)
      {
         _vector_add(rs.c1,(*so).c1,(*so).c3);
         _vector_mul(rs.c3,-1.0,rs.c1);
         _vector_add(rs.c2,(*so).c2,(*so).c4);
         _vector_mul(rs.c4,-1.0,rs.c2);
    
         _vector_mul_assign(rs.c1,c);
         _vector_mul_assign(rs.c2,c);
         _vector_mul_assign(rs.c3,c);
         _vector_mul_assign(rs.c4,c);
      }
      else if (t==(NPROC0*L0-1))
      {
         _vector_sub(rs.c1,(*so).c1,(*so).c3);
         rs.c3=rs.c1;
         _vector_sub(rs.c2,(*so).c2,(*so).c4);
         rs.c4=rs.c2;

         _vector_mul_assign(rs.c1,c);
         _vector_mul_assign(rs.c2,c);
         _vector_mul_assign(rs.c3,c);
         _vector_mul_assign(rs.c4,c);
      }
      else
         rs=sd0;

      (*ro)=rs;         

      ro+=1;
      ix+=1;
   }
}


double rwtm1eo_xsf(double mu,int isp,double *sqn,int *status)
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
      tmcgeo_xsf(sp.nmx,sp.res,tau3,0.0,eta,phi,status);

      error_root(status[0]<0,1,"rwtm1eo_xsf [rwtmeo_xsf.c]",
                 "CGNE solver failed (mu = 0.0, parameter set no %d, "
                 "status = %d)",isp,status[0]);

      Dwoe_xsf_dble(phi,phi);
      apply_dQee_dble(mu,phi,phi);
      apply_dQoo_dble(mu,phi,phi); 
      Dweo_xsf_dble(phi,phi);
      mulg5_dble(VOLUME/2,phi);

      lnr=spinor_prod_re_dble(VOLUME/2,1,eta,phi);
   }
   else
   {
      lnr=0.0;
      error_root(1,1,"rwtm1eo_xsf [rwtmeo_xsf.c]","Unknown solver only CG is supported");
   }
   
   release_wsd();

   return lnr;
}


double rwtm1eo_act(double mu,int isp,double *sqn,int *status)
{
   int ifail;
   double lnr;
   spinor_dble *eta,*phi,*psi,**wsd;
   solver_parms_t sp;
   tm_parms_t tm;

   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   ifail=sw_term_xsf(ODD_PTS);
   error_root(ifail!=0,1,"rwtm1eo_act [rwtmeo_xsf.c]",
              "Inversion of the SW term was not safe");

   wsd=reserve_wsd(3);   
   eta=wsd[0];
   phi=wsd[1];
   psi=wsd[2];
   (*sqn)=set_eta(eta);
   sp=solver_parms(isp);

   set_xsf_parms(-1);
   Dwhat_xsf_dble(-mu,eta,psi);
   mulg5_dble(VOLUME/2,psi);

   if (sp.solver==CGNE)
   {
      tmcgeo_xsf(sp.nmx,sp.res,tau3,0.0,psi,phi,status);

      error_root(status[0]<0,1,"rwtm1eo_act [rwtmeo_xsf.c]",
                 "CGNE solver failed (mu = 0.0, parameter set no %d, "
                 "status = %d)",isp,status[0]);

      set_xsf_parms(1);
      Dwhat_xsf_dble(mu,phi,psi);
      mulg5_dble(VOLUME/2,psi);
      lnr=spinor_prod_re_dble(VOLUME/2,1,eta,psi);
   }
   else
   {
      lnr=0.0;
      error_root(1,1,"rwtm1eo_act [rwtmeo_xsf.c]","Unknown solver only CG is supported");
   }
 
   release_wsd();

   return lnr;
}


