
/*******************************************************************************
*
* File ptsplit.c
*
* Copyright (C) 2014 Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic routine for the computation of correlation functions involving 
* the point split vector current.
*
* The externally accessible functions are
*
*   void ptsplit(int dir,spinor_dble *s,spinor_dble *r)
*     Apply to the source field s the linear transformation
*      
*      r(x) = U(x,mu)*P_mu*s(x+mu)
*
*     where 
*
*      P_mu = (1/2)*(1-gamma_mu) 
*
*     and mu=0,...3, corresponds to one of the four space-time directions.
*     Note that on exit s is left unchanged.
*
* Notes:
*
* The input and output fields can not coincide. The program performs global 
* operations and must be called simultaneously on all processes.
*
*******************************************************************************/

#define PTSPLIT_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "flags.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "sw_term.h"
#include "dirac.h"
#include "global.h"


static void mulPmu_su3_dble(int mu,su3_dble *u,spinor_dble *s,spinor_dble *r)
{
   spinor_dble rs;
   su3_vector_dble psi;

   if (mu==0)
   {
   /******************************* direction +0 *********************************/

      _vector_add(psi,(*s).c1,(*s).c3);
      _su3_multiply(rs.c1,(*u),psi);
      rs.c3=rs.c1;

      _vector_add(psi,(*s).c2,(*s).c4);
      _su3_multiply(rs.c2,(*u),psi);
      rs.c4=rs.c2;
   }
   else if (mu==1)
   {
   /******************************* direction +1 *********************************/
   
      _vector_i_add(psi,(*s).c1,(*s).c4);
      _su3_multiply(rs.c1,(*u),psi);
      _vector_imul(rs.c4,-1.0,rs.c1);

      _vector_i_add(psi,(*s).c2,(*s).c3);
      _su3_multiply(rs.c2,(*u),psi);
      _vector_imul(rs.c3,-1.0,rs.c2);
   }
   else if (mu==2)
   {
   /******************************* direction +2 *********************************/

      _vector_add(psi,(*s).c1,(*s).c4);
      _su3_multiply(rs.c1,(*u),psi);
      rs.c4=rs.c1;

      _vector_sub(psi,(*s).c2,(*s).c3);
      _su3_multiply(rs.c2,(*u),psi);
      _vector_mul(rs.c3,-1.0,rs.c2);
   }
   else
   {
   /******************************* direction +3 *********************************/

      _vector_i_add(psi,(*s).c1,(*s).c3);
      _su3_multiply(rs.c1,(*u),psi);
      _vector_imul(rs.c3,-1.0,rs.c1);

      _vector_i_sub(psi,(*s).c2,(*s).c4);
      _su3_multiply(rs.c2,(*u),psi);
      _vector_imul(rs.c4,1.0,rs.c2);
   }

   _vector_mul((*r).c1,0.5,rs.c1);
   _vector_mul((*r).c2,0.5,rs.c2);
   _vector_mul((*r).c3,0.5,rs.c3);
   _vector_mul((*r).c4,0.5,rs.c4);
}


void ptsplit(int mu,spinor_dble *s,spinor_dble *r)
{
   int ix,t;
   int iprms[1];
   int *piup,*pidn;
   spinor_dble *so,*ro;
   spinor_dble *sp,*rm;
   su3_dble *u,*um;

   if (NPROC>1)
   {
      iprms[0]=mu;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=mu,1,
            "ptsplit [ptsplit.c]","Parameters are not global");   
   }

   error_root(((mu<0)||(mu>3)),1,
               "ptsplit [ptsplit.c]","Improper argument mu");

   so=s+VOLUME/2;
   ro=r+VOLUME/2;
   piup=iup[VOLUME/2];
   pidn=idn[VOLUME/2];

   u=udfld();
   um=u+4*VOLUME;

   cpsd_int_bnd_xsf(0x1,s);   
   set_sd2zero(NSPIN,r);
   
   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=VOLUME/2;
      
      for (;u<um;u+=8)
      {
         t=global_time(ix);
         ix+=1;

         if (((t==0)||(t==(NPROC0*L0-1)))&&(mu==0))
         {
            if (t==0)
            {
               sp=s+piup[mu];
               mulPmu_su3_dble(mu,u+2*mu,sp,ro);
            }
            else 
            {
               rm=r+pidn[mu];
               mulPmu_su3_dble(mu,u+2*mu+1,so,rm);
            }
         }
         else
         {
            sp=s+piup[mu];
            rm=r+pidn[mu];
            
            mulPmu_su3_dble(mu,u+2*mu,sp,ro);
            mulPmu_su3_dble(mu,u+2*mu+1,so,rm);
         }

         piup+=4;
         pidn+=4;
         so+=1;
         ro+=1;
      }
   }
   else
   {
      for (;u<um;u+=8)
      {
         sp=s+piup[mu];
         rm=r+pidn[mu];
         
         mulPmu_su3_dble(mu,u+2*mu,sp,ro);
         mulPmu_su3_dble(mu,u+2*mu+1,so,rm);

         piup+=4;
         pidn+=4;
         so+=1;
         ro+=1;
      }
   }

   cpsd_ext_bnd_xsf(0x1,r);
}


