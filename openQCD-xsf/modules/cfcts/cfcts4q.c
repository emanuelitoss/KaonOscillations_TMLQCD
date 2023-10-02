
/*******************************************************************************
*
* File cfcts4q.c
*
* Copyright (C) 2007, 2008, 2009, 2013, 2015 Martin Luescher, 
*                        Leonardo Giusti, Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* General programs for the calculation of correlation functions involving
* 4 fermion interpolators
*
* The externally accessible functions are
*
*  void cfcts4q1(dirac_t A,dirac_t B,dirac_t C,dirac_t D,spinor_dble **sk,
*  spinor_dble **sl,spinor_dble **skp,spinor_dble **slp,complex_dble ab[])
*     Calculates the time-slice sums ab[x0] at time x0=0,..,NPROC0*L0-1 of
*     the connected part of the correlation function 
*
*      <C_da(x) (A_ab B_cd)(y) D_bc(z)>   (1)
*
*     assuming the quark propagators S_{a,b,c,d}(x,y) for the quark flavours 
*     "a", "b", "c" and "d" are stored in the spinor fields sk[i], i=0,..,11,
*     sl[i], skp[i], and slp[i], respectively (see the notes below).
*
*  void cfcts4q2(dirac_t A,dirac_t B,dirac_t C,dirac_t D,spinor_dble **sk,
*  spinor_dble **sl,spinor_dble **skp,spinor_dble **slp,complex_dble ab[])
*     Calculates the time-slice sums ab[x0] at time x0=0,..,NPROC0*L0-1 of
*     the connected part of the correlation function 
*
*      <C_ba(x) (A_ab B_cd)(y) D_dc(z)>   (2)
*
*     assuming the quark propagators S_{a,b,c,d}(x,y) for the quark flavours 
*     "a", "b", "c" and "d" are stored in the spinor fields sk[i], i=0,..,11,
*     sl[i], skp[i], and slp[i], respectively (see the notes below).
*
* Notes:
*
* All programs in this file act globally and should be called 
* simultaneously on all processes with the same parameters.
* 
* Currently the supported Dirac matrices A are
*
*  S=1, P=gamma_5, Vmu=gamma_mu, Amu=gamma_mu*gamma_5
*
* where mu=0,..,3 (the possible arguments A are thus S,P,..,A2,A3),
* and the anti-hermitian tensors,
*
* T_{mu,nu}  = sigma_{mu,nu},
*
* Tt_{mu,nu} = gamma_5*sigma_{mu,nu},
*
* where mu=0,1,2, nu=1,2,3, and sigma_{mu,nu}=(i/2)*[gamma_mu,gamma_nu]
* (the possible arguments A are thus T01,...,Tt23).
*
* The symbol A_ab(x) stands for psibar_a(x)*A*psi_b(x), where "a" and
* "b" are flavour indices and A any of the known Dirac matrices. The
* calculated three-point correlation functions thus correspond to 
*
*  (1) = -tr{S_a(x,y)*A*S_b(y,z)*D*S_c(z,y)*B*S_d(y,x)*C} 
*
*  and
*   
*  (2) = tr{S_a(x,y)*A*S_b(y,x)*C} x tr{S_c(z,y)*B*S_d(y,z)*D}
*
* which include the overall fermion minus sign.
*
* For any given source point y, Dirac index id=1,2,3,4 and colour index
* ic=1,2,3 at y, the quark propagator S(x,y) is a spinor field (as a
* function of x). The programs in this module assume the 12 fields that
* correspond to all possible combinations of id and ic to be stored in 
* sequences of fields psd[3*(id-1)+ic-1].
*
*******************************************************************************/

#define CFCTS4Q_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "cfcts.h"
#include "dfl.h"
#include "flags.h"
#include "sap.h"
#include "utils.h"
#include "sflds.h"
#include "lattice.h"
#include "global.h"

static int *tms=NULL;
static complex_dble *lsm,spt;
static complex_dble tslice[NPROC0*L0];


static void cmpnts(dirac_t A,int j[],complex_dble g[])
{
   g[0].re=0.0;
   g[0].im=0.0;   
   g[1].re=0.0;
   g[1].im=0.0;   
   g[2].re=0.0;
   g[2].im=0.0;
   g[3].re=0.0;
   g[3].im=0.0;

   if ((A==S)||(A==P)||(A==T03)||(A==Tt03)||
       (A==T12)||(A==Tt12))
   {
      j[0]=0;
      j[1]=1;
      j[2]=2;
      j[3]=3;

      if (A==S)
      {
         g[0].re=1.0;
         g[1].re=1.0;
         g[2].re=1.0;
         g[3].re=1.0;
      }
      else if (A==P)
      {
         g[0].re=1.0;
         g[1].re=1.0;
         g[2].re=-1.0;
         g[3].re=-1.0;
      }
      else if (A==T03)
      {
         g[0].re=1.0;
         g[1].re=-1.0;
         g[2].re=-1.0;
         g[3].re=1.0;
      }
      else if (A==Tt03)
      {
         g[0].re=1.0;  
         g[1].re=-1.0;
         g[2].re=1.0;
         g[3].re=-1.0;
      } 
      else if (A==T12)
      {
         g[0].re=-1.0;
         g[1].re=1.0;
         g[2].re=-1.0;
         g[3].re=1.0;
      }
      else
      {
         g[0].re=-1.0;
         g[1].re=1.0;
         g[2].re=1.0;
         g[3].re=-1.0;
      }
   }
   else if ((A==V0)||(A==A0)||(A==V3)||(A==A3))
   {
      j[0]=2;
      j[1]=3;
      j[2]=0;
      j[3]=1;

      if (A==V0)
      {
         g[0].re=-1.0;
         g[1].re=-1.0;
         g[2].re=-1.0;
         g[3].re=-1.0;
      }
      else if (A==A0)
      {
         g[0].re=1.0;
         g[1].re=1.0;
         g[2].re=-1.0;
         g[3].re=-1.0;
      }
      else if (A==V3)
      {
         g[0].im=-1.0;
         g[1].im=1.0;
         g[2].im=1.0;
         g[3].im=-1.0;
      }
      else
      {
         g[0].im=1.0;
         g[1].im=-1.0;
         g[2].im=1.0;
         g[3].im=-1.0;
      }
   }
   else if ((A==V1)||(A==A1)||(A==V2)||(A==A2))
   {
      j[0]=3;
      j[1]=2;
      j[2]=1;
      j[3]=0;

      if (A==V1)
      {
         g[0].im=-1.0;
         g[1].im=-1.0;
         g[2].im=1.0;
         g[3].im=1.0;
      }
      else if (A==A1)
      {
         g[0].im=1.0;
         g[1].im=1.0;
         g[2].im=1.0;
         g[3].im=1.0;
      }      
      else if (A==V2)
      {
         g[0].re=-1.0;
         g[1].re=1.0;
         g[2].re=1.0;
         g[3].re=-1.0;
      }
      else
      {
         g[0].re=1.0;
         g[1].re=-1.0;
         g[2].re=1.0;
         g[3].re=-1.0;
      }
   }
   else if ((A==T01)||(A==Tt01)||(A==T02)||(A==Tt02)||
            (A==T13)||(A==Tt13)||(A==T23)||(A==Tt23))
   {
      j[0]=1;
      j[1]=0;
      j[2]=3;
      j[3]=2;

      if (A==T01)
      {
         g[0].re=1.0;
         g[1].re=1.0;
         g[2].re=-1.0;
         g[3].re=-1.0;
      }
      else if (A==Tt01)
      {
         g[0].re=1.0;
         g[1].re=1.0;
         g[2].re=1.0;
         g[3].re=1.0;
      }
      else if ((A==T02)||(A==Tt13))
      {
         g[0].im=-1.0;
         g[1].im=1.0;
         g[2].im=1.0;
         g[3].im=-1.0;
      }
      else if ((A==Tt02)||(A==T13))
      {
         g[0].im=-1.0;
         g[1].im=1.0;
         g[2].im=-1.0;
         g[3].im=1.0;
      }
      else if (A==T23)
      {
         g[0].re=-1.0;
         g[1].re=-1.0;
         g[2].re=-1.0;
         g[3].re=-1.0;
      }
      else 
      {
         g[0].re=-1.0;
         g[1].re=-1.0;
         g[2].re=1.0;
         g[3].re=1.0;
      }
   }
   else
   {
      j[0]=0;
      j[1]=1;
      j[2]=2;
      j[3]=3;

      error_loc(1,1,"cmpnts [cfcts4q.c]","Unknown Dirac matrix");
   }
}


static void alloc_tms(void)
{
   int ix,iy,x0;

   tms=amalloc(VOLUME*sizeof(int),3);
   lsm=amalloc(NPROC0*L0*sizeof(complex_dble),ALIGN);
   
   error((tms==NULL)||(lsm==NULL),1,"alloc_tms [cfcts4q.c]",
         "Unable to allocate auxiliary arrays");

   for (iy=0;iy<VOLUME;iy++)
   {
      x0=iy/(L1*L2*L3);
      ix=ipt[iy];

      tms[ix]=x0+cpr[0]*L0;
   }
}


static void sp_S(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_re((*r).c1,(*s).c1));
   spt.re+=(_vector_prod_re((*r).c2,(*s).c2));
   spt.re+=(_vector_prod_re((*r).c3,(*s).c3));
   spt.re+=(_vector_prod_re((*r).c4,(*s).c4));

   spt.im+=(_vector_prod_im((*r).c1,(*s).c1));
   spt.im+=(_vector_prod_im((*r).c2,(*s).c2));
   spt.im+=(_vector_prod_im((*r).c3,(*s).c3));
   spt.im+=(_vector_prod_im((*r).c4,(*s).c4));
}


static void sp_P(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_re((*r).c1,(*s).c1));
   spt.re+=(_vector_prod_re((*r).c2,(*s).c2));
   spt.re-=(_vector_prod_re((*r).c3,(*s).c3));
   spt.re-=(_vector_prod_re((*r).c4,(*s).c4));

   spt.im+=(_vector_prod_im((*r).c1,(*s).c1));
   spt.im+=(_vector_prod_im((*r).c2,(*s).c2));
   spt.im-=(_vector_prod_im((*r).c3,(*s).c3));
   spt.im-=(_vector_prod_im((*r).c4,(*s).c4));
}


static void sp_V0(spinor_dble *r,spinor_dble *s)
{
   spt.re-=(_vector_prod_re((*r).c1,(*s).c3));
   spt.re-=(_vector_prod_re((*r).c2,(*s).c4));
   spt.re-=(_vector_prod_re((*r).c3,(*s).c1));
   spt.re-=(_vector_prod_re((*r).c4,(*s).c2));

   spt.im-=(_vector_prod_im((*r).c1,(*s).c3));
   spt.im-=(_vector_prod_im((*r).c2,(*s).c4));
   spt.im-=(_vector_prod_im((*r).c3,(*s).c1));
   spt.im-=(_vector_prod_im((*r).c4,(*s).c2));
}


static void sp_A0(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_re((*r).c1,(*s).c3));
   spt.re+=(_vector_prod_re((*r).c2,(*s).c4));
   spt.re-=(_vector_prod_re((*r).c3,(*s).c1));
   spt.re-=(_vector_prod_re((*r).c4,(*s).c2));

   spt.im+=(_vector_prod_im((*r).c1,(*s).c3));
   spt.im+=(_vector_prod_im((*r).c2,(*s).c4));
   spt.im-=(_vector_prod_im((*r).c3,(*s).c1));
   spt.im-=(_vector_prod_im((*r).c4,(*s).c2));
}


static void sp_V1(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_im((*r).c1,(*s).c4));
   spt.re+=(_vector_prod_im((*r).c2,(*s).c3));
   spt.re-=(_vector_prod_im((*r).c3,(*s).c2));
   spt.re-=(_vector_prod_im((*r).c4,(*s).c1));

   spt.im-=(_vector_prod_re((*r).c1,(*s).c4));
   spt.im-=(_vector_prod_re((*r).c2,(*s).c3));
   spt.im+=(_vector_prod_re((*r).c3,(*s).c2));
   spt.im+=(_vector_prod_re((*r).c4,(*s).c1));
}


static void sp_A1(spinor_dble *r,spinor_dble *s)
{
   spt.re-=(_vector_prod_im((*r).c1,(*s).c4));
   spt.re-=(_vector_prod_im((*r).c2,(*s).c3));
   spt.re-=(_vector_prod_im((*r).c3,(*s).c2));
   spt.re-=(_vector_prod_im((*r).c4,(*s).c1));

   spt.im+=(_vector_prod_re((*r).c1,(*s).c4));
   spt.im+=(_vector_prod_re((*r).c2,(*s).c3));
   spt.im+=(_vector_prod_re((*r).c3,(*s).c2));
   spt.im+=(_vector_prod_re((*r).c4,(*s).c1));
}


static void sp_V2(spinor_dble *r,spinor_dble *s)
{
   spt.re-=(_vector_prod_re((*r).c1,(*s).c4));
   spt.re+=(_vector_prod_re((*r).c2,(*s).c3));
   spt.re+=(_vector_prod_re((*r).c3,(*s).c2));
   spt.re-=(_vector_prod_re((*r).c4,(*s).c1));

   spt.im-=(_vector_prod_im((*r).c1,(*s).c4));
   spt.im+=(_vector_prod_im((*r).c2,(*s).c3));
   spt.im+=(_vector_prod_im((*r).c3,(*s).c2));
   spt.im-=(_vector_prod_im((*r).c4,(*s).c1));
}


static void sp_A2(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_re((*r).c1,(*s).c4));
   spt.re-=(_vector_prod_re((*r).c2,(*s).c3));
   spt.re+=(_vector_prod_re((*r).c3,(*s).c2));
   spt.re-=(_vector_prod_re((*r).c4,(*s).c1));

   spt.im+=(_vector_prod_im((*r).c1,(*s).c4));
   spt.im-=(_vector_prod_im((*r).c2,(*s).c3));
   spt.im+=(_vector_prod_im((*r).c3,(*s).c2));
   spt.im-=(_vector_prod_im((*r).c4,(*s).c1));
}


static void sp_V3(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_im((*r).c1,(*s).c3));
   spt.re-=(_vector_prod_im((*r).c2,(*s).c4));
   spt.re-=(_vector_prod_im((*r).c3,(*s).c1));
   spt.re+=(_vector_prod_im((*r).c4,(*s).c2));

   spt.im-=(_vector_prod_re((*r).c1,(*s).c3));
   spt.im+=(_vector_prod_re((*r).c2,(*s).c4));
   spt.im+=(_vector_prod_re((*r).c3,(*s).c1));
   spt.im-=(_vector_prod_re((*r).c4,(*s).c2));
}


static void sp_A3(spinor_dble *r,spinor_dble *s)
{
   spt.re-=(_vector_prod_im((*r).c1,(*s).c3));
   spt.re+=(_vector_prod_im((*r).c2,(*s).c4));
   spt.re-=(_vector_prod_im((*r).c3,(*s).c1));
   spt.re+=(_vector_prod_im((*r).c4,(*s).c2));

   spt.im+=(_vector_prod_re((*r).c1,(*s).c3));
   spt.im-=(_vector_prod_re((*r).c2,(*s).c4));
   spt.im+=(_vector_prod_re((*r).c3,(*s).c1));
   spt.im-=(_vector_prod_re((*r).c4,(*s).c2));
}


static void sp_T01(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_re((*r).c1,(*s).c2));
   spt.re+=(_vector_prod_re((*r).c2,(*s).c1));
   spt.re-=(_vector_prod_re((*r).c3,(*s).c4));
   spt.re-=(_vector_prod_re((*r).c4,(*s).c3));

   spt.im+=(_vector_prod_im((*r).c1,(*s).c2));
   spt.im+=(_vector_prod_im((*r).c2,(*s).c1));
   spt.im-=(_vector_prod_im((*r).c3,(*s).c4));
   spt.im-=(_vector_prod_im((*r).c4,(*s).c3));
}


static void sp_Tt01(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_re((*r).c1,(*s).c2));
   spt.re+=(_vector_prod_re((*r).c2,(*s).c1));
   spt.re+=(_vector_prod_re((*r).c3,(*s).c4));
   spt.re+=(_vector_prod_re((*r).c4,(*s).c3));

   spt.im+=(_vector_prod_im((*r).c1,(*s).c2));
   spt.im+=(_vector_prod_im((*r).c2,(*s).c1));
   spt.im+=(_vector_prod_im((*r).c3,(*s).c4));
   spt.im+=(_vector_prod_im((*r).c4,(*s).c3));
}


static void sp_T02(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_im((*r).c1,(*s).c2));
   spt.re-=(_vector_prod_im((*r).c2,(*s).c1));
   spt.re-=(_vector_prod_im((*r).c3,(*s).c4));
   spt.re+=(_vector_prod_im((*r).c4,(*s).c3));

   spt.im-=(_vector_prod_re((*r).c1,(*s).c2));
   spt.im+=(_vector_prod_re((*r).c2,(*s).c1));
   spt.im+=(_vector_prod_re((*r).c3,(*s).c4));
   spt.im-=(_vector_prod_re((*r).c4,(*s).c3));
}


static void sp_Tt02(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_im((*r).c1,(*s).c2));
   spt.re-=(_vector_prod_im((*r).c2,(*s).c1));
   spt.re+=(_vector_prod_im((*r).c3,(*s).c4));
   spt.re-=(_vector_prod_im((*r).c4,(*s).c3));

   spt.im-=(_vector_prod_re((*r).c1,(*s).c2));
   spt.im+=(_vector_prod_re((*r).c2,(*s).c1));
   spt.im-=(_vector_prod_re((*r).c3,(*s).c4));
   spt.im+=(_vector_prod_re((*r).c4,(*s).c3));
}


static void sp_T03(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_re((*r).c1,(*s).c1));
   spt.re-=(_vector_prod_re((*r).c2,(*s).c2));
   spt.re-=(_vector_prod_re((*r).c3,(*s).c3));
   spt.re+=(_vector_prod_re((*r).c4,(*s).c4));

   spt.im+=(_vector_prod_im((*r).c1,(*s).c1));
   spt.im-=(_vector_prod_im((*r).c2,(*s).c2));
   spt.im-=(_vector_prod_im((*r).c3,(*s).c3));
   spt.im+=(_vector_prod_im((*r).c4,(*s).c4));
}


static void sp_Tt03(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_re((*r).c1,(*s).c1));
   spt.re-=(_vector_prod_re((*r).c2,(*s).c2));
   spt.re+=(_vector_prod_re((*r).c3,(*s).c3));
   spt.re-=(_vector_prod_re((*r).c4,(*s).c4));

   spt.im+=(_vector_prod_im((*r).c1,(*s).c1));
   spt.im-=(_vector_prod_im((*r).c2,(*s).c2));
   spt.im+=(_vector_prod_im((*r).c3,(*s).c3));
   spt.im-=(_vector_prod_im((*r).c4,(*s).c4));
}


static void sp_T12(spinor_dble *r,spinor_dble *s)
{
   spt.re-=(_vector_prod_re((*r).c1,(*s).c1));
   spt.re+=(_vector_prod_re((*r).c2,(*s).c2));
   spt.re-=(_vector_prod_re((*r).c3,(*s).c3));
   spt.re+=(_vector_prod_re((*r).c4,(*s).c4));

   spt.im-=(_vector_prod_im((*r).c1,(*s).c1));
   spt.im+=(_vector_prod_im((*r).c2,(*s).c2));
   spt.im-=(_vector_prod_im((*r).c3,(*s).c3));
   spt.im+=(_vector_prod_im((*r).c4,(*s).c4));
}


static void sp_Tt12(spinor_dble *r,spinor_dble *s)
{
   spt.re-=(_vector_prod_re((*r).c1,(*s).c1));
   spt.re+=(_vector_prod_re((*r).c2,(*s).c2));
   spt.re+=(_vector_prod_re((*r).c3,(*s).c3));
   spt.re-=(_vector_prod_re((*r).c4,(*s).c4));

   spt.im-=(_vector_prod_im((*r).c1,(*s).c1));
   spt.im+=(_vector_prod_im((*r).c2,(*s).c2));
   spt.im+=(_vector_prod_im((*r).c3,(*s).c3));
   spt.im-=(_vector_prod_im((*r).c4,(*s).c4));
}


static void sp_T13(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_im((*r).c1,(*s).c2));
   spt.re-=(_vector_prod_im((*r).c2,(*s).c1));
   spt.re+=(_vector_prod_im((*r).c3,(*s).c4));
   spt.re-=(_vector_prod_im((*r).c4,(*s).c3));

   spt.im-=(_vector_prod_re((*r).c1,(*s).c2));
   spt.im+=(_vector_prod_re((*r).c2,(*s).c1));
   spt.im-=(_vector_prod_re((*r).c3,(*s).c4));
   spt.im+=(_vector_prod_re((*r).c4,(*s).c3));
}


static void sp_Tt13(spinor_dble *r,spinor_dble *s)
{
   spt.re+=(_vector_prod_im((*r).c1,(*s).c2));
   spt.re-=(_vector_prod_im((*r).c2,(*s).c1));
   spt.re-=(_vector_prod_im((*r).c3,(*s).c4));
   spt.re+=(_vector_prod_im((*r).c4,(*s).c3));

   spt.im-=(_vector_prod_re((*r).c1,(*s).c2));
   spt.im+=(_vector_prod_re((*r).c2,(*s).c1));
   spt.im+=(_vector_prod_re((*r).c3,(*s).c4));
   spt.im-=(_vector_prod_re((*r).c4,(*s).c3));
}


static void sp_T23(spinor_dble *r,spinor_dble *s)
{
   spt.re-=(_vector_prod_re((*r).c1,(*s).c2));
   spt.re-=(_vector_prod_re((*r).c2,(*s).c1));
   spt.re-=(_vector_prod_re((*r).c3,(*s).c4));
   spt.re-=(_vector_prod_re((*r).c4,(*s).c3));

   spt.im-=(_vector_prod_im((*r).c1,(*s).c2));
   spt.im-=(_vector_prod_im((*r).c2,(*s).c1));
   spt.im-=(_vector_prod_im((*r).c3,(*s).c4));
   spt.im-=(_vector_prod_im((*r).c4,(*s).c3));
}


static void sp_Tt23(spinor_dble *r,spinor_dble *s)
{
   spt.re-=(_vector_prod_re((*r).c1,(*s).c2));
   spt.re-=(_vector_prod_re((*r).c2,(*s).c1));
   spt.re+=(_vector_prod_re((*r).c3,(*s).c4));
   spt.re+=(_vector_prod_re((*r).c4,(*s).c3));

   spt.im-=(_vector_prod_im((*r).c1,(*s).c2));
   spt.im-=(_vector_prod_im((*r).c2,(*s).c1));
   spt.im+=(_vector_prod_im((*r).c3,(*s).c4));
   spt.im+=(_vector_prod_im((*r).c4,(*s).c3));
}


static void slices(dirac_t A,dirac_t B,spinor_dble *sk,spinor_dble *sl,
                   spinor_dble *skp,spinor_dble *slp,complex_dble *tsl)
{
   int iprms[2],x0,*t,*tm;
   complex_dble sptA,sptB;
   void (*spA)(spinor_dble *r,spinor_dble *s);
   void (*spB)(spinor_dble *r,spinor_dble *s);

   if (NPROC>1)
   {   
      iprms[0]=(int)(A);
      iprms[1]=(int)(B);
      
      MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=(int)(A))||(iprms[1]!=(int)(B)),1,
            "slices [cfcts4q.c]","Parameters are not global");    
   }
   
   error_root((A<S)||(A>Tt23)||(B<S)||(B>Tt23),1,
              "slices [cfcts4q.c]","Parameters are out of range"); 
   
   if (tms==NULL)
      alloc_tms();
   
   if (A==S)
      spA=sp_S;
   else if (A==P)
      spA=sp_P;
   else if (A==V0)
      spA=sp_V0;
   else if (A==A0)
      spA=sp_A0;
   else if (A==V1)
      spA=sp_V1;
   else if (A==A1)
      spA=sp_A1;
   else if (A==V2)
      spA=sp_V2;
   else if (A==A2)
      spA=sp_A2;
   else if (A==V3)
      spA=sp_V3;
   else if (A==A3)
      spA=sp_A3;
   else if (A==T01)
      spA=sp_T01;
   else if (A==Tt01)
      spA=sp_Tt01;
   else if (A==T02)
      spA=sp_T02;
   else if (A==Tt02)
      spA=sp_Tt02;
   else if (A==T03)
      spA=sp_T03;
   else if (A==Tt03)
      spA=sp_Tt03;
   else if (A==T12)
      spA=sp_T12;
   else if (A==Tt12)
      spA=sp_Tt12;
   else if (A==T13)
      spA=sp_T13;
   else if (A==Tt13)
      spA=sp_Tt13;
   else if (A==T23)
      spA=sp_T23;
   else if (A==Tt23)
      spA=sp_Tt23;
   else 
      spA=NULL;
   
   if (B==S)
      spB=sp_S;
   else if (B==P)
      spB=sp_P;
   else if (B==V0)
      spB=sp_V0;
   else if (B==A0)
      spB=sp_A0;
   else if (B==V1)
      spB=sp_V1;
   else if (B==A1)
      spB=sp_A1;
   else if (B==V2)
      spB=sp_V2;
   else if (B==A2)
      spB=sp_A2;
   else if (B==V3)
      spB=sp_V3;
   else if (B==A3)
      spB=sp_A3;
   else if (B==T01)
      spB=sp_T01;
   else if (B==Tt01)
      spB=sp_Tt01;
   else if (B==T02)
      spB=sp_T02;
   else if (B==Tt02)
      spB=sp_Tt02;
   else if (B==T03)
      spB=sp_T03;
   else if (B==Tt03)
      spB=sp_Tt03;
   else if (B==T12)
      spB=sp_T12;
   else if (B==Tt12)
      spB=sp_Tt12;
   else if (B==T13)
      spB=sp_T13;
   else if (B==Tt13)
      spB=sp_Tt13;
   else if (B==T23)
      spB=sp_T23;
   else if (B==Tt23)
      spB=sp_Tt23; 
   else 
      spB=NULL;

   for (x0=0;x0<(NPROC0*L0);x0++)
   {
      lsm[x0].re=0.0;
      lsm[x0].im=0.0;
   }

   t=tms;
   tm=t+VOLUME;
   
   for (;t<tm;t++)
   {
      spt.re=0.0;
      spt.im=0.0;

      spA(sk,sl);
      sptA=spt;

      spt.re=0.0;
      spt.im=0.0;

      spB(skp,slp);
      sptB=spt;

      lsm[*t].re+=sptA.re*sptB.re-sptA.im*sptB.im;
      lsm[*t].im+=sptA.im*sptB.re+sptA.re*sptB.im;

      sk+=1; skp+=1;
      sl+=1; slp+=1;
   }

   MPI_Reduce((double*)(lsm),(double*)(tsl),2*NPROC0*L0,MPI_DOUBLE,
              MPI_SUM,0,MPI_COMM_WORLD);
   MPI_Bcast((double*)(tsl),2*NPROC0*L0,MPI_DOUBLE,0,MPI_COMM_WORLD);
}


static dirac_t AxP(dirac_t A)
{
   if (A==S)
      return P;
   else if (A==P)
      return S;
   else if (A==V0)
      return A0;
   else if (A==V1)
      return A1;
   else if (A==V2)
      return A2;
   else if (A==V3)
      return A3;
   else if (A==A0)
      return V0;
   else if (A==A1)
      return V1;
   else if (A==A2)
      return V2;
   else if (A==A3)
      return V3;
   else if (A==T01)
      return Tt01;
   else if (A==T02)
      return Tt02;
   else if (A==T03)
      return Tt03;
   else if (A==T12)
      return Tt12;
   else if (A==T13)
      return Tt13;
   else if (A==T23)
      return Tt23;
   else if (A==Tt01)
      return T01;
   else if (A==Tt02)
      return T02;
   else if (A==Tt03) 
      return T03;
   else if (A==Tt12) 
      return T12;
   else if (A==Tt13) 
      return T13;
   else 
      return T23;
}


void cfcts4q1(dirac_t A,dirac_t B,dirac_t C,dirac_t D,spinor_dble **sk,spinor_dble **sl,
                                  spinor_dble **skp,spinor_dble **slp,complex_dble ab[])
{
   int iprms[4],x0,i,j,a,b,k[4],l[4];
   complex_dble z,w,g[4],h[4];
   dirac_t AP,BP,CP,DP;

   if (NPROC>1)
   {
      iprms[0]=(int)(A);
      iprms[1]=(int)(B);
      iprms[2]=(int)(C);
      iprms[3]=(int)(D);

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=(int)(A))||(iprms[1]!=(int)(B))||
            (iprms[2]!=(int)(C))||(iprms[3]!=(int)(D)),1,
            "cfcts [cfcts4q.c]","Parameter A,B,C or D is not global"); 
   }

   error_root((A<S)||(A>Tt23)||(B<S)||(B>Tt23)||
              (C<S)||(C>Tt23)||(D<S)||(D>Tt23),1,
              "cfcts [cfcts4q.c]","Parameter A,B,C or D is out of range");
   
   AP=AxP(A);
   BP=AxP(B);
   CP=AxP(C);
   DP=AxP(D);

   cmpnts(BP,l,g);
   cmpnts(DP,k,h);

   for (x0=0;x0<(NPROC0*L0);x0++)
   {
      ab[x0].re=0.0;
      ab[x0].im=0.0;
   }

   for (i=0;i<4;i++)
   {
      if ((A==S)||(A==P)||(A>=T01))
      {
         z.re=g[i].re;
         z.im=g[i].im;
      }
      else
      {
         z.re=-g[i].re;
         z.im=-g[i].im;
      }

      for (j=0;j<4;j++)
      {
         if ((C==S)||(C==P)||(C>=T01))
         {
            w.re=h[j].re*z.re-h[j].im*z.im;
            w.im=h[j].im*z.re+h[j].re*z.im;
         }
         else
         {
            w.re=-h[j].re*z.re+h[j].im*z.im;
            w.im=-h[j].im*z.re-h[j].re*z.im;
         }

         for (a=0;a<3;a++)
         {
            for (b=0;b<3;b++)
            {
               slices(AP,CP,*(sk+3*k[j]+a),*(sl+3*i+b),*(skp+3*l[i]+b),*(slp+3*j+a),tslice);
               
               for (x0=0;x0<(NPROC0*L0);x0++)
               {
                  ab[x0].re-=w.re*tslice[x0].re-w.im*tslice[x0].im;
                  ab[x0].im-=w.re*tslice[x0].im+w.im*tslice[x0].re;
               }
            }
         }
      }
   }
}


void cfcts4q2(dirac_t A,dirac_t B,dirac_t C,dirac_t D,spinor_dble **sk,spinor_dble **sl,
                                  spinor_dble **skp,spinor_dble **slp,complex_dble ab[])
{
   int iprms[4],x0,i,j,a,b,k[4],l[4];
   complex_dble z,w,g[4],h[4];
   dirac_t AP,BP,CP,DP;

   if (NPROC>1)
   {
      iprms[0]=(int)(A);
      iprms[1]=(int)(B);
      iprms[2]=(int)(C);
      iprms[3]=(int)(D);

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=(int)(A))||(iprms[1]!=(int)(B))||
            (iprms[2]!=(int)(C))||(iprms[3]!=(int)(D)),1,
            "cfcts [cfcts4q.c]","Parameter A,B,C or D is not global"); 
   }

   error_root((A<S)||(A>Tt23)||(B<S)||(B>Tt23)||
              (C<S)||(C>Tt23)||(D<S)||(D>Tt23),1,
              "cfcts [cfcts4q.c]","Parameter A,B,C or D is out of range");
   
   AP=AxP(A);
   BP=AxP(B);
   CP=AxP(C);
   DP=AxP(D);

   cmpnts(BP,l,g);
   cmpnts(DP,k,h);

   for (x0=0;x0<(NPROC0*L0);x0++)
   {
      ab[x0].re=0.0;
      ab[x0].im=0.0;
   }

   for (i=0;i<4;i++)
   {
      if ((A==S)||(A==P)||(A>=T01))
      {
         z.re=g[i].re;
         z.im=g[i].im;
      }
      else
      {
         z.re=-g[i].re;
         z.im=-g[i].im;
      }

      for (j=0;j<4;j++)
      {
         if ((C==S)||(C==P)||(C>=T01))
         {
            w.re=h[j].re*z.re-h[j].im*z.im;
            w.im=h[j].im*z.re+h[j].re*z.im;
         }
         else
         {
            w.re=-h[j].re*z.re+h[j].im*z.im;
            w.im=-h[j].im*z.re-h[j].re*z.im;
         }

         for (a=0;a<3;a++)
         {
            for (b=0;b<3;b++)
            {
               slices(AP,CP,*(sk+3*l[i]+a),*(sl+3*i+a),*(skp+3*k[j]+b),*(slp+3*j+b),tslice);
               
               for (x0=0;x0<(NPROC0*L0);x0++)
               {
                  ab[x0].re+=w.re*tslice[x0].re-w.im*tslice[x0].im;
                  ab[x0].im+=w.re*tslice[x0].im+w.im*tslice[x0].re;
               }
            }
         }
      }
   }
}

