
/*******************************************************************************
*
* File cfcts2q.c
*
* Copyright (C) 2007, 2008, 2009 2013 Martin Luescher, Leonardo Giusti
*                                     Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* General programs for the calculation of correlation functions involving
* 2 fermion interpolators
*
* The externally accessible functions are
*
*   void cfcts2q(dirac_t A,dirac_t B,spinor_dble **sk,spinor_dble **sl,
*                complex_dble ab[])
*     Calculates the time-slice sums ab[x0] at time x0=0,..,NPROC0*L0-1 
*     of the connected part of the correlation function <A_ab(x)B_ba(y)>,
*     assuming the quark propagators S_{a,b}(x,y) for quark flavour "a" 
*     and "b" are stored in the spinor fields sk[i], i=0,..,11, and sl[i],
*     respectively (see the notes below).
*
*   void ctrcts2q(dirac_t A,dirac_t B,spinor_dble *s,spinor_dble *r,
*                complex_dble* ab)
*     Calculates the contraction tr{(S*B^dag)^dag*A*R} between the spinor 
*     matrices S and R, assuming that the corresponding dirac and color
*     components are stored in s[i] and r[i], i=0,...,11, respectively.
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
* where mu=0, nu=1,2,3, and sigma_{mu,nu}=(i/2)*[gamma_mu,gamma_nu]
* (the possible arguments A are thus T01,...,Tt03).
*
* The symbol A_ab(x) stands for psibar_a(x)*A*psi_b(x), where "a" and
* "b" are flavour indices and A any of the known Dirac matrices. The
* calculated two-point correlation functions
*
*  <A_ab(x)B_ba(y)> = -tr{S_a(y,x)*A*S_b(x,y)*B} 
*
* include the overall fermion minus sign.
*
* For any given source point y, Dirac index id=1,2,3,4 and colour index
* ic=1,2,3 at y, the quark propagator S(x,y) is a spinor field (as a
* function of x). The programs in this module assume the 12 fields that
* correspond to all possible combinations of id and ic to be stored in 
* sequences of fields psd[3*(id-1)+ic-1].
*
* Finally, some useful internally accessible functions are given by
*
*   void cmpnts(dirac_t A,int j[],complex_dble g[])
*     For a given Dirac matrix A, this program initializes the array
*     elements j[0],..,j[3] and g[0],..,g[3] in such a way that the 
*     non-zero components of A are given by A_{i,j[i]}=g[i]
*
*   void slices(dirac_t A,spinor_dble *sk,spinor_dble *sl,complex_dble tsl[])
*     Calculates the fixed-time scalar products (sk,A*sl) at all
*     times x0=0,..,NPROC0*L0-1 and assigns them to the elements of the
*     array tsl[]
*
*******************************************************************************/

#define CFCTS2Q_C

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

   if ((A==S)||(A==P)||(A==T03)||(A==Tt03))
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
      else
      {
         g[0].re=1.0;  
         g[1].re=-1.0;
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
   else if ((A==T01)||(A==Tt01)||(A==T02)||(A==Tt02))
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
      else if (A==T02)
      {
         g[0].im=-1.0;
         g[1].im=1.0;
         g[2].im=1.0;
         g[3].im=-1.0;
      }
      else 
      {
         g[0].im=-1.0;
         g[1].im=1.0;
         g[2].im=-1.0;
         g[3].im=1.0;
      }
   }
   else
   {
      j[0]=0;
      j[1]=1;
      j[2]=2;
      j[3]=3;

      error_loc(1,1,"cmpnts [cfcts2q.c]","Unknown Dirac matrix");
   }
}


static void alloc_tms(void)
{
   int ix,iy,x0;

   tms=amalloc(VOLUME*sizeof(int),3);
   lsm=amalloc(NPROC0*L0*sizeof(complex_dble),ALIGN);
   
   error((tms==NULL)||(lsm==NULL),1,"alloc_tms [cfcts2q.c]",
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


static void slices(dirac_t A,spinor_dble *sk,spinor_dble *sl,complex_dble *tsl)
{
   int iprms[1],x0,*t,*tm;
   void (*sp)(spinor_dble *r,spinor_dble *s);

   if (NPROC>1)
   {   
      iprms[0]=(int)(A);
      
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=(int)(A)),1,
            "slices [cfcts2q.c]","Parameters are not global");    
   }
   
   error_root((A<S)||(A>Tt03),1,
              "slices [cfcts2q.c]","Parameters are out of range"); 
   
   if (tms==NULL)
      alloc_tms();
   
   if (A==S)
      sp=sp_S;
   else if (A==P)
      sp=sp_P;
   else if (A==V0)
      sp=sp_V0;
   else if (A==A0)
      sp=sp_A0;
   else if (A==V1)
      sp=sp_V1;
   else if (A==A1)
      sp=sp_A1;
   else if (A==V2)
      sp=sp_V2;
   else if (A==A2)
      sp=sp_A2;
   else if (A==V3)
      sp=sp_V3;
   else if (A==A3)
      sp=sp_A3;
   else if (A==T01)
      sp=sp_T01;
    else if (A==Tt01)
      sp=sp_Tt01;
   else if (A==T02)
      sp=sp_T02;
  else if (A==Tt02)
      sp=sp_Tt02;
   else if (A==T03)
      sp=sp_T03;
   else if (A==Tt03)
      sp=sp_Tt03;
   else 
      sp=NULL;
   
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

      sp(sk,sl);

      lsm[*t].re+=spt.re;
      lsm[*t].im+=spt.im;
      sk+=1;
      sl+=1;
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
   else if (A==Tt01)
      return T01;
   else if (A==Tt02)
      return T02;
   else 
      return T03;
}


void cfcts2q(dirac_t A,dirac_t B,spinor_dble **sk,spinor_dble **sl,complex_dble ab[])
{
   int iprms[2],x0,i,a,j[4];
   complex_dble z,g[4];
   dirac_t AP,BP;

   if (NPROC>1)
   {
      iprms[0]=(int)(A);
      iprms[1]=(int)(B);

      MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=(int)(A))||(iprms[1]!=(int)(B)),1,
            "cfcts2q [cfcts2q.c]","Parameter A or B is not global"); 
   }

   error_root((A<S)||(A>Tt03)||(B<S)||(B>Tt03),1,
              "cfcts2q [cfcts2q.c]","Parameter A or B is out of range");
   
   AP=AxP(A);
   BP=AxP(B);
   cmpnts(BP,j,g);

   for (x0=0;x0<(NPROC0*L0);x0++)
   {
      ab[x0].re=0.0;
      ab[x0].im=0.0;
   }

   for (i=0;i<4;i++)
   {
      if ((A==S)||(A==P)||
          (A==T01)||(A==T02)||(A==T03)||
          (A==Tt01)||(A==Tt02)||(A==Tt03))
      {
         z.re=-g[i].re;
         z.im=-g[i].im;
      }
      else
      {
         z.re=g[i].re;
         z.im=g[i].im;
      }

      for (a=0;a<3;a++)
      {
	      slices(AP,*(sk+3*j[i]+a),*(sl+3*i+a),tslice);
         
         for (x0=0;x0<(NPROC0*L0);x0++)
         {
            ab[x0].re+=z.re*tslice[x0].re-z.im*tslice[x0].im;
            ab[x0].im+=z.re*tslice[x0].im+z.im*tslice[x0].re;
         }
      }
   }
}


void ctrcts2q(dirac_t A,dirac_t B,spinor_dble *s,spinor_dble *r,complex_dble *ab)
{
   int iprms[2];
   int i,a,j[4];
   complex_dble g[4];
   void (*sp)(spinor_dble *r,spinor_dble *s);

   if (NPROC>1)
   {
      iprms[0]=(int)(A);
      iprms[1]=(int)(B);

      MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=(int)(A))||(iprms[1]!=(int)(B)),1,
            "ctrcts2q [cfcts2q.c]","Parameter A or B is not global"); 
   }

   error_root((A<S)||(A>Tt03)||(B<S)||(B>Tt03),1,
              "ctrcts2q [cfcts2q.c]","Parameter A or B is out of range");
 
   if (A==S)
      sp=sp_S;
   else if (A==P)
      sp=sp_P;
   else if (A==V0)
      sp=sp_V0;
   else if (A==A0)
      sp=sp_A0;
   else if (A==V1)
      sp=sp_V1;
   else if (A==A1)
      sp=sp_A1;
   else if (A==V2)
      sp=sp_V2;
   else if (A==A2)
      sp=sp_A2;
   else if (A==V3)
      sp=sp_V3;
   else if (A==A3)
      sp=sp_A3;
   else if (A==T01)
      sp=sp_T01;
    else if (A==Tt01)
      sp=sp_Tt01;
   else if (A==T02)
      sp=sp_T02;
  else if (A==Tt02)
      sp=sp_Tt02;
   else if (A==T03)
      sp=sp_T03;
   else if (A==Tt03)
      sp=sp_Tt03;
   else 
      sp=NULL;
 
   (*ab).re=0.0;
   (*ab).im=0.0;

   cmpnts(B,j,g);

   for (i=0;i<4;i++)
   {
      spt.re=0.0;
      spt.im=0.0;

      for (a=0;a<3;a++)
         sp(s+3*j[i]+a,r+3*i+a);

      (*ab).re+=g[i].re*spt.re-g[i].im*spt.im;
      (*ab).im+=g[i].re*spt.im+g[i].im*spt.re;
   }
}

