
/*******************************************************************************
*
* File xsfcfcts.c
*
* Copyright (C) 2013, 2014 Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* General programs for the calculation of XSF correlation functions
*
* The externally accessible functions are
*
*   void xsfprop(int x0,int tau3,int isp,spinor_dble *sk,int *status)
*     Inverts the XSF Dirac operator with flavour structure defined by
*     tau3 on the boundary source field located at the time slice x0. 
*     The variable x0 can thus only take the values x0=0,NPROC0*L0-1. 
*     The solver is specified by the solver index isp, and the exit 
*     status is returned in status. The 12 components of the XSF propagator,
*     one for each color and dirac index ic and id, are then accessed 
*     as sk[3*(id-1)+(ic-1)].
*
*   void pull_slice_xsf(int x0,int tau3,spinor_dble *s,spinor_dble *r)
*     Applies to the source field s at time x0 the linear transformation
*
*      r(y,x0-1) = U([y,x0-1],0)*Q_-/+*s(y,x0)
*   
*     where the projector structure depends on the flavour flag tau3=+1/-1.
*     The result is stored in the spinor field r at the time slice x0-1. 
*     The other time slices of r instead are set equal to the ones of s.
*     Note that the spinors s and r can be the same, and x0 can only take
*     the values x0>0, and x0<=NPROC0*L0-1.
*
*   void push_slice_xsf(int x0,int tau3,spinor_dble *s,spinor_dble *r)
*     Applies to the source field s at time x0 the linear transformation
*
*      r(y,x0+1) = U([y,x0],0)^dag*Q_+/-*s(y,x0)
*   
*     where the projector structure depends on the flavour flag tau3=+1/-1.
*     The result is stored in the spinor field r at the time slice x0+1. 
*     The other time slices of r instead are set equal to the ones of s. 
*     Note that the spinors s and r can be the same, and x0 can only take
*     the values x0>=0, and x0<NPROC0*L0-1.
*
* Notes:
*
* All programs in this module involve global communications and must be
* called simultaneously on all processes.
*
* Before the program xsfprop() is launched, the lattice, sf and solver
* parameters must be set through set_lat_parms(), set_sf_parms(),
* set_solver_parms() and set_sw_parms().  
*
*******************************************************************************/

#define XSFCFCTS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "cfcts.h"
#include "dfl.h"
#include "flags.h"
#include "lattice.h"
#include "sap.h"
#include "utils.h"
#include "uflds.h"
#include "sflds.h"
#include "su3.h"
#include "dirac.h"
#include "sw_term.h"
#include "forces.h"
#include "linalg.h"
#include "global.h"


static void scale_complex_dble(int vol,double c,spinor_dble *s)
{
   spinor_dble rs,*sm;

   sm=s+vol;
                                                                                                                
   for (;s<sm;s++)
   {
      _vector_imul(rs.c1,c,(*s).c1);
      _vector_imul(rs.c2,c,(*s).c2);
      _vector_imul(rs.c3,c,(*s).c3);                                                                            
      _vector_imul(rs.c4,c,(*s).c4);                                                                            

      (*s)=rs;
   }                                                                                                            
}      

                                     
static void wallsrc_bnd(int x0,int id,int ic,spinor_dble *s)
{
   int ix,t;
   int iprms[3];
   spinor_dble *sm;
   su3_vector_dble *v;

   if (NPROC>1)
   {
      iprms[0]=x0;
      iprms[1]=id;
      iprms[2]=ic;
      MPI_Bcast(iprms,3,MPI_INT,0,MPI_COMM_WORLD);
      error((iprms[0]!=x0)||(iprms[1]!=id)||(iprms[2]!=ic),1,
            "wallsrc_bnd [xsfcfcts.c]","Parameters are not global");   
   }
   
   error_root((id<1)||(id>4)||(ic<1)||(ic>3),1,
              "wallsrc_bnd [xsfcfcts.c]","Improper argument id,ic"); 
   
   error_root((x0!=0)&&(x0!=(NPROC0*L0-1)),1,
            "wallsrc_bnd [xsfcfcts.c]","Improper argument x0");
 
   sm=s+VOLUME;
   set_sd2zero(VOLUME,s);

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=0;
 
      for(;s<sm;s++)
      {
         t=global_time(ix);
         ix+=1;

         if (t==x0)
         {
            if (id==1) 
               v=&((*s).c1);
            else if (id==2) 
               v=&((*s).c2);
            else if (id==3) 
               v=&((*s).c3);
            else 
               v=&((*s).c4);

            if (ic==1) 
               (*v).c1.re=1.0;
            else if (ic==2)
               (*v).c2.re=1.0; 
            else 
               (*v).c3.re=1.0;
         }
      }
   }
}


static void mulQp_su3_dble(su3_dble *u,spinor_dble *s,spinor_dble *r)
{
   spinor_dble rs;
   su3_vector_dble psi;

   _vector_i_add(psi,(*s).c1,(*s).c3);
   _su3_multiply(rs.c1,(*u),psi);
   _vector_imul(rs.c3,-1.0,rs.c1);

   _vector_i_add(psi,(*s).c2,(*s).c4);
   _su3_multiply(rs.c2,(*u),psi);
   _vector_imul(rs.c4,-1.0,rs.c2);

   _vector_mul((*r).c1,0.5,rs.c1);
   _vector_mul((*r).c2,0.5,rs.c2);
   _vector_mul((*r).c3,0.5,rs.c3);
   _vector_mul((*r).c4,0.5,rs.c4);
}


static void mulQm_su3_dble(su3_dble *u,spinor_dble *s,spinor_dble *r)
{
   spinor_dble rs;
   su3_vector_dble psi;

   _vector_i_sub(psi,(*s).c1,(*s).c3);
   _su3_multiply(rs.c1,(*u),psi);
   _vector_imul(rs.c3,1.0,rs.c1);

   _vector_i_sub(psi,(*s).c2,(*s).c4);
   _su3_multiply(rs.c2,(*u),psi);
   _vector_imul(rs.c4,1.0,rs.c2);

   _vector_mul((*r).c1,0.5,rs.c1);
   _vector_mul((*r).c2,0.5,rs.c2);
   _vector_mul((*r).c3,0.5,rs.c3);
   _vector_mul((*r).c4,0.5,rs.c4);
}


static void mulQp_su3dag_dble(su3_dble *u,spinor_dble *s,spinor_dble *r)
{
   spinor_dble rs;
   su3_vector_dble psi;

   _vector_i_add(psi,(*s).c1,(*s).c3);
   _su3_inverse_multiply(rs.c1,(*u),psi);
   _vector_imul(rs.c3,-1.0,rs.c1);

   _vector_i_add(psi,(*s).c2,(*s).c4);
   _su3_inverse_multiply(rs.c2,(*u),psi);
   _vector_imul(rs.c4,-1.0,rs.c2);

   _vector_mul((*r).c1,0.5,rs.c1);
   _vector_mul((*r).c2,0.5,rs.c2);
   _vector_mul((*r).c3,0.5,rs.c3);
   _vector_mul((*r).c4,0.5,rs.c4);
}


static void mulQm_su3dag_dble(su3_dble *u,spinor_dble *s,spinor_dble *r)
{
   spinor_dble rs;
   su3_vector_dble psi;

   _vector_i_sub(psi,(*s).c1,(*s).c3);
   _su3_inverse_multiply(rs.c1,(*u),psi);
   _vector_imul(rs.c3,1.0,rs.c1);

   _vector_i_sub(psi,(*s).c2,(*s).c4);
   _su3_inverse_multiply(rs.c2,(*u),psi);
   _vector_imul(rs.c4,1.0,rs.c2);

   _vector_mul((*r).c1,0.5,rs.c1);
   _vector_mul((*r).c2,0.5,rs.c2);
   _vector_mul((*r).c3,0.5,rs.c3);
   _vector_mul((*r).c4,0.5,rs.c4);
}


void pull_slice_xsf(int x0,int tau3,spinor_dble *s,spinor_dble *r)
{
   int ix,t;
   int iprms[2];
   int *piup,*pidn;
   spinor_dble *so,*ro;
   spinor_dble *sp,*rm;
   su3_dble *u,*um;
  
   if (NPROC>1)
   {
      iprms[0]=x0;
      iprms[1]=tau3;

      MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=x0)||(iprms[1]!=tau3),1,
            "pull_slice_xsf [xsfcfcts.c]","Parameters are not global");   
   }

   error_root(((x0<1)||(x0>(NPROC0*L0-1))),1,
               "pull_slice_xsf [xsfcfcts.c]","Improper argument x0");

   error_root((tau3!=1)&&(tau3!=-1),1,"pull_slice_xsf [xsfcfcts.c]",
              "Flavour tau3 flag set to an improper value");

   so=s+(VOLUME/2);
   ro=r+(VOLUME/2);
   piup=iup[VOLUME/2];
   pidn=idn[VOLUME/2];

   u=udfld();
   um=u+4*VOLUME;

   assign_sd2sd(VOLUME/2,s,r);
   ix=VOLUME/2;

   for (;u<um;u+=8)
   {
      t=global_time(ix);
      ix+=1;

      if (t==x0)
      {
         rm=r+(*pidn);

         if (tau3==1)
            mulQp_su3_dble(u+1,so,rm);
         else
            mulQm_su3_dble(u+1,so,rm);

         (*ro)=(*so);   
      }
      else if (t==(x0-1))
      {
         sp=s+(*piup);

         if (tau3==1)
            mulQp_su3_dble(u,sp,ro);
         else
            mulQm_su3_dble(u,sp,ro);
      }
      else
         (*ro)=(*so);   

      so+=1;
      ro+=1;
      piup+=4;
      pidn+=4;
   }
}


void push_slice_xsf(int x0,int tau3,spinor_dble *s,spinor_dble *r)
{
   int ix,t;
   int iprms[2];
   int *piup,*pidn;
   spinor_dble *so,*ro;
   spinor_dble *sm,*rp;
   su3_dble *u,*um;
  
   if (NPROC>1)
   {
      iprms[0]=x0;
      iprms[1]=tau3;

      MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=x0)||(iprms[1]!=tau3),1,
            "push_slice_xsf [xsfcfcts.c]","Parameters are not global");   
   }

   error_root(((x0<0)||(x0>(NPROC0*L0-2))),1,
               "push_slice_xsf [xsfcfcts.c]","Improper argument x0");

   error_root((tau3!=1)&&(tau3!=-1),1,"push_slice_xsf [xsfcfcts.c]",
              "Flavour tau3 flag set to an improper value");

   so=s+(VOLUME/2);
   ro=r+(VOLUME/2);
   piup=iup[VOLUME/2];
   pidn=idn[VOLUME/2];

   u=udfld();
   um=u+4*VOLUME;

   assign_sd2sd(VOLUME/2,s,r);
   ix=VOLUME/2;

   for (;u<um;u+=8)
   {
      t=global_time(ix);
      ix+=1;

      if (t==x0)
      {
         rp=r+(*piup);

         if (tau3==1)
            mulQm_su3dag_dble(u,so,rp);
         else
            mulQp_su3dag_dble(u,so,rp);

         (*ro)=(*so);
      }
      else if (t==(x0+1))
      {
         sm=s+(*pidn);

         if (tau3==1)
            mulQm_su3dag_dble(u+1,sm,ro);
         else
            mulQp_su3dag_dble(u+1,sm,ro);
      }
      else
         (*ro)=(*so);

      so+=1;
      ro+=1;
      piup+=4;
      pidn+=4;
   }
}


void xsfprop(int x0,int tau3,int isp,spinor_dble **sk,int *status)
{
   int iprms[3];
   int ic,id,stat;
   double tau3d;
   solver_parms_t sp;
   spinor_dble **wsd,**psd;
   sf_parms_t sf;

   sf=sf_parms();
   error_root((sf.flg!=1),1,"xsfprop [xsfcfcts.c]",
              "SF boundary values are not set");
  
   if (NPROC>1)
   {
      iprms[0]=x0;
      iprms[1]=tau3;
      iprms[2]=isp;

      MPI_Bcast(iprms,3,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=x0)||(iprms[1]!=tau3)||(iprms[2]!=isp),1,
            "xsfprop [xsfcfcts.c]","Parameters are not global");   
   }
 
   error_root(((x0!=0)&&(x0!=(NPROC0*L0-1))),1,"xsfprop [xsfcfcts.c]",
             "Improper argument x0");

   error_root((tau3!=1)&&(tau3!=-1),1,"xsfprop [xsfcfcts.c]",
              "Flavour tau3 flag set to an improper value");

   stat=0;
   tau3d=(double)(tau3);
   sp=solver_parms(isp);
   wsd=reserve_wsd(1);

   for (id=1;id<=2;id++)
   {
      for (ic=1;ic<=3;ic++)
      {
         psd=sk+3*(id-1)+(ic-1);
         wallsrc_bnd(x0,id,ic,wsd[0]);

         if(x0==0)
            push_slice_xsf(x0,tau3,wsd[0],wsd[0]); 
         else 
            pull_slice_xsf(x0,tau3,wsd[0],wsd[0]); 

         bnd_sd2zero(ALL_PTS,wsd[0]);

         if (sp.solver==CGNE)
         {
            mulg5_dble(VOLUME,wsd[0]);
            tmcg_xsf(sp.nmx,sp.res,-tau3,0.0,wsd[0],wsd[0],&stat);
            set_xsf_parms(-tau3);
            Dw_xsf_dble(0.0,wsd[0],(*psd));
            mulg5_dble(VOLUME,(*psd));
         }
         else
            error_root(1,1,"xsfprop [xsfcfcts.c]","Unknown solver only CG is supported");

         assign_sd2sd(VOLUME,psd[0],psd[6]); 

         if (x0==0)
            scale_complex_dble(VOLUME,-tau3d,psd[6]);            
         else 
            scale_complex_dble(VOLUME, tau3d,psd[6]);

         stat+=stat/6;         
      }
   }

   status[0]=stat;
   release_wsd();
}


void xsfpropeo(int x0,int tau3,int isp,spinor_dble **sk,int *status)
{
   int ifail,iprms[3];
   int ic,id,stat;
   double tau3d;
   solver_parms_t sp;
   spinor_dble **wsd,**psd;
   sf_parms_t sf;

   sf=sf_parms();
   error_root((sf.flg!=1),1,"xsfpropeo [xsfcfcts.c]",
              "SF boundary values are not set");
  
   if (NPROC>1)
   {
      iprms[0]=x0;
      iprms[1]=tau3;
      iprms[2]=isp;

      MPI_Bcast(iprms,3,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=x0)||(iprms[1]!=tau3)||(iprms[2]!=isp),1,
            "xsfpropeo [xsfcfcts.c]","Parameters are not global");   
   }
 
   error_root(((x0!=0)&&(x0!=(NPROC0*L0-1))),1,"xsfpropeo [xsfcfcts.c]",
             "Improper argument x0");

   error_root((tau3!=1)&&(tau3!=-1),1,"xsfpropeo [xsfcfcts.c]",
              "Flavour tau3 flag set to an improper value");

   ifail=sw_term_xsf(ODD_PTS);
   error_root(ifail!=0,1,"xsfpropeo [xsfcfcts.c]",
              "Inversion of the SW term was not safe");

   stat=0;
   tau3d=(double)(tau3);
   sp=solver_parms(isp);
   wsd=reserve_wsd(2);

   for (id=1;id<=2;id++)
   {
      for (ic=1;ic<=3;ic++)
      {
         psd=sk+3*(id-1)+(ic-1);
         wallsrc_bnd(x0,id,ic,wsd[0]);

         if(x0==0)
            push_slice_xsf(x0,tau3,wsd[0],wsd[0]); 
         else 
            pull_slice_xsf(x0,tau3,wsd[0],wsd[0]); 

         bnd_sd2zero(ALL_PTS,wsd[0]);

         if (sp.solver==CGNE)
         {
            set_xsf_parms(tau3);
            Dwoo_xsf_dble(0.0,wsd[0],wsd[0]);            
            Dweo_xsf_dble(wsd[0],wsd[0]);            
            mulg5_dble(VOLUME/2,wsd[0]);

            tmcgeo_xsf(sp.nmx,sp.res,-tau3,0.0,wsd[0],wsd[1],&stat);
            set_xsf_parms(-tau3);
            Dwhat_xsf_dble(0.0,wsd[1],(*psd));
            mulg5_dble(VOLUME/2,(*psd));

            set_xsf_parms(tau3);
            Dwoe_xsf_dble((*psd),(*psd));            
            Dwoo_xsf_dble(0.0,(*psd),(*psd));            

            combine_spinor_dble(VOLUME/2,(*psd)+(VOLUME/2),wsd[0]+(VOLUME/2),-1.0,1.0);
         }
         else
            error_root(1,1,"xsfpropeo [xsfcfcts.c]","Unknown solver only CG is supported");

         assign_sd2sd(VOLUME,psd[0],psd[6]); 

         if (x0==0)
            scale_complex_dble(VOLUME,-tau3d,psd[6]);            
         else 
            scale_complex_dble(VOLUME, tau3d,psd[6]);

         stat+=stat/6;         
      }
   }

   status[0]=stat;
   release_wsd();
}

