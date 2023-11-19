
/*******************************************************************************
*
* File gauge_tranforms.c
*
* Copyright (C) 2023 Emanuele Rosi
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Gauge transformations of Gauge and spinor fields
*
*******************************************************************************
*******************************************************************************
*  EXTERN FUNCTIONS
*  
*  extern void generate_g_trnsfrms(void)
*     generates the Gauge trnaformations acting on u-fields.
*     Dynamical memory is allocated.     
*
*  extern void free_g_trnsfrms(void)
*     de-allocates the memory reserved to g-transformations.
*
*  extern void transform_ud(void)
*     transforms the actual Gauge fields throught transformations g.
*     If g is not initialized, it automatically generates the
*     tranformations and free them.
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "forces.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int nfc[8],ofs[8];
static su3_dble *g,*gbuf;

static void pack_gbuf(void)
{
   int n,ix,iy,io;

   nfc[0]=FACE0/2;
   nfc[1]=FACE0/2;
   nfc[2]=FACE1/2;
   nfc[3]=FACE1/2;
   nfc[4]=FACE2/2;
   nfc[5]=FACE2/2;
   nfc[6]=FACE3/2;
   nfc[7]=FACE3/2;

   ofs[0]=0;
   ofs[1]=ofs[0]+nfc[0];
   ofs[2]=ofs[1]+nfc[1];
   ofs[3]=ofs[2]+nfc[2];
   ofs[4]=ofs[3]+nfc[3];
   ofs[5]=ofs[4]+nfc[4];
   ofs[6]=ofs[5]+nfc[5];
   ofs[7]=ofs[6]+nfc[6];

   for (n=0;n<8;n++)
   {
      io=ofs[n];

      for (ix=0;ix<nfc[n];ix++)
      {
         iy=map[io+ix];
         gbuf[io+ix]=g[iy];
      }
   }
}

static void send_gbuf(void)
{
   int n,mu,np,saddr,raddr;
   int nbf,tag;
   su3_dble *sbuf,*rbuf;
   MPI_Status stat;

   for (n=0;n<8;n++)
   {
      nbf=18*nfc[n];

      if (nbf>0)
      {
         tag=mpi_tag();
         mu=n/2;
         np=cpr[mu];

         if (n==(2*mu))
         {
            saddr=npr[n+1];
            raddr=npr[n];
         }
         else
         {
            saddr=npr[n-1];
            raddr=npr[n];
         }

         sbuf=gbuf+ofs[n];
         rbuf=g+ofs[n]+VOLUME;

         if ((np|0x1)!=np)
         {
            MPI_Send((double*)(sbuf),nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
            MPI_Recv((double*)(rbuf),nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,
                     &stat);
         }
         else
         {
            MPI_Recv((double*)(rbuf),nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,
                     &stat);
            MPI_Send((double*)(sbuf),nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
         }
      }
   }
}

static void random_g(void)
{
   su3_dble *gx,*gm;

   gm=g+VOLUME;

   for (gx=g;gx<gm;gx++)
      random_su3_dble(gx);

   if (BNDRY>0)
   {
      pack_gbuf();
      send_gbuf();
   }
}

extern void generate_g_trnsfrms(void)
{
   g=amalloc(NSPIN*sizeof(*g),4);
   if (BNDRY!=0)
       gbuf=amalloc((BNDRY/2)*sizeof(*gbuf),4);
    
   error((g==NULL)||((BNDRY!=0)&&(gbuf==NULL)),1,"generate_g_trnsfrms [uflds_trnsfrm.c]","Unable to allocate auxiliary arrays");

   random_g();
}

extern void free_g_trnsfrms(void)
{
   free(g);
}

extern void g_transform_ud(void)
{
   int ix,iy,mu,allocate_check;
   su3_dble *ub,u,v,w;
   su3_dble gx,gxi,gy,gyi;

   ub=udfld();

   allocate_check=(g==NULL);
   if(allocate_check)
      generate_g_trnsfrms();
   
   for (ix=(VOLUME/2);ix<VOLUME;ix++)
   {
      gx=g[ix];

      for (mu=0;mu<4;mu++)
      {
         iy=iup[ix][mu];
         gy=g[iy];
         u=ub[2*mu];
         _su3_dagger(gyi,gy);
         _su3_times_su3(v,u,gyi);
         _su3_times_su3(w,gx,v);
         ub[2*mu]=w;

         iy=idn[ix][mu];
         gy=g[iy];
         u=ub[2*mu+1];
         _su3_dagger(gxi,gx);
         _su3_times_su3(v,u,gxi);
         _su3_times_su3(w,gy,v);
         ub[2*mu+1]=w;
      }

      ub+=8;
   }

   if(allocate_check)
      free_g_trnsfrms(void);

   set_flags(UPDATED_UD);
}

/*
extern void g_transform_sdble(int volume,spinor_dble *sp){

   su3_vector_dble *vec_iterate,*vec_max;
   int dirac;
   
   for(dirac=0;dirac<4;dirac++)
   {
      switch (dirac)
      {
      case 0:
         vec_iterate=&((*sp).c1);
         break;
      case 1:
         vec_iterate=&((*sp).c2);
         break;
      case 2:
         vec_iterate=&((*sp).c3);
         break;
      case 3:
         vec_iterate=&((*sp).c4);
         break;
      default:
         break;
      }

      vec_max=vec_iterate+volume;

      error(vec_iterate==NULL||vec_max==NULL,1,"g_transform_sdble [gauge_trasnsforms.c]","Unable to allocate su3_vector_dble types.");

      for(;vec_iterate<vec_max;vec_iterate++)
      {
         _su3_multiply(r,u,s)
      }
   }

}*/
