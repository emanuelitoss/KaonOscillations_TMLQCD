
/**********************************************************************************
*
* File ms6_xsf.c
*
* Copyright (C) 2015 Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the xSF correlation functions involving bulk 4-fermion interpolators
* with Dirac structure: VA,AV,SP,PS,TTt, and boundary source fields Qcal5, and Qcalk.
* The flavour combinations considered are: uuud,uudu,uduu,duuu.
*
* Syntax: ms6_xsf -i <input file> [-noexp]
*
* For usage instructions see the file README.ms6_xsf
*
**********************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "sflds.h"
#include "uflds.h"
#include "linalg.h"
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
#include "cfcts.h"
#include "forces.h"
#include "version.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

#define MAX(n,m) \
   if ((n)<(m)) \
      (n)=(m)

static struct
{
   int tmax;
   double kappa,csw,dF,zF;
} file_head;

static struct
{
   int nc;
   complex_dble *GVA2,*GAV2,*GSP2,*GPS2,*GTTt2;
   complex_dble *GVA1,*GAV1,*GSP1,*GPS1,*GTTt1;
   complex_dble *LVA2,*LAV2,*LSP2,*LPS2,*LTTt2;
   complex_dble *LVA1,*LAV1,*LSP1,*LPS1,*LTTt1;
} data;

static int my_rank,noexp,append,endian;
static int first,last,step;
static complex_dble GXY[N0];

static char line[NAME_SIZE];
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char loc_dir[NAME_SIZE],cnfg_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE],end_file[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char dat_file_uuud[NAME_SIZE],dat_save_uuud[NAME_SIZE];
static char dat_file_uudu[NAME_SIZE],dat_save_uudu[NAME_SIZE];
static char dat_file_uduu[NAME_SIZE],dat_save_uduu[NAME_SIZE];
static char dat_file_duuu[NAME_SIZE],dat_save_duuu[NAME_SIZE];
static char cnfg_file[NAME_SIZE],nbase[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fdat=NULL,*fend=NULL;


static void alloc_data(void)
{
   int tmax;
   complex_dble *p;

   tmax=file_head.tmax;

   p=amalloc((20*tmax)*sizeof(*p),4);
   
   error((p==NULL),1,"alloc_data [ms6_xsf.c]",
         "Unable to allocate data arrays");

   data.GVA2 =p;
   data.GVA1 =p+tmax;
   data.GAV2 =p+2*tmax;
   data.GAV1 =p+3*tmax;
   data.GSP2 =p+4*tmax;
   data.GSP1 =p+5*tmax;
   data.GPS2 =p+6*tmax;
   data.GPS1 =p+7*tmax;
   data.GTTt2=p+8*tmax;
   data.GTTt1=p+9*tmax;

   data.LVA2 =p+10*tmax;
   data.LVA1 =p+11*tmax;
   data.LAV2 =p+12*tmax;
   data.LAV1 =p+13*tmax;
   data.LSP2 =p+14*tmax;
   data.LSP1 =p+15*tmax;
   data.LPS2 =p+16*tmax;
   data.LPS1 =p+17*tmax;
   data.LTTt2=p+18*tmax;
   data.LTTt1=p+19*tmax;
}


static void write_file_head(void)
{
   int iw;
   double dstd[4];   
   stdint_t istd[1];

   dstd[0]=file_head.kappa;
   dstd[1]=file_head.csw;
   dstd[2]=file_head.dF;
   dstd[3]=file_head.zF;
   istd[0]=(stdint_t)(file_head.tmax);   
   
   if (endian==BIG_ENDIAN)
   {
      bswap_double(4,dstd);
      bswap_int(1,istd);
   }
   
   iw=fwrite(dstd,sizeof(double),4,fdat);   
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   error_root(iw!=5,1,"write_file_head [ms6_xsf.c]",
              "Incorrect write count");
}


static void check_file_head(void)
{
   int ir;
   double dstd[4];
   stdint_t istd[1];

   ir=fread(dstd,sizeof(double),4,fdat);   
   ir+=fread(istd,sizeof(stdint_t),1,fdat);

   error_root(ir!=5,1,"check_file_head [ms6_xsf.c]",
              "Incorrect read count");
   
   if (endian==BIG_ENDIAN)
   {
      bswap_double(4,dstd);      
      bswap_int(1,istd);
   }
   
   error_root((dstd[0]!=file_head.kappa)||
              (dstd[1]!=file_head.csw)||
              (dstd[2]!=file_head.dF)||
              (dstd[3]!=file_head.zF)||
              ((int)(istd[0])!=file_head.tmax),1,"check_file_head [ms6_xsf.c]",
              "Unexpected value of kappa,csw,dF,zF,tmax or bnd");
}


static void write_data(void)
{
   int iw,t,tmax;
   stdint_t istd[1];
   double dstd[1];   

   istd[0]=(stdint_t)(data.nc);

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   iw=fwrite(istd,sizeof(stdint_t),1,fdat);

   tmax=file_head.tmax;   

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.GVA2[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.GVA2[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.GVA1[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.GVA1[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.GAV2[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.GAV2[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.GAV1[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.GAV1[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.GSP2[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.GSP2[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)   
   {
      dstd[0]=data.GSP1[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.GSP1[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.GPS2[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.GPS2[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.GPS1[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.GPS1[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.GTTt2[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.GTTt2[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.GTTt1[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.GTTt1[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.LVA2[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.LVA2[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.LVA1[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.LVA1[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.LAV2[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.LAV2[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.LAV1[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.LAV1[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.LSP2[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.LSP2[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)   
   {
      dstd[0]=data.LSP1[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.LSP1[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.LPS2[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.LPS2[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.LPS1[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.LPS1[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.LTTt2[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.LTTt2[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.LTTt1[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.LTTt1[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   error_root(iw!=(1+40*tmax),1,"write_data [ms6_xsf.c]",
              "Incorrect write count");
}


static int read_data(void)
{
   int ir,t,tmax;
   stdint_t istd[1];
   double dstd[1];
   
   ir=fread(istd,sizeof(stdint_t),1,fdat);

   if (ir!=1)
      return 0;

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   
   data.nc=(int)(istd[0]);

   tmax=file_head.tmax;

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GVA2[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GVA2[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GVA1[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GVA1[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GAV2[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GAV2[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GAV1[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GAV1[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GSP2[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GSP2[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GSP1[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GSP1[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GPS2[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GPS2[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GPS1[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GPS1[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GTTt2[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GTTt2[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GTTt1[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.GTTt1[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LVA2[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LVA2[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LVA1[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LVA1[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LAV2[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LAV2[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LAV1[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LAV1[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LSP2[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LSP2[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LSP1[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LSP1[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LPS2[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LPS2[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LPS1[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LPS1[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LTTt2[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LTTt2[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LTTt1[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.LTTt1[t].im=dstd[0];
   }

   error_root(ir!=(1+40*tmax),1,"read_data [ms6_xsf.c]",
              "Read error or incomplete data record");

   return 1;
}


static void read_dirs(void)
{
   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

      find_section("Directories");
      read_line("log_dir","%s",log_dir);
      read_line("dat_dir","%s",dat_dir);

      if (noexp)
      {
         read_line("loc_dir","%s",loc_dir);
         cnfg_dir[0]='\0';
      }
      else
      {
         read_line("cnfg_dir","%s",cnfg_dir);         
         loc_dir[0]='\0';
      }

      find_section("Configurations");
      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);

      error_root((last<first)||(step<1)||(((last-first)%step)!=0),1,
                 "read_dirs [ms6_xsf.c]","Improper configuration range");
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(dat_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(loc_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   
   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);   
}


static void setup_files(void)
{
   if (noexp)
      error_root(name_size("%s/%sn%d_%d",loc_dir,nbase,last,NPROC-1)>=NAME_SIZE,
                 1,"setup_files [ms6_xsf.c]","loc_dir name is too long");
   else
      error_root(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,
                 1,"setup_files [ms6_xsf.c]","cnfg_dir name is too long");

   check_dir_root(log_dir);   
   check_dir_root(dat_dir);
   error_root(name_size("%s/%s.ms6_xsf.log~",log_dir,nbase)>=NAME_SIZE,
              1,"setup_files [ms6_xsf.c]","log_dir name is too long");
   error_root(name_size("%s/%s.ms6_xsf_xxxx.dat~",dat_dir,nbase)>=NAME_SIZE,
              1,"setup_files [ms6_xsf.c]","dat_dir name is too long");   

   sprintf(log_file,"%s/%s.ms6_xsf.log",log_dir,nbase);
   sprintf(par_file,"%s/%s.ms6_xsf.par",dat_dir,nbase);   
   sprintf(dat_file_uuud,"%s/%s.ms6_xsf_uuud.dat",dat_dir,nbase);
   sprintf(dat_file_uudu,"%s/%s.ms6_xsf_uudu.dat",dat_dir,nbase);
   sprintf(dat_file_uduu,"%s/%s.ms6_xsf_uduu.dat",dat_dir,nbase);
   sprintf(dat_file_duuu,"%s/%s.ms6_xsf_duuu.dat",dat_dir,nbase);
   sprintf(end_file,"%s/%s.ms6_xsf.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);   
   sprintf(dat_save_uuud,"%s~",dat_file_uuud);
   sprintf(dat_save_uudu,"%s~",dat_file_uudu);
   sprintf(dat_save_uduu,"%s~",dat_file_uduu);
   sprintf(dat_save_duuu,"%s~",dat_file_duuu);
}


static void read_sf_parms(void) 
{
   double phi[7];

   if (my_rank==0)
   {
      find_section("Boundary values");
      read_dprms("phi",2,phi);
      read_dprms("phi'",2,phi+2);
      read_dprms("theta",3,phi+4);
   }

   MPI_Bcast(phi,7,MPI_DOUBLE,0,MPI_COMM_WORLD);   

   set_sf_parms(phi,phi+2,phi+4);

   if (append)
      check_sf_parms(fdat);      
   else
      write_sf_parms(fdat);
}


static void read_lat_parms(void)
{
   int ie,ir,iw;
   double kappa,csw,dF,zF;
   double dstd[4];

   if (my_rank==0)
   {
      find_section("Dirac operator");
      read_line("kappa","%lf",&kappa);
      read_line("csw","%lf",&csw);
      read_line("dF","%lf",&dF);   
      read_line("zF","%lf",&zF);   
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&dF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&zF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   set_lat_parms(0.0,1.0,kappa,0.0,0.0,csw,1.0,1.0,dF,zF);
   set_sw_parms(sea_quark_mass(0));

   file_head.kappa=kappa;
   file_head.csw=csw;
   file_head.dF=dF;
   file_head.zF=zF;
   file_head.tmax=N0;

   if (my_rank==0)
   {
      if (append)
      {
         ir=fread(dstd,sizeof(double),4,fdat);
         error_root(ir!=4,1,"read_lat_parms [ms6_xsf.c]",
                    "Incorrect read count");         

         if (endian==BIG_ENDIAN)
            bswap_double(4,dstd);

         ie=0;
         ie|=(dstd[0]!=kappa);
         ie|=(dstd[1]!=csw);
         ie|=(dstd[2]!=dF);
         ie|=(dstd[3]!=zF);

         error_root(ie!=0,1,"read_lat_parms [ms6_xsf.c]",
                    "Parameters do not match previous run");
      }
      else
      {
         dstd[0]=kappa;
         dstd[1]=csw;         
         dstd[2]=dF;         
         dstd[3]=zF;         

         if (endian==BIG_ENDIAN)
            bswap_double(4,dstd);

         iw=fwrite(dstd,sizeof(double),4,fdat);
         error_root(iw!=4,1,"read_lat_parms [ms6_xsf.c]",
                    "Incorrect write count");
      }
   }
}


static void read_sap_parms(void)
{
   int bs[4];

   if (my_rank==0)
   {
      find_section("SAP");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   set_sap_parms(bs,1,4,5);
}


static void read_dfl_parms(void)
{
   int bs[4],Ns;
   int ninv,nmr,ncy,nkv,nmx;
   double kappa,mu,res,resd;

   if (my_rank==0)
   {
      find_section("Deflation subspace");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      read_line("Ns","%d",&Ns);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);   
   set_dfl_parms(bs,Ns);
   
   if (my_rank==0)
   {
      find_section("Deflation subspace generation");
      read_line("kappa","%lf",&kappa);
      read_line("mu","%lf",&mu);
      read_line("ninv","%d",&ninv);     
      read_line("nmr","%d",&nmr);
      read_line("ncy","%d",&ncy);
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);           
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&ninv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_gen_parms(kappa,mu,ninv,nmr,ncy,nkv,nmx,res);
   
   if (my_rank==0)
   {
      find_section("Deflation projectors");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);           
      read_line("resd","%lf",&resd);
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&resd,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_pro_parms(nkv,nmx,resd,res);
}


static void read_solver(void)
{
   solver_parms_t sp;

   read_solver_parms(0);
   sp=solver_parms(0);

   if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
      read_sap_parms();
      
   if (sp.solver==DFL_SAP_GCR)
      read_dfl_parms();
}


static void read_infile(int argc,char *argv[])
{
   int ifile;

   if (my_rank==0)
   {
      flog=freopen("STARTUP_ERROR","w",stdout);
 
      ifile=find_opt(argc,argv,"-i");      
      endian=endianness();

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [ms6_xsf.c]",
                 "Syntax: ms6_xsf -i <input file> [-noexp]");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms6_xsf.c]",
                 "Machine has unknown endianness");

      noexp=find_opt(argc,argv,"-noexp");      
      append=find_opt(argc,argv,"-a");
      
      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ms6_xsf.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&noexp,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&append,1,MPI_INT,0,MPI_COMM_WORLD);
   
   read_dirs();
   setup_files();

   if (my_rank==0)
   {
      if (append)
         fdat=fopen(par_file,"rb");
      else
         fdat=fopen(par_file,"wb");

      error_root(fdat==NULL,1,"read_infile [ms6_xsf.c]",
                 "Unable to open parameter file");
   }

   read_lat_parms();
   read_sf_parms();
   read_solver();

   if (my_rank==0)
   {
      fclose(fin);
      fclose(fdat);

      if (append==0)
         copy_file(par_file,par_save);
   }
}


static void check_old_log(int *fst,int *lst,int *stp)
{
   int ie,ic,isv;
   int fc,lc,dc,pc;
   
   fend=fopen(log_file,"r");
   error_root(fend==NULL,1,"check_old_log [ms6_xsf.c]",
              "Unable to open log file");

   fc=0;
   lc=0;
   dc=0;
   pc=0;

   ie=0x0;
   ic=0;      
   isv=0;
         
   while (fgets(line,NAME_SIZE,fend)!=NULL)
   {
      if (strstr(line,"fully processed")!=NULL)
      {
         pc=lc;
         
         if (sscanf(line,"Configuration no %d",&lc)==1)
         {
            ic+=1;
            isv=1;
         }
         else
            ie|=0x1;
         
         if (ic==1)
            fc=lc;
         else if (ic==2)
            dc=lc-fc;
         else if ((ic>2)&&(lc!=(pc+dc)))
            ie|=0x2;
      }
      else if (strstr(line,"Configuration no")!=NULL)
         isv=0;
   }

   fclose(fend);

   error_root((ie&0x1)!=0x0,1,"check_old_log [ms6_xsf.c]",
              "Incorrect read count");   
   error_root((ie&0x2)!=0x0,1,"check_old_log [ms6_xsf.c]",
              "Configuration numbers are not equally spaced");
   error_root(isv==0,1,"check_old_log [ms6_xsf.c]",
              "Log file extends beyond the last configuration save");

   (*fst)=fc;
   (*lst)=lc;
   (*stp)=dc;
}


static void check_old_dat(int fst,int lst,int stp,char *dat_file)
{
   int ie,ic;
   int fc,lc,dc,pc;
   
   fdat=fopen(dat_file,"rb");
   error_root(fdat==NULL,1,"check_old_dat [ms6_xsf.c]",
              "Unable to open data file");

   check_file_head();

   fc=0;
   lc=0;
   dc=0;
   pc=0;

   ie=0x0;
   ic=0;

   while (read_data()==1)
   {
      pc=lc;
      lc=data.nc;
      ic+=1;
      
      if (ic==1)
         fc=lc;
      else if (ic==2)
         dc=lc-fc;
      else if ((ic>2)&&(lc!=(pc+dc)))
         ie|=0x1;
   }
   
   fclose(fdat);

   error_root(ic==0,1,"check_old_dat [ms6_xsf.c]",
              "No data records found");
   error_root((ie&0x1)!=0x0,1,"check_old_dat [ms6_xsf.c]",
              "Configuration numbers are not equally spaced");
   error_root((fst!=fc)||(lst!=lc)||(stp!=dc),1,"check_old_dat [ms6_xsf.c]",
              "Configuration range is not as reported in the log file");
}


static void check_files(void)
{
   int fst,lst,stp;
   
   if (my_rank==0)
   {
      if (append)
      {
         check_old_log(&fst,&lst,&stp);

         check_old_dat(fst,lst,stp,dat_file_uuud);
         check_old_dat(fst,lst,stp,dat_file_uudu);
         check_old_dat(fst,lst,stp,dat_file_uduu);
         check_old_dat(fst,lst,stp,dat_file_duuu);

         error_root((fst!=lst)&&(stp!=step),1,"check_files [ms6_xsf.c]",
                    "Continuation run:\n"
                    "Previous run had a different configuration separation");
         error_root(first!=lst+step,1,"check_files [ms6_xsf.c]",
                    "Continuation run:\n"
                    "Configuration range does not continue the previous one");
      }
      else
      {
         fin=fopen(log_file,"r");

         fdat=fopen(dat_file_uuud,"rb");
         error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms6_xsf.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file_uudu,"rb");
         error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms6_xsf.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file_uduu,"rb");
         error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms6_xsf.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file_duuu,"rb");
         error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms6_xsf.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file_uuud,"wb");
         error_root(fdat==NULL,1,"check_files [ms6_xsf.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);

         fdat=fopen(dat_file_uudu,"wb");
         error_root(fdat==NULL,1,"check_files [ms6_xsf.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);

         fdat=fopen(dat_file_uduu,"wb");
         error_root(fdat==NULL,1,"check_files [ms6_xsf.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);

         fdat=fopen(dat_file_duuu,"wb");
         error_root(fdat==NULL,1,"check_files [ms6_xsf.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);
      }
   }
}


static void print_info(void)
{
   int isap,idfl;
   long ip;   
   
   if (my_rank==0)
   {
      ip=ftell(flog);
      fclose(flog);

      if (ip==0L)
         remove("STARTUP_ERROR");
      
      if (append)
         flog=freopen(log_file,"a",stdout);
      else
         flog=freopen(log_file,"w",stdout);

      error_root(flog==NULL,1,"print_info [ms6_xsf.c]","Unable to open log file");
      printf("\n");

      if (append)
         printf("Continuation run\n\n");
      else
      {
         printf("Computation of xSF correlation functions\n");         
         printf("----------------------------------\n\n");
      }

      printf("Program version %s\n",openQCD_RELEASE);         

      if (endian==LITTLE_ENDIAN)
         printf("The machine is little endian\n");
      else
         printf("The machine is big endian\n");
      if (noexp)
         printf("Configurations are read in imported file format\n\n");
      else
         printf("Configurations are read in exported file format\n\n");
         
      if (append==0)
      {
         printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
         printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
         printf("%dx%dx%dx%d local lattice\n",L0,L1,L2,L3);

         printf("Chirally rotated Schroedinger functional boundary conditions\n\n");
         print_sf_parms();
          
         printf("Dirac operator:\n");
         printf("kappa = %.6f\n",file_head.kappa);
         printf("csw = %.6f\n",file_head.csw);      
         printf("dF = %.6f\n",file_head.dF);
         printf("zF = %.6f\n\n",file_head.zF);

         print_solver_parms(&isap,&idfl);

         if (isap)
            print_sap_parms(0);

         if (idfl)
            print_dfl_parms(0);
      }

      printf("Configurations no %d -> %d in steps of %d\n\n",
             first,last,step);      
      fflush(flog);
   }
}


static void dfl_wsize(int *nws,int *nwv,int *nwvd)
{
   dfl_parms_t dp;
   dfl_pro_parms_t dpp;
   dfl_gen_parms_t dgp;

   dp=dfl_parms();
   dpp=dfl_pro_parms();
   dgp=dfl_gen_parms();

   MAX(*nws,dp.Ns+2);
   MAX(*nwv,2*dpp.nkv+2);
   MAX(*nwv,2*dgp.nkv+2);
   MAX(*nwvd,4);
}


static void wsize(int *nws,int *nwsd,int *nwv,int *nwvd)
{
   int nsd;
   solver_parms_t sp;

   (*nws)=0;
   (*nwsd)=0;
   (*nwv)=0;
   (*nwvd)=0;

   sp=solver_parms(0);
   nsd=2;

   if (sp.solver==CGNE)
   {
      MAX(*nws,5);
      MAX(*nwsd,nsd+3);
   }
   else if (sp.solver==SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+1);
      MAX(*nwsd,nsd+2);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+1);      
      MAX(*nwsd,nsd+4);
      dfl_wsize(nws,nwv,nwvd);
   }
   else
      error_root(1,1,"wsize [ms6_xsf.c]",
                 "Unknown or unsupported solver");   
}


static void save_data(char *dat_file)
{
   if (my_rank==0)
   {
      fdat=fopen(dat_file,"ab");
      error_root(fdat==NULL,1,"save_data [ms6_xsf.c]",
                 "Unable to open data file");
      write_data();
      fclose(fdat);
   }
}


static void check_endflag(int *iend)
{
   if (my_rank==0)
   {
      fend=fopen(end_file,"r");

      if (fend!=NULL)
      {
         fclose(fend);
         remove(end_file);
         (*iend)=1;
         printf("End flag set, run stopped\n\n");
      }
      else
         (*iend)=0;
   }

   MPI_Bcast(iend,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void print_log(char flv[])
{
   int t,tmax;

   tmax=file_head.tmax;

   if (my_rank==0)
   {
      printf("\n#### GVA2_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.GVA2[t].re,data.GVA2[t].im);

      printf("\n#### GVA1_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.GVA1[t].re,data.GVA1[t].im);

      printf("\n#### GAV2_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.GAV2[t].re,data.GAV2[t].im);

      printf("\n#### GAV1_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.GAV1[t].re,data.GAV1[t].im);

      printf("\n#### GSP2_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.GSP2[t].re,data.GSP2[t].im);

      printf("\n#### GSP1_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.GSP1[t].re,data.GSP1[t].im);

      printf("\n#### GPS2_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.GPS2[t].re,data.GPS2[t].im);

      printf("\n#### GPS1_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.GPS1[t].re,data.GPS1[t].im);

      printf("\n#### GTTt2_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.GTTt2[t].re,data.GTTt2[t].im);

      printf("\n#### GTTt1_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.GTTt1[t].re,data.GTTt1[t].im);

      printf("\n#### LVA2_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.LVA2[t].re,data.LVA2[t].im);

      printf("\n#### LVA1_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.LVA1[t].re,data.LVA1[t].im);

      printf("\n#### LAV2_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.LAV2[t].re,data.LAV2[t].im);

      printf("\n#### LAV1_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.LAV1[t].re,data.LAV1[t].im);

      printf("\n#### LSP2_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.LSP2[t].re,data.LSP2[t].im);

      printf("\n#### LSP1_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.LSP1[t].re,data.LSP1[t].im);

      printf("\n#### LPS2_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.LPS2[t].re,data.LPS2[t].im);

      printf("\n#### LPS1_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.LPS1[t].re,data.LPS1[t].im);

      printf("\n#### LTTt2_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.LTTt2[t].re,data.LTTt2[t].im);

      printf("\n#### LTTt1_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.LTTt1[t].re,data.LTTt1[t].im);
   }
}


static void xsfcfcts(int nc,int *status)
{
   int i,k,l;
   int x0,t,tmax;
   double nrm;
   spinor_dble **usd,**dsd;
   spinor_dble **upsd,**dpsd;

   usd=reserve_wsd(12);
   dsd=reserve_wsd(12);

   upsd=reserve_wsd(12);
   dpsd=reserve_wsd(12);

   data.nc=nc;
   tmax=file_head.tmax;
   nrm=(double)(N1*N2*N3);

   for (i=0;i<4;i++)
      status[i]=0;

   mult_phase(1);

   x0=0;
   xsfpropeo(x0, 1,0,usd,status);
   xsfpropeo(x0,-1,0,dsd,status);

   x0=N0-1;
   xsfpropeo(x0, 1,0,upsd,status);
   xsfpropeo(x0,-1,0,dpsd,status);

   mult_phase(-1);

   /* O5uu QXuuud O5du */
   for(t=0;t<tmax;t++)
   {
      data.GVA2[t].re=0.0;
      data.GVA2[t].im=0.0;
      data.GVA1[t].re=0.0;
      data.GVA1[t].im=0.0;

      data.GAV2[t].re=0.0;
      data.GAV2[t].im=0.0;
      data.GAV1[t].re=0.0;
      data.GAV1[t].im=0.0;

      data.GSP2[t].re=0.0;
      data.GSP2[t].im=0.0;
      data.GSP1[t].re=0.0;
      data.GSP1[t].im=0.0;

      data.GPS2[t].re=0.0;
      data.GPS2[t].im=0.0;
      data.GPS1[t].re=0.0;
      data.GPS1[t].im=0.0;

      data.GTTt2[t].re=0.0;
      data.GTTt2[t].im=0.0;
      data.GTTt1[t].re=0.0;
      data.GTTt1[t].im=0.0;
   }

   for (i=0; i<4; i++)
   {
      /* VA */ 
      cfcts4q2(V0+i,A0,A0+i,P,dpsd,upsd,dsd,dsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GVA2[t].re-=GXY[t].re/nrm;
         data.GVA2[t].im-=GXY[t].im/nrm;
      }
 
      cfcts4q1(V0+i,P,A0+i,A0,dpsd,dsd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GVA1[t].re-=GXY[t].re/nrm;
         data.GVA1[t].im-=GXY[t].im/nrm;
      }
   
      /* AV */ 
      cfcts4q2(A0+i,A0,V0+i,P,dpsd,upsd,dsd,dsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GAV2[t].re-=GXY[t].re/nrm;
         data.GAV2[t].im-=GXY[t].im/nrm;
      }

      cfcts4q1(A0+i,P,V0+i,A0,dpsd,dsd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GAV1[t].re-=GXY[t].re/nrm;
         data.GAV1[t].im-=GXY[t].im/nrm;
      }
   }

   /* SP */
   cfcts4q2(S,A0,P,P,dpsd,upsd,dsd,dsd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GSP2[t].re-=GXY[t].re/nrm;
      data.GSP2[t].im-=GXY[t].im/nrm;
   }

   cfcts4q1(S,P,P,A0,dpsd,dsd,dsd,upsd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GSP1[t].re-=GXY[t].re/nrm;
      data.GSP1[t].im-=GXY[t].im/nrm;
   }

   /* PS */
   cfcts4q2(P,A0,S,P,dpsd,upsd,dsd,dsd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GPS2[t].re-=GXY[t].re/nrm;
      data.GPS2[t].im-=GXY[t].im/nrm;
   }
   
   cfcts4q1(P,P,S,A0,dpsd,dsd,dsd,upsd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GPS1[t].re-=GXY[t].re/nrm;
      data.GPS1[t].im-=GXY[t].im/nrm;
   }

   /* TTt */
   for (l=0; l<3; l++)
   {
      cfcts4q2(T01+l,A0,Tt01+l,P,dpsd,upsd,dsd,dsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt2[t].re-=4.0*GXY[t].re/nrm;
         data.GTTt2[t].im-=4.0*GXY[t].im/nrm;
      }

      cfcts4q1(T01+l,P,Tt01+l,A0,dpsd,dsd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt1[t].re-=4.0*GXY[t].re/nrm;
         data.GTTt1[t].im-=4.0*GXY[t].im/nrm;
      }

      cfcts4q2(T12+l,A0,Tt12+l,P,dpsd,upsd,dsd,dsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt2[t].re-=4.0*GXY[t].re/nrm;
         data.GTTt2[t].im-=4.0*GXY[t].im/nrm;
      }

      cfcts4q1(T12+l,P,Tt12+l,A0,dpsd,dsd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt1[t].re-=4.0*GXY[t].re/nrm;
         data.GTTt1[t].im-=4.0*GXY[t].im/nrm;
      }
   }

   /* Okuu QXuuud Okdu */
   for(t=0;t<tmax;t++)
   {
      data.LVA2[t].re=0.0;
      data.LVA2[t].im=0.0;
      data.LVA1[t].re=0.0;
      data.LVA1[t].im=0.0;

      data.LAV2[t].re=0.0;
      data.LAV2[t].im=0.0;
      data.LAV1[t].re=0.0;
      data.LAV1[t].im=0.0;

      data.LSP2[t].re=0.0;
      data.LSP2[t].im=0.0;
      data.LSP1[t].re=0.0;
      data.LSP1[t].im=0.0;

      data.LPS2[t].re=0.0;
      data.LPS2[t].im=0.0;
      data.LPS1[t].re=0.0;
      data.LPS1[t].im=0.0;

      data.LTTt2[t].re=0.0;
      data.LTTt2[t].im=0.0;
      data.LTTt1[t].re=0.0;
      data.LTTt1[t].im=0.0;
   }

   for (k=0; k<3; k++)
   {
      for (i=0; i<4; i++)
      {
         /* VA */ 
         cfcts4q2(V0+i,V1+k,A0+i,T01+k,dpsd,upsd,dsd,dsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LVA2[t].re+=GXY[t].im/(3.0*nrm);
            data.LVA2[t].im-=GXY[t].re/(3.0*nrm);
         }
    
         cfcts4q1(V0+i,T01+k,A0+i,V1+k,dpsd,dsd,dsd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LVA1[t].re+=GXY[t].im/(3.0*nrm);
            data.LVA1[t].im-=GXY[t].re/(3.0*nrm);
         }
      
         /* AV */ 
         cfcts4q2(A0+i,V1+k,V0+i,T01+k,dpsd,upsd,dsd,dsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LAV2[t].re+=GXY[t].im/(3.0*nrm);
            data.LAV2[t].im-=GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(A0+i,T01+k,V0+i,V1+k,dpsd,dsd,dsd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LAV1[t].re+=GXY[t].im/(3.0*nrm);
            data.LAV1[t].im-=GXY[t].re/(3.0*nrm);
         }
      }

      /* SP */
      cfcts4q2(S,V1+k,P,T01+k,dpsd,upsd,dsd,dsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LSP2[t].re+=GXY[t].im/(3.0*nrm);
         data.LSP2[t].im-=GXY[t].re/(3.0*nrm);
      }

      cfcts4q1(S,T01+k,P,V1+k,dpsd,dsd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LSP1[t].re+=GXY[t].im/(3.0*nrm);
         data.LSP1[t].im-=GXY[t].re/(3.0*nrm);
      }

      /* PS */
      cfcts4q2(P,V1+k,S,T01+k,dpsd,upsd,dsd,dsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LPS2[t].re+=GXY[t].im/(3.0*nrm);
         data.LPS2[t].im-=GXY[t].re/(3.0*nrm);
      }
      
      cfcts4q1(P,T01+k,S,V1+k,dpsd,dsd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LPS1[t].re+=GXY[t].im/(3.0*nrm);
         data.LPS1[t].im-=GXY[t].re/(3.0*nrm);
      }

      /* TTt */
      for (l=0; l<3; l++)
      {
         cfcts4q2(T01+l,V1+k,Tt01+l,T01+k,dpsd,upsd,dsd,dsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt2[t].re+=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt2[t].im-=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(T01+l,T01+k,Tt01+l,V1+k,dpsd,dsd,dsd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt1[t].re+=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt1[t].im-=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q2(T12+l,V1+k,Tt12+l,T01+k,dpsd,upsd,dsd,dsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt2[t].re+=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt2[t].im-=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(T12+l,T01+k,Tt12+l,V1+k,dpsd,dsd,dsd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt1[t].re+=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt1[t].im-=4.0*GXY[t].re/(3.0*nrm);
         }
      }
   }

   /* save */
   save_data(dat_file_uuud);
   /* print */
   print_log("uuud");

   /* O5uu QXuudu O5ud */
   for(t=0;t<tmax;t++)
   {
      data.GVA2[t].re=0.0;
      data.GVA2[t].im=0.0;
      data.GVA1[t].re=0.0;
      data.GVA1[t].im=0.0;

      data.GAV2[t].re=0.0;
      data.GAV2[t].im=0.0;
      data.GAV1[t].re=0.0;
      data.GAV1[t].im=0.0;

      data.GSP2[t].re=0.0;
      data.GSP2[t].im=0.0;
      data.GSP1[t].re=0.0;
      data.GSP1[t].im=0.0;

      data.GPS2[t].re=0.0;
      data.GPS2[t].im=0.0;
      data.GPS1[t].re=0.0;
      data.GPS1[t].im=0.0;

      data.GTTt2[t].re=0.0;
      data.GTTt2[t].im=0.0;
      data.GTTt1[t].re=0.0;
      data.GTTt1[t].im=0.0;
   }

   for (i=0; i<4; i++)
   {
      /* VA */ 
      cfcts4q2(V0+i,A0,A0+i,P,dpsd,upsd,usd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GVA2[t].re-=GXY[t].re/nrm;
         data.GVA2[t].im-=GXY[t].im/nrm;
      }
 
      cfcts4q1(V0+i,P,A0+i,A0,dpsd,usd,usd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GVA1[t].re-=GXY[t].re/nrm;
         data.GVA1[t].im-=GXY[t].im/nrm;
      }
   
      /* AV */ 
      cfcts4q2(A0+i,A0,V0+i,P,dpsd,upsd,usd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GAV2[t].re-=GXY[t].re/nrm;
         data.GAV2[t].im-=GXY[t].im/nrm;
      }

      cfcts4q1(A0+i,P,V0+i,A0,dpsd,usd,usd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GAV1[t].re-=GXY[t].re/nrm;
         data.GAV1[t].im-=GXY[t].im/nrm;
      }
   }

   /* SP */
   cfcts4q2(S,A0,P,P,dpsd,upsd,usd,usd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GSP2[t].re-=GXY[t].re/nrm;
      data.GSP2[t].im-=GXY[t].im/nrm;
   }

   cfcts4q1(S,P,P,A0,dpsd,usd,usd,upsd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GSP1[t].re-=GXY[t].re/nrm;
      data.GSP1[t].im-=GXY[t].im/nrm;
   }

   /* PS */
   cfcts4q2(P,A0,S,P,dpsd,upsd,usd,usd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GPS2[t].re-=GXY[t].re/nrm;
      data.GPS2[t].im-=GXY[t].im/nrm;
   }
   
   cfcts4q1(P,P,S,A0,dpsd,usd,usd,upsd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GPS1[t].re-=GXY[t].re/nrm;
      data.GPS1[t].im-=GXY[t].im/nrm;
   }

   /* TTt */
   for (l=0; l<3; l++)
   {
      cfcts4q2(T01+l,A0,Tt01+l,P,dpsd,upsd,usd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt2[t].re-=4.0*GXY[t].re/nrm;
         data.GTTt2[t].im-=4.0*GXY[t].im/nrm;
      }

      cfcts4q1(T01+l,P,Tt01+l,A0,dpsd,usd,usd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt1[t].re-=4.0*GXY[t].re/nrm;
         data.GTTt1[t].im-=4.0*GXY[t].im/nrm;
      }

      cfcts4q2(T12+l,A0,Tt12+l,P,dpsd,upsd,usd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt2[t].re-=4.0*GXY[t].re/nrm;
         data.GTTt2[t].im-=4.0*GXY[t].im/nrm;
      }

      cfcts4q1(T12+l,P,Tt12+l,A0,dpsd,usd,usd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt1[t].re-=4.0*GXY[t].re/nrm;
         data.GTTt1[t].im-=4.0*GXY[t].im/nrm;
      }
   }

   /* Okuu QXuudu Okud */
   for(t=0;t<tmax;t++)
   {
      data.LVA2[t].re=0.0;
      data.LVA2[t].im=0.0;
      data.LVA1[t].re=0.0;
      data.LVA1[t].im=0.0;

      data.LAV2[t].re=0.0;
      data.LAV2[t].im=0.0;
      data.LAV1[t].re=0.0;
      data.LAV1[t].im=0.0;

      data.LSP2[t].re=0.0;
      data.LSP2[t].im=0.0;
      data.LSP1[t].re=0.0;
      data.LSP1[t].im=0.0;

      data.LPS2[t].re=0.0;
      data.LPS2[t].im=0.0;
      data.LPS1[t].re=0.0;
      data.LPS1[t].im=0.0;

      data.LTTt2[t].re=0.0;
      data.LTTt2[t].im=0.0;
      data.LTTt1[t].re=0.0;
      data.LTTt1[t].im=0.0;
   }

   for (k=0; k<3; k++)
   {
      for (i=0; i<4; i++)
      {
         /* VA */ 
         cfcts4q2(V0+i,V1+k,A0+i,T01+k,dpsd,upsd,usd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LVA2[t].re+=GXY[t].im/(3.0*nrm);
            data.LVA2[t].im-=GXY[t].re/(3.0*nrm);
         }
    
         cfcts4q1(V0+i,T01+k,A0+i,V1+k,dpsd,usd,usd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LVA1[t].re+=GXY[t].im/(3.0*nrm);
            data.LVA1[t].im-=GXY[t].re/(3.0*nrm);
         }
      
         /* AV */ 
         cfcts4q2(A0+i,V1+k,V0+i,T01+k,dpsd,upsd,usd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LAV2[t].re+=GXY[t].im/(3.0*nrm);
            data.LAV2[t].im-=GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(A0+i,T01+k,V0+i,V1+k,dpsd,usd,usd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LAV1[t].re+=GXY[t].im/(3.0*nrm);
            data.LAV1[t].im-=GXY[t].re/(3.0*nrm);
         }
      }

      /* SP */
      cfcts4q2(S,V1+k,P,T01+k,dpsd,upsd,usd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LSP2[t].re+=GXY[t].im/(3.0*nrm);
         data.LSP2[t].im-=GXY[t].re/(3.0*nrm);
      }

      cfcts4q1(S,T01+k,P,V1+k,dpsd,usd,usd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LSP1[t].re+=GXY[t].im/(3.0*nrm);
         data.LSP1[t].im-=GXY[t].re/(3.0*nrm);
      }

      /* PS */
      cfcts4q2(P,V1+k,S,T01+k,dpsd,upsd,usd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LPS2[t].re+=GXY[t].im/(3.0*nrm);
         data.LPS2[t].im-=GXY[t].re/(3.0*nrm);
      }
      
      cfcts4q1(P,T01+k,S,V1+k,dpsd,usd,usd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LPS1[t].re+=GXY[t].im/(3.0*nrm);
         data.LPS1[t].im-=GXY[t].re/(3.0*nrm);
      }

      /* TTt */
      for (l=0; l<3; l++)
      {
         cfcts4q2(T01+l,V1+k,Tt01+l,T01+k,dpsd,upsd,usd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt2[t].re+=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt2[t].im-=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(T01+l,T01+k,Tt01+l,V1+k,dpsd,usd,usd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt1[t].re+=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt1[t].im-=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q2(T12+l,V1+k,Tt12+l,T01+k,dpsd,upsd,usd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt2[t].re+=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt2[t].im-=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(T12+l,T01+k,Tt12+l,V1+k,dpsd,usd,usd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt1[t].re+=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt1[t].im-=4.0*GXY[t].re/(3.0*nrm);
         }
      }
   }

   /* save */
   save_data(dat_file_uudu);
   /* print */
   print_log("uudu");

   /* O5du QXuduu O5uu */
   for(t=0;t<tmax;t++)
   {
      data.GVA2[t].re=0.0;
      data.GVA2[t].im=0.0;
      data.GVA1[t].re=0.0;
      data.GVA1[t].im=0.0;

      data.GAV2[t].re=0.0;
      data.GAV2[t].im=0.0;
      data.GAV1[t].re=0.0;
      data.GAV1[t].im=0.0;

      data.GSP2[t].re=0.0;
      data.GSP2[t].im=0.0;
      data.GSP1[t].re=0.0;
      data.GSP1[t].im=0.0;

      data.GPS2[t].re=0.0;
      data.GPS2[t].im=0.0;
      data.GPS1[t].re=0.0;
      data.GPS1[t].im=0.0;

      data.GTTt2[t].re=0.0;
      data.GTTt2[t].im=0.0;
      data.GTTt1[t].re=0.0;
      data.GTTt1[t].im=0.0;
   }

   for (i=0; i<4; i++)
   {
      /* VA */ 
      cfcts4q2(V0+i,P,A0+i,A0,dpsd,dpsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GVA2[t].re+=GXY[t].re/nrm;
         data.GVA2[t].im+=GXY[t].im/nrm;
      }
 
      cfcts4q1(V0+i,A0,A0+i,P,dpsd,usd,dsd,dpsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GVA1[t].re+=GXY[t].re/nrm;
         data.GVA1[t].im+=GXY[t].im/nrm;
      }
   
      /* AV */ 
      cfcts4q2(A0+i,P,V0+i,A0,dpsd,dpsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GAV2[t].re+=GXY[t].re/nrm;
         data.GAV2[t].im+=GXY[t].im/nrm;
      }

      cfcts4q1(A0+i,A0,V0+i,P,dpsd,usd,dsd,dpsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GAV1[t].re+=GXY[t].re/nrm;
         data.GAV1[t].im+=GXY[t].im/nrm;
      }
   }

   /* SP */
   cfcts4q2(S,P,P,A0,dpsd,dpsd,dsd,usd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GSP2[t].re+=GXY[t].re/nrm;
      data.GSP2[t].im+=GXY[t].im/nrm;
   }

   cfcts4q1(S,A0,P,P,dpsd,usd,dsd,dpsd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GSP1[t].re+=GXY[t].re/nrm;
      data.GSP1[t].im+=GXY[t].im/nrm;
   }

   /* PS */
   cfcts4q2(P,P,S,A0,dpsd,dpsd,dsd,usd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GPS2[t].re+=GXY[t].re/nrm;
      data.GPS2[t].im+=GXY[t].im/nrm;
   }
   
   cfcts4q1(P,A0,S,P,dpsd,usd,dsd,dpsd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GPS1[t].re+=GXY[t].re/nrm;
      data.GPS1[t].im+=GXY[t].im/nrm;
   }

   /* TTt */
   for (l=0; l<3; l++)
   {
      cfcts4q2(T01+l,P,Tt01+l,A0,dpsd,dpsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt2[t].re+=4.0*GXY[t].re/nrm;
         data.GTTt2[t].im+=4.0*GXY[t].im/nrm;
      }

      cfcts4q1(T01+l,A0,Tt01+l,P,dpsd,usd,dsd,dpsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt1[t].re+=4.0*GXY[t].re/nrm;
         data.GTTt1[t].im+=4.0*GXY[t].im/nrm;
      }

      cfcts4q2(T12+l,P,Tt12+l,A0,dpsd,dpsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt2[t].re+=4.0*GXY[t].re/nrm;
         data.GTTt2[t].im+=4.0*GXY[t].im/nrm;
      }

      cfcts4q1(T12+l,A0,Tt12+l,P,dpsd,usd,dsd,dpsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt1[t].re+=4.0*GXY[t].re/nrm;
         data.GTTt1[t].im+=4.0*GXY[t].im/nrm;
      }
   }

   /* Okdu QXuduu Okuu */
   for(t=0;t<tmax;t++)
   {
      data.LVA2[t].re=0.0;
      data.LVA2[t].im=0.0;
      data.LVA1[t].re=0.0;
      data.LVA1[t].im=0.0;

      data.LAV2[t].re=0.0;
      data.LAV2[t].im=0.0;
      data.LAV1[t].re=0.0;
      data.LAV1[t].im=0.0;

      data.LSP2[t].re=0.0;
      data.LSP2[t].im=0.0;
      data.LSP1[t].re=0.0;
      data.LSP1[t].im=0.0;

      data.LPS2[t].re=0.0;
      data.LPS2[t].im=0.0;
      data.LPS1[t].re=0.0;
      data.LPS1[t].im=0.0;

      data.LTTt2[t].re=0.0;
      data.LTTt2[t].im=0.0;
      data.LTTt1[t].re=0.0;
      data.LTTt1[t].im=0.0;
   }

   for (k=0; k<3; k++)
   {
      for (i=0; i<4; i++)
      {
         /* VA */ 
         cfcts4q2(V0+i,T01+k,A0+i,V1+k,dpsd,dpsd,dsd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LVA2[t].re-=GXY[t].im/(3.0*nrm);
            data.LVA2[t].im+=GXY[t].re/(3.0*nrm);
         }
    
         cfcts4q1(V0+i,V1+k,A0+i,T01+k,dpsd,usd,dsd,dpsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LVA1[t].re-=GXY[t].im/(3.0*nrm);
            data.LVA1[t].im+=GXY[t].re/(3.0*nrm);
         }
      
         /* AV */ 
         cfcts4q2(A0+i,T01+k,V0+i,V1+k,dpsd,dpsd,dsd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LAV2[t].re-=GXY[t].im/(3.0*nrm);
            data.LAV2[t].im+=GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(A0+i,V1+k,V0+i,T01+k,dpsd,usd,dsd,dpsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LAV1[t].re-=GXY[t].im/(3.0*nrm);
            data.LAV1[t].im+=GXY[t].re/(3.0*nrm);
         }
      }

      /* SP */
      cfcts4q2(S,T01+k,P,V1+k,dpsd,dpsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LSP2[t].re-=GXY[t].im/(3.0*nrm);
         data.LSP2[t].im+=GXY[t].re/(3.0*nrm);
      }

      cfcts4q1(S,V1+k,P,T01+k,dpsd,usd,dsd,dpsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LSP1[t].re-=GXY[t].im/(3.0*nrm);
         data.LSP1[t].im+=GXY[t].re/(3.0*nrm);
      }

      /* PS */
      cfcts4q2(P,T01+k,S,V1+k,dpsd,dpsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LPS2[t].re-=GXY[t].im/(3.0*nrm);
         data.LPS2[t].im+=GXY[t].re/(3.0*nrm);
      }
      
      cfcts4q1(P,V1+k,S,T01+k,dpsd,usd,dsd,dpsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LPS1[t].re-=GXY[t].im/(3.0*nrm);
         data.LPS1[t].im+=GXY[t].re/(3.0*nrm);
      }

      /* TTt */
      for (l=0; l<3; l++)
      {
         cfcts4q2(T01+l,T01+k,Tt01+l,V1+k,dpsd,dpsd,dsd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt2[t].re-=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt2[t].im+=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(T01+l,V1+k,Tt01+l,T01+k,dpsd,usd,dsd,dpsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt1[t].re-=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt1[t].im+=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q2(T12+l,T01+k,Tt12+l,V1+k,dpsd,dpsd,dsd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt2[t].re-=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt2[t].im+=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(T12+l,V1+k,Tt12+l,T01+k,dpsd,usd,dsd,dpsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt1[t].re-=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt1[t].im+=4.0*GXY[t].re/(3.0*nrm);
         }
      }
   }

   /* save */
   save_data(dat_file_uduu);
   /* print */
   print_log("uduu");

   /* O5ud QXduuu O5uu */
   for(t=0;t<tmax;t++)
   {
      data.GVA2[t].re=0.0;
      data.GVA2[t].im=0.0;
      data.GVA1[t].re=0.0;
      data.GVA1[t].im=0.0;

      data.GAV2[t].re=0.0;
      data.GAV2[t].im=0.0;
      data.GAV1[t].re=0.0;
      data.GAV1[t].im=0.0;

      data.GSP2[t].re=0.0;
      data.GSP2[t].im=0.0;
      data.GSP1[t].re=0.0;
      data.GSP1[t].im=0.0;

      data.GPS2[t].re=0.0;
      data.GPS2[t].im=0.0;
      data.GPS1[t].re=0.0;
      data.GPS1[t].im=0.0;

      data.GTTt2[t].re=0.0;
      data.GTTt2[t].im=0.0;
      data.GTTt1[t].re=0.0;
      data.GTTt1[t].im=0.0;
   }

   for (i=0; i<4; i++)
   {
      /* VA */ 
      cfcts4q2(V0+i,P,A0+i,A0,upsd,upsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GVA2[t].re+=GXY[t].re/nrm;
         data.GVA2[t].im+=GXY[t].im/nrm;
      }
 
      cfcts4q1(V0+i,A0,A0+i,P,upsd,usd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GVA1[t].re+=GXY[t].re/nrm;
         data.GVA1[t].im+=GXY[t].im/nrm;
      }
   
      /* AV */ 
      cfcts4q2(A0+i,P,V0+i,A0,upsd,upsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GAV2[t].re+=GXY[t].re/nrm;
         data.GAV2[t].im+=GXY[t].im/nrm;
      }

      cfcts4q1(A0+i,A0,V0+i,P,upsd,usd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GAV1[t].re+=GXY[t].re/nrm;
         data.GAV1[t].im+=GXY[t].im/nrm;
      }
   }

   /* SP */
   cfcts4q2(S,P,P,A0,upsd,upsd,dsd,usd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GSP2[t].re+=GXY[t].re/nrm;
      data.GSP2[t].im+=GXY[t].im/nrm;
   }

   cfcts4q1(S,A0,P,P,upsd,usd,dsd,upsd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GSP1[t].re+=GXY[t].re/nrm;
      data.GSP1[t].im+=GXY[t].im/nrm;
   }

   /* PS */
   cfcts4q2(P,P,S,A0,upsd,upsd,dsd,usd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GPS2[t].re+=GXY[t].re/nrm;
      data.GPS2[t].im+=GXY[t].im/nrm;
   }
   
   cfcts4q1(P,A0,S,P,upsd,usd,dsd,upsd,GXY);

   for(t=0;t<tmax;t++)
   {
      data.GPS1[t].re+=GXY[t].re/nrm;
      data.GPS1[t].im+=GXY[t].im/nrm;
   }

   /* TTt */
   for (l=0; l<3; l++)
   {
      cfcts4q2(T01+l,P,Tt01+l,A0,upsd,upsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt2[t].re+=4.0*GXY[t].re/nrm;
         data.GTTt2[t].im+=4.0*GXY[t].im/nrm;
      }

      cfcts4q1(T01+l,A0,Tt01+l,P,upsd,usd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt1[t].re+=4.0*GXY[t].re/nrm;
         data.GTTt1[t].im+=4.0*GXY[t].im/nrm;
      }

      cfcts4q2(T12+l,P,Tt12+l,A0,upsd,upsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt2[t].re+=4.0*GXY[t].re/nrm;
         data.GTTt2[t].im+=4.0*GXY[t].im/nrm;
      }

      cfcts4q1(T12+l,A0,Tt12+l,P,upsd,usd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.GTTt1[t].re+=4.0*GXY[t].re/nrm;
         data.GTTt1[t].im+=4.0*GXY[t].im/nrm;
      }
   }

   /* Okdu QXuduu Okuu */
   for(t=0;t<tmax;t++)
   {
      data.LVA2[t].re=0.0;
      data.LVA2[t].im=0.0;
      data.LVA1[t].re=0.0;
      data.LVA1[t].im=0.0;

      data.LAV2[t].re=0.0;
      data.LAV2[t].im=0.0;
      data.LAV1[t].re=0.0;
      data.LAV1[t].im=0.0;

      data.LSP2[t].re=0.0;
      data.LSP2[t].im=0.0;
      data.LSP1[t].re=0.0;
      data.LSP1[t].im=0.0;

      data.LPS2[t].re=0.0;
      data.LPS2[t].im=0.0;
      data.LPS1[t].re=0.0;
      data.LPS1[t].im=0.0;

      data.LTTt2[t].re=0.0;
      data.LTTt2[t].im=0.0;
      data.LTTt1[t].re=0.0;
      data.LTTt1[t].im=0.0;
   }

   for (k=0; k<3; k++)
   {
      for (i=0; i<4; i++)
      {
         /* VA */ 
         cfcts4q2(V0+i,T01+k,A0+i,V1+k,upsd,upsd,dsd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LVA2[t].re-=GXY[t].im/(3.0*nrm);
            data.LVA2[t].im+=GXY[t].re/(3.0*nrm);
         }
    
         cfcts4q1(V0+i,V1+k,A0+i,T01+k,upsd,usd,dsd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LVA1[t].re-=GXY[t].im/(3.0*nrm);
            data.LVA1[t].im+=GXY[t].re/(3.0*nrm);
         }
      
         /* AV */ 
         cfcts4q2(A0+i,T01+k,V0+i,V1+k,upsd,upsd,dsd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LAV2[t].re-=GXY[t].im/(3.0*nrm);
            data.LAV2[t].im+=GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(A0+i,V1+k,V0+i,T01+k,upsd,usd,dsd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LAV1[t].re-=GXY[t].im/(3.0*nrm);
            data.LAV1[t].im+=GXY[t].re/(3.0*nrm);
         }
      }

      /* SP */
      cfcts4q2(S,T01+k,P,V1+k,upsd,upsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LSP2[t].re-=GXY[t].im/(3.0*nrm);
         data.LSP2[t].im+=GXY[t].re/(3.0*nrm);
      }

      cfcts4q1(S,V1+k,P,T01+k,upsd,usd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LSP1[t].re-=GXY[t].im/(3.0*nrm);
         data.LSP1[t].im+=GXY[t].re/(3.0*nrm);
      }

      /* PS */
      cfcts4q2(P,T01+k,S,V1+k,upsd,upsd,dsd,usd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LPS2[t].re-=GXY[t].im/(3.0*nrm);
         data.LPS2[t].im+=GXY[t].re/(3.0*nrm);
      }
      
      cfcts4q1(P,V1+k,S,T01+k,upsd,usd,dsd,upsd,GXY);

      for(t=0;t<tmax;t++)
      {
         data.LPS1[t].re-=GXY[t].im/(3.0*nrm);
         data.LPS1[t].im+=GXY[t].re/(3.0*nrm);
      }

      /* TTt */
      for (l=0; l<3; l++)
      {
         cfcts4q2(T01+l,T01+k,Tt01+l,V1+k,upsd,upsd,dsd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt2[t].re-=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt2[t].im+=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(T01+l,V1+k,Tt01+l,T01+k,upsd,usd,dsd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt1[t].re-=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt1[t].im+=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q2(T12+l,T01+k,Tt12+l,V1+k,upsd,upsd,dsd,usd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt2[t].re-=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt2[t].im+=4.0*GXY[t].re/(3.0*nrm);
         }

         cfcts4q1(T12+l,V1+k,Tt12+l,T01+k,upsd,usd,dsd,upsd,GXY);

         for(t=0;t<tmax;t++)
         {
            data.LTTt1[t].re-=4.0*GXY[t].im/(3.0*nrm);
            data.LTTt1[t].im+=4.0*GXY[t].re/(3.0*nrm);
         }
      }
   }

   /* save */
   save_data(dat_file_duuu);
   /* print */
   print_log("duuu");

   /* free */
   release_wsd();
   release_wsd();
   release_wsd();
   release_wsd();
}


int main(int argc,char *argv[])
{
   int ie,nc,iend,status[4];
   int nws,nwsd,nwsdc,nwv,nwvd,n;
   double wt1,wt2,wtavg;
   dfl_parms_t dfl;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   alloc_data();
   check_files();
   print_info();
   dfl=dfl_parms();

   geometry();
   nwsdc=48;

   wsize(&nws,&nwsd,&nwv,&nwvd);
   nwsd+=nwsdc;
   alloc_ws(nws);
   alloc_wsd(nwsd);
   alloc_wv(nwv);
   alloc_wvd(nwvd);   
   
   iend=0;   
   wtavg=0.0;
   
   for (nc=first;(iend==0)&&(nc<=last);nc+=step)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();
      
      if (my_rank==0)
         printf("Configuration no %d\n",nc);

      if (noexp)
      {
         sprintf(cnfg_file,"%s/%sn%d_%d",loc_dir,nbase,nc,my_rank);
         read_cnfg(cnfg_file);
      }
      else
      {
         sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,nc);
         import_cnfg(cnfg_file);
      }

      ie=check_sfbcd();
      error_root(ie!=1,1,"main [ms6_xsf.c]",
                 "Initial configuration has incorrect boundary values");

      if (dfl.Ns)
      {
         dfl_modes(status);
         error_root(status[0]<0,1,"main [ms6_xsf.c]",
                    "Deflation subspace generation failed (status = %d)",
                    status[0]);

         if (my_rank==0)
            printf("Deflation subspace generation: status = %d\n",status[0]);
      }


      xsfcfcts(nc,status);

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);
      error_chk();
      
      if (my_rank==0)
      {
         printf("Computation of xSF correlation functions completed\n");

         if (dfl.Ns)
         {
            printf("status = %d,%d,%d",status[0],status[1],status[2]);

            if (status[3])
               printf(" (no of subspace regenerations = %d)\n",status[3]);
            else
               printf("\n");
         }
         else
            printf("status = %d\n",status[0]);

         n=(nc-first)/step+1;
         
         printf("Configuration no %d fully processed in %.2e sec ",
                nc,wt2-wt1);
         printf("(average = %.2e sec)\n\n",wtavg/(double)(n));

         fflush(flog);         
         copy_file(log_file,log_save);
         copy_file(dat_file_uuud,dat_save_uuud);
         copy_file(dat_file_uudu,dat_save_uudu);
         copy_file(dat_file_uduu,dat_save_uduu);
         copy_file(dat_file_duuu,dat_save_duuu);
      }

      check_endflag(&iend);
   }

   error_chk();
   
   if (my_rank==0)
   {
      fflush(flog);
      copy_file(log_file,log_save);
      fclose(flog);
   }
   
   MPI_Finalize();    
   exit(0);
}
