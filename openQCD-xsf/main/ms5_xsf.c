
/**********************************************************************************
*
* File ms5_xsf.c
*
* Copyright (C) 2014 Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the xSF correlation functions gS,gP,gA,gV,g1, and lA,lV,lT,lTt,l1,
* for the flavour combinations uu',dd',ud, and du.
*
* Syntax: ms5_xsf -i <input file> [-noexp]
*
* For usage instructions see the file README.ms5_xsf
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

typedef union
{
   spinor_dble s;
   double w[24];
} spin_t;

static struct
{
   int tmax,bnd;
   double kappa,csw,dF,zF;
} file_head;

static struct
{
   int nc;
   complex_dble *gS,*gP,*gA,*gV,*gVt;
   complex_dble *lA,*lV,*lVt,*lT,*lTt;
   complex_dble *g1,*l1;
} data;

static int my_rank,noexp,append,endian;
static int first,last,step,bnd;
static complex_dble gX[N0],lX[N0];
static const spinor_dble sd0={{{0.0}}};

static char line[NAME_SIZE];
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char loc_dir[NAME_SIZE],cnfg_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE],end_file[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char dat_file_uu[NAME_SIZE],dat_save_uu[NAME_SIZE];
static char dat_file_dd[NAME_SIZE],dat_save_dd[NAME_SIZE];
static char dat_file_ud[NAME_SIZE],dat_save_ud[NAME_SIZE];
static char dat_file_du[NAME_SIZE],dat_save_du[NAME_SIZE];
static char cnfg_file[NAME_SIZE],nbase[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fdat=NULL,*fend=NULL;


static void alloc_data(void)
{
   int tmax;
   complex_dble *p;

   tmax=file_head.tmax;

   p=amalloc((10*tmax+2)*sizeof(*p),4);
   
   error((p==NULL),1,"alloc_data [ms5_xsf.c]",
         "Unable to allocate data arrays");

   data.gS =p;
   data.gP =p+tmax;
   data.gA =p+2*tmax;
   data.gV =p+3*tmax;
   data.gVt=p+4*tmax;

   data.lA =p+5*tmax;
   data.lV =p+6*tmax;
   data.lVt=p+7*tmax;
   data.lT =p+8*tmax;
   data.lTt=p+9*tmax;

   data.g1 =p+10*tmax;
   data.l1 =p+10*tmax+1;
}


/*
   CHECK HERE:
   Add a specifier for the flavour content in the 
   header file in case that the files get renamed
*/


static void write_file_head(void)
{
   int iw;
   double dstd[4];   
   stdint_t istd[2];

   dstd[0]=file_head.kappa;
   dstd[1]=file_head.csw;
   dstd[2]=file_head.dF;
   dstd[3]=file_head.zF;
   istd[0]=(stdint_t)(file_head.tmax);   
   istd[1]=(stdint_t)(file_head.bnd);   
   
   if (endian==BIG_ENDIAN)
   {
      bswap_double(4,dstd);
      bswap_int(2,istd);
   }
   
   iw=fwrite(dstd,sizeof(double),4,fdat);   
   iw+=fwrite(istd,sizeof(stdint_t),2,fdat);

   error_root(iw!=6,1,"write_file_head [ms5_xsf.c]",
              "Incorrect write count");
}


static void check_file_head(void)
{
   int ir;
   double dstd[4];
   stdint_t istd[2];

   ir=fread(dstd,sizeof(double),4,fdat);   
   ir+=fread(istd,sizeof(stdint_t),2,fdat);

   error_root(ir!=6,1,"check_file_head [ms5_xsf.c]",
              "Incorrect read count");
   
   if (endian==BIG_ENDIAN)
   {
      bswap_double(4,dstd);      
      bswap_int(2,istd);
   }
   
   error_root((dstd[0]!=file_head.kappa)||
              (dstd[1]!=file_head.csw)||
              (dstd[2]!=file_head.dF)||
              (dstd[3]!=file_head.zF)||
              ((int)(istd[0])!=file_head.tmax)||
              ((int)(istd[1])!=file_head.bnd),1,"check_file_head [ms5_xsf.c]",
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
      dstd[0]=data.gS[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.gS[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.gP[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.gP[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.gA[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.gA[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.gV[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.gV[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.gVt[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.gVt[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)   
   {
      dstd[0]=data.lA[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.lA[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.lV[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.lV[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.lVt[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.lVt[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.lT[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.lT[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.lTt[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.lTt[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   dstd[0]=data.g1[0].re;

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);

   iw+=fwrite(dstd,sizeof(double),1,fdat);

   dstd[0]=data.g1[0].im;

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);

   iw+=fwrite(dstd,sizeof(double),1,fdat);

   dstd[0]=data.l1[0].re;

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);

   iw+=fwrite(dstd,sizeof(double),1,fdat);

   dstd[0]=data.l1[0].im;

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);

   iw+=fwrite(dstd,sizeof(double),1,fdat);

   error_root(iw!=(1+4+20*tmax),1,"write_data [ms5_xsf.c]",
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
         
      data.gS[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.gS[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.gP[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.gP[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.gA[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.gA[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.gV[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.gV[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.gVt[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.gVt[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.lA[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.lA[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.lV[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.lV[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.lVt[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.lVt[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.lT[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.lT[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.lTt[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.lTt[t].im=dstd[0];
   }

   ir+=fread(dstd,sizeof(double),1,fdat);

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);
         
   data.g1[0].re=dstd[0];

   ir+=fread(dstd,sizeof(double),1,fdat);

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);
         
   data.g1[0].im=dstd[0];

   ir+=fread(dstd,sizeof(double),1,fdat);

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);
         
   data.l1[0].re=dstd[0];

   ir+=fread(dstd,sizeof(double),1,fdat);

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);
         
   data.l1[0].im=dstd[0];


   error_root(ir!=(1+4+20*tmax),1,"read_data [ms5_xsf.c]",
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
      read_line("bnd","%d",&bnd);

      error_root((last<first)||(step<1)||(((last-first)%step)!=0),1,
                 "read_dirs [ms5_xsf.c]","Improper configuration range");
      error_root((bnd<0)||(bnd>1),1,"read_dirs [ms5_xsf.c]",
                 "Parameter bnd must be 0 or 1");
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(dat_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(loc_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   
   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&bnd,1,MPI_INT,0,MPI_COMM_WORLD);   

   file_head.bnd=bnd;
}


static void setup_files(void)
{
   if (noexp)
      error_root(name_size("%s/%sn%d_%d",loc_dir,nbase,last,NPROC-1)>=NAME_SIZE,
                 1,"setup_files [ms5_xsf.c]","loc_dir name is too long");
   else
      error_root(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,
                 1,"setup_files [ms5_xsf.c]","cnfg_dir name is too long");

   check_dir_root(log_dir);   
   check_dir_root(dat_dir);
   error_root(name_size("%s/%s.ms5_xsf.log~",log_dir,nbase)>=NAME_SIZE,
              1,"setup_files [ms5_xsf.c]","log_dir name is too long");
   error_root(name_size("%s/%s.ms5_xsf_xx.dat~",dat_dir,nbase)>=NAME_SIZE,
              1,"setup_files [ms5_xsf.c]","dat_dir name is too long");   

   sprintf(log_file,"%s/%s.ms5_xsf.log",log_dir,nbase);
   sprintf(par_file,"%s/%s.ms5_xsf.par",dat_dir,nbase);   
   sprintf(dat_file_uu,"%s/%s.ms5_xsf_uu.dat",dat_dir,nbase);
   sprintf(dat_file_dd,"%s/%s.ms5_xsf_dd.dat",dat_dir,nbase);
   sprintf(dat_file_ud,"%s/%s.ms5_xsf_ud.dat",dat_dir,nbase);
   sprintf(dat_file_du,"%s/%s.ms5_xsf_du.dat",dat_dir,nbase);
   sprintf(end_file,"%s/%s.ms5_xsf.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);   
   sprintf(dat_save_uu,"%s~",dat_file_uu);
   sprintf(dat_save_dd,"%s~",dat_file_dd);
   sprintf(dat_save_ud,"%s~",dat_file_ud);
   sprintf(dat_save_du,"%s~",dat_file_du);
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
         error_root(ir!=4,1,"read_lat_parms [ms5_xsf.c]",
                    "Incorrect read count");         

         if (endian==BIG_ENDIAN)
            bswap_double(4,dstd);

         ie=0;
         ie|=(dstd[0]!=kappa);
         ie|=(dstd[1]!=csw);
         ie|=(dstd[2]!=dF);
         ie|=(dstd[3]!=zF);

         error_root(ie!=0,1,"read_lat_parms [ms5_xsf.c]",
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
         error_root(iw!=4,1,"read_lat_parms [ms5_xsf.c]",
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

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [ms5_xsf.c]",
                 "Syntax: ms5_xsf -i <input file> [-noexp]");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms5_xsf.c]",
                 "Machine has unknown endianness");

      noexp=find_opt(argc,argv,"-noexp");      
      append=find_opt(argc,argv,"-a");
      
      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ms5_xsf.c]",
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

      error_root(fdat==NULL,1,"read_infile [ms5_xsf.c]",
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
   error_root(fend==NULL,1,"check_old_log [ms5_xsf.c]",
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

   error_root((ie&0x1)!=0x0,1,"check_old_log [ms5_xsf.c]",
              "Incorrect read count");   
   error_root((ie&0x2)!=0x0,1,"check_old_log [ms5_xsf.c]",
              "Configuration numbers are not equally spaced");
   error_root(isv==0,1,"check_old_log [ms5_xsf.c]",
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
   error_root(fdat==NULL,1,"check_old_dat [ms5_xsf.c]",
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

   error_root(ic==0,1,"check_old_dat [ms5_xsf.c]",
              "No data records found");
   error_root((ie&0x1)!=0x0,1,"check_old_dat [ms5_xsf.c]",
              "Configuration numbers are not equally spaced");
   error_root((fst!=fc)||(lst!=lc)||(stp!=dc),1,"check_old_dat [ms5_xsf.c]",
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

         check_old_dat(fst,lst,stp,dat_file_uu);
         check_old_dat(fst,lst,stp,dat_file_dd);
         check_old_dat(fst,lst,stp,dat_file_ud);
         check_old_dat(fst,lst,stp,dat_file_du);

         error_root((fst!=lst)&&(stp!=step),1,"check_files [ms5_xsf.c]",
                    "Continuation run:\n"
                    "Previous run had a different configuration separation");
         error_root(first!=lst+step,1,"check_files [ms5_xsf.c]",
                    "Continuation run:\n"
                    "Configuration range does not continue the previous one");
      }
      else
      {
         fin=fopen(log_file,"r");

         fdat=fopen(dat_file_uu,"rb");
         error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms5_xsf.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file_dd,"rb");
         error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms5_xsf.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file_ud,"rb");
         error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms5_xsf.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file_du,"rb");
         error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms5_xsf.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file_uu,"wb");
         error_root(fdat==NULL,1,"check_files [ms5_xsf.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);

         fdat=fopen(dat_file_dd,"wb");
         error_root(fdat==NULL,1,"check_files [ms5_xsf.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);

         fdat=fopen(dat_file_ud,"wb");
         error_root(fdat==NULL,1,"check_files [ms5_xsf.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);

         fdat=fopen(dat_file_du,"wb");
         error_root(fdat==NULL,1,"check_files [ms5_xsf.c]",
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

      error_root(flog==NULL,1,"print_info [ms5_xsf.c]","Unable to open log file");
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

         if (bnd==0)
            printf("Correlation functions from x0=0 to bulk\n\n");
         else
            printf("Correlation functions from x0=%i to bulk\n\n",N0-1);

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
      error_root(1,1,"wsize [ms5_xsf.c]",
                 "Unknown or unsupported solver");   
}


static void save_data(char *dat_file)
{
   if (my_rank==0)
   {
      fdat=fopen(dat_file,"ab");
      error_root(fdat==NULL,1,"save_data [ms5_xsf.c]",
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
      printf("\n#### gS_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.gS[t].re,data.gS[t].im);

      printf("\n#### gP_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.gP[t].re,data.gP[t].im);

      printf("\n#### gA_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.gA[t].re,data.gA[t].im);

      printf("\n#### gV_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.gV[t].re,data.gV[t].im);

      printf("\n#### gVt_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.gVt[t].re,data.gVt[t].im);

      printf("\n#### lA_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.lA[t].re,data.lA[t].im);

      printf("\n#### lV_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.lV[t].re,data.lV[t].im);

      printf("\n#### lVt_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.lVt[t].re,data.lVt[t].im);

      printf("\n#### lT_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.lT[t].re,data.lT[t].im);

      printf("\n#### lTt_%s ####\n\n",flv);
      for (t=0;t<tmax;t++)
         printf("x0 = %2i, % .6e   % .6e\n",t,data.lTt[t].re,data.lTt[t].im);

      printf("\n#### g1_%s ####\n\n",flv);
      printf("x0 = -, % .6e   % .6e\n",data.g1[0].re,data.g1[0].im);

      printf("\n#### l1_%s ####\n\n",flv);
      printf("x0 = -, % .6e   % .6e\n\n",data.l1[0].re,data.l1[0].im);
   }
}


static void spinor_sum_bnd(int x0,spinor_dble *s,spinor_dble *r)
{
   int ix,t;
   int iprms[1];
   spin_t rs,ra;
   spinor_dble *sm;

   if (NPROC>1)
   {
      iprms[0]=x0;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=x0),1,"spinor_sum_bnd [ms5_xsf.c]",
            "Parameters are not global");   
   }
 
   error_root(((x0!=0)&&(x0!=(NPROC0*L0-1))),1,
             "spinor_sum_bnd [ms5_xsf.c]","Improper argument x0");

   sm=s+VOLUME;
   rs.s=sd0;

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=0;

      for(;s<sm;s++)
      {
         t=global_time(ix);
         ix+=1;

         if (t==x0)
         {
            _vector_add_assign(rs.s.c1,(*s).c1);
            _vector_add_assign(rs.s.c2,(*s).c2);
            _vector_add_assign(rs.s.c3,(*s).c3);
            _vector_add_assign(rs.s.c4,(*s).c4);
         }            
      }
   }

   MPI_Reduce(rs.w,ra.w,24,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   MPI_Bcast(ra.w,24,MPI_DOUBLE,0,MPI_COMM_WORLD);

   (*r)=ra.s;
}


static void bnd2bnd_prop_xsf(int tau3,spinor_dble **psd,spinor_dble *r)
{
   int iprms[1],i,x0;
   spinor_dble **wsd;

   if (NPROC>1)
   {
      iprms[0]=tau3;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=tau3),1,
            "bnd2bnd_prop_xsf [ms5_xsf.c]","Parameters are not global");   
   }

   error_root((tau3!=1)&&(tau3!=-1),1,"bnd2bnd_prop_xsf [ms5_xsf.c]",
              "Flavour tau3 flag set to an improper value");

   wsd=reserve_wsd(1);

   if (bnd==0)
      x0=N0-1;
   else
      x0=0;

   for(i=0;i<12;i++)
   {
      if (bnd==0)
         push_slice_xsf(x0-1,-tau3,psd[i],wsd[0]);
      else
         pull_slice_xsf(x0+1,-tau3,psd[i],wsd[0]);

      spinor_sum_bnd(x0,wsd[0],r+i);      
   }

   release_wsd();
}


static void point_split_prop(int dir,spinor_dble **ps,spinor_dble **pr)
{
   int i;

   for (i=0;i<12;i++)
      ptsplit(dir,ps[i],pr[i]);
}


static void xsfcfcts(int nc,int *status)
{
   int i,k;
   int x0,t,tmax;
   double nrm;
   complex_dble z;
   spinor_dble u[12],d[12];
   spinor_dble **usd,**dsd;
   spinor_dble **pusd,**pdsd;

   usd=reserve_wsd(12);
   pusd=reserve_wsd(12);

   dsd=reserve_wsd(12);
   pdsd=reserve_wsd(12);

   data.nc=nc;
   tmax=file_head.tmax;
   nrm=(double)(N1*N2*N3);

   for (i=0;i<4;i++)
      status[i]=0;

   if (bnd==0)
      x0=0;
   else 
      x0=N0-1;

   mult_phase(1);

   xsfpropeo(x0,-1,0,dsd,status);
   xsfpropeo(x0, 1,0,usd,status);

   mult_phase(-1);

   bnd2bnd_prop_xsf(-1,dsd,d);
   bnd2bnd_prop_xsf( 1,usd,u);

   /* uu */
   /* gS,gP,gA,gV */
   cfcts2q(S ,A0,dsd,usd,data.gS);
   cfcts2q(P ,A0,dsd,usd,data.gP);
   cfcts2q(A0,A0,dsd,usd,data.gA);
   cfcts2q(V0,A0,dsd,usd,data.gV);
   /* gVt */
   point_split_prop(0,usd,pusd);
   point_split_prop(0,dsd,pdsd);
   cfcts2q(S,A0,dsd,pusd,gX);  
   cfcts2q(S,A0,pdsd,usd,data.gVt);
   /* g1 */
   ctrcts2q(V0,V0,d,u,&z);

   for(t=0;t<tmax;t++)
   {
      data.gS[t].re/=-2.0*nrm;
      data.gS[t].im/=-2.0*nrm;

      data.gP[t].re/=-2.0*nrm;
      data.gP[t].im/=-2.0*nrm;

      data.gA[t].re/=-2.0*nrm;
      data.gA[t].im/=-2.0*nrm;

      data.gV[t].re/=-2.0*nrm;
      data.gV[t].im/=-2.0*nrm;

      data.gVt[t].re-=gX[t].re;
      data.gVt[t].im-=gX[t].im;

      data.gVt[t].re/=-2.0*nrm;
      data.gVt[t].im/=-2.0*nrm;
   }

   data.g1[0].re=z.re/(2.0*nrm*nrm);
   data.g1[0].im=z.im/(2.0*nrm*nrm);

   /* lA,lA,lT,lTt,l1 */
   for(t=0;t<tmax;t++)
   {
      data.lA[t].re=0.0;
      data.lA[t].im=0.0;

      data.lV[t].re=0.0;
      data.lV[t].im=0.0;

      data.lVt[t].re=0.0;
      data.lVt[t].im=0.0;

      data.lT[t].re=0.0;
      data.lT[t].im=0.0;

      data.lTt[t].re=0.0;
      data.lTt[t].im=0.0;
   }
 
   data.l1[0].re=0.0;
   data.l1[0].im=0.0;

   for (k=0;k<3;k++)
   {      
      cfcts2q(A1+k,V1+k,dsd,usd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lA[t].re-=lX[t].re/(6.0*nrm);
         data.lA[t].im-=lX[t].im/(6.0*nrm);
      }
   
      cfcts2q(V1+k,V1+k,dsd,usd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lV[t].re-=lX[t].re/(6.0*nrm);
         data.lV[t].im-=lX[t].im/(6.0*nrm);
      }
   
      point_split_prop(k+1,usd,pusd);
      point_split_prop(k+1,dsd,pdsd);

      cfcts2q(S,V1+k,dsd,pusd,gX);  
      cfcts2q(S,V1+k,pdsd,usd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lVt[t].re-=(lX[t].re-gX[t].re)/(6.0*nrm);
         data.lVt[t].im-=(lX[t].im-gX[t].im)/(6.0*nrm);
      }

      cfcts2q(T01+k,V1+k,dsd,usd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lT[t].re-=lX[t].im/(6.0*nrm);
         data.lT[t].im+=lX[t].re/(6.0*nrm);
      }
   
      cfcts2q(Tt01+k,V1+k,dsd,usd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lTt[t].re-=lX[t].im/(6.0*nrm);
         data.lTt[t].im+=lX[t].re/(6.0*nrm);
      }

      ctrcts2q(A1+k,A1+k,d,u,&z);

      data.l1[0].re-=z.re/(6.0*nrm*nrm);
      data.l1[0].im-=z.im/(6.0*nrm*nrm);
   }   

   save_data(dat_file_uu);
   print_log("uu");

   /* dd */
   /* gS,gP,gA,gV */
   cfcts2q(S ,A0,usd,dsd,data.gS);
   cfcts2q(P ,A0,usd,dsd,data.gP);
   cfcts2q(A0,A0,usd,dsd,data.gA);
   cfcts2q(V0,A0,usd,dsd,data.gV);
   /* gVt */
   point_split_prop(0,usd,pusd);
   point_split_prop(0,dsd,pdsd);
   cfcts2q(S,A0,usd,pdsd,gX);
   cfcts2q(S,A0,pusd,dsd,data.gVt);
   /* g1 */
   ctrcts2q(V0,V0,u,d,&z);
  
   for(t=0;t<tmax;t++)
   {
      data.gS[t].re/=-2.0*nrm;
      data.gS[t].im/=-2.0*nrm;

      data.gP[t].re/=-2.0*nrm;
      data.gP[t].im/=-2.0*nrm;

      data.gA[t].re/=-2.0*nrm;
      data.gA[t].im/=-2.0*nrm;

      data.gV[t].re/=-2.0*nrm;
      data.gV[t].im/=-2.0*nrm;

      data.gVt[t].re-=gX[t].re;
      data.gVt[t].im-=gX[t].im;

      data.gVt[t].re/=-2.0*nrm;
      data.gVt[t].im/=-2.0*nrm;
   }

   data.g1[0].re=z.re/(2.0*nrm*nrm);
   data.g1[0].im=z.im/(2.0*nrm*nrm);

   /* lA,lA,lT,lTt,l1 */
   for(t=0;t<tmax;t++)
   {
      data.lA[t].re=0.0;
      data.lA[t].im=0.0;

      data.lV[t].re=0.0;
      data.lV[t].im=0.0;

      data.lVt[t].re=0.0;
      data.lVt[t].im=0.0;

      data.lT[t].re=0.0;
      data.lT[t].im=0.0;

      data.lTt[t].re=0.0;
      data.lTt[t].im=0.0;
   }
 
   data.l1[0].re=0.0;
   data.l1[0].im=0.0;

   for (k=0;k<3;k++)
   {      
      cfcts2q(A1+k,V1+k,usd,dsd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lA[t].re-=lX[t].re/(6.0*nrm);
         data.lA[t].im-=lX[t].im/(6.0*nrm);
      }
   
      cfcts2q(V1+k,V1+k,usd,dsd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lV[t].re-=lX[t].re/(6.0*nrm);
         data.lV[t].im-=lX[t].im/(6.0*nrm);
      }

      point_split_prop(k+1,usd,pusd);
      point_split_prop(k+1,dsd,pdsd);

      cfcts2q(S,V1+k,usd,pdsd,gX);  
      cfcts2q(S,V1+k,pusd,dsd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lVt[t].re-=(lX[t].re-gX[t].re)/(6.0*nrm);
         data.lVt[t].im-=(lX[t].im-gX[t].im)/(6.0*nrm);
      }
  
      cfcts2q(T01+k,V1+k,usd,dsd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lT[t].re-=lX[t].im/(6.0*nrm);
         data.lT[t].im+=lX[t].re/(6.0*nrm);
      }
   
      cfcts2q(Tt01+k,V1+k,usd,dsd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lTt[t].re-=lX[t].im/(6.0*nrm);
         data.lTt[t].im+=lX[t].re/(6.0*nrm);
      }

      ctrcts2q(A1+k,A1+k,u,d,&z);

      data.l1[0].re-=z.re/(6.0*nrm*nrm);
      data.l1[0].im-=z.im/(6.0*nrm*nrm);
   }   

   save_data(dat_file_dd);
   print_log("dd");

   /* ud */
   /* gS,gP,gA,gV */
   cfcts2q(S ,P,dsd,dsd,data.gS);
   cfcts2q(P ,P,dsd,dsd,data.gP);
   cfcts2q(A0,P,dsd,dsd,data.gA);
   cfcts2q(V0,P,dsd,dsd,data.gV);
   /* gVt */
   point_split_prop(0,dsd,pdsd);
   cfcts2q(S,P,pdsd,dsd,data.gVt);
   /* g1 */
   ctrcts2q(S,S,u,u,&z);

   for(t=0;t<tmax;t++)
   {
      data.gS[t].re/=-2.0*nrm;
      data.gS[t].im/=-2.0*nrm;

      data.gP[t].re/=-2.0*nrm;
      data.gP[t].im/=-2.0*nrm;

      data.gA[t].re/=-2.0*nrm;
      data.gA[t].im/=-2.0*nrm;

      data.gV[t].re/=-2.0*nrm;
      data.gV[t].im/=-2.0*nrm;

      data.gVt[t].re = 0.0;
      data.gVt[t].im/=-1.0*nrm;
   }

   data.g1[0].re=z.re/(2.0*nrm*nrm);
   data.g1[0].im=z.im/(2.0*nrm*nrm);

   /* lA,lA,lT,lTt,l1 */
   for(t=0;t<tmax;t++)
   {
      data.lA[t].re=0.0;
      data.lA[t].im=0.0;

      data.lV[t].re=0.0;
      data.lV[t].im=0.0;

      data.lVt[t].re=0.0;
      data.lVt[t].im=0.0;

      data.lT[t].re=0.0;
      data.lT[t].im=0.0;

      data.lTt[t].re=0.0;
      data.lTt[t].im=0.0;
   }
 
   data.l1[0].re=0.0;
   data.l1[0].im=0.0;

   for (k=0;k<3;k++)
   {      
      cfcts2q(A1+k,T01+k,dsd,dsd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lA[t].re-=lX[t].im/(6.0*nrm);
         data.lA[t].im+=lX[t].re/(6.0*nrm);
      }
   
      cfcts2q(V1+k,T01+k,dsd,dsd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lV[t].re-=lX[t].im/(6.0*nrm);
         data.lV[t].im+=lX[t].re/(6.0*nrm);
      }
   
      point_split_prop(k+1,dsd,pdsd);
      cfcts2q(S,T01+k,pdsd,dsd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lVt[t].re-=lX[t].im/(3.0*nrm);
         data.lVt[t].im=0.0;
      }
/* 
      point_split_prop(k+1,dsd,pdsd);
      cfcts2q(S,T01+k,dsd,pdsd,gX);
      cfcts2q(S,T01+k,pdsd,dsd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lVt[t].re-=(lX[t].im-gX[t].im)/(6.0*nrm);
         data.lVt[t].im+=(lX[t].re-gX[t].re)/(6.0*nrm);
      }
*/
      cfcts2q(T01+k,T01+k,dsd,dsd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lT[t].re+=lX[t].re/(6.0*nrm);
         data.lT[t].im+=lX[t].im/(6.0*nrm);
      }
   
      cfcts2q(Tt01+k,T01+k,dsd,dsd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lTt[t].re+=lX[t].re/(6.0*nrm);
         data.lTt[t].im+=lX[t].im/(6.0*nrm);
      }

      ctrcts2q(Tt01+k,Tt01+k,u,u,&z);

      data.l1[0].re+=z.re/(6.0*nrm*nrm);
      data.l1[0].im+=z.im/(6.0*nrm*nrm);
   }   

   save_data(dat_file_ud);
   print_log("ud");

   /* pdsd */
   release_wsd();
   /* dsd */
   release_wsd();

   /* du */
   /* gS,gP,gA,gV */
   cfcts2q(S ,P,usd,usd,data.gS);
   cfcts2q(P ,P,usd,usd,data.gP);
   cfcts2q(A0,P,usd,usd,data.gA);
   cfcts2q(V0,P,usd,usd,data.gV);
   /* gVt */ 
   point_split_prop(0,usd,pusd);
   cfcts2q(S,P,pusd,usd,data.gVt);
   /* g1 */
   ctrcts2q(S,S,d,d,&z);

   for(t=0;t<tmax;t++)
   {
      data.gS[t].re/=-2.0*nrm;
      data.gS[t].im/=-2.0*nrm;

      data.gP[t].re/=-2.0*nrm;
      data.gP[t].im/=-2.0*nrm;

      data.gA[t].re/=-2.0*nrm;
      data.gA[t].im/=-2.0*nrm;

      data.gV[t].re/=-2.0*nrm;
      data.gV[t].im/=-2.0*nrm;

      data.gVt[t].re = 0.0;
      data.gVt[t].im/=-1.0*nrm;
   }

   data.g1[0].re=z.re/(2.0*nrm*nrm);
   data.g1[0].im=z.im/(2.0*nrm*nrm);

   /* lA,lA,lT,lTt,l1 */
   for(t=0;t<tmax;t++)
   {
      data.lA[t].re=0.0;
      data.lA[t].im=0.0;

      data.lV[t].re=0.0;
      data.lV[t].im=0.0;

      data.lVt[t].re=0.0;
      data.lVt[t].im=0.0;

      data.lT[t].re=0.0;
      data.lT[t].im=0.0;

      data.lTt[t].re=0.0;
      data.lTt[t].im=0.0;
   }
 
   data.l1[0].re=0.0;
   data.l1[0].im=0.0;

   for (k=0;k<3;k++)
   {      
      cfcts2q(A1+k,T01+k,usd,usd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lA[t].re-=lX[t].im/(6.0*nrm);
         data.lA[t].im+=lX[t].re/(6.0*nrm);
      }
   
      cfcts2q(V1+k,T01+k,usd,usd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lV[t].re-=lX[t].im/(6.0*nrm);
         data.lV[t].im+=lX[t].re/(6.0*nrm);
      }
   
      point_split_prop(k+1,usd,pusd);
      cfcts2q(S,T01+k,pusd,usd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lVt[t].re-=lX[t].im/(3.0*nrm);
         data.lVt[t].im=0.0;
      } 

      cfcts2q(T01+k,T01+k,usd,usd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lT[t].re+=lX[t].re/(6.0*nrm);
         data.lT[t].im+=lX[t].im/(6.0*nrm);
      }
   
      cfcts2q(Tt01+k,T01+k,usd,usd,lX);

      for(t=0;t<tmax;t++)
      {
         data.lTt[t].re+=lX[t].re/(6.0*nrm);
         data.lTt[t].im+=lX[t].im/(6.0*nrm);
      }

      ctrcts2q(Tt01+k,Tt01+k,d,d,&z);

      data.l1[0].re+=z.re/(6.0*nrm*nrm);
      data.l1[0].im+=z.im/(6.0*nrm*nrm);
   }   

   /* save */
   save_data(dat_file_du);
   print_log("du");

   /* pusd */
   release_wsd();
   /* usd */
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
      error_root(ie!=1,1,"main [ms5_xsf.c]",
                 "Initial configuration has incorrect boundary values");

      if (dfl.Ns)
      {
         dfl_modes(status);
         error_root(status[0]<0,1,"main [ms5_xsf.c]",
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
         copy_file(dat_file_uu,dat_save_uu);
         copy_file(dat_file_dd,dat_save_dd);
         copy_file(dat_file_ud,dat_save_ud);
         copy_file(dat_file_du,dat_save_du);
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
