/*******************************************************************************
*
* File correlators.c
* Copyright (C) 2023 Emanuele Rosi
*
* Based on tm mesons
* Copyright (C) 2016 David Preti
*
* Based on mesons 
* Copyright (C) 2013, 2014 Tomasz Korzec
*
* Based on openQCD, ms1 and ms4 
* Copyright (C) 2012 Martin Luescher and Stefan Schaefer
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*******************************************************************************
*
* Computation of three points correlators for mesons oscillations
*
* Syntax: mesons -i <input file> [-noexp] [-a] -rndmgauge -nogauge
*
* For usage instructions see the file README.correlators
*
*******************************************************************************
*
* NOTES:
* Some comments as 'modified','unchanged','new function',... refer to file
* correlators.c, the originary one. Some comments explain some functions.
* Other comments are just my notes.
*
*******************************************************************************
*
* DATA:  static struct data.
* for each correlator I store each single contribution from 
* stochastic vectors eta1 and eta2 with each intermediate time t.
*
*******************************************************************************/

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
#include "linalg.h"
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "version.h"
#include "global.h"
#include "mesons.h"

/* new includes */
#include "OutputColors.h"
#include "uflds.h"
#include <time.h>
#include "gauge_transforms.h"

/* lattice sizes */
#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

/* available operator types */
#define SS 0
#define PP 1
#define SP 2
#define PS 3
#define VV 4
#define AA 5
#define VA 6
#define AV 7
#define TT 8
#define TTt 9
#define OPERATOR_MAX_TYPE 10

#define N_COLOURS 3
#define N_DIRAC 4

#define MAX(n,m) \
   if ((n)<(m)) \
      (n)=(m)

/************************ STRUCTURES ************************/

static struct
{
   complex_dble *corr;        /* correlators */
   complex_dble *corr_tmp;    /* temp. correlators */
   int nc;                    /* configuration number */
   int offset;                /* to switch to gauge transformed data */
} data;

static struct
{
   int ncorr;           /* number of correlators */
   int nnoise;          /* number of noise vectors */
   int noisetype;       /* noise type */
   int tvals;           /* number of timeslices */

   int x0;              /* x0 value - end time slice */
   int z0;              /* z0 value - source time slice */

   double *kappa1;      /* hopping parameters of the four choosen quark lines */
   double *kappa2;
   double *kappa3;
   double *kappa4;
   double *mus1;        /* twisted masses of the four choosen quark lines */
   double *mus2;
   double *mus3;
   double *mus4;

   int *typeA_x;        /* Dirac matrices A */
   int *typeC_z;        /* Dirac matrices C */
   int *operator_type;  /* Dirac matrices XY = DB */

   int *isreal;         /* =1 if the correlator is real. I do not use it */
} file_head;

static struct
{
   /* to evaluate xi^1_A (y)*/
   int *matrix_typeA;
   int *prop_type1;
   int len_1A;
   int *idx_1A;         /* idx_1A[ncorr] contains, for each correlator, the index idx[i] for which
                           typeA[i]=matrix_type1[idx[i]] and prop1[i]=prop_type1[idx[i]]
                           similar for the rest of the proplist */

   /* to evaluate xi^3_C (y)*/
   int *matrix_typeC;
   int *prop_type3;
   int len_3C;
   int *idx_3C;

   /* to evaluate zeta^2 (y) */
   int *prop_type2;
   int len2;
   int *idx_2;

   /* to evaluate zeta^4 (y) */
   int *prop_type4;
   int len4;
   int *idx_4;

} proplist;

/************************ STATICS ************************/

static char line[NAME_SIZE+1];   /* useful string */
static char str_type[4];         /* useful string */

static int my_rank;                                      /* run rank */
static int noexp,append,norng,endian,nogauge,rndmgauge;  /* running options */
static int first,last,step;                              /* configurations indices */
static int level,seed;                                   /* random number generator */
static int ipgrd[2],*rlxs_state=NULL,*rlxd_state=NULL;   /* random number generator */

static int nprop,ncorr,nnoise,noisetype;                 /* file_head */         
static int fixed_x0,fixed_z0;                            /* file_head */
static int tvals;                                        /* number of timeslices */
static int *typeA,*typeC,*operator_type;                 /* Dirac matrices types*/   
static int *isps,*props1,*props2,*props3,*props4;        /* propagators types */    
static double *kappas,*mus;                              /* propagators parameters */

/* directories, files, file names and similars */
static char log_dir[NAME_SIZE],loc_dir[NAME_SIZE], cnfg_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE],end_file[NAME_SIZE];
static char dat_file[NAME_SIZE],dat_save[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char rng_file[NAME_SIZE],rng_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],run_name[NAME_SIZE],output_name[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fend=NULL,*fdat=NULL;

/************************ FUNCTIONS ************************/

static void lonfo(void)
{
   error(1,1,"[correlators.c]","Il lonfo non vaterca, né brigatta.");
}

static int check_null_spinor(spinor_dble *psi,char *str)
{
   double variable;
   int i_pos,i_time,idx,counter=0;

   printf("%s\n",str);
   printf("Prima locazione: %p\n",(void *)&(psi[0]));
   for(i_time=0;i_time<L0;i_time++)
   {
      for (i_pos=0;i_pos<L1*L2*L3;i_pos++)
      {
         idx=ipt[i_pos+i_time*L1*L2*L3];

         variable = psi[idx].c1.c1.re;
         if(variable!=0)
         {
            printf("Non null value c1.c1.re: %f\n",variable);
            counter++;  
         }
         variable = psi[idx].c2.c2.re;
         if(variable!=0)
         {
            printf("Non null value c2.c1.re: %f\n",variable);
            counter++;
         }
         if(i_time==(L0-1)&&i_pos==(L1*L2*L3-1))
         printf("Ultima locazione: %p\n",(void *)&(psi[idx]));
      }
   }
   
   if(counter==0)
      return 0;
   else
      return 1;
}

/* this function turns back the color index from an integer */

extern int colour_index(int idx)
{
   int colour;
   if(idx<0||idx>=N_COLOURS*N_DIRAC)
      error(1,1,"colour_index [correlators.c]","Invalid index number.");

   colour=idx%N_COLOURS;

   if(colour<0||colour>2)
      error(1,1,"colour_index [correlators.c]","We are working in SU(3).");

   return colour;
}

/* this function turns back the Dirac index from an integer */

extern int dirac_index(int idx)
{
   int remainder,result;

   if(idx<0||idx>=N_COLOURS*N_DIRAC)
      error(1,1,"[correlators.c]","Invalid index number.");

   remainder=idx%N_COLOURS;
   result=(int)(idx-remainder)/N_COLOURS;

   if(result<0 || result>3)
      error(1,1,"dirac_index [correlators.c]","Dirac spinor has 4 components.");

   return result;
}

/************************ CORRELATORS FUNCTIONS ************************/

static void alloc_data(void)  /*modified*/
{
   int number_of_data = file_head.ncorr*file_head.nnoise*file_head.nnoise*file_head.tvals;

   data.offset=0;
   data.corr=malloc(2*number_of_data*sizeof(complex_dble));
      /* additive x2 because I havo two sets of data:
         - first set as usual
         - second set for gauge transformed data */
   data.corr_tmp=malloc(number_of_data*sizeof(complex_dble));

   error((data.corr==NULL)||(data.corr_tmp==NULL),1,"alloc_data [correlators.c]", "Unable to allocate data arrays");
}

static char* operator_to_string(int type) /*new function*/
{
   switch (type)
   {
   case SS:
      sprintf(str_type,"SS");
      break;
   case PP:
      sprintf(str_type,"PP");
      break;
   case SP:
      sprintf(str_type,"SP");
      break;
   case PS:
      sprintf(str_type,"PS");
      break;
   case VV:
      sprintf(str_type,"VV");
      break;
   case AA:
      sprintf(str_type,"AA");
      break;
   case VA:
      sprintf(str_type,"VA");
      break;
   case AV:
      sprintf(str_type,"AV");
      break;
   case TT:
      sprintf(str_type,"TT");
      break;
   case TTt:
      sprintf(str_type,"TTt");
      break;
   default:
      error(1,1,"operator_to_string [correlators.c]","Unknown operator type");
      break;
   }
   return str_type;
}

static void write_file_head(void)   /*modified*/
{
   stdint_t istd[1];
   int iw=0;
   int i;
   double dbl[1];

   istd[0]=(stdint_t)(file_head.ncorr);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw=fwrite(istd,sizeof(stdint_t),1,fdat);

   istd[0]=(stdint_t)(file_head.nnoise);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   istd[0]=(stdint_t)(file_head.tvals);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   istd[0]=(stdint_t)(file_head.noisetype);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   istd[0]=(stdint_t)(file_head.x0);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   istd[0]=(stdint_t)(file_head.z0);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   error_root(iw!=6,1,"write_file_head [correlators.c]", "Incorrect write count");

   for (i=0;i<file_head.ncorr;i++)
   {
      dbl[0] = file_head.kappa1[i];
      if (endian==BIG_ENDIAN)
         bswap_double(1,dbl);
      iw=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.kappa2[i];
      if (endian==BIG_ENDIAN)
         bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.kappa3[i];
      if (endian==BIG_ENDIAN)
         bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.kappa4[i];
      if (endian==BIG_ENDIAN)
         bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.mus1[i];
      if (endian==BIG_ENDIAN)
         bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.mus2[i];
      if (endian==BIG_ENDIAN)
         bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.mus3[i];
      if (endian==BIG_ENDIAN)
         bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.mus4[i];
      if (endian==BIG_ENDIAN)
         bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      istd[0]=(stdint_t)(file_head.typeA_x[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

      istd[0]=(stdint_t)(file_head.typeC_z[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

      istd[0]=(stdint_t)(file_head.operator_type[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

      istd[0]=(stdint_t)(file_head.isreal[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

      error_root(iw!=12,1,"write_file_head [correlators.c]","Incorrect write count");
   }
}

static void check_file_head(void)   /*modified*/
{
   int i,ir,ie;
   stdint_t istd[1];
   double dbl[1];

   ir=fread(istd,sizeof(stdint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie=(istd[0]!=(stdint_t)(file_head.ncorr));

   ir+=fread(istd,sizeof(stdint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie+=(istd[0]!=(stdint_t)(file_head.nnoise));

   ir+=fread(istd,sizeof(stdint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie+=(istd[0]!=(stdint_t)(file_head.tvals));

   ir+=fread(istd,sizeof(stdint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie+=(istd[0]!=(stdint_t)(file_head.noisetype));

   ir+=fread(istd,sizeof(stdint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie+=(istd[0]!=(stdint_t)(file_head.x0));

   ir+=fread(istd,sizeof(stdint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie+=(istd[0]!=(stdint_t)(file_head.z0));

   error_root(ir!=6,1,"check_file_head [correlators.c]","Incorrect read count");
   error_root(ie!=0,1,"check_file_head [correlators.c]", "Unexpected value of ncorr, nnoise, tvals or noisetype");
   
   for (i=0;i<file_head.ncorr;i++)
   {
      ir=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie=(dbl[0]!=file_head.kappa1[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.kappa2[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.kappa3[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.kappa4[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.mus1[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.mus2[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.mus3[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.mus4[i]);

      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.typeA_x[i]));

      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.typeC_z[i]));

      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.operator_type[i]));

      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.isreal[i]));

      error_root(ir!=12,1,"check_file_head [correlators.c]","Incorrect read count");
      error_root(ie!=0,1,"check_file_head [correlators.c]","Unexpected value of kappa, mu, typeA, typeC, OperatorType");
   }
}

static void write_data(void)  /*modified*/
{
   int iw;
   int nw;
   int chunk;
   int icorr,i;
   int offset=0;

   if (my_rank==0)
   {
      fdat=fopen(dat_file,"ab");
      error_root(fdat==NULL,1,"write_data [correlators.c]","Unable to open dat file");

      nw = 1;
      if(endian==BIG_ENDIAN)
      {
         bswap_double(2*file_head.nnoise*file_head.nnoise*file_head.tvals*file_head.ncorr*2,
                      data.corr);   /* additive x2*/
         bswap_int(1,&(data.nc));
      }
      iw=fwrite(&(data.nc),sizeof(int),1,fdat);
      for (icorr=0;icorr<file_head.ncorr;icorr++)
      {
         chunk=file_head.nnoise*file_head.nnoise*file_head.tvals*(2-file_head.isreal[icorr]);
         nw+=chunk;
         if (file_head.isreal[icorr])
         {
            for (i=0;i<chunk;i++)
               iw+=fwrite(&(data.corr[icorr*file_head.tvals*file_head.nnoise*file_head.nnoise+i]),
                       sizeof(double),1,fdat);
         }else
         {
            iw+=fwrite(&(data.corr[icorr*file_head.tvals*file_head.nnoise*file_head.nnoise]),
                       sizeof(double),chunk,fdat);
         }
      }
      /* BEGIN: write Gauge tranformed data */
      offset=file_head.nnoise*file_head.nnoise*file_head.tvals*file_head.ncorr;
      for (icorr=0;icorr<file_head.ncorr;icorr++)
      {
         chunk=file_head.nnoise*file_head.nnoise*file_head.tvals*(2-file_head.isreal[icorr]);
         nw+=chunk;
         if (file_head.isreal[icorr])
         {
            for (i=0;i<chunk;i++)
               iw+=fwrite(&(data.corr[offset+icorr*file_head.tvals*file_head.nnoise*file_head.nnoise+i]),
                       sizeof(double),1,fdat);
         }else
         {
            iw+=fwrite(&(data.corr[offset+icorr*file_head.tvals*file_head.nnoise*file_head.nnoise]),
                       sizeof(double),chunk,fdat);
         }
      }
      /* END: write Gauge tranformed data */
      if(endian==BIG_ENDIAN)
      {
         bswap_double(2*file_head.nnoise*file_head.nnoise*file_head.tvals*file_head.ncorr*2,
                      data.corr);
         bswap_int(1,&(data.nc));
      }
      error_root(iw!=nw,1,"write_data [correlators.c]","Incorrect write count");
      fclose(fdat);
   }
}

static int read_data(void) /*modified*/
{
   int ir;
   int nr;
   int chunk;
   int icorr,i;
   double zero;

   zero=0;
   if(endian==BIG_ENDIAN)
      bswap_double(1,&zero);
   nr=1;
   ir=fread(&(data.nc),sizeof(int),1,fdat);

   for (icorr=0;icorr<file_head.ncorr;icorr++)
   {
      chunk=file_head.nnoise*file_head.nnoise*file_head.tvals*(2-file_head.isreal[icorr]);
      nr+=chunk;
      if (file_head.isreal[icorr])
      {
         for (i=0;i<chunk;i++)
         {
            ir+=fread(&(data.corr[icorr*file_head.tvals*file_head.nnoise*file_head.nnoise+i]),
                    sizeof(double),1,fdat);
            data.corr[icorr*file_head.tvals*file_head.nnoise*file_head.nnoise+i].im=zero;
         }
      }else
      {
         ir+=fread(&(data.corr[icorr*file_head.tvals*file_head.nnoise*file_head.nnoise]),
                    sizeof(double),chunk,fdat);
      }
   }

   if (ir==0)
      return 0;

   error_root(ir!=nr,1,"read_data [correlators.c]","Read error or incomplete data record");
   if(endian==BIG_ENDIAN)
   {
      bswap_double(nr,data.corr);
      bswap_int(1,&(data.nc));
   }
   return 1;
}

static void read_dirs(void)   /*untouched*/
{
   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",run_name);
      read_line_opt("output",run_name,"%s",output_name);

      find_section("Directories");
      read_line("log_dir","%s",log_dir);
      read_line("dat_dir","%s",dat_dir);
      if (noexp)
      {
         read_line("loc_dir","%s",loc_dir);
         cnfg_dir[0]='\0';
      }
      else if(nogauge||rndmgauge)
      {
         cnfg_dir[0]='\0';
         loc_dir[0]='\0';
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
                 "read_dirs [correlators.c]","Improper configuration range");

      find_section("Random number generator");
      read_line("level","%d",&level);
      read_line("seed","%d",&seed);
   }

   MPI_Bcast(run_name,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(output_name,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(loc_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(dat_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&level,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);
}

static void setup_files(void) /*untouched*/
{
   if (noexp)
      error_root(name_size("%s/%sn%d_%d",loc_dir,run_name,last,NPROC-1)>=NAME_SIZE,
                 1,"setup_files [correlators.c]","loc_dir name is too long");
   else if ((!nogauge)&&(!rndmgauge))
      error_root(name_size("%s/%sn%d",cnfg_dir,run_name,last)>=NAME_SIZE,
                 1,"setup_files [correlators.c]","cnfg_dir name is too long");

   check_dir_root(dat_dir);
   error_root(name_size("%s/%s.mesons.dat~",dat_dir,output_name)>=NAME_SIZE,
              1,"setup_files [correlators.c]","dat_dir name is too long");

   check_dir_root(log_dir);
   error_root(name_size("%s/%s.mesons.log~",log_dir,output_name)>=NAME_SIZE,
              1,"setup_files [correlators.c]","log_dir name is too long");

   sprintf(log_file,"%s/%s.correlators.log",log_dir,output_name);
   sprintf(end_file,"%s/%s.correlators.end",log_dir,output_name);
   sprintf(par_file,"%s/%s.correlators.par",dat_dir,output_name);
   sprintf(dat_file,"%s/%s.correlators.dat",dat_dir,output_name);
   sprintf(rng_file,"%s/%s.correlators.rng",dat_dir,output_name);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);
   sprintf(dat_save,"%s~",dat_file);
   sprintf(rng_save,"%s~",rng_file);
}

static void read_lat_parms(void) /*modified*/
{
   double csw,cF;
   char tmpstring[NAME_SIZE],tmpstring2[NAME_SIZE];
   int iprop,icorr,eoflg;

   if (my_rank==0)
   {
      find_section("Measurements");
      read_line("nprop","%d",&nprop);
      read_line("ncorr","%d",&ncorr);
      /*read_line("nnoise","%d",&nnoise);*/
      nnoise=N_DIRAC*N_COLOURS;
      error_root(((nprop<1)||(ncorr<1)||(nnoise<1)),1,"read_lat_parms [correlators.c]",
                 "Specified nprop/ncorr/nnoise must be larger than zero");
      
      read_line("noise_type","%s",tmpstring);
      noisetype=-1;
      if(strcmp(tmpstring,"Z2")==0)
         noisetype=Z2_NOISE;
      if(strcmp(tmpstring,"GAUSS")==0)
         noisetype=GAUSS_NOISE;
      if(strcmp(tmpstring,"U1")==0)
         noisetype=U1_NOISE;
      error_root(noisetype==-1,1,"read_lat_parms [correlators.c]",
                 "Unknown noise type");

      read_line("csw","%lf",&csw);
      read_line("cF","%lf",&cF);
      read_line("eoflg","%d",&eoflg);
      error_root(((eoflg!=0)&&(eoflg!=1)),1,"read_lat_parms [correlators.c]",
		 "Specified eoflg must be 0,1");
      
      read_line("x0","%d",&fixed_x0);
      read_line("z0","%d",&fixed_z0);
      error_root((fixed_x0>N0)||(fixed_x0<0)||(fixed_z0>N0)||(fixed_z0<0)||(fixed_z0>fixed_x0),1,"read_lat_parms [correlators.c]",
                 "x0 and z0 must be in the range [0,N0], with x0>z0");
   }

   MPI_Bcast(&nprop,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncorr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nnoise,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&noisetype,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&eoflg,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&fixed_x0,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&fixed_z0,1,MPI_INT,0,MPI_COMM_WORLD);

   kappas=malloc(nprop*sizeof(double));
   mus=malloc(nprop*sizeof(double));
   isps=malloc(nprop*sizeof(int));
   props1=malloc(ncorr*sizeof(int));
   props2=malloc(ncorr*sizeof(int));
   props3=malloc(ncorr*sizeof(int));
   props4=malloc(ncorr*sizeof(int));
   typeA=malloc(ncorr*sizeof(int));
   typeC=malloc(ncorr*sizeof(int));
   operator_type=malloc(ncorr*sizeof(int));
   file_head.kappa1=malloc(ncorr*sizeof(double));
   file_head.kappa2=malloc(ncorr*sizeof(double));
   file_head.kappa3=malloc(ncorr*sizeof(double));
   file_head.kappa4=malloc(ncorr*sizeof(double));
   file_head.mus1=malloc(ncorr*sizeof(double));
   file_head.mus2=malloc(ncorr*sizeof(double));
   file_head.mus3=malloc(ncorr*sizeof(double));
   file_head.mus4=malloc(ncorr*sizeof(double));
   file_head.typeA_x=typeA;
   file_head.typeC_z=typeC;
   file_head.operator_type=operator_type;
   file_head.x0=fixed_x0;
   file_head.z0=fixed_z0;
   file_head.isreal=malloc(ncorr*sizeof(int));

   error((kappas==NULL)||(mus==NULL)||(isps==NULL)||
         (props1==NULL)||(props2==NULL)||(props3==NULL)||(props4==NULL)||
         (typeA==NULL)||(typeC==NULL)||
         (file_head.kappa1==NULL)||(file_head.kappa2==NULL)||(file_head.kappa3==NULL)||(file_head.kappa4==NULL)||
         (file_head.mus1==NULL)||(file_head.mus2==NULL)||(file_head.mus3==NULL)||(file_head.mus4==NULL)||
         (file_head.operator_type==NULL)||(file_head.isreal==NULL),
         1,"read_lat_parms [correlators.c]","Out of memory");

   if (my_rank==0)
   {
      for(iprop=0; iprop<nprop; iprop++)
      {
         sprintf(tmpstring,"Propagator %i",iprop);
         find_section(tmpstring);
         read_line("kappa","%lf",&kappas[iprop]);
         read_line("isp","%d",&isps[iprop]);
	      read_line("mus","%lf",&mus[iprop]);
      }
      for(icorr=0; icorr<ncorr; icorr++)
      {
         sprintf(tmpstring,"Correlator %i",icorr);
         find_section(tmpstring);

         read_line("iprop","%d %d %d %d",&props1[icorr],&props2[icorr],&props3[icorr],&props4[icorr]);
         error_root((props1[icorr]<0)||(props1[icorr]>=nprop),1,"read_lat_parms [correlators.c]",
                 "Propagator index out of range");
         error_root((props2[icorr]<0)||(props2[icorr]>=nprop),1,"read_lat_parms [correlators.c]",
                 "Propagator index out of range");
         error_root((props3[icorr]<0)||(props3[icorr]>=nprop),1,"read_lat_parms [correlators.c]",
                 "Propagator index out of range");
         error_root((props4[icorr]<0)||(props4[icorr]>=nprop),1,"read_lat_parms [correlators.c]",
                 "Propagator index out of range");

         read_line("type_sources","%s %s",tmpstring,tmpstring2);
         typeA[icorr]=-1;
         typeC[icorr]=-1;

         if(strncmp(tmpstring,"1",1)==0)
            typeA[icorr]=ONE_TYPE;
         else if(strncmp(tmpstring,"G0G1",4)==0)
            typeA[icorr]=GAMMA0GAMMA1_TYPE;
         else if(strncmp(tmpstring,"G0G2",4)==0)
            typeA[icorr]=GAMMA0GAMMA2_TYPE;
         else if(strncmp(tmpstring,"G0G3",4)==0)
            typeA[icorr]=GAMMA0GAMMA3_TYPE;
         else if(strncmp(tmpstring,"G0G5",4)==0)
            typeA[icorr]=GAMMA0GAMMA5_TYPE;
         else if(strncmp(tmpstring,"G1G2",4)==0)
            typeA[icorr]=GAMMA1GAMMA2_TYPE;
         else if(strncmp(tmpstring,"G1G3",4)==0)
            typeA[icorr]=GAMMA1GAMMA3_TYPE;
         else if(strncmp(tmpstring,"G1G5",4)==0)
            typeA[icorr]=GAMMA1GAMMA5_TYPE;
         else if(strncmp(tmpstring,"G2G3",4)==0)
            typeA[icorr]=GAMMA2GAMMA3_TYPE;
         else if(strncmp(tmpstring,"G2G5",4)==0)
            typeA[icorr]=GAMMA2GAMMA5_TYPE;
         else if(strncmp(tmpstring,"G3G5",4)==0)
            typeA[icorr]=GAMMA3GAMMA5_TYPE;
         else if(strncmp(tmpstring,"G0",2)==0)
            typeA[icorr]=GAMMA0_TYPE;
         else if(strncmp(tmpstring,"G1",2)==0)
            typeA[icorr]=GAMMA1_TYPE;
         else if(strncmp(tmpstring,"G2",2)==0)
            typeA[icorr]=GAMMA2_TYPE;
         else if(strncmp(tmpstring,"G3",2)==0)
            typeA[icorr]=GAMMA3_TYPE;
         else if(strncmp(tmpstring,"G5",2)==0)
            typeA[icorr]=GAMMA5_TYPE;

         if(strncmp(tmpstring2,"1",1)==0)
            typeC[icorr]=ONE_TYPE;
         else if(strncmp(tmpstring2,"G0G1",4)==0)
            typeC[icorr]=GAMMA0GAMMA1_TYPE;
         else if(strncmp(tmpstring2,"G0G2",4)==0)
            typeC[icorr]=GAMMA0GAMMA2_TYPE;
         else if(strncmp(tmpstring2,"G0G3",4)==0)
            typeC[icorr]=GAMMA0GAMMA3_TYPE;
         else if(strncmp(tmpstring2,"G0G5",4)==0)
            typeC[icorr]=GAMMA0GAMMA5_TYPE;
         else if(strncmp(tmpstring2,"G1G2",4)==0)
            typeC[icorr]=GAMMA1GAMMA2_TYPE;
         else if(strncmp(tmpstring2,"G1G3",4)==0)
            typeC[icorr]=GAMMA1GAMMA3_TYPE;
         else if(strncmp(tmpstring2,"G1G5",4)==0)
            typeC[icorr]=GAMMA1GAMMA5_TYPE;
         else if(strncmp(tmpstring2,"G2G3",4)==0)
            typeC[icorr]=GAMMA2GAMMA3_TYPE;
         else if(strncmp(tmpstring2,"G2G5",4)==0)
            typeC[icorr]=GAMMA2GAMMA5_TYPE;
         else if(strncmp(tmpstring2,"G3G5",4)==0)
            typeC[icorr]=GAMMA3GAMMA5_TYPE;
         else if(strncmp(tmpstring2,"G0",2)==0)
            typeC[icorr]=GAMMA0_TYPE;
         else if(strncmp(tmpstring2,"G1",2)==0)
            typeC[icorr]=GAMMA1_TYPE;
         else if(strncmp(tmpstring2,"G2",2)==0)
            typeC[icorr]=GAMMA2_TYPE;
         else if(strncmp(tmpstring2,"G3",2)==0)
            typeC[icorr]=GAMMA3_TYPE;
         else if(strncmp(tmpstring2,"G5",2)==0)
            typeC[icorr]=GAMMA5_TYPE;

         read_line("operator","%s",tmpstring);
         operator_type[icorr]=-1;

         if(strncmp(tmpstring,"SS",2)==0)
            operator_type[icorr]=SS;
         else if(strncmp(tmpstring,"PP",2)==0)
            operator_type[icorr]=PP;
         else if(strncmp(tmpstring,"SP",2)==0)
            operator_type[icorr]=SP;
         else if(strncmp(tmpstring,"PS",2)==0)
            operator_type[icorr]=PS;
         else if(strncmp(tmpstring,"VV",2)==0)
            operator_type[icorr]=VV;
         else if(strncmp(tmpstring,"AA",2)==0)
            operator_type[icorr]=AA;
         else if(strncmp(tmpstring,"VA",2)==0)
            operator_type[icorr]=VA;
         else if(strncmp(tmpstring,"AV",2)==0)
            operator_type[icorr]=AV;
         else if(strncmp(tmpstring,"TTt",3)==0)
            operator_type[icorr]=TTt;
         else if(strncmp(tmpstring,"TT",2)==0)
            operator_type[icorr]=TT;

         error_root((typeA[icorr]==-1)||(typeC[icorr]==-1)||(operator_type[icorr]==-1),1,"read_lat_parms [correlators.c]",
                 "Unknown or unsupported Dirac structures or intermediate operator");
      }
   }

   MPI_Bcast(kappas,nprop,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(mus,nprop,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&eoflg,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(isps,nprop,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(props1,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(props2,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(props3,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(props4,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(typeA,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(typeC,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(operator_type,ncorr,MPI_INT,0,MPI_COMM_WORLD);

   set_lat_parms(0.0,1.0,kappas[0],0.0,0.0,csw,1.0,cF);
   set_sw_parms(sea_quark_mass(0));
   set_tm_parms(eoflg);

   file_head.ncorr = ncorr;
   file_head.nnoise = nnoise;
   file_head.tvals = NPROC0*L0;
   tvals = NPROC0*L0;
   file_head.noisetype = noisetype;
   for(icorr=0; icorr<ncorr; icorr++)
   {
      file_head.kappa1[icorr]=kappas[props1[icorr]];
      file_head.kappa2[icorr]=kappas[props2[icorr]];
      file_head.kappa3[icorr]=kappas[props3[icorr]];
      file_head.kappa4[icorr]=kappas[props4[icorr]];

      file_head.mus1[icorr]=mus[props1[icorr]];
      file_head.mus2[icorr]=mus[props2[icorr]];
      file_head.mus3[icorr]=mus[props3[icorr]];
      file_head.mus4[icorr]=mus[props4[icorr]];

      /*if ((typeA[icorr]==GAMMA5_TYPE)&&(typeC[icorr]==GAMMA5_TYPE)&&
          (props1[icorr]==props2[icorr]))
         file_head.isreal[icorr]=1;
      else
         file_head.isreal[icorr]=0;*/
      /* substituted with: */
      file_head.isreal[icorr]=0;
   }
   if (append)
      check_lat_parms(fdat);
   else
      write_lat_parms(fdat);
}

static void read_sap_parms(void) /*untouched*/
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

static void read_dfl_parms(void) /*untouched*/
{
   int bs[4],Ns;
   int ninv,nmr,ncy,nkv,nmx;
   double kappa,mudfl,res;

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
      read_line("mu","%lf",&mudfl);
      read_line("ninv","%d",&ninv);
      read_line("nmr","%d",&nmr);
      read_line("ncy","%d",&ncy);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mudfl,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&ninv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);
   set_dfl_gen_parms(kappa,mudfl,ninv,nmr,ncy);

   if (my_rank==0)
   {
      find_section("Deflation projection");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_pro_parms(nkv,nmx,res);
}

static void read_solvers(void)   /*untouched*/
{
   solver_parms_t sp;
   int i,j;
   int isap=0,idfl=0;

   for (i=0;i<nprop;i++)
   {
      j=isps[i];
      sp=solver_parms(j);
      if (sp.solver==SOLVERS)
      {
         read_solver_parms(j);
         sp=solver_parms(j);
         if (sp.solver==SAP_GCR)
            isap=1;
         if (sp.solver==DFL_SAP_GCR)
         {
            isap=1;
            idfl=1;
         }
      }
   }

   if (isap)
      read_sap_parms();

   if (idfl)
      read_dfl_parms();
}

static void read_infile(int argc,char *argv[])  /*modified*/
{
   int ifile;
   int error_exclusive_options;

   if (my_rank==0)
   {
      flog=freopen("STARTUP_ERROR","w",stdout);
 
      ifile=find_opt(argc,argv,"-i");
      endian=endianness();

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [correlators.c]",
                 "Syntax: mesons -i <input file> [-noexp] [-nogauge] [-rndmgauge] [-a [-norng]]");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [correlators.c]",
                 "Machine has unknown endianness");

      noexp=find_opt(argc,argv,"-noexp");
      nogauge=find_opt(argc,argv,"-nogauge");
      rndmgauge=find_opt(argc,argv,"-rndmgauge");
      append=find_opt(argc,argv,"-a");
      norng=find_opt(argc,argv,"-norng");

      /* the options -noexp -nogauge -rndmgauge must be exclusive */
      error_exclusive_options=(int)(noexp>0)+(int)(nogauge>0)+(int)(rndmgauge>0);
      error_root(error_exclusive_options>1,1,"read_infile [correlators.c]",
            "Invalid flags. Remember that you can use only one between -noexp,-nogauge,-rndmgauge ");

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [correlators.c]","Unable to open input file");
   }

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&noexp,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nogauge,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&rndmgauge,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&append,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&norng,1,MPI_INT,0,MPI_COMM_WORLD);

   read_dirs();
   setup_files();

   if (my_rank==0)
   {
      if (append)
         fdat=fopen(par_file,"rb");
      else
         fdat=fopen(par_file,"wb");

      error_root(fdat==NULL,1,"read_infile [correlators.c]",
                 "Unable to open parameter file");
   }
   read_lat_parms();
   read_solvers();

   if (my_rank==0)
   {
      fclose(fin);
      fclose(fdat);

      if (append==0)
         copy_file(par_file,par_save);
   }
}

static void check_old_log(int *fst,int *lst,int *stp) /*untouched*/
{
   int ie,ic,isv;
   int fc,lc,dc,pc;
   int np[4],bp[4];

   fend=fopen(log_file,"r");
   error_root(fend==NULL,1,"check_old_log [correlators.c]","Unable to open log file");

   fc=0;
   lc=0;
   dc=0;
   pc=0;

   ie=0x0;
   ic=0;
   isv=0;

   while (fgets(line,NAME_SIZE,fend)!=NULL)
   {
      if (strstr(line,"process grid")!=NULL)
      {
         if (sscanf(line,"%dx%dx%dx%d process grid, %dx%dx%dx%d",
                    np,np+1,np+2,np+3,bp,bp+1,bp+2,bp+3)==8)
         {
            ipgrd[0]=((np[0]!=NPROC0)||(np[1]!=NPROC1)||
                      (np[2]!=NPROC2)||(np[3]!=NPROC3));
            ipgrd[1]=((bp[0]!=NPROC0_BLK)||(bp[1]!=NPROC1_BLK)||
                      (bp[2]!=NPROC2_BLK)||(bp[3]!=NPROC3_BLK));
         }
         else
            ie|=0x1;
      }
      
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

   error_root((ie&0x1)!=0x0,1,"check_old_log [correlators.c]",
              "Incorrect read count");
   error_root((ie&0x2)!=0x0,1,"check_old_log [correlators.c]",
              "Configuration numbers are not equally spaced");
   error_root(isv==0,1,"check_old_log [correlators.c]",
              "Log file extends beyond the last configuration save");

   (*fst)=fc;
   (*lst)=lc;
   (*stp)=dc;
}

static void check_old_dat(int fst,int lst,int stp) /*untouched*/
{
   int ie,ic;
   int fc,lc,dc,pc;

   fdat=fopen(dat_file,"rb");
   error_root(fdat==NULL,1,"check_old_dat [correlators.c]",
              "Unable to open data file");

   check_file_head();

   fc=0;
   ic=0;
   lc=0;
   dc=0;
   pc=0;
   ie=0x0;

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

   error_root(ic==0,1,"check_old_dat [correlators.c]","No data records found");
   error_root((ie&0x1)!=0x0,1,"check_old_dat [correlators.c]","Configuration numbers are not equally spaced");
   error_root((fst!=fc)||(lst!=lc)||(stp!=dc),1,"check_old_dat [correlators.c]","Configuration range is not as reported in the log file");
}

static void check_files(void) /*untouched*/
{
   int fst,lst,stp;

   ipgrd[0]=0;
   ipgrd[1]=0;
   
   if (my_rank==0)
   {
      if (append)
      {
         check_old_log(&fst,&lst,&stp);
         check_old_dat(fst,lst,stp);

         error_root((fst!=lst)&&(stp!=step),1,"check_files [correlators.c]",
                    "Continuation run:\n"
                    "Previous run had a different configuration separation");
         error_root(first!=lst+step,1,"check_files [correlators.c]",
                    "Continuation run:\n"
                    "Configuration range does not continue the previous one");
      }
      else
      {
         fin=fopen(log_file,"r");
         error_root(fin!=NULL,1,"check_files [correlators.c]",
                    "Attempt to overwrite old *.log file");
         fdat=fopen(dat_file,"r");
         error_root(fdat!=NULL,1,"check_files [correlators.c]",
                    "Attempt to overwrite old *.dat file");
         fdat=fopen(dat_file,"wb");
         error_root(fdat==NULL,1,"check_files [correlators.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);
      }
   }
}

static void print_info(void)  /*modified*/
{
   int i,isap,idfl;
   long ip;
   lat_parms_t lat;
   tm_parms_t tm;

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

      error_root(flog==NULL,1,"print_info [correlators.c]",
                 "Unable to open log file");
      printf("\n");

      if (append)
         printf("Continuation run\n\n");
      else
      {
         printf("Computation of meson correlators\n");
         printf("--------------------------------\n\n");
         printf("cnfg   base name: %s\n",run_name);
         printf("output base name: %s\n\n",output_name);
      }

      printf("openQCD version: %s, correlator version: %s\n",openQCD_RELEASE,correlators_RELEASE);
      if (endian==LITTLE_ENDIAN)
         printf("The machine is little endian\n");
      else
         printf("The machine is big endian\n");
      if (noexp)
         printf("Configurations are read in imported file format\n\n");
      else
         printf("Configurations are read in exported file format\n\n");

      if ((ipgrd[0]!=0)&&(ipgrd[1]!=0))
         printf("Process grid and process block size changed:\n");
      else if (ipgrd[0]!=0)
         printf("Process grid changed:\n");
      else if (ipgrd[1]!=0)
         printf("Process block size changed:\n");

      if ((append==0)||(ipgrd[0]!=0)||(ipgrd[1]!=0))
      {
         printf("%dx%dx%dx%d lattice, ",N0,N1,N2,N3);
         printf("%dx%dx%dx%d local lattice\n",L0,L1,L2,L3);
         printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
         printf("%dx%dx%dx%d process block size\n",NPROC0_BLK,NPROC1_BLK,NPROC2_BLK,NPROC3_BLK);

         if (append)
            printf("\n");
         else
            printf("SF boundary conditions on the quark fields\n\n");   /*???*/
      }
      
      
      if (append)
      {
         printf("Random number generator:\n");

         if (norng)
            printf("level = %d, seed = %d, effective seed = %d\n\n",
                   level,seed,seed^(first-step));
         else
         {
            printf("State of ranlxs and ranlxd reset to the\n");
            printf("last exported state\n\n");
         }
      }
      else
      {
         printf("Random number generator:\n");
         printf("level = %d, seed = %d\n\n",level,seed);

         lat=lat_parms();

         printf("Measurements:\n");
         printf("nprop     = %i\n",nprop);
         printf("ncorr     = %i\n",ncorr);
         printf("nnoise    = %i\n",nnoise);
         if (noisetype==Z2_NOISE)
            printf("noisetype = Z2\n");
         if (noisetype==GAUSS_NOISE)
            printf("noisetype = GAUSS\n");
         if (noisetype==U1_NOISE)
            printf("noisetype = U1\n");
         printf("csw       = %.6f\n",lat.csw);
         printf("cF        = %.6f\n\n",lat.cF);
	      printf("eoflg     = %i\n",tm.eoflg);
         printf("x0        = %i\n",fixed_x0);
         printf("z0        = %i\n\n",fixed_z0);

         for (i=0; i<nprop; i++)
         {
            printf("Propagator %i:\n",i);
            printf("kappa  = %.6f\n",kappas[i]);
            printf("isp    = %i\n",isps[i]);
            printf("mu     = %.6f\n\n",mus[i]);
         }
         for (i=0; i<ncorr; i++)
         {
            printf("Correlator %i:\n",i);
            printf("iprop = %i %i %i %i\n",props1[i],props2[i],props3[i],props4[i]);
            printf("type_sources = %i %i\n",typeA[i],typeC[i]);
            printf("type_operator = %i\n\n",operator_type[i]);
         }
      }
      print_solver_parms(&isap,&idfl);

      if (isap)
         print_sap_parms(0);

      if (idfl)
         print_dfl_parms(0);

      printf("Configurations no %d -> %d in steps of %d\n\n",
             first,last,step);
      fflush(flog);
   }
}

static void dfl_wsize(int *nws,int *nwv,int *nwvd) /*untouched*/
{
   dfl_parms_t dp;
   dfl_pro_parms_t dpp;

   dp=dfl_parms();
   dpp=dfl_pro_parms();

   MAX(*nws,dp.Ns+2);
   MAX(*nwv,2*dpp.nkv+2);
   MAX(*nwvd,4);
}

/* removed:          static void make_proplist(void) 
*  exchanged with:   static void make_proplist(void) (new)   */

static void make_proplist(void) /* new function */
{
   int icorr,j;

   proplist.idx_2=malloc(ncorr*sizeof(int));
   proplist.idx_4=malloc(ncorr*sizeof(int));
   proplist.idx_1A=malloc(ncorr*sizeof(int));
   proplist.idx_3C=malloc(ncorr*sizeof(int));

   proplist.prop_type1=malloc(ncorr*sizeof(int));
   proplist.prop_type2=malloc(ncorr*sizeof(int));
   proplist.prop_type3=malloc(ncorr*sizeof(int));
   proplist.prop_type4=malloc(ncorr*sizeof(int));

   proplist.matrix_typeA=malloc(ncorr*sizeof(int));
   proplist.matrix_typeC=malloc(ncorr*sizeof(int));

   error((proplist.idx_1A==NULL)||(proplist.idx_3C==NULL)||(proplist.idx_2==NULL)||(proplist.idx_4==NULL)
      ||(proplist.prop_type1==NULL)||(proplist.prop_type2==NULL)||(proplist.prop_type3==NULL)||(proplist.prop_type4==NULL)
      ||(proplist.matrix_typeA==NULL)||(proplist.matrix_typeC==NULL),1,"make_proplist [correlators.c]","Out of memory");

   /* set to zero all the indices and lengths */
   proplist.len_1A=0;
   proplist.len_3C=0;
   proplist.len2=0;
   proplist.len4=0;

   /* NOTA: questo potrebbe essere inutile, ci devo pensare un attimo*/
   for(icorr=0;icorr<ncorr;icorr++)
   {
      proplist.idx_2[icorr]=0;
      proplist.idx_4[icorr]=0;
      proplist.idx_1A[icorr]=0;
      proplist.idx_3C[icorr]=0;
      proplist.prop_type1[icorr]=0;
      proplist.prop_type2[icorr]=0;
      proplist.prop_type3[icorr]=0;
      proplist.prop_type4[icorr]=0;
      proplist.matrix_typeA[icorr]=0;
      proplist.matrix_typeC[icorr]=0;
   }

   /* main routine */
   for(icorr=0;icorr<ncorr;icorr++)
   {
      /* propagator of type 2 */
      for (j=0;j<proplist.len2;j++)
      {
         if(proplist.prop_type2[j]==props2[icorr])
         {
            proplist.idx_2[icorr]=j;
            break;
         }
      }
      if(j==proplist.len2)
      {
         proplist.prop_type2[j]=props2[icorr];
         proplist.idx_2[icorr]=j;
         proplist.len2++;
      }

      /* propagator of type 4 */
      for (j=0;j<proplist.len4;j++)
      {
         if(proplist.prop_type4[j]==props4[icorr])
         {
            proplist.idx_4[icorr]=j;
            break;
         }
      }
      if(j==proplist.len4)
      {
         proplist.prop_type4[j]=props4[icorr];
         proplist.idx_4[icorr]=j;
         proplist.len4++;
      }

      /* propagator of type 1 + matrix A */
      for (j=0;j<proplist.len_1A;j++)
      {
         if((proplist.prop_type1[j]==props1[icorr])&&(proplist.matrix_typeA[j]==typeA[icorr]))
         {
            proplist.idx_1A[icorr]=j;
            break;
         }
      }
      if(j==proplist.len_1A)
      {
         proplist.prop_type1[j]=props1[icorr];
         proplist.matrix_typeA[j]=typeA[icorr];
         proplist.idx_2[icorr]=j;
         proplist.len_1A++;
      }

      /* propagator of type 3 + matrix C */
      for (j=0;j<proplist.len_3C;j++)
      {
         if((proplist.prop_type3[j]==props3[icorr])&&(proplist.matrix_typeC[j]==typeC[icorr]))
         {
            proplist.idx_3C[icorr]=j;
            break;
         }
      }
      if(j==proplist.len_3C)
      {
         proplist.prop_type3[j]=props3[icorr];
         proplist.matrix_typeC[j]=typeC[icorr];
         proplist.idx_3C[icorr]=j;
         proplist.len_3C++;
      }
   }   


}

static void free_proplist(void)  /* new function */
{
   free(proplist.idx_1A);
   free(proplist.idx_3C);
   free(proplist.idx_2);
   free(proplist.idx_4);
   free(proplist.prop_type1);
   free(proplist.prop_type2);
   free(proplist.prop_type3);
   free(proplist.prop_type4);
   free(proplist.matrix_typeA);
   free(proplist.matrix_typeC);
}

static void wsize(int *nws,int *nwsd,int *nwv,int *nwvd) /*modified*/
{
   int nsd;
   int proplist_nmax;
   solver_parms_t sp;

   (*nws)=0;
   (*nwsd)=0;
   (*nwv)=0;
   (*nwvd)=0;

   sp=solver_parms(0);

   proplist_nmax=0;  /* modified part */
   MAX(proplist_nmax,proplist.len_1A);
   MAX(proplist_nmax,proplist.len_3C);
   MAX(proplist_nmax,proplist.len2);
   MAX(proplist_nmax,proplist.len4);
   nsd=2*(2*proplist_nmax+2); /* not completely sure */

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
      MAX(*nws,2*sp.nkv+2);
      MAX(*nwsd,nsd+4);
      dfl_wsize(nws,nwv,nwvd);
   }
   else
      error_root(1,1,"wsize [correlators.c]","Unknown or unsupported solver");
}

/*
static void random_source(spinor_dble *eta, int x0)
{
   int y0,iy,ix;

   set_sd2zero(VOLUME,eta);
   y0=x0-cpr[0]*L0;

   if ((y0>=0)&&(y0<L0))
   {
      if (noisetype==Z2_NOISE)
      {
         for (iy=0;iy<(L1*L2*L3);iy++)
         {
            ix=ipt[iy+y0*L1*L2*L3];
            random_Z2_sd(1,eta+ix);
         }
      }
      else if (noisetype==GAUSS_NOISE)
      {
         for (iy=0;iy<(L1*L2*L3);iy++)
         {
            ix=ipt[iy+y0*L1*L2*L3];
            random_sd(1,eta+ix,1.0);
         }
      }
      else if (noisetype==U1_NOISE)
      {
         for (iy=0;iy<(L1*L2*L3);iy++)
         {
            ix=ipt[iy+y0*L1*L2*L3];
            random_U1_sd(1,eta+ix);
         }
      }
   }
}
*/

/* It generates an ETA source in
   - timeslice x0
   - lattice space position (0,0,0)
   - fixed colour index and dirac index
   The value of ETA in that point and in the indices is = 1+i
   Otherwise, it in null.
*/

static void set_fixed_point(int vol,spinor_dble *sd,int dc_index)
{
   su3_vector_dble *colour_ptr=NULL;
   spinor_dble *sm=NULL;
   int col_idx,dir_idx;

   col_idx = colour_index(dc_index);
   dir_idx = dirac_index(dc_index);

   sm=sd+vol;

   for (;sd<sm;sd++)
   {

      switch (dir_idx)  /* choose the right Dirac index */
      {
      case 0:
         colour_ptr = &((*sd).c1);
         break;
      case 1:
         colour_ptr = &((*sd).c2);
         break;
      case 2:
         colour_ptr = &((*sd).c3);
         break;
      case 3:
         colour_ptr = &((*sd).c4);
         break;
      default:
         error_root(colour_ptr==NULL,1,"set_fixed_point [correlators.c]","Unable to allocate an su3_vector_dble pointer");
         error_root(1,1,"set_fixed_point [correlators.c]","Invalid Dirac index");
         break;
      }

      switch (col_idx)  /* choose the right color index */
      {
      case 0:
         (*colour_ptr).c1.re=1.0;
         (*colour_ptr).c1.im=1.0;
         break;
      case 1:
         (*colour_ptr).c2.re=1.0;
         (*colour_ptr).c2.im=1.0;
         break;
      case 2:
         (*colour_ptr).c3.re=1.0;
         (*colour_ptr).c3.im=1.0;
         break;
      default:
         error_root(1,1,"set_fixed_point [correlators.c]","Invalid colour index");
         break;
      }
   }
}

static void pointlike_source(spinor_dble *eta, int x0, int dc_index)
{
   int y0,ix;

   set_sd2zero(VOLUME,eta);
   y0=x0-cpr[0]*L0;

   if ((y0>=0)&&(y0<L0))
   {
      ix=ipt[y0*L1*L2*L3];  /* spatial point (0,0,0) */
      set_fixed_point(1,eta+ix,dc_index);
   }
   else
      lonfo();
}

static void solve_dirac(int prop, spinor_dble *eta, spinor_dble *psi, int *status)  /* untouched */
{
   solver_parms_t sp;
   sap_parms_t sap;

   sp=solver_parms(isps[prop]);
   set_sw_parms(0.5/kappas[prop]-4.0); /* set c_{SW} parm, it need the bare quark mass */

   if (sp.solver==CGNE)
   {
      mulg5_dble(VOLUME,eta);

      tmcg(sp.nmx,sp.res,mus[prop],eta,eta,status);
      
      if (my_rank==0)
         printf("%i\n",status[0]);
      error_root(status[0]<0,1,"solve_dirac [correlators.c]",
                 "CGNE solver failed (status = %d)",status[0]);

      Dw_dble(-mus[prop],eta,psi);
      mulg5_dble(VOLUME,psi);
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      sap_gcr(sp.nkv,sp.nmx,sp.res,mus[prop],eta,psi,status);
      if (my_rank==0)
         printf("%i\n",status[0]);
      error_root(status[0]<0,1,"solve_dirac [correlators.c]",
                 "SAP_GCR solver failed (status = %d)",status[0]);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.res,mus[prop],eta,psi,status);
      if (my_rank==0)
         printf("%i %i\n",status[0],status[1]);
      error_root((status[0]<0)||(status[1]<0),1,
                 "solve_dirac [correlators.c]","DFL_SAP_GCR solver failed "
                 "(status = %d,%d)",status[0],status[1]);
   }
   else
      error_root(1,1,"solve_dirac [correlators.c]",
                 "Unknown or unsupported solver");
}

void make_source(spinor_dble *eta, int type, spinor_dble *xi)  /* untouched */
{
   switch (type) 
   {
      case GAMMA0_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg0g5_dble(VOLUME,xi);
         break;
      case GAMMA1_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg1g5_dble(VOLUME,xi);
         break;
      case GAMMA2_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg2g5_dble(VOLUME,xi);
         break;
      case GAMMA3_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg3g5_dble(VOLUME,xi);
         break;
      case GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         break;
      case GAMMA0GAMMA1_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg2g3_dble(VOLUME,xi);
         break;
      case GAMMA0GAMMA2_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg1g3_dble(VOLUME,xi);
         break;
      case GAMMA0GAMMA3_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg1g2_dble(VOLUME,xi);
         break;
      case GAMMA0GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg0_dble(VOLUME,xi);
         break;
      case GAMMA1GAMMA2_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg0g3_dble(VOLUME,xi);
         break;
      case GAMMA1GAMMA3_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg0g2_dble(VOLUME,xi);
         break;
      case GAMMA1GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg1_dble(VOLUME,xi);
         break;
      case GAMMA2GAMMA3_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg0g1_dble(VOLUME,xi);
         break;
      case GAMMA2GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg2_dble(VOLUME,xi);
         break;
      case GAMMA3GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg3_dble(VOLUME,xi);
         break;
      case ONE_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg5_dble(VOLUME,xi);
         break;
      default:
         error_root(1,1,"make_source [correlators.c]",
                 "Unknown or unsupported type");
   }
}

/* removed:    void make_xi(spinor_dble *eta,int type,spinor_dble *xi)  */

/* removed:    static void correlators(void) */

static void mul_type_sd(spinor_dble *psi,int type) /* new function */
{
/* This function must be put into LINALG_SALG_DBLE_C */
   switch (type)
   {
   case GAMMA0_TYPE:
      mulg0_dble(VOLUME,psi);
      break;
   case GAMMA1_TYPE:
      mulg1_dble(VOLUME,psi);
      break;
   case GAMMA2_TYPE:
      mulg2_dble(VOLUME,psi);
      break;
   case GAMMA3_TYPE:
      mulg3_dble(VOLUME,psi);
      break;
   case GAMMA5_TYPE:
      mulg5_dble(VOLUME,psi);
      break;
   case ONE_TYPE:
      break;
   case GAMMA0GAMMA1_TYPE:
      mulg0g1_dble(VOLUME,psi);
      break;
   case GAMMA0GAMMA2_TYPE:
      mulg0g2_dble(VOLUME,psi);
      break;
   case GAMMA0GAMMA3_TYPE:
      mulg0g3_dble(VOLUME,psi);
      break;
   case GAMMA0GAMMA5_TYPE:
      mulg0g5_dble(VOLUME,psi);
      break;
   case GAMMA1GAMMA2_TYPE:
      mulg1g2_dble(VOLUME,psi);
      break;
   case GAMMA1GAMMA3_TYPE:
      mulg1g3_dble(VOLUME,psi);
      break;
   case GAMMA1GAMMA5_TYPE:
      mulg1g5_dble(VOLUME,psi);
      break;
   case GAMMA2GAMMA3_TYPE:
      mulg2g3_dble(VOLUME,psi);
      break;
   case GAMMA2GAMMA5_TYPE:
      mulg2g5_dble(VOLUME,psi);
      break;
   case GAMMA3GAMMA5_TYPE:
      mulg3g5_dble(VOLUME,psi);
      break;
   default:
      error(1,1,"mul_type_sd [correlators.c]","Invalid Dirac matrix type");
      break;
   }
}

static void contraction_single_trace(spinor_dble *xi1,spinor_dble *xi2,spinor_dble *zeta1,spinor_dble *zeta2,spinor_dble *psi1,spinor_dble *psi2,int idx_noise1,int idx_noise2,int idx_corr)   /* new function */
{
   int y0,iy,l,mu,nu,data_index;
   complex_dble complex1,complex2,contribution;

   int type;
   type=file_head.operator_type[idx_corr];

   assign_sd2sd(VOLUME,zeta1,psi1);
   assign_sd2sd(VOLUME,zeta2,psi2);

   if(type<VV)
   {
      switch (type)
      {
         case SS:
            mul_type_sd(psi1,ONE_TYPE);
            mul_type_sd(psi2,ONE_TYPE);
            break;
         case PP:
            mul_type_sd(psi1,GAMMA5_TYPE);
            mul_type_sd(psi2,GAMMA5_TYPE);
            break;
         case PS:
            mul_type_sd(psi1,ONE_TYPE);
            mul_type_sd(psi2,GAMMA5_TYPE);
            break;
         case SP:
            mul_type_sd(psi1,GAMMA5_TYPE);
            mul_type_sd(psi2,ONE_TYPE);
            break;
         default:
            break;
      }
      for(y0=0;y0<L0;y0++) /* non sono molto sicuro di questa scelta!!! Non avrebbe senso solo per x0 < y0 < z0? */
      {
         for (l=0;l<L1*L2*L3;l++)
         {
            iy=ipt[l+y0*L1*L2*L3];

            complex1 = spinor_prod_dble(1,0,xi1+iy,psi2+iy);
            complex2 = spinor_prod_dble(1,0,xi2+iy,psi1+iy);
            contribution.re = complex1.re*complex2.re - complex1.im*complex2.im;
            contribution.im = complex1.re*complex2.im + complex1.im*complex2.re;

            /* I choose to structure my data in this way.
               I need to check that all the functions behave coherently with this. */
            data_index = idx_noise2+nnoise*(idx_noise1+nnoise*(cpr[0]*L0+y0+file_head.tvals*idx_corr));
            data.corr_tmp[data_index].re -= contribution.re;
            data.corr_tmp[data_index].im -= contribution.im;
         }
      }
   }
   else if(type>=VV && type<TT)
   {
      for(mu=0;mu<4;mu++)
      {
         if(mu!=0)
         {
            assign_sd2sd(VOLUME,zeta1,psi1);
            assign_sd2sd(VOLUME,zeta2,psi2);
         }
         switch (type)
         {
            mul_type_sd(psi1,GAMMA0_TYPE+mu);
            mul_type_sd(psi2,GAMMA0_TYPE+mu);
            case VV:
               mul_type_sd(psi1,ONE_TYPE);
               mul_type_sd(psi2,ONE_TYPE);
               break;
            case AA:
               mul_type_sd(psi1,GAMMA5_TYPE);
               mul_type_sd(psi2,GAMMA5_TYPE);
               break;
            case VA:
               mul_type_sd(psi1,GAMMA5_TYPE);
               mul_type_sd(psi2,ONE_TYPE);
               break;
            case AV:
               mul_type_sd(psi1,ONE_TYPE);
               mul_type_sd(psi2,GAMMA5_TYPE);
               break;
            default:
               break;
         }
         for(y0=0;y0<L0;y0++)
         {
            for (l=0;l<L1*L2*L3;l++)
            {
               iy=ipt[l+y0*L1*L2*L3];

               complex1 = spinor_prod_dble(1,0,xi1+iy,psi2+iy);
               complex2 = spinor_prod_dble(1,0,xi2+iy,psi1+iy);
               contribution.re = complex1.re*complex2.re - complex1.im*complex2.im;
               contribution.im = complex1.re*complex2.im + complex1.im*complex2.re;

               data_index = idx_noise2+nnoise*(idx_noise1+nnoise*(cpr[0]*L0+y0+file_head.tvals*idx_corr));
               data.corr_tmp[data_index].re -= contribution.re;
               data.corr_tmp[data_index].im -= contribution.im;
            }
         }
      }
   }
   else if(type>=TT && type<OPERATOR_MAX_TYPE)
   {
      for(nu=0;nu<4;nu++)
      {
         for(mu=0;mu<nu;mu++)
         {
            assign_sd2sd(VOLUME,zeta1,psi1);
            assign_sd2sd(VOLUME,zeta2,psi2);
            if(mu==0)
            {
               mul_type_sd(psi1,GAMMA0GAMMA1_TYPE+nu-1);
               mul_type_sd(psi2,GAMMA0GAMMA1_TYPE+nu-1);
            }
            else if(mu==1)
            {
               mul_type_sd(psi1,GAMMA1GAMMA2_TYPE+nu-2);
               mul_type_sd(psi2,GAMMA1GAMMA2_TYPE+nu-2);
            }
            else if(mu==2)
            {
               mul_type_sd(psi1,GAMMA2GAMMA3_TYPE+nu-3);
               mul_type_sd(psi2,GAMMA2GAMMA3_TYPE+nu-3);
            }
            switch (type)
            {
               case TT:
                  mul_type_sd(psi1,ONE_TYPE);
                  mul_type_sd(psi2,ONE_TYPE);
                  break;
               case TTt:
                  mul_type_sd(psi1,GAMMA5_TYPE);
                  mul_type_sd(psi2,ONE_TYPE);
                  break;
               default:
                  error(1,1,"contraction_single_trace [correlators.c]","Invalid operator type");
                  break;
            }

            for(y0=0;y0<L0;y0++)
            {
               for (l=0;l<L1*L2*L3;l++)
               {
                  iy=ipt[l+y0*L1*L2*L3];

                  complex1 = spinor_prod_dble(1,0,xi1+iy,psi2+iy);
                  complex2 = spinor_prod_dble(1,0,xi2+iy,psi1+iy);
                  contribution.re = complex1.re*complex2.re - complex1.im*complex2.im;
                  contribution.im = complex1.re*complex2.im + complex1.im*complex2.re;

                  data_index = idx_noise2+nnoise*(idx_noise1+nnoise*(cpr[0]*L0+y0+file_head.tvals*idx_corr));
                  data.corr_tmp[data_index].re -= contribution.re;
                  data.corr_tmp[data_index].im -= contribution.im;
               }
            }
         }
      }
   }
   else error(1,1,"contraction_single_trace [correlators.c]","Invalid Dirac-gamma types");
}

static void contraction_double_trace(spinor_dble *xi1,spinor_dble *xi2,spinor_dble *zeta1,spinor_dble *zeta2,spinor_dble *psi1,spinor_dble *psi2,int idx_noise1,int idx_noise2,int idx_corr)   /* new function */
{
   int y0,iy,l,mu,nu,data_index;
   complex_dble complex1,complex2,contribution;

   int type;
   type=file_head.operator_type[idx_corr];

   assign_sd2sd(VOLUME,zeta1,psi1);
   assign_sd2sd(VOLUME,zeta2,psi2);

   if(type<VV)
   {
      switch (type)
      {
         case SS:
            mul_type_sd(psi1,ONE_TYPE);
            mul_type_sd(psi2,ONE_TYPE);
            break;
         case PP:
            mul_type_sd(psi1,GAMMA5_TYPE);
            mul_type_sd(psi2,GAMMA5_TYPE);
            break;
         case PS:
            mul_type_sd(psi1,ONE_TYPE);
            mul_type_sd(psi2,GAMMA5_TYPE);
            break;
         case SP:
            mul_type_sd(psi1,GAMMA5_TYPE);
            mul_type_sd(psi2,ONE_TYPE);
            break;
         default:
            break;
      }
      for(y0=0;y0<L0;y0++) /* non sono molto sicuro di questa scelta!!! Non avrebbe senso solo per x0 < y0 < z0? */
      {
         for (l=0;l<L1*L2*L3;l++)
         {
            iy=ipt[l+y0*L1*L2*L3];

            complex1 = spinor_prod_dble(1,0,xi1+iy,psi1+iy);
            complex2 = spinor_prod_dble(1,0,xi2+iy,psi2+iy);
            contribution.re = complex1.re*complex2.re - complex1.im*complex2.im;
            contribution.im = complex1.re*complex2.im + complex1.im*complex2.re;

            data_index = idx_noise2+nnoise*(idx_noise1+nnoise*(cpr[0]*L0+y0+file_head.tvals*idx_corr));
            data.corr_tmp[data_index].re += contribution.re;
            data.corr_tmp[data_index].im += contribution.im;
         }
      }
   }
   else if(type>=VV && type<TT)
   {
      for(mu=0;mu<4;mu++)
      {
         if(mu!=0)
         {
            assign_sd2sd(VOLUME,zeta1,psi1);
            assign_sd2sd(VOLUME,zeta2,psi2);
         }
         switch (type)
         {
            mul_type_sd(psi1,GAMMA0_TYPE+mu);
            mul_type_sd(psi2,GAMMA0_TYPE+mu);
            case VV:
               mul_type_sd(psi1,ONE_TYPE);
               mul_type_sd(psi2,ONE_TYPE);
               break;
            case AA:
               mul_type_sd(psi1,GAMMA5_TYPE);
               mul_type_sd(psi2,GAMMA5_TYPE);
               break;
            case VA:
               mul_type_sd(psi1,GAMMA5_TYPE);
               mul_type_sd(psi2,ONE_TYPE);
               break;
            case AV:
               mul_type_sd(psi1,ONE_TYPE);
               mul_type_sd(psi2,GAMMA5_TYPE);
               break;
            default:
               break;
         }
         for(y0=0;y0<L0;y0++)
         {
            for (l=0;l<L1*L2*L3;l++)
            {
               iy=ipt[l+y0*L1*L2*L3];

               complex1 = spinor_prod_dble(1,0,xi1+iy,psi1+iy);
               complex2 = spinor_prod_dble(1,0,xi2+iy,psi2+iy);
               contribution.re = complex1.re*complex2.re - complex1.im*complex2.im;
               contribution.im = complex1.re*complex2.im + complex1.im*complex2.re;

               data_index = idx_noise2+nnoise*(idx_noise1+nnoise*(cpr[0]*L0+y0+file_head.tvals*idx_corr));
               data.corr_tmp[data_index].re += contribution.re;
               data.corr_tmp[data_index].im += contribution.im;
            }
         }
      }
   }
   else if(type>=TT && type<OPERATOR_MAX_TYPE)
   {
      for(nu=0;nu<4;nu++)
      {
         for(mu=0;mu<nu;mu++)
         {
            assign_sd2sd(VOLUME,zeta1,psi1);
            assign_sd2sd(VOLUME,zeta2,psi2);
            if(mu==0)
            {
               mul_type_sd(psi1,GAMMA0GAMMA1_TYPE+nu-1);
               mul_type_sd(psi2,GAMMA0GAMMA1_TYPE+nu-1);
            }
            else if(mu==1)
            {
               mul_type_sd(psi1,GAMMA1GAMMA2_TYPE+nu-2);
               mul_type_sd(psi2,GAMMA1GAMMA2_TYPE+nu-2);
            }
            else if(mu==2)
            {
               mul_type_sd(psi1,GAMMA2GAMMA3_TYPE+nu-3);
               mul_type_sd(psi2,GAMMA2GAMMA3_TYPE+nu-3);
            }
            switch (type)
            {
               case TT:
                  mul_type_sd(psi1,ONE_TYPE);
               case TTt:
                  mul_type_sd(psi1,GAMMA5_TYPE);
                  break;
               default:
                  error(1,1,"contraction_double_trace [correlators.c]","Invalid operator type");
                  break;
            }

            for(y0=0;y0<L0;y0++)
            {
               for (l=0;l<L1*L2*L3;l++)
               {
                  iy=ipt[l+y0*L1*L2*L3];

                  complex1 = spinor_prod_dble(1,0,xi1+iy,psi1+iy);
                  complex2 = spinor_prod_dble(1,0,xi2+iy,psi2+iy);
                  contribution.re = complex1.re*complex2.re - complex1.im*complex2.im;
                  contribution.im = complex1.re*complex2.im + complex1.im*complex2.re;

                  data_index = idx_noise2+nnoise*(idx_noise1+nnoise*(cpr[0]*L0+y0+file_head.tvals*idx_corr));
                  data.corr_tmp[data_index].re += contribution.re;
                  data.corr_tmp[data_index].im += contribution.im;
               }
            }
         }
      }
   }
   else error(1,1,"contraction_double_trace [correlators.c]","Invalid Dirac-gamma types");
}

static void correlators_contractions(void)  /*new function*/
{
   int idx,noise_idx1,noise_idx2,stat[4],l,transform_idx;
   spinor_dble *eta1,*eta2,*tmp_spinor,*tmp_spinor2;
   spinor_dble **xi1,**xi2,**zeta1,**zeta2,**wsd;

   wsd=reserve_wsd(4+proplist.len_1A+proplist.len_3C+proplist.len4+proplist.len2);
   eta1=wsd[0];
   eta2=wsd[1];
   tmp_spinor=wsd[2];
   tmp_spinor2=wsd[3];
   xi1=malloc(proplist.len_1A*sizeof(spinor_dble*));
   xi2=malloc(proplist.len_3C*sizeof(spinor_dble*));
   zeta1=malloc(proplist.len4*sizeof(spinor_dble*));
   zeta2=malloc(proplist.len2*sizeof(spinor_dble*));
   error((xi1==NULL)||(xi2==NULL)||(zeta1==NULL)||(zeta2==NULL),1,"correlators [correlators.c]","Out of memory");

   for(l=0;l<proplist.len_1A;l++)
      xi1[l]=wsd[4+l];
   for(l=0;l<proplist.len_3C;l++)
      xi2[l]=wsd[4+l+proplist.len_1A];
   for(l=0;l<proplist.len4;l++)
      zeta1[l]=wsd[4+l+proplist.len_1A+proplist.len_3C];
   for(l=0;l<proplist.len2;l++)
      zeta2[l]=wsd[4+l+proplist.len_1A+proplist.len_3C+proplist.len4];

   for(transform_idx=0;transform_idx<1;transform_idx++)
   {   
      for (l=0;l<nnoise*nnoise*ncorr*tvals;l++)
      {
         data.corr_tmp[l].re=0.0;
         data.corr_tmp[l].im=0.0;
      }

      /* ETA_1 noise vectors */
      for(noise_idx1=0;noise_idx1<nnoise;noise_idx1++)
      {
         pointlike_source(eta1,fixed_x0,noise_idx1);
         check_null_spinor(eta1,"Checking ETA1");

         if(my_rank==0) printf("\tNoise vector eta1 number %i\n",noise_idx1);

         /* evaluate the needed \xi_1 */
         for(idx=0;idx<proplist.len_1A;idx++)
         {
            if (my_rank==0)
               printf("\t\tXi_{1A}^{1,-} evaluation:\n\t\t\ttype=%s, prop=%i, status:\n",dirac_type_to_string(proplist.matrix_typeA[idx]), proplist.prop_type1[idx]);

            make_source(eta1,proplist.matrix_typeA[idx],tmp_spinor);
            solve_dirac(proplist.prop_type1[idx],tmp_spinor,xi1[idx],stat);
            mulg5_dble(VOLUME,xi1[idx]);
         }

         /* evaluate the needed ZETA_1 s */
         for(idx=0;idx<proplist.len4;idx++)
         {
            if (my_rank==0)
               printf("\t\tZeta_1^{4,+} evaluation:\n\t\t\tprop=%i, status:\n",proplist.prop_type4[idx]);

            assign_sd2sd(VOLUME,eta1,tmp_spinor);
            solve_dirac(proplist.prop_type4[idx],tmp_spinor,zeta1[idx],stat);
         }
      }
      /* ETA_2 noise vectors */
      for(noise_idx2=0;noise_idx2<nnoise;noise_idx2++)
      {
         pointlike_source(eta2,fixed_z0,noise_idx2);

         if(my_rank==0) printf("\tNoise vector eta2 number %i\n",noise_idx2);

         /* evaluate the needed XI_2 s */
         for(idx=0;idx<proplist.len_3C;idx++)
         {
            if (my_rank==0)
               printf("\t\tXi_{3C}^{3,-} evaluation:\n\t\t\ttype=%s, prop=%i, status:\n",dirac_type_to_string(proplist.matrix_typeC[idx]),proplist.prop_type3[idx]);

            make_source(eta2,proplist.matrix_typeC[idx],tmp_spinor);
            solve_dirac(proplist.prop_type3[idx],tmp_spinor,xi2[idx],stat);
            mulg5_dble(VOLUME,xi2[idx]);
         }

         /* evaluate the needed ZETA_2 s */
         for(idx=0;idx<proplist.len2;idx++)
         {
            if (my_rank==0)
               printf("\t\tZeta_2^{2,+} evaluation:\n\t\t\tprop=%i, status:\n",proplist.prop_type2[idx]);

            assign_sd2sd(VOLUME,eta2,tmp_spinor);
            solve_dirac(proplist.prop_type2[idx],tmp_spinor,zeta2[idx],stat);
         }
      }

      if(my_rank==0) printf("Evaluation of Wick contractions:\n");

      for(noise_idx1=0;noise_idx1<nnoise;noise_idx1++)
      {
         for(noise_idx2=0;noise_idx2<nnoise;noise_idx2++)
         {
            if (my_rank==0)   printf("\tStohcastic vectors eta1 = %i\teta2 = %i\n",noise_idx1,noise_idx2);

            /* contractions */
            for(idx=0;idx<ncorr;idx++)
            {
               if (my_rank==0)   printf("\t\tOperator XY = %s",operator_to_string(file_head.operator_type[idx]));
               contraction_single_trace(xi1[proplist.idx_1A[idx]],xi2[proplist.idx_3C[idx]],zeta1[proplist.idx_4[idx]],zeta2[proplist.idx_2[idx]],tmp_spinor,tmp_spinor2,noise_idx1,noise_idx2,idx);
               contraction_double_trace(xi1[proplist.idx_1A[idx]],xi2[proplist.idx_3C[idx]],zeta1[proplist.idx_4[idx]],zeta2[proplist.idx_2[idx]],tmp_spinor,tmp_spinor2,noise_idx1,noise_idx2,idx);
               if (my_rank==0)   printf("\t---> Work done.\n");
            }
         }
      }

      printf("DATA OFFSET: %i\n",data.offset);
      MPI_Allreduce(data.corr_tmp,data.corr+data.offset,nnoise*nnoise*ncorr*file_head.tvals*2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      if(data.offset==0)
      {
         generate_g_trnsfrms();
         g_transform_ud();
      }
      data.offset=file_head.ncorr*file_head.nnoise*file_head.nnoise*file_head.tvals;
   }

   free_g_trnsfrms();
   
   MPI_Allreduce(data.corr_tmp,data.corr,nnoise*nnoise*ncorr*file_head.tvals*2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
   free(xi1);
   free(xi2);
   free(zeta1);
   free(zeta2);
   release_wsd();
}

static void set_data(int nc)  /*untouched*/
{
   data.nc=nc;
   correlators_contractions();

   if (my_rank==0)
   {
      printf("G(t) =  %.4e%+.4ei",data.corr[0].re,data.corr[0].im);
      printf(",%.4e%+.4ei,...",data.corr[1].re,data.corr[1].im);
      printf(",%.4e%+.4ei",data.corr[file_head.tvals-1].re,data.corr[file_head.tvals-1].im);
      printf("\n");
      fflush(flog);
   }
}

static void init_rng(void) /*untouched*/
{
   int ic;

   if (append)
   {
      if (norng)
         start_ranlux(level,seed^(first-step));
      else
      {
         ic=import_ranlux(rng_file);
         error_root(ic!=(first-step),1,"init_rng [correlators.c]",
                    "Configuration number mismatch (*.rng file)");
      }
   }
   else
      start_ranlux(level,seed);
}

static void save_ranlux(void) /*untouched*/
{
   int nlxs,nlxd;

   if (rlxs_state==NULL)
   {
      nlxs=rlxs_size();
      nlxd=rlxd_size();

      rlxs_state=malloc((nlxs+nlxd)*sizeof(int));
      rlxd_state=rlxs_state+nlxs;

      error(rlxs_state==NULL,1,"save_ranlux [correlators.c]",
            "Unable to allocate state arrays");
   }

   rlxs_get(rlxs_state);
   rlxd_get(rlxd_state);
}

static void restore_ranlux(void) /*untouched*/
{
   rlxs_reset(rlxs_state);
   rlxd_reset(rlxd_state);
}

static void check_endflag(int *iend)   /*untouched*/
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

/************************ MAIN FUNCTION ************************/

int main(int argc,char *argv[])
{
   int nc,iend,status[4];
   int nws,nwsd,nwv,nwvd;
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
   init_rng();

   make_proplist();
   wsize(&nws,&nwsd,&nwv,&nwvd);
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
         save_ranlux();
         sprintf(cnfg_file,"%s/%sn%d_%d",loc_dir,run_name,nc,my_rank);
         read_cnfg(cnfg_file);
         restore_ranlux();
      }
      else if((!nogauge)&&(!rndmgauge))
      {
         sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,run_name,nc);
         import_cnfg(cnfg_file);
      }
      else
      {
         if(nogauge)
         {
            sprintf(cnfg_file,"# No interactions, Gauge fields are set to 0 and link variables to 1 #\n");
            alloc_ud_to_identity();
         }
         if(rndmgauge)
         {
            sprintf(cnfg_file,"# Gauge configurations are randomly generated #\n");
            random_ud();
         }
         set_flags(UPDATED_UD);
      }


      if (dfl.Ns)
      {
         dfl_modes(status);

         error_root(status[0]<0,1,"main [correlators.c]",
                    "Deflation subspace generation failed (status = %d)",
                    status[0]);

         if (my_rank==0)
            printf("Deflation subspace generation: status = %d\n",status[0]);
      }

      set_data(nc);
      write_data();

      free_proplist();

      export_ranlux(nc,rng_file);
      error_chk();
      
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);
      

      if (my_rank==0)
      {
         printf("Configuration no %d fully processed in %.2e sec ",nc,wt2-wt1);
         printf("(average = %.2e sec)\n\n",wtavg/(double)((nc-first)/step+1));
      }
      check_endflag(&iend);

      if (my_rank==0)
      {
         fflush(flog);
         copy_file(log_file,log_save);
         copy_file(dat_file,dat_save);
         copy_file(rng_file,rng_save);
      }
   }

   if (my_rank==0)
   {
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
