
/*******************************************************************************
*
* File ms5.c
*
* Copyright (C) 2014 Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the SF correlation functions fS,fP,fA,fV,f1
*
* Syntax: ms5 -i <input file> [-noexp]
*
* For usage instructions see the file README.ms5
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
   double kappa,csw,cF;
} file_head;

static struct
{
   int nc;
   complex_dble *fS,*fP,*fA,*fV,*fVt,*f1;
} data;

static int my_rank,noexp,append,endian;
static int first,last,step,bnd;
static const spinor_dble sd0={{{0.0}}};

static char line[NAME_SIZE];
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char loc_dir[NAME_SIZE],cnfg_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE],end_file[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char dat_file[NAME_SIZE],dat_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],nbase[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fdat=NULL,*fend=NULL;


static void alloc_data(void)
{
   int tmax;
   complex_dble *p;

   tmax=file_head.tmax;

   p=amalloc((5*tmax+1)*sizeof(*p),4);
   
   error((p==NULL),1,"alloc_data [ms5.c]",
         "Unable to allocate data arrays");

   data.fS=p;
   data.fP=p+tmax;
   data.fA=p+2*tmax;
   data.fV=p+3*tmax;
   data.fVt=p+4*tmax;
   data.f1=p+5*tmax;
}


static void write_file_head(void)
{
   int iw;
   double dstd[3];   
   stdint_t istd[2];

   dstd[0]=file_head.kappa;
   dstd[1]=file_head.csw;
   dstd[2]=file_head.cF;
   istd[0]=(stdint_t)(file_head.tmax);   
   istd[1]=(stdint_t)(file_head.bnd);   
   
   if (endian==BIG_ENDIAN)
   {
      bswap_double(3,dstd);
      bswap_int(2,istd);
   }
   
   iw=fwrite(dstd,sizeof(double),3,fdat);   
   iw+=fwrite(istd,sizeof(stdint_t),2,fdat);

   error_root(iw!=5,1,"write_file_head [ms5.c]",
              "Incorrect write count");
}


static void check_file_head(void)
{
   int ir;
   double dstd[3];
   stdint_t istd[2];

   ir=fread(dstd,sizeof(double),3,fdat);   
   ir+=fread(istd,sizeof(stdint_t),2,fdat);

   error_root(ir!=5,1,"check_file_head [ms5.c]",
              "Incorrect read count");
   
   if (endian==BIG_ENDIAN)
   {
      bswap_double(3,dstd);      
      bswap_int(2,istd);
   }
   
   error_root((dstd[0]!=file_head.kappa)||
              (dstd[1]!=file_head.csw)||
              (dstd[2]!=file_head.cF)||
              ((int)(istd[0])!=file_head.tmax)||
              ((int)(istd[1])!=file_head.bnd),1,"check_file_head [ms5.c]",
              "Unexpected value of kappa,csw,cF,tmax or bnd");
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
      dstd[0]=data.fS[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.fS[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.fP[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.fP[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.fA[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.fA[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.fV[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.fV[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
  }

   for (t=0;t<tmax;t++)
   {
      dstd[0]=data.fVt[t].re;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);

      dstd[0]=data.fVt[t].im;

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
  }

   dstd[0]=data.f1[0].re;

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);

   iw+=fwrite(dstd,sizeof(double),1,fdat);

   dstd[0]=data.f1[0].im;

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);

   iw+=fwrite(dstd,sizeof(double),1,fdat);

   error_root(iw!=(1+2+10*tmax),1,"write_data [ms5.c]",
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
         
      data.fS[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.fS[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.fP[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.fP[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.fA[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.fA[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.fV[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.fV[t].im=dstd[0];
   }

   for (t=0;t<tmax;t++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.fVt[t].re=dstd[0];

      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
         
      data.fVt[t].im=dstd[0];
   }

   ir+=fread(dstd,sizeof(double),1,fdat);

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);
         
   data.f1[0].re=dstd[0];

   ir+=fread(dstd,sizeof(double),1,fdat);

   if (endian==BIG_ENDIAN)
      bswap_double(1,dstd);
         
   data.f1[0].im=dstd[0];

   error_root(ir!=(1+2+10*tmax),1,"read_data [ms5.c]",
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
                 "read_dirs [ms5.c]","Improper configuration range");
      error_root((bnd<0)||(bnd>1),1,"read_dirs [ms5.c]",
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
                 1,"setup_files [ms5.c]","loc_dir name is too long");
   else
      error_root(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,
                 1,"setup_files [ms5.c]","cnfg_dir name is too long");

   check_dir_root(log_dir);   
   check_dir_root(dat_dir);
   error_root(name_size("%s/%s.ms5.log~",log_dir,nbase)>=NAME_SIZE,
              1,"setup_files [ms5.c]","log_dir name is too long");
   error_root(name_size("%s/%s.ms5.dat~",dat_dir,nbase)>=NAME_SIZE,
              1,"setup_files [ms5.c]","dat_dir name is too long");   

   sprintf(log_file,"%s/%s.ms5.log",log_dir,nbase);
   sprintf(par_file,"%s/%s.ms5.par",dat_dir,nbase);   
   sprintf(dat_file,"%s/%s.ms5.dat",dat_dir,nbase);
   sprintf(end_file,"%s/%s.ms5.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);   
   sprintf(dat_save,"%s~",dat_file);
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
   double kappa,csw,cF;
   double dstd[3];

   if (my_rank==0)
   {
      find_section("Dirac operator");
      read_line("kappa","%lf",&kappa);
      read_line("csw","%lf",&csw);
      read_line("cF","%lf",&cF);   
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   set_lat_parms(0.0,1.0,kappa,0.0,0.0,csw,1.0,cF,0.5,1.0);
   set_sw_parms(sea_quark_mass(0));

   file_head.kappa=kappa;
   file_head.csw=csw;
   file_head.cF=cF;
   file_head.tmax=N0;

   if (my_rank==0)
   {
      if (append)
      {
         ir=fread(dstd,sizeof(double),3,fdat);
         error_root(ir!=3,1,"read_lat_parms [ms5.c]",
                    "Incorrect read count");         

         if (endian==BIG_ENDIAN)
            bswap_double(3,dstd);

         ie=0;
         ie|=(dstd[0]!=kappa);
         ie|=(dstd[1]!=csw);
         ie|=(dstd[2]!=cF);

         error_root(ie!=0,1,"read_lat_parms [ms5.c]",
                    "Parameters do not match previous run");
      }
      else
      {
         dstd[0]=kappa;
         dstd[1]=csw;         
         dstd[2]=cF;         

         if (endian==BIG_ENDIAN)
            bswap_double(3,dstd);

         iw=fwrite(dstd,sizeof(double),3,fdat);
         error_root(iw!=3,1,"read_lat_parms [ms5.c]",
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

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [ms5.c]",
                 "Syntax: ms5 -i <input file> [-noexp]");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms5.c]",
                 "Machine has unknown endianness");

      noexp=find_opt(argc,argv,"-noexp");      
      append=find_opt(argc,argv,"-a");
      
      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ms5.c]",
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

      error_root(fdat==NULL,1,"read_infile [ms5.c]",
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
   error_root(fend==NULL,1,"check_old_log [ms5.c]",
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

   error_root((ie&0x1)!=0x0,1,"check_old_log [ms5.c]",
              "Incorrect read count");   
   error_root((ie&0x2)!=0x0,1,"check_old_log [ms5.c]",
              "Configuration numbers are not equally spaced");
   error_root(isv==0,1,"check_old_log [ms5.c]",
              "Log file extends beyond the last configuration save");

   (*fst)=fc;
   (*lst)=lc;
   (*stp)=dc;
}


static void check_old_dat(int fst,int lst,int stp)
{
   int ie,ic;
   int fc,lc,dc,pc;
   
   fdat=fopen(dat_file,"rb");
   error_root(fdat==NULL,1,"check_old_dat [ms5.c]",
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

   error_root(ic==0,1,"check_old_dat [ms5.c]",
              "No data records found");
   error_root((ie&0x1)!=0x0,1,"check_old_dat [ms5.c]",
              "Configuration numbers are not equally spaced");
   error_root((fst!=fc)||(lst!=lc)||(stp!=dc),1,"check_old_dat [ms5.c]",
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
         check_old_dat(fst,lst,stp);

         error_root((fst!=lst)&&(stp!=step),1,"check_files [ms5.c]",
                    "Continuation run:\n"
                    "Previous run had a different configuration separation");
         error_root(first!=lst+step,1,"check_files [ms5.c]",
                    "Continuation run:\n"
                    "Configuration range does not continue the previous one");
      }
      else
      {
         fin=fopen(log_file,"r");
         fdat=fopen(dat_file,"rb");

         error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms5.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file,"wb");
         error_root(fdat==NULL,1,"check_files [ms5.c]",
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

      error_root(flog==NULL,1,"print_info [ms1.c]","Unable to open log file");
      printf("\n");

      if (append)
         printf("Continuation run\n\n");
      else
      {
         printf("Computation of SF correlation functions\n");         
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

         printf("Schroedinger functional boundary conditions\n\n");
         print_sf_parms();
          
         printf("Dirac operator:\n");
         printf("kappa = %.6f\n",file_head.kappa);
         printf("csw = %.6f\n",file_head.csw);      
         printf("cF = %.6f\n\n",file_head.cF);

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
      error_root(1,1,"wsize [ms5.c]",
                 "Unknown or unsupported solver");   
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

      error((iprms[0]!=x0),1,"spinor_sum_bnd [ms5.c]",
            "Parameters are not global");   
   }
 
   error_root(((x0!=0)&&(x0!=(NPROC0*L0-1))),1,
             "spinor_sum_bnd [ms5.c]","Improper argument x0");

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


static void bnd2bnd_prop(spinor_dble **psd,spinor_dble *r)
{
   int i,x0;
   spinor_dble **wsd;

   wsd=reserve_wsd(1);

   if (bnd==0)
      x0=N0-1;
   else
      x0=0;

   for(i=0;i<12;i++)
   {
      if (bnd==0)
         push_slice(x0-1,psd[i],wsd[0]);
      else
         pull_slice(x0+1,psd[i],wsd[0]);

      spinor_sum_bnd(x0,wsd[0],r+i);      
   }

   release_wsd();
}


static void point_split_prop(int x0,spinor_dble **ps,spinor_dble **pr)
{
   int i,t;

   for (i=0;i<12;i++)
   {
      assign_sd2sd(VOLUME,ps[i],pr[i]);

      if (x0==0)
      {
         for (t=1;t<NPROC0*L0;t++)
            pull_slice(t,pr[i],pr[i]);
      }
      else 
      {
         for (t=NPROC0*L0-2;t==0;t--)
            push_slice(t,pr[i],pr[i]);
      }
   }
}


static void sfcfcts(int nc,int *status)
{
   int i,x0,t,tmax;
   double cF2,nrm;
   complex_dble z;
   spinor_dble r[12];
   spinor_dble **wsd,**psd;

   data.nc=nc;
   wsd=reserve_wsd(12);

   cF2=file_head.cF;
   cF2*=cF2;
   tmax=file_head.tmax;
   nrm=(double)(N1*N2*N3);

   for (i=0;i<4;i++)
      status[i]=0;

   if (bnd==0)
      x0=0;
   else 
      x0=N0-1;

   sfprop(x0,0,wsd,status);

   cfcts2q(S ,P,wsd,wsd,data.fS);
   cfcts2q(P ,P,wsd,wsd,data.fP);
   cfcts2q(A0,P,wsd,wsd,data.fA);
   cfcts2q(V0,P,wsd,wsd,data.fV);

   psd=reserve_wsd(12);

   point_split_prop(x0,wsd,psd);
   cfcts2q(S,P,psd,wsd,data.fVt);
   release_wsd();

   for(t=0;t<tmax;t++)
   {
      data.fS[t].re*=-cF2/(2.0*nrm);
      data.fS[t].im*=-cF2/(2.0*nrm);

      data.fP[t].re*=-cF2/(2.0*nrm);
      data.fP[t].im*=-cF2/(2.0*nrm);

      data.fA[t].re*=-cF2/(2.0*nrm);
      data.fA[t].im*=-cF2/(2.0*nrm);

      data.fV[t].re*=-cF2/(2.0*nrm);
      data.fV[t].im*=-cF2/(2.0*nrm);

      data.fVt[t].re = 0.0;
      data.fVt[t].im*=-cF2/nrm;
   }

   bnd2bnd_prop(wsd,r);
   ctrcts2q(S,S,r,r,&z);

   data.f1[0].re=(cF2*cF2*z.re)/(2.0*nrm*nrm);
   data.f1[0].im=(cF2*cF2*z.im)/(2.0*nrm*nrm);

   release_wsd();
}


static void save_data(void)
{
   if (my_rank==0)
   {
      fdat=fopen(dat_file,"ab");
      error_root(fdat==NULL,1,"save_data [ms5.c]",
                 "Unable to open data file");
      write_data();
      fclose(fdat);
   }
}


static void print_log(void)
{
   int t,tmax;

   tmax=file_head.tmax;

   if (my_rank==0)
   {
      printf("\n#### fS ####\n\n");
      for (t=0;t<tmax;t++)
         printf("x0 = %i, %.14e   %.14e\n",t,data.fS[t].re,data.fS[t].im);

      printf("\n#### fP ####\n\n");
      for (t=0;t<tmax;t++)
         printf("x0 = %i, %.14e   %.14e\n",t,data.fP[t].re,data.fP[t].im);

      printf("\n#### fA ####\n\n");
      for (t=0;t<tmax;t++)
         printf("x0 = %i, %.14e   %.14e\n",t,data.fA[t].re,data.fA[t].im);

      printf("\n#### fV ####\n\n");
      for (t=0;t<tmax;t++)
         printf("x0 = %i, %.14e   %.14e\n",t,data.fV[t].re,data.fV[t].im);

      printf("\n#### fVt ####\n\n");
      for (t=0;t<tmax;t++)
         printf("x0 = %i, %.14e   %.14e\n",t,data.fVt[t].re,data.fVt[t].im);

      printf("\n#### f1 ####\n\n");
      printf("x0 = -, %.14e   %.14e\n\n",data.f1[0].re,data.f1[0].im);
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
   nwsdc=21;

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
      error_root(ie!=1,1,"main [ms5.c]",
                 "Initial configuration has incorrect boundary values");

      mult_phase(1);

      if (dfl.Ns)
      {
         dfl_modes(status);
         error_root(status[0]<0,1,"main [ms5.c]",
                    "Deflation subspace generation failed (status = %d)",
                    status[0]);

         if (my_rank==0)
            printf("Deflation subspace generation: status = %d\n",status[0]);
      }

      sfcfcts(nc,status);
      save_data();
      print_log();

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);
      error_chk();
      
      if (my_rank==0)
      {
         printf("Computation of SF correlation functions completed\n");

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
         copy_file(dat_file,dat_save);
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
