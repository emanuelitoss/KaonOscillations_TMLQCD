
/*******************************************************************************
*
* File check12.c
*
* Copyright (C) 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check and performance of the CG solver
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "linsolv.h"
#include "forces.h"
#include "global.h"

static int my_rank,first,last,step,nmx;
static double kappa,csw,cF,zF;
static double mu,m0,res;
static char cnfg_dir[NAME_SIZE],cnfg_file[NAME_SIZE],nbase[NAME_SIZE];


int main(int argc,char *argv[])
{
   int nsize,icnfg,status,tau3;
   double rho,nrm,del;
   double wt1,wt2,wdt;
   double phi[2],phi_prime[2],theta[3];
   complex_dble z;
   spinor_dble **psd;
   lat_parms_t lat;
   sw_parms_t sw;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check12.log","w",stdout);
      fin=freopen("check12.in","r",stdin);

      printf("\n");
      printf("Check and performance of the CG solver\n");
      printf("--------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      find_section("Configurations");
      read_line("name","%s",nbase);
      read_line("cnfg_dir","%s",cnfg_dir);
      read_line("first","%d",&first);
      read_line("last","%d",&last);  
      read_line("step","%d",&step);  

      find_section("Lattice parameters");
      read_line("kappa","%lf",&kappa);
      read_line("csw","%lf",&csw);
      read_line("cF","%lf",&cF);
      read_line("zF","%lf",&zF);
      read_line("mu","%lf",&mu);            

      find_section("CG");
      read_line("nmx","%d",&nmx);
      read_line("res","%lf",&res);

      fclose(fin);
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&zF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   
   start_ranlux(0,1234);
   geometry();

   lat=set_lat_parms(6.0,1.0,kappa,0.0,0.0,csw,1.0,cF,0.5,1.0);
   phi[0]=0.0;
   phi[1]=0.0;
   phi_prime[0]=0.0;
   phi_prime[1]=0.0;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;
  
   set_sf_parms(phi,phi_prime,theta);

   m0=lat.m0u;
   sw=set_sw_parms(m0);

   if (my_rank==0)
   {
      printf("kappa = %.6f\n",lat.kappa_u);
      printf("csw = %.6f\n",sw.csw);
      printf("cF = %.6f\n",sw.cF);
      printf("sf = %d\n",sf_flg());
      printf("mu = %.6f\n\n",mu);
      printf("CG parameters:\n");
      printf("nmx = %d\n",nmx);      
      printf("res = %.2e\n\n",res);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   alloc_ws(5);
   alloc_wsd(6);
   psd=reserve_wsd(3);
   
   error_root(((last-first)%step)!=0,1,"main [check12.c]",
              "last-first is not a multiple of step");
   check_dir_root(cnfg_dir);   
   nsize=name_size("%s/%sn%d",cnfg_dir,nbase,last);
   error_root(nsize>=NAME_SIZE,1,"main [check12.c]",
              "configuration file name is too long");

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      random_ud();
/*
      import_cnfg(cnfg_file);
*/
      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      } 

      random_sd(VOLUME,psd[0],1.0);
/*    bnd_sd2zero(ALL_PTS,psd[0]); */
      nrm=sqrt(norm_square_dble(VOLUME,1,psd[0]));
      assign_sd2sd(VOLUME,psd[0],psd[2]);         

      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();              

      tau3=1;
      rho=tmcg_xsf(nmx,res,tau3,mu,psd[0],psd[1],&status);

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wdt=wt2-wt1;
      
      error_chk();
      z.re=-1.0;
      z.im=0.0;
      mulc_spinor_add_dble(VOLUME,psd[2],psd[0],z);
      del=norm_square_dble(VOLUME,1,psd[2]);
      error_root(del!=0.0,1,"main [check12.c]",
                 "Source field is not preserved");

      set_xsf_parms(1);
      Dw_xsf_dble(mu,psd[1],psd[2]);
      mulg5_dble(VOLUME,psd[2]);
      set_xsf_parms(-1);
      Dw_xsf_dble(-mu,psd[2],psd[1]);
      mulg5_dble(VOLUME,psd[1]);
      mulc_spinor_add_dble(VOLUME,psd[1],psd[0],z);
      del=sqrt(norm_square_dble(VOLUME,1,psd[1]));
      
      if (my_rank==0)
      {
         printf("Solution w/o eo-preconditioning:\n");
         printf("status = %d\n",status);
         printf("rho   = %.2e, res   = %.2e\n",rho,res);
         printf("check = %.2e, check = %.2e\n",del,del/nrm);
         printf("time = %.2e sec (total)\n",wdt);
         if (status>0)
            printf("     = %.2e usec (per point and CG iteration)",
                   (1.0e6*wdt)/((double)(status)*(double)(VOLUME)));
         printf("\n\n");
         fflush(flog);
      }

      random_sd(VOLUME/2,psd[0],1.0);
/*    bnd_sd2zero(ALL_PTS,psd[0]); */
      nrm=sqrt(norm_square_dble(VOLUME/2,1,psd[0]));
      assign_sd2sd(VOLUME/2,psd[0],psd[2]);         

      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();              

      rho=tmcgeo_xsf(nmx,res,tau3,mu,psd[0],psd[1],&status);

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wdt=wt2-wt1;
      
      error_chk();
      z.re=-1.0;
      z.im=0.0;
      mulc_spinor_add_dble(VOLUME/2,psd[2],psd[0],z);
      del=norm_square_dble(VOLUME/2,1,psd[2]);
      error_root(del!=0.0,1,"main [check12.c]",
                 "Source field is not preserved");

      set_xsf_parms(1);
      Dwhat_xsf_dble(mu,psd[1],psd[2]);
      mulg5_dble(VOLUME/2,psd[2]);
      set_xsf_parms(-1);
      Dwhat_xsf_dble(-mu,psd[2],psd[1]);
      mulg5_dble(VOLUME/2,psd[1]);
      mulc_spinor_add_dble(VOLUME/2,psd[1],psd[0],z);
      del=sqrt(norm_square_dble(VOLUME/2,1,psd[1]));
 
      if (my_rank==0)
      {
         printf("Solution with eo-preconditioning:\n");
         printf("status = %d\n",status);
         printf("rho   = %.2e, res   = %.2e\n",rho,res);
         printf("check = %.2e, check = %.2e\n",del,del/nrm);
         printf("time = %.2e sec (total)\n",wdt);
         if (status>0)
            printf("     = %.2e usec (per point and CG iteration)",
                   (1.0e6*wdt)/((double)(status)*(double)(VOLUME)));
         printf("\n\n");
         fflush(flog);
      }
   }

   if (my_rank==0)
      fclose(flog);
   
   MPI_Finalize();    
   exit(0);
}
