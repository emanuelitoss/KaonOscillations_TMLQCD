
/*******************************************************************************
*
* File check13.c
*
* Copyright (C) 2012, 2013 Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Hermiticity of Dw_xsf_dble() and comparison with Dwee_xsf_dble(),...
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "global.h"


int main(int argc,char *argv[])
{
   int my_rank,i,ifail;
   double mu,d;
   double m0,kappa;
   complex_dble z1,z2;
   spinor_dble **psd;
   pauli_dble *m;
   sw_parms_t swp;
   xsf_parms_t xsf;
   FILE *flog=NULL;
   double phi[2],phip[2];
   double theta[3];

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check13.log","w",stdout);
      printf("\n");
      printf("Hermiticity of Dw_dble() and comparison with Dwee_dble(),...\n");
      printf("------------------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,12345);
   geometry();
   alloc_wsd(5);
   psd=reserve_wsd(5);

   m0=-0.0123;
   kappa=1./(2.*m0+8.0);

   set_lat_parms(5.5,1.0,kappa,0.0,0.0,0.456,1.0,1.234,0.5,1.0);
   swp=set_sw_parms(m0);
   phi[0]=0.0; phi[1]=0.0; 
   phip[0]=0.0; phip[1]=0.0;
	
   theta[0]=0.0; theta[1]=0.0; theta[2]=0.0;
   set_sf_parms(phi,phip,theta);
   mu=0.0376;

   if (my_rank==0)
   {
      printf("m0 = %.4e, mu= %.4e, csw = %.4e, sf = %d cF = %.4e\n\n",
             swp.m0,mu,swp.csw,sf_flg(),swp.cF);
      printf("Deviations should be at most 10^(-15) or so in these tests\n\n");
   }

   random_ud();
   sw_term(NO_PTS);

   for (i=0;i<4;i++)
      random_sd(NSPIN,psd[i],1.0);

   set_xsf_parms(1);
   Dw_xsf_dble(mu,psd[0],psd[2]);
   mulg5_dble(VOLUME,psd[2]);
   set_xsf_parms(-1);
   Dw_xsf_dble(-mu,psd[1],psd[3]);
   mulg5_dble(VOLUME,psd[3]);

   z1=spinor_prod_dble(VOLUME,1,psd[0],psd[3]);
   z2=spinor_prod_dble(VOLUME,1,psd[2],psd[1]);

   d=sqrt((z1.re-z2.re)*(z1.re-z2.re)+
          (z1.im-z2.im)*(z1.im-z2.im));
   d/=sqrt((double)(12*NPROC)*(double)(VOLUME));
   error_chk();

   if (my_rank==0)
      printf("Deviation from gamma5-Hermiticity             = %.1e\n",d);

   for (i=0;i<4;i++)
      random_sd(NSPIN,psd[i],1.0);

   m=swdfld();

   for (i=0;i<4;i++)
      random_sd(NSPIN,psd[i],1.0);

   set_xsf_parms(1);
   assign_sd2sd(VOLUME,psd[0],psd[1]);
   assign_sd2sd(VOLUME,psd[2],psd[3]);   
   Dwee_xsf_dble(mu,psd[1],psd[2]);

/*   bnd_sd2zero(EVEN_PTS,psd[0]); */
   mulr_spinor_add_dble(VOLUME,psd[1],psd[0],-1.0);   
   d=norm_square_dble(VOLUME,1,psd[1]);

   error(d!=0.0,1,"main [check13.c]",
         "Dwee_dble() changes the input field in unexpected ways");

   mulr_spinor_add_dble(VOLUME/2,psd[2]+(VOLUME/2),psd[3]+(VOLUME/2),-1.0);   
   assign_sd2sd(VOLUME/2,psd[2],psd[4]);
/*   bnd_sd2zero(EVEN_PTS,psd[4]);   */
   mulr_spinor_add_dble(VOLUME/2,psd[2],psd[4],-1.0);   
   d=norm_square_dble(VOLUME,1,psd[2]);
   
   error(d!=0.0,1,"main [check13.c]",
         "Dwee_dble() changes the output field where it should not");


   xsf=xsf_parms();
   assign_sd2sd(VOLUME,psd[0],psd[1]);
   apply_sw_xsf_dble(VOLUME/2,xsf.tau3,mu,m,psd[0],psd[2]);
   Dwee_xsf_dble(mu,psd[1],psd[3]);   
 
   mulr_spinor_add_dble(VOLUME/2,psd[3],psd[2],-1.0f);
   d=norm_square_dble(VOLUME/2,1,psd[3])/norm_square_dble(VOLUME/2,1,psd[2]);
   d=(float)(sqrt((double)(d)));

   if (my_rank==0)
      printf("Deviation of Dwee() from apply_sw,..          = %.1e\n",d);

   for (i=0;i<4;i++)
      random_sd(NSPIN,psd[i],1.0);

   assign_sd2sd(VOLUME,psd[0],psd[1]);
   assign_sd2sd(VOLUME,psd[2],psd[3]);   
   Dwoo_xsf_dble(mu,psd[1],psd[2]);

/*   bnd_sd2zero(ODD_PTS,psd[0]); */
   mulr_spinor_add_dble(VOLUME,psd[1],psd[0],-1.0);   
   d=norm_square_dble(VOLUME,1,psd[1]);

   error(d!=0.0,1,"main [check13.c]",
         "Dwoo_dble() changes the input field in unexpected ways");

   mulr_spinor_add_dble(VOLUME/2,psd[2],psd[3],-1.0);   
   assign_sd2sd(VOLUME/2,psd[2]+(VOLUME/2),psd[4]+(VOLUME/2));
/*   bnd_sd2zero(ODD_PTS,psd[4]);   */
   mulr_spinor_add_dble(VOLUME/2,psd[2]+(VOLUME/2),psd[4]+(VOLUME/2),-1.0);   
   d=norm_square_dble(VOLUME,1,psd[2]);
   
   error(d!=0.0,1,"main [check13.c]",
         "Dwoo_dble() changes the output field where it should not");

   for (i=0;i<4;i++)
      random_sd(NSPIN,psd[i],1.0);

   assign_sd2sd(VOLUME,psd[0],psd[1]);

   m=swdfld();
   apply_sw_xsf_dble(VOLUME/2,xsf.tau3,mu,m+VOLUME,psd[0]+VOLUME/2,psd[2]+VOLUME/2);
   Dwoo_xsf_dble(mu,psd[1],psd[3]);   
 
   mulr_spinor_add_dble(VOLUME/2,psd[3]+VOLUME/2,psd[2]+VOLUME/2,-1.0f);
   d=norm_square_dble(VOLUME/2,1,psd[3]+VOLUME/2)/norm_square_dble(VOLUME/2,1,psd[2]+VOLUME/2);
   d=(float)(sqrt((double)(d)));

   if (my_rank==0)
      printf("Deviation of Dwoo() from apply_sw,..          = %.1e\n",d);

   for (i=0;i<2;i++)
      random_sd(NSPIN,psd[i],1.0f);

/*   bnd_s2zero(ODD_PTS,ps[0]);*/
   assign_sd2sd(VOLUME,psd[0],psd[1]);

   Dwoo_xsf_dble(0,psd[0],psd[0]);   

   ifail=sw_term(ODD_PTS);
   error_root(ifail!=0,1,"main [check13.c]",
              "Inversion of the SW term was not safe");

   Dwoo_xsf_dble(0,psd[0],psd[0]);   
 
   mulr_spinor_add_dble(VOLUME,psd[0],psd[1],-1.0f);
   d=norm_square_dble(VOLUME,1,psd[0])/norm_square_dble(VOLUME,1,psd[1]);
   d=sqrt((double)(d));

   if (my_rank==0)
      printf("Check on Dwoo()^-1                            = %.1e\n",d);

   sw_term(NO_PTS);

   for (i=0;i<4;i++)
      random_sd(NSPIN,psd[i],1.0);

   assign_sd2sd(VOLUME,psd[0],psd[1]);
   assign_sd2sd(VOLUME,psd[2],psd[3]);   
   Dwoe_xsf_dble(psd[1],psd[2]);

/*   bnd_sd2zero(EVEN_PTS,psd[0]); */
   mulr_spinor_add_dble(VOLUME,psd[1],psd[0],-1.0);   
   d=norm_square_dble(VOLUME,1,psd[1]);

   error(d!=0.0,1,"main [check13.c]",
         "Dwoe_dble() changes the input field in unexpected ways");

   mulr_spinor_add_dble(VOLUME/2,psd[2],psd[3],-1.0);   
   assign_sd2sd(VOLUME/2,psd[2]+(VOLUME/2),psd[4]+(VOLUME/2));
/*   bnd_sd2zero(ODD_PTS,psd[4]);   */
   mulr_spinor_add_dble(VOLUME/2,psd[2]+(VOLUME/2),psd[4]+(VOLUME/2),-1.0);   
   d=norm_square_dble(VOLUME,1,psd[2]);
   
   error(d!=0.0,1,"main [check13.c]",
         "Dwoe_dble() changes the output field where it should not");

   for (i=0;i<4;i++)
      random_sd(NSPIN,psd[i],1.0);

   assign_sd2sd(VOLUME,psd[0],psd[1]);
   assign_sd2sd(VOLUME,psd[2],psd[3]);   
   Dweo_xsf_dble(psd[1],psd[2]);

/*   bnd_sd2zero(ODD_PTS,psd[0]); */
   mulr_spinor_add_dble(VOLUME,psd[1],psd[0],-1.0);   
   d=norm_square_dble(VOLUME,1,psd[1]);

   error(d!=0.0,1,"main [check13.c]",
         "Dweo_dble() changes the input field in unexpected ways");

   mulr_spinor_add_dble(VOLUME/2,psd[2]+(VOLUME/2),psd[3]+(VOLUME/2),-1.0);   
   assign_sd2sd(VOLUME/2,psd[2],psd[4]);
/*   bnd_sd2zero(EVEN_PTS,psd[4]);    */
   mulr_spinor_add_dble(VOLUME/2,psd[2],psd[4],-1.0);   
   d=norm_square_dble(VOLUME,1,psd[2]);
   
   error(d!=0.0,1,"main [check13.c]",
         "Dweo_dble() changes the output field where it should not");
   
   for (i=0;i<4;i++)
      random_sd(NSPIN,psd[i],1.0);

   assign_sd2sd(VOLUME,psd[0],psd[1]);
   assign_sd2sd(VOLUME,psd[2],psd[3]);   
   Dwhat_xsf_dble(mu,psd[1],psd[2]);

/*   bnd_sd2zero(EVEN_PTS,psd[0]); */
   mulr_spinor_add_dble(VOLUME,psd[1],psd[0],-1.0);   
   d=norm_square_dble(VOLUME,1,psd[1]);

   error(d!=0.0,1,"main [check13.c]",
         "Dwhat_dble() changes the input field in unexpected ways");

   mulr_spinor_add_dble(VOLUME/2,psd[2]+(VOLUME/2),psd[3]+(VOLUME/2),-1.0);   
   assign_sd2sd(VOLUME/2,psd[2],psd[4]);
/*   bnd_sd2zero(EVEN_PTS,psd[4]);   */
   mulr_spinor_add_dble(VOLUME/2,psd[2],psd[4],-1.0);   
   d=norm_square_dble(VOLUME,1,psd[2]);
   
   error(d!=0.0,1,"main [check13.c]",
         "Dwhat_dble() changes the output field where it should not");

   for (i=0;i<4;i++)
      random_sd(NSPIN,psd[i],1.0);

   assign_sd2sd(VOLUME,psd[0],psd[2]);
   Dw_xsf_dble(mu,psd[0],psd[1]);
   Dwee_xsf_dble(mu,psd[2],psd[3]);
   set_sd2zero(VOLUME/2,psd[0]);
   mulr_spinor_add_dble(VOLUME/2,psd[0],psd[3],-1.0);    
   Dweo_xsf_dble(psd[2],psd[0]);
   set_sd2zero(VOLUME/2,psd[3]);
   mulr_spinor_add_dble(VOLUME/2,psd[3],psd[0],-1.0);

   Dwoo_xsf_dble(mu,psd[2],psd[3]);   
   Dwoe_xsf_dble(psd[2],psd[4]);
   mulr_spinor_add_dble(VOLUME/2,psd[3]+(VOLUME/2),psd[4]+(VOLUME/2),1.0);      

   mulr_spinor_add_dble(VOLUME,psd[3],psd[1],-1.0);   
   d=norm_square_dble(VOLUME,1,psd[3])/norm_square_dble(VOLUME,1,psd[1]);   
   d=sqrt(d);
   
   if (my_rank==0)
      printf("Deviation of Dw_dble() from Dwee_dble(),..    = %.1e\n",d);

   for (i=0;i<4;i++)
      random_sd(NSPIN,psd[i],1.0);

  sw_term(ODD_PTS);

   assign_sd2sd(NSPIN,psd[0],psd[1]);
   Dwhat_xsf_dble(mu,psd[0],psd[2]);

   Dwoe_xsf_dble(psd[1],psd[1]);
   Dwee_xsf_dble(mu,psd[1],psd[1]);   
   Dwoo_xsf_dble(0.0,psd[1],psd[1]);
   Dweo_xsf_dble(psd[1],psd[1]);

   mulr_spinor_add_dble(VOLUME/2,psd[1],psd[2],-1.0);
   d=norm_square_dble(VOLUME/2,1,psd[1])/norm_square_dble(VOLUME/2,1,psd[2]);
   d=sqrt(d);

   if (my_rank==0)
      printf("Deviation of Dwhat_dble() from Dwee_dble(),.. = %.1e\n",d);
   
   for (i=0;i<4;i++)
      random_sd(NSPIN,psd[i],1.0);

   assign_sd2sd(VOLUME,psd[0],psd[2]);

   set_tm_parms(1);
   Dw_xsf_dble(mu,psd[0],psd[1]);
   set_tm_parms(0);
   
   Dwee_xsf_dble(mu,psd[2],psd[3]);
   mulr_spinor_add_dble(VOLUME/2,psd[1],psd[3],-1.0);    
   Dweo_xsf_dble(psd[2],psd[1]);
   Dwoe_xsf_dble(psd[2],psd[3]);
   mulr_spinor_add_dble(VOLUME/2,psd[1]+(VOLUME/2),psd[3]+(VOLUME/2),-1.0);
   Dwoo_xsf_dble(0.0,psd[2],psd[3]);   
   mulr_spinor_add_dble(VOLUME/2,psd[1]+(VOLUME/2),psd[3]+(VOLUME/2),-1.0);
   d=norm_square_dble(VOLUME,1,psd[1])/norm_square_dble(VOLUME,1,psd[2]);   
   d=sqrt(d);

   error_chk();

   if (my_rank==0)
   {
      printf("Check of Dw_dble()|eoflg=1                    = %.1e\n\n",d);      
      fclose(flog);
   }
   
   MPI_Finalize();
   exit(0);
}
