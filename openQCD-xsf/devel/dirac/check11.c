
/*******************************************************************************
*
* File check11.c
*
* Copyright (C) 2005, 2008, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Hermiticity of Dw() and comparison with Dwee(),...
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
   float mu,d;
   float kappa,m0;
   complex z1,z2;
   spinor **ps;
   sw_parms_t swp;
   xsf_parms_t xsf;
   pauli *m;
   FILE *flog=NULL;
   double phi[2],phip[2];
   double theta[3];

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check11.log","w",stdout);
      printf("\n");
      printf("Hermiticity of Dw() and comparison with Dwee(),...\n");
      printf("--------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,12345);
   geometry();
   alloc_ws(5);
   ps=reserve_ws(5);

   m0=-0.0123f;
   kappa=1.0f/(2.0f*m0+8.0f);

   set_lat_parms(5.5,1.0,kappa,0.0,0.0,0.456,1.0,1.234,1.0,0.5);
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
      printf("Deviations should be at most 10^(-6) or so in these tests\n\n");
   }

   random_ud();
   sw_term(NO_PTS);
   assign_ud2u();
   assign_swd2sw();

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   set_xsf_parms(1);
   Dw_xsf(mu,ps[0],ps[2]);
   mulg5(VOLUME,ps[2]);
   set_xsf_parms(-1);
   Dw_xsf(-mu,ps[1],ps[3]);
   mulg5(VOLUME,ps[3]);

   z1=spinor_prod(VOLUME,1,ps[0],ps[3]);
   z2=spinor_prod(VOLUME,1,ps[2],ps[1]);

   d=(float)(sqrt((double)((z1.re-z2.re)*(z1.re-z2.re)+
                           (z1.im-z2.im)*(z1.im-z2.im))));
   d/=(float)(sqrt((double)(12*NPROC)*(double)(VOLUME)));
   error_chk();

   if (my_rank==0)
      printf("Deviation from gamma5-Hermiticity    = %.1e\n",d);

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);
   assign_s2s(VOLUME,ps[2],ps[3]);   
   set_xsf_parms(1);
   Dwee_xsf(mu,ps[1],ps[2]);

/* bnd_s2zero(EVEN_PTS,ps[0]); */
   mulr_spinor_add(VOLUME,ps[1],ps[0],-1.0f);   
   d=norm_square(VOLUME,1,ps[1]);

   error(d!=0.0f,1,"main [check11.c]",
         "Dwee() changes the input field in unexpected ways");

   mulr_spinor_add(VOLUME/2,ps[2]+(VOLUME/2),ps[3]+(VOLUME/2),-1.0f);   
   assign_s2s(VOLUME/2,ps[2],ps[4]);
/* bnd_s2zero(EVEN_PTS,ps[4]);   */
   mulr_spinor_add(VOLUME/2,ps[2],ps[4],-1.0f);   
   d=norm_square(VOLUME,1,ps[2]);
   
   error(d!=0.0f,1,"main [check11.c]",
         "Dwee() changes the output field where it should not");

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);

   m=swfld();
   xsf=xsf_parms();
   apply_sw_xsf(VOLUME/2,xsf.tau3,mu,m,ps[0],ps[2]);
   Dwee_xsf(mu,ps[1],ps[3]);   
 
   mulr_spinor_add(VOLUME/2,ps[3],ps[2],-1.0f);
   d=norm_square(VOLUME/2,1,ps[3])/norm_square(VOLUME/2,1,ps[2]);
   d=(float)(sqrt((double)(d)));

   if (my_rank==0)
      printf("Deviation of Dwee() from apply_sw,.. = %.1e\n",d);

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);
   assign_s2s(VOLUME,ps[2],ps[3]);   
   Dwoo_xsf(mu,ps[1],ps[2]);

/* bnd_s2zero(ODD_PTS,ps[0]); */
   mulr_spinor_add(VOLUME,ps[1],ps[0],-1.0f);   
   d=norm_square(VOLUME,1,ps[1]);

   error(d!=0.0f,1,"main [check11.c]",
         "Dwoo() changes the input field in unexpected ways");

   mulr_spinor_add(VOLUME/2,ps[2],ps[3],-1.0f);   
   assign_s2s(VOLUME/2,ps[2]+(VOLUME/2),ps[4]+(VOLUME/2));
/* bnd_s2zero(ODD_PTS,ps[4]);  */ 
   mulr_spinor_add(VOLUME/2,ps[2]+(VOLUME/2),ps[4]+(VOLUME/2),-1.0f);   
   d=norm_square(VOLUME,1,ps[2]);
   
   error(d!=0.0f,1,"main [check11.c]",
         "Dwoo() changes the output field where it should not");

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);

   set_xsf_parms(-1);
   xsf=xsf_parms(); 
   apply_sw_xsf(VOLUME/2,xsf.tau3,mu,m+VOLUME,ps[0]+VOLUME/2,ps[2]+VOLUME/2);
   Dwoo_xsf(mu,ps[1],ps[3]);   
 
   mulr_spinor_add(VOLUME/2,ps[3]+VOLUME/2,ps[2]+VOLUME/2,-1.0f);
   d=norm_square(VOLUME/2,1,ps[3]+VOLUME/2)/norm_square(VOLUME/2,1,ps[2]+VOLUME/2);
   d=(float)(sqrt((double)(d)));

   if (my_rank==0)
      printf("Deviation of Dwoo() from apply_sw,.. = %.1e\n",d);
    
   for (i=0;i<2;i++)
      random_s(NSPIN,ps[i],1.0f);

/*   bnd_s2zero(ODD_PTS,ps[0]);*/
   assign_s2s(VOLUME,ps[0],ps[1]);

   set_xsf_parms(1);
   Dwoo_xsf(0,ps[0],ps[0]);   

   ifail=sw_term(ODD_PTS);
   error_root(ifail!=0,1,"main [check12.c]",
              "Inversion of the SW term was not safe");

   assign_swd2sw();

   Dwoo_xsf(0,ps[0],ps[0]);   
 
   mulr_spinor_add(VOLUME,ps[0],ps[1],-1.0f);
   d=norm_square(VOLUME,1,ps[0])/norm_square(VOLUME,1,ps[1]);
   d=(float)(sqrt((double)(d)));

   sw_term(NO_PTS);
   assign_swd2sw();

   if (my_rank==0)
      printf("Check on Dwoo()^-1                   = %.1e\n",d);

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);
   assign_s2s(VOLUME,ps[2],ps[3]);   
   Dwoe_xsf(ps[1],ps[2]);

/* bnd_s2zero(EVEN_PTS,ps[0]); */
   mulr_spinor_add(VOLUME,ps[1],ps[0],-1.0f);   
   d=norm_square(VOLUME,1,ps[1]);

   error(d!=0.0f,1,"main [check11.c]",
         "Dwoe() changes the input field in unexpected ways");

   mulr_spinor_add(VOLUME/2,ps[2],ps[3],-1.0f);   
   assign_s2s(VOLUME/2,ps[2]+(VOLUME/2),ps[4]+(VOLUME/2));
/* bnd_s2zero(ODD_PTS,ps[4]);  */ 
   mulr_spinor_add(VOLUME/2,ps[2]+(VOLUME/2),ps[4]+(VOLUME/2),-1.0f);   
   d=norm_square(VOLUME,1,ps[2]);
   
   error(d!=0.0f,1,"main [check11.c]",
         "Dwoe() changes the output field where it should not");

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);
   assign_s2s(VOLUME,ps[2],ps[3]);   
   Dweo_xsf(ps[1],ps[2]);

/* bnd_s2zero(ODD_PTS,ps[0]); */
   mulr_spinor_add(VOLUME,ps[1],ps[0],-1.0f);   
   d=norm_square(VOLUME,1,ps[1]);

   error(d!=0.0f,1,"main [check11.c]",
         "Dweo() changes the input field in unexpected ways");

   mulr_spinor_add(VOLUME/2,ps[2]+(VOLUME/2),ps[3]+(VOLUME/2),-1.0f);   
   assign_s2s(VOLUME/2,ps[2],ps[4]);
/* bnd_s2zero(EVEN_PTS,ps[4]); */ 
   mulr_spinor_add(VOLUME/2,ps[2],ps[4],-1.0f);   
   d=norm_square(VOLUME,1,ps[2]);
   
   error(d!=0.0f,1,"main [check11.c]",
         "Dweo() changes the output field where it should not");
   
   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);
   assign_s2s(VOLUME,ps[2],ps[3]);   
   Dwhat_xsf(mu,ps[1],ps[2]);

/*   bnd_s2zero(EVEN_PTS,ps[0]); */
   mulr_spinor_add(VOLUME,ps[1],ps[0],-1.0f);   
   d=norm_square(VOLUME,1,ps[1]);

   error(d!=0.0f,1,"main [check11.c]",
         "Dwhat() changes the input field in unexpected ways");

   mulr_spinor_add(VOLUME/2,ps[2]+(VOLUME/2),ps[3]+(VOLUME/2),-1.0f);   
   assign_s2s(VOLUME/2,ps[2],ps[4]);
/*   bnd_s2zero(EVEN_PTS,ps[4]); */
   mulr_spinor_add(VOLUME/2,ps[2],ps[4],-1.0f);   
   d=norm_square(VOLUME,1,ps[2]);
   
   error(d!=0.0f,1,"main [check11.c]",
         "Dwhat() changes the output field where it should not");

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[2]);
   set_xsf_parms(-1);
   Dw_xsf(mu,ps[0],ps[1]);
   Dwee_xsf(mu,ps[2],ps[3]);
   set_s2zero(VOLUME/2,ps[0]);
   mulr_spinor_add(VOLUME/2,ps[0],ps[3],-1.0f);    
   Dweo_xsf(ps[2],ps[0]);
   set_s2zero(VOLUME/2,ps[3]);
   mulr_spinor_add(VOLUME/2,ps[3],ps[0],-1.0f);

   Dwoo_xsf(mu,ps[2],ps[3]);   
   Dwoe_xsf(ps[2],ps[4]);
   mulr_spinor_add(VOLUME/2,ps[3]+(VOLUME/2),ps[4]+(VOLUME/2),1.0f);      
   mulr_spinor_add(VOLUME,ps[3],ps[1],-1.0f);   
   d=norm_square(VOLUME,1,ps[3])/norm_square(VOLUME,1,ps[1]);   
   d=(float)(sqrt((double)(d)));
   
   if (my_rank==0)
      printf("Deviation of Dw() from Dwee(),..     = %.1e\n",d);

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(NSPIN,ps[0],ps[1]);

   set_xsf_parms(1);
   Dwhat_xsf(mu,ps[0],ps[2]);

   Dwoe_xsf(ps[1],ps[1]);
   Dwee_xsf(mu,ps[1],ps[1]);   
   Dwoo_xsf(0.0,ps[1],ps[1]);
   Dweo_xsf(ps[1],ps[1]);
   
   mulr_spinor_add(VOLUME/2,ps[1],ps[2],-1.0f);
   d=norm_square(VOLUME/2,1,ps[1])/norm_square(VOLUME/2,1,ps[2]);
   d=(float)(sqrt((double)(d)));

   if (my_rank==0)
      printf("Deviation of Dwhat() from Dwee(),..  = %.1e\n",d);

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[2]);

   set_tm_parms(1);
   set_xsf_parms(-1);
   Dw_xsf(mu,ps[0],ps[1]);
   set_tm_parms(0);
   
   Dwee_xsf(mu,ps[2],ps[3]);
   mulr_spinor_add(VOLUME/2,ps[1],ps[3],-1.0f);    
   Dweo_xsf(ps[2],ps[1]);
   Dwoe_xsf(ps[2],ps[3]);
   mulr_spinor_add(VOLUME/2,ps[1]+(VOLUME/2),ps[3]+(VOLUME/2),-1.0f);
   Dwoo_xsf(0.0f,ps[2],ps[3]);   
   mulr_spinor_add(VOLUME/2,ps[1]+(VOLUME/2),ps[3]+(VOLUME/2),-1.0f);
   d=norm_square(VOLUME,1,ps[1])/norm_square(VOLUME,1,ps[2]);   
   d=(float)(sqrt((double)(d)));
   
   error_chk();

   if (my_rank==0)
   {
      printf("Check of Dw()|eoflg=1                = %.1e\n\n",d);
      fclose(flog);
   }
   
   MPI_Finalize();
   exit(0);
}
