
/*******************************************************************************
*
* File check4_ss.c
*
* Copyright (C) 2005, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Application of Dw_dble()
*
*
* This code only works in serial!
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "archive.h"
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
#include "linsolv.h"
#include "forces.h"
/*
static const su3_dble ud0={{0.0}};
#define N0 (NPROC0*L0)
#define MAX_LEVELS 8
#define BLK_LENGTH 8

static double smE[L0][MAX_LEVELS];
static int cnt[L0][MAX_LEVELS];
static su3_dble *udb;
static su3_dble wd1,wd2 ALIGNED16;

static double plaq_dble_ss(int n,int ix)
{
   int ip[4];
   double sm;

   plaq_uidx(n,ix,ip);

   su3xsu3(udb+ip[0],udb+ip[1],&wd1);
   su3dagxsu3dag(udb+ip[3],udb+ip[2],&wd2);
   cm3x3_retr(&wd1,&wd2,&sm);
   
   return sm;
}


static double local_plaq_sum_dble_ss(int iw)
{
   int n,ix,t,*cnt0;
   double pa,*smx0;

	 udb=udfld();
   cnt0=cnt[0];
   smx0=smE[0];
   
   for (n=0;n<MAX_LEVELS;n++)
   {
      cnt0[n]=0;
      smx0[n]=0.0;
   }
   
   for (ix=0;ix<VOLUME;ix++)
   {
      t=global_time(ix);
      pa=0.0;

      for (n=0;n<6;n++)
      {
         if (iw==0)
            pa+=plaq_dble_ss(n,ix);
         else
         {
            if (((t>0)&&(t<(N0-1)))||((t==0)&&(n<3)))
               pa+=plaq_dble_ss(n,ix);
         }
      }

      cnt0[0]+=1;
      smx0[0]+=pa;

      for (n=1;(cnt0[n-1]>=BLK_LENGTH)&&(n<MAX_LEVELS);n++)
      {
         cnt0[n]+=1;
         smx0[n]+=smx0[n-1];

         cnt0[n-1]=0;
         smx0[n-1]=0.0;
      }               
   }

   for (n=1;n<MAX_LEVELS;n++)
      smx0[0]+=smx0[n]; 
   
   return smx0[0];
}

static double plaq_sum_dble_ss(int icom)
{
   double p,pa;

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   p=local_plaq_sum_dble_ss(1);

   if (icom==1)
   {
      MPI_Reduce(&p,&pa,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&pa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
      pa=p;
   
   return pa;
}



static void write_gauge(char *out) {

	FILE *fout=NULL;
	int t,x,y,zl,d,s;
	int ix;
	double plaq;
	complex_dble z;

	su3_dble *ud,utemp,unity;

	unity=ud0;
	unity.c11.re=1.0;
	unity.c22.re=1.0;
	unity.c33.re=1.0;

	ud=udfld();
	
	plaq=plaq_sum_dble_ss(1)/(9.0*(double)(2*L0 - 3)*(double)(L1*L2*L3));

	fout=fopen(out,"w");
	error_root(fout==NULL,1,"write_gauge [check4_ss.c]",
			"Unable to open out file");

	fprintf(fout, "%e\n", plaq);
	fflush(stdout);

	for (t=0; t<L0 ;t++) 
		for (x=0; x<L1 ;x++) 
			for (y=0; y<L2 ;y++) 
				for (zl=0; zl <L3 ;zl++) {

					s=t+x+y+zl;
					ix=zl+L3*(y+L2*(x + L1*t));

					fprintf(fout,"%d %d %d %d\n", t, x+1, y+1, zl+1);

					for (d=0 ; d<4 ; d++) {

							if  (!((t==L0-1)&&(d==0))) { 
							if ((s%2)!=0)
								utemp=*(ud+8*(ipt[ix]-VOLUME/2)+2*d);
							else
								utemp=*(ud+8*(iup[ipt[ix]][d]-VOLUME/2)+2*d+1);
							} else 
								utemp=unity; 

						z=utemp.c11;
						fprintf(fout, "%e %e\n", z.re, z.im);
						
						z=utemp.c12;
						fprintf(fout, "%e  %e\n", z.re, z.im);
						
						z=utemp.c13;
						fprintf(fout, "%e  %e\n", z.re, z.im);
						
						z=utemp.c21;
						fprintf(fout, "%e  %e\n", z.re, z.im);
						
						z=utemp.c22;
						fprintf(fout, "%e  %e\n", z.re, z.im);
						
						z=utemp.c23;
						fprintf(fout, "%e  %e\n", z.re, z.im);
						
						z=utemp.c31;
						fprintf(fout, "%e  %e\n", z.re, z.im);
						
						z=utemp.c32;
						fprintf(fout, "%e  %e\n", z.re, z.im);
						
						z=utemp.c33;
						fprintf(fout, "%e  %e\n", z.re, z.im);

					}
				}

	fclose(fout);
	}


static void read_gauge (char *in) {

	FILE *fin=NULL;
	int ir,t,x,y,zl,d,s;
	int tr,xr,yr,zr,ix;
	double plaq_in,plaq;
	complex_dble z;
	su3_dble *ud,utemp;

	ud=udfld();
	
	fin=fopen(in,"r");
	error_root(fin==NULL,1,"read_gauge [check4_ss.c]",
			"Unable to open input file");

	ir=fscanf(fin, "%lf", &plaq_in);

	error_root(ir!=1, 1, "read_gauge [check4_ss.c]", "incorrect read count");
	
	printf("plaq read from input gauge configuration    = %f\n", plaq_in);

	for (t=0; t<L0 ;t++) 
		for (x=0; x<L1 ;x++) 
			for (y=0; y<L2 ;y++) 
				for (zl=0; zl <L3 ;zl++) {

					s=t+x+y+zl;
					ix=zl+L3*(y+L2*(x + L1*t));

					ir=fscanf(fin, "%d %d %d %d", &tr,&xr,&yr,&zr);
	

					error_root(ir!=4, 1, "read_gauge [check4_ss.c]", 
							"incorrect read count dims");

					for (d=0 ; d<4 ; d++) {

						ir=fscanf(fin, "%lf  %lf", &z.re, &z.im);
						utemp.c11 = z;

						ir+=fscanf(fin, "%lf %lf", &z.re, &z.im);
						utemp.c12 = z;

						ir+=fscanf(fin, "%lf %lf", &z.re, &z.im);
						utemp.c13 = z;

						ir+=fscanf(fin, "%lf %lf", &z.re, &z.im);
						utemp.c21 = z;

						ir+=fscanf(fin, "%lf %lf", &z.re, &z.im);
						utemp.c22 = z;

						ir+=fscanf(fin, "%lf %lf", &z.re, &z.im);
						utemp.c23 = z;

						ir+=fscanf(fin, "%lf %lf", &z.re, &z.im);
						utemp.c31 = z;

						ir+=fscanf(fin, "%lf %lf", &z.re, &z.im);
						utemp.c32 = z;

						ir+=fscanf(fin, "%lf %lf", &z.re, &z.im);
						utemp.c33 = z;

						error_root(ir!=18, 1, "read_gauge [check4_ss.c]", 
								"incorrect read count, u");

						if  (!((t==L0-1)&&(d==0))) { 
							if ((s%2)!=0)
								*(ud+8*(ipt[ix]-VOLUME/2)+2*d)=utemp;
							else
								*(ud+8*(iup[ipt[ix]][d]-VOLUME/2)+2*d+1)=utemp;
						}

					}
				}

	fclose(fin);

	plaq=plaq_sum_dble_ss(1)/(9.0*(double)(2*L0 - 3)*(double)(L1*L2*L3));

	printf("plaq calculated from the read configuration = %f\n\n", plaq);
}

static void write_spinor(char *out, spinor_dble *sp) {

	FILE *fout=NULL;
	int t,x,y,zl,ix;
	complex_dble z;
        double plaq;

	spinor_dble stemp;


	plaq=plaq_sum_dble_ss(1)/(9.0*(double)(2*L0 - 3)*(double)(L1*L2*L3));


	fout=fopen(out,"w");
	error_root(fout==NULL,1,"write_spinor [check4_ss.c]",
			"Unable to open output file");

	fprintf(fout, "%e\n", plaq);

	for (t=0; t<L0 ;t++) 
		for (x=0; x<L1 ;x++) 
			for (y=0; y<L2 ;y++) 
				for (zl=0; zl <L3 ;zl++) {

					ix=zl+L3*(y+L2*(x + L1*t));
					stemp=*(sp+ipt[ix]);
					
					fprintf(fout, "%d %d %d %d\n", t,x+1,y+1,zl+1);

					z=stemp.c1.c1;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
					z=stemp.c1.c2;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
					z=stemp.c1.c3;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
					z=stemp.c2.c1;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
					z=stemp.c2.c2;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
					z=stemp.c2.c3;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
					z=stemp.c3.c1;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
					z=stemp.c3.c2;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
					z=stemp.c3.c3;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
					z=stemp.c4.c1;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
					z=stemp.c4.c2;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
					z=stemp.c4.c3;
					fprintf(fout, "%e  %e\n", z.re, z.im);
					
				}

	fclose(fout);
}


static void read_spinor(char *in, spinor_dble *sp) {

	FILE *fin=NULL;
	int ir,t,x,y,zl;
	int tr,xr,yr,zr,ix;
	complex_dble z;
  double plaq_in;

	spinor_dble stemp;
	
	fin=fopen(in,"r");
	error_root(fin==NULL,1,"read_spinor [check4_ss.c]",
			"Unable to open input file");

	ir=fscanf(fin, "%lf", &plaq_in);

	error_root(ir!=1, 1, "read_spinor [check4_ss.c]", "incorrect read count");

	for (t=0; t<L0 ;t++) 
		for (x=0; x<L1 ;x++) 
			for (y=0; y<L2 ;y++) 
				for (zl=0; zl <L3 ;zl++) {

					ix=zl+L3*(y+L2*(x + L1*t));

					ir=fscanf(fin, "%d %d %d %d", &tr,&xr,&yr,&zr);
	

					error_root(ir!=4, 1, "read_gauge [check4_ss.c]", 
							"incorrect read count dims");

					ir=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c1.c1 = z;
					
					ir+=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c1.c2 = z;
					
					ir+=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c1.c3 = z;
					
					ir+=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c2.c1 = z;
					
					ir+=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c2.c2 = z;
					
					ir+=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c2.c3 = z;
					
					ir+=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c3.c1 = z;
					
					ir+=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c3.c2 = z;
					
					ir+=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c3.c3 = z;
					
					ir+=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c4.c1 = z;
					
					ir+=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c4.c2 = z;
					
					ir+=fscanf(fin, "%lf  %lf", &z.re, &z.im);
					stemp.c4.c3 = z;

					error_root(ir!=24, 1, "read_gauge [check4_ss.c]", 
								"incorrect read count, u");

					*(sp+ipt[ix])=stemp;

				}

	fclose(fin);
}
*/
int main(int argc,char *argv[])
{
	int my_rank,nmx,status;
	double mu,c0s,fac,d,rho,res,nrm;
	complex_dble z;
	spinor_dble **psd;
	sw_parms_t swp;   
	FILE *flog=NULL;

	double phi[2],phip[2],theta[3];

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	if (my_rank==0)
	{
		flog=freopen("check4_ss.log","w",stdout);
		printf("\n");
		printf("Application of Dw_dble() (random fields)\n");
		printf("---------------------------------------------\n\n");

		printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
		printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
		printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

		fflush(stdout);
	}

	start_ranlux(0,12345);
	geometry();
	alloc_ws(5);
	alloc_wsd(10);
	psd=reserve_wsd(4);

	set_lat_parms(5.5,1.0,0.135499,0.0,0.0,1.584177689,1.0,1.0,0.5,1.5);
	swp=set_sw_parms(-0.3099358667);

	phi[0]=0.0; phi[1]=0.0; 
	phip[0]=0.0; phip[1]=0.0;
	theta[0]=0.5; theta[1]=0.5; theta[2]=0.5;
	set_sf_parms(phi,phip,theta);
	mu=0.0;

	if (my_rank==0)
		printf("m0 = %.4e, mu= %.4e, csw = %.4e, cF = %.4e\n\n",
				swp.m0,mu,swp.csw,swp.cF);
/*
	read_gauge("8x8/test.conf");
   export_cnfg("8x8/test.conf_cern");
	write_gauge("out.conf");
*/
   import_cnfg("8x8/test.conf_cern");
	mult_phase(1);
/*
	read_spinor("source", psd[0]);
	write_spinor("source_test", psd[0]);

	read_spinor("solution", psd[1]);
*/
	sw_term_xsf(NO_PTS);
	c0s = 1.0/(1.0+8.0*0.135499);
	fac = 2.0*0.135499*c0s;
/*
	z.re=-1.0*fac;
	z.im=0.0;

	set_xsf_parms(1);
	Dw_xsf_dble(mu, psd[0], psd[2]); 

	mulg5_dble(VOLUME,psd[2]);

	set_sd2zero(VOLUME,psd[3]);
	mulc_spinor_add_dble(VOLUME,psd[3],psd[2],z);
	write_spinor("solution_jm", psd[3]);

	mulc_spinor_add_dble(VOLUME,psd[1],psd[2],z);
	d=norm_square_dble(VOLUME,1,psd[1])/norm_square_dble(VOLUME,1,psd[2]);

	error_chk();

	if (my_rank==0)
	{
		printf("Application of Dw_up for mu=0: Normalized difference = %.2e\n",sqrt(d));
		printf("(should be around 1*10^(-15) or so)\n\n");
	}

	read_spinor("source", psd[0]);
	write_spinor("source_test", psd[0]);

	read_spinor("solution", psd[1]);

	sw_term_xsf(NO_PTS);
	c0s = 1.0/(1.0+8.0*0.135499);
	fac = 2.0*0.135499*c0s;

	z.re=-1.0*fac;
	z.im=0.0;

	set_xsf_parms(-1);
	Dw_xsf_dble(mu, psd[0], psd[2]); 

	mulg5_dble(VOLUME,psd[2]);

	set_sd2zero(VOLUME,psd[3]);
	mulc_spinor_add_dble(VOLUME,psd[3],psd[2],z);
	write_spinor("solution_jm", psd[3]);

	mulc_spinor_add_dble(VOLUME,psd[1],psd[2],z);
	d=norm_square_dble(VOLUME,1,psd[1])/norm_square_dble(VOLUME,1,psd[2]);

	error_chk();

	if (my_rank==0)
	{
		printf("Application of Dw_down for mu=0: Normalized difference = %.2e\n",sqrt(d));
		printf("(should be around 1*10^(-15) or so)\n\n");
	}

	read_spinor("8x8/down_inv_solution", psd[1]);
   export_sfld("8x8/down_inv_solution_cern",psd[1]);

	res=1e-12; nmx=5000;
	rho=tmcg_xsf(nmx,res,1,mu,psd[0],psd[2],&status);

	set_xsf_parms(1);
	Dw_xsf_dble(mu, psd[2], psd[3]); 
	mulg5_dble(VOLUME,psd[3]);

	z.re=-1.0/fac;
	z.im=0.0;

	mulc_spinor_add_dble(VOLUME,psd[1],psd[3],z);
	d=norm_square_dble(VOLUME,1,psd[1]);
	nrm=norm_square_dble(VOLUME,1,psd[3]);
	if (my_rank==0)
	{
		printf("Solution w/o eo-preconditioning for Dw_down^{-1} mu=0 :\n");
		printf("status = %d\n",status);
		printf("rho   = %.2e, res   = %.2e\n",rho,res);
		printf("check = %.2e, check = %.2e\n",sqrt(d),sqrt(d/nrm));
		printf("(should be around 1*10^(-15) or so)\n\n");
	}

	read_spinor("8x8/up_inv_solution", psd[1]);
   export_sfld("8x8/up_inv_solution_cern",psd[1]);

	res=1e-12; nmx=5000;
	rho=tmcg_xsf(nmx,res,-1,mu,psd[0],psd[2],&status);

	set_xsf_parms(-1);
	Dw_xsf_dble(mu, psd[2], psd[3]); 
	mulg5_dble(VOLUME,psd[3]);

	z.re=-1.0/fac;
	z.im=0.0;

	mulc_spinor_add_dble(VOLUME,psd[1],psd[3],z);
	d=norm_square_dble(VOLUME,1,psd[1]);
	nrm=norm_square_dble(VOLUME,1,psd[3]);
	if (my_rank==0)
	{
		printf("Solution w/o eo-preconditioning for Dw_up^{-1} mu=0 :\n");
		printf("status = %d\n",status);
		printf("rho   = %.2e, res   = %.2e\n",rho,res);
		printf("check = %.2e, check = %.2e\n",sqrt(d),sqrt(d/nrm));
		printf("(should be around 1*10^(-15) or so)\n\n");
	}
*/
/*
	read_spinor("8x8/source_mu", psd[0]);
   export_sfld("8x8/source_mu_cern",psd[0]);
	read_spinor("8x8/solution_up_mu", psd[1]);
   export_sfld("8x8/solution_up_mu_cern",psd[1]);
*/
   import_sfld("8x8/source_mu_cern",psd[0]);
   import_sfld("8x8/solution_up_mu_cern",psd[1]);

	mu=0.05;
	z.re=-1.0*fac;
	z.im=0.0;

	set_xsf_parms(1);
	Dw_xsf_dble(mu, psd[0], psd[2]); 

	mulg5_dble(VOLUME,psd[2]);

	set_sd2zero(VOLUME,psd[3]);
	mulc_spinor_add_dble(VOLUME,psd[3],psd[2],z);

	mulc_spinor_add_dble(VOLUME,psd[1],psd[2],z);
	d=norm_square_dble(VOLUME,1,psd[1])/norm_square_dble(VOLUME,1,psd[2]);

	error_chk();

	if (my_rank==0)
	{
		printf("Application of Dw_up for mu!=0: Normalized difference = %.2e\n",sqrt(d));
		printf("(should be around 1*10^(-15) or so)\n\n");
	}
/*
	read_spinor("8x8/source_mu", psd[0]);
   export_sfld("8x8/source_mu_cern",psd[0]);
	read_spinor("8x8/solution_down_mu", psd[1]);
   export_sfld("8x8/solution_down_mu_cern",psd[1]);
*/
   import_sfld("8x8/source_mu_cern",psd[0]);
   import_sfld("8x8/solution_down_mu_cern",psd[1]);

	mu=-0.05;
	z.re=-1.0*fac;
	z.im=0.0;

	set_xsf_parms(-1);
	Dw_xsf_dble(mu, psd[0], psd[2]); 

	mulg5_dble(VOLUME,psd[2]);

	set_sd2zero(VOLUME,psd[3]);
	mulc_spinor_add_dble(VOLUME,psd[3],psd[2],z);

	mulc_spinor_add_dble(VOLUME,psd[1],psd[2],z);
	d=norm_square_dble(VOLUME,1,psd[1])/norm_square_dble(VOLUME,1,psd[2]);

	error_chk();

	if (my_rank==0)
	{
		printf("Application of Dw_down for mu!=0: Normalized difference = %.2e\n",sqrt(d));
		printf("(should be around 1*10^(-15) or so)\n\n");
	}
/*
	read_spinor("8x8/down_inv_solution_mu", psd[1]);
   export_sfld("8x8/down_inv_solution_mu_cern",psd[1]);
*/
   import_sfld("8x8/down_inv_solution_mu_cern",psd[1]);

   mu=0.05;
	res=1e-12; nmx=5000;
	rho=tmcg_xsf(nmx,res,1,mu,psd[0],psd[2],&status);

	set_xsf_parms(1);
	Dw_xsf_dble(mu, psd[2], psd[3]); 
	mulg5_dble(VOLUME,psd[3]);

	z.re=-1.0/fac;
	z.im=0.0;

	mulc_spinor_add_dble(VOLUME,psd[1],psd[3],z);
	d=norm_square_dble(VOLUME,1,psd[1]);
	nrm=norm_square_dble(VOLUME,1,psd[3]);
	if (my_rank==0)
	{
		printf("Solution w/o eo-preconditioning for Dw_down^{-1} mu!=0 :\n");
		printf("status = %d\n",status);
		printf("rho   = %.2e, res   = %.2e\n",rho,res);
		printf("check = %.2e, check = %.2e\n",sqrt(d),sqrt(d/nrm));
		printf("(should be around 1*10^(-15) or so)\n\n");
	}
/*
	read_spinor("8x8/up_inv_solution_mu", psd[1]);
   export_sfld("8x8/up_inv_solution_mu_cern",psd[1]);
*/   
   import_sfld("8x8/up_inv_solution_mu_cern",psd[1]);

        mu=-0.05;
	res=1e-12; nmx=5000;
	rho=tmcg_xsf(nmx,res,-1,mu,psd[0],psd[2],&status);

	set_xsf_parms(-1);
	Dw_xsf_dble(mu, psd[2], psd[3]); 
	mulg5_dble(VOLUME,psd[3]);

	z.re=-1.0/fac;
	z.im=0.0;

	mulc_spinor_add_dble(VOLUME,psd[1],psd[3],z);
	d=norm_square_dble(VOLUME,1,psd[1]);
	nrm=norm_square_dble(VOLUME,1,psd[3]);
	if (my_rank==0)
	{
		printf("Solution w/o eo-preconditioning Dw_up^{-1} mu!=0 :\n");
		printf("status = %d\n",status);
		printf("rho   = %.2e, res   = %.2e\n",rho,res);
		printf("check = %.2e, check = %.2e\n",sqrt(d),sqrt(d/nrm));
		printf("(should be around 1*10^(-15) or so)\n\n");
	}
/*
	read_spinor("8x8/source_random", psd[0]);
   export_sfld("8x8/source_random_cern",psd[0]);
	read_spinor("8x8/solution_up_random", psd[1]);
   export_sfld("8x8/solution_up_random_cern",psd[1]);
*/

   import_sfld("8x8/source_random_cern",psd[0]);
   import_sfld("8x8/solution_up_random_cern",psd[1]);

	mu=0.05;
	z.re=-1.0*fac;
	z.im=0.0;

	set_xsf_parms(1);
	Dw_xsf_dble(mu, psd[0], psd[2]); 

	mulg5_dble(VOLUME,psd[2]);

	set_sd2zero(VOLUME,psd[3]);
	mulc_spinor_add_dble(VOLUME,psd[3],psd[2],z);

	mulc_spinor_add_dble(VOLUME,psd[1],psd[2],z);
	d=norm_square_dble(VOLUME,1,psd[1])/norm_square_dble(VOLUME,1,psd[2]);

	error_chk();

	if (my_rank==0)
	{
		printf("Application of Dw_up for random source: Normalized difference = %.2e\n",sqrt(d));
		printf("(should be around 1*10^(-15) or so)\n\n");
	}
/*
	read_spinor("8x8/source_random", psd[0]);
   export_sfld("8x8/source_random_cern",psd[0]);
	read_spinor("8x8/solution_down_random", psd[1]);
   export_sfld("8x8/solution_down_random_cern",psd[1]);
*/

   import_sfld("8x8/source_random_cern",psd[0]);
   import_sfld("8x8/solution_down_random_cern",psd[1]);

	mu=-0.05;
	z.re=-1.0*fac;
	z.im=0.0;

	set_xsf_parms(-1);
	Dw_xsf_dble(mu, psd[0], psd[2]); 

	mulg5_dble(VOLUME,psd[2]);

	set_sd2zero(VOLUME,psd[3]);
	mulc_spinor_add_dble(VOLUME,psd[3],psd[2],z);

	mulc_spinor_add_dble(VOLUME,psd[1],psd[2],z);
	d=norm_square_dble(VOLUME,1,psd[1])/norm_square_dble(VOLUME,1,psd[2]);

	error_chk();

	if (my_rank==0)
	{
		printf("Application of Dw_down for random source: Normalized difference = %.2e\n",sqrt(d));
		printf("(should be around 1*10^(-15) or so)\n\n");
	}
/*
	read_spinor("8x8/down_inv_solution_random", psd[1]);
   export_sfld("8x8/down_inv_solution_random_cern",psd[1]);
*/

   import_sfld("8x8/down_inv_solution_random_cern",psd[1]);

   mu=0.05;
	res=1e-12; nmx=5000;
	rho=tmcg_xsf(nmx,res,1,mu,psd[0],psd[2],&status);

	set_xsf_parms(1);
	Dw_xsf_dble(mu, psd[2], psd[3]); 
	mulg5_dble(VOLUME,psd[3]);

	z.re=-1.0/fac;
	z.im=0.0;

	mulc_spinor_add_dble(VOLUME,psd[1],psd[3],z);
	d=norm_square_dble(VOLUME,1,psd[1]);
	nrm=norm_square_dble(VOLUME,1,psd[3]);
	if (my_rank==0)
	{
		printf("Solution w/o eo-preconditioning for Dw_down^{-1} for random source :\n");
		printf("status = %d\n",status);
		printf("rho   = %.2e, res   = %.2e\n",rho,res);
		printf("check = %.2e, check = %.2e\n",sqrt(d),sqrt(d/nrm));
		printf("(should be around 1*10^(-15) or so)\n\n");
	}
/*
	read_spinor("8x8/up_inv_solution_random", psd[1]);
   export_sfld("8x8/up_inv_solution_random_cern",psd[1]);
*/   

   import_sfld("8x8/up_inv_solution_random_cern",psd[1]);

   mu=-0.05;
	res=1e-12; nmx=5000;
	rho=tmcg_xsf(nmx,res,-1,mu,psd[0],psd[2],&status);

	set_xsf_parms(-1);
	Dw_xsf_dble(mu, psd[2], psd[3]); 
	mulg5_dble(VOLUME,psd[3]);

	z.re=-1.0/fac;
	z.im=0.0;

	mulc_spinor_add_dble(VOLUME,psd[1],psd[3],z);
	d=norm_square_dble(VOLUME,1,psd[1]);
	nrm=norm_square_dble(VOLUME,1,psd[3]);
	if (my_rank==0)
	{
		printf("Solution w/o eo-preconditioning Dw_up^{-1} for random source:\n");
		printf("status = %d\n",status);
		printf("rho   = %.2e, res   = %.2e\n",rho,res);
		printf("check = %.2e, check = %.2e\n",sqrt(d),sqrt(d/nrm));
		printf("(should be around 1*10^(-15) or so)\n\n");
	   fclose(flog);
	}


	MPI_Finalize();
	exit(0);
}
