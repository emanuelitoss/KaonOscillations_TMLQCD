
/*******************************************************************************
*
* File archive.c
*
* Copyright (C) 2005, 2007, 2009, 2010, 2011, 2012 Martin Luescher, 2013 Hubert Simma
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs to read and write gauge-field configurations
*
* The externally accessible functions are
*
*   void write_cnfg(char *out)
*     Writes the lattice sizes, the processor grid, the rank of the
*     calling process, the state of the random number generator and the
*     local double-precision gauge field to the file "out".
*
*   void read_cnfg(char *in)
*     Reads the data previously written by the program write_cnfg from
*     the file "in" and resets the random number generator and the local
*     double-precision gauge field accordingly. The program checks that
*     the configuration satisfies open boundary conditions.
*
*   void export_cnfg(char *out)
*     Writes the lattice sizes and the global double-precision gauge field to
*     the file "out" from process 0 either in the universal "CERN" format 
*     (if sf_flg==0) or in lexicographic order as specified below.
*
*   void import_cnfg(char *in)
*     Reads the global double-precision gauge field from the file "in" from
*     process 0. If sf_flg==1 it is assumed that the that the field is stored 
*     in lexicographic order. The unused links sticking out of the boundaries
*     at t=0 and t=L0 are forced to zero if FORCE_LINKS is defined.
*     Otherwise, if sf_flg==0,  it is assumed that the field is stored
*     in the universal "CERN" format as specified below. In this case the
*     field is periodically extended if needed and the program imposes 
*     open boundary conditions when they are not already satisfied.
*
* Notes:
*
* All programs in this module may involve global communications and must be
* called simultaneously on all processes.
*
* The program export_cnfg_cern() first writes the lattice sizes and the average of
* the plaquette Re tr{U_p} to the output file. Then follow the 8 link variables
* in the directions +0,-0,...,+3,-3 at the first odd point, the second odd
* point, and so on. The order of the point (x0,x1,x2,x3) with coordinates in
* the range 0<=x0<N0,...,0<=x3<N3 is determined by the index
*
*   ix=x3+N3*x2+N2*N3*x1+N1*N2*N3*x0
*
* where N0,N1,N2,N3 are the lattice sizes.
*
* Independently of the machine, the export functions write the data to the
* output file in little-endian byte order. Integers and double-precision
* numbers on the output file occupy 4 and 8 bytes, respectively, the latter
* being formatted according to the IEEE-754 standard. The import functions
* assume the data on the input file to be little endian and converts them
* to big-endian order if the machine is big endian. Exported configurations
* can thus be safely exchanged between different machines.
*
* In the case of the write and read functions, no byte reordering is applied
* and the data are written and read respecting the endianness of the machine.
* The copy_file() program copies characters one by one and therefore preserves
* the byte ordering.
*
* It is permissible to import field configurations that do not satisfy open
* boundary conditions. Fields satisfying open boundary conditions cannot be
* periodically extended in the time direction (an error occurs in this case).
*
* The function export_cnfg_lex writes only the gauge links U(t,x,y,z,mu) 
* in the order of a multi-dimensional array in C, i.e. with index t running 
* slowest and index mu running fastest.
*  
*******************************************************************************/

#define ARCHIVE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "archive.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int ns,nd,*state=NULL;
static su3_dble *ubuf=NULL,*vbuf,*udb;
static const su3_dble ud0={{0.0}};

static void alloc_state(void)
{
   int n;

   ns=rlxs_size();
   nd=rlxd_size();

   if (ns<nd)
      n=nd;
   else
      n=ns;

   state=amalloc(n*sizeof(int),3);
   error(state==NULL,1,"alloc_state [archive.c]",
         "Unable to allocate auxiliary array");
}


void write_cnfg(char *out)
{
   int ldat[9],iw;
   FILE *fout;

   if (state==NULL)
      alloc_state();

   fout=fopen(out,"wb");
   error_loc(fout==NULL,1,"write_cnfg [archive.c]",
             "Unable to open output file");
   error_chk();

   ldat[0]=NPROC0;
   ldat[1]=NPROC1;
   ldat[2]=NPROC2;
   ldat[3]=NPROC3;

   ldat[4]=L0;
   ldat[5]=L1;
   ldat[6]=L2;
   ldat[7]=L3;

   MPI_Comm_rank(MPI_COMM_WORLD,ldat+8);

   iw=fwrite(ldat,sizeof(int),9,fout);
   rlxs_get(state);
   iw+=fwrite(state,sizeof(int),ns,fout);
   rlxd_get(state);
   iw+=fwrite(state,sizeof(int),nd,fout);
   udb=udfld();
   iw+=fwrite(udb,sizeof(su3_dble),4*VOLUME,fout);

   error_loc(iw!=(9+ns+nd+4*VOLUME),1,"write_cnfg [archive.c]",
             "Incorrect write count");
   error_chk();
   fclose(fout);
}


void read_cnfg(char *in)
{
   int n,ldat[9],ir;
   FILE *fin;

   if (state==NULL)
      alloc_state();

   fin=fopen(in,"rb");
   error_loc(fin==NULL,1,"read_cnfg [archive.c]",
             "Unable to open input file");
   error_chk();

   ir=fread(ldat,sizeof(int),9,fin);
   MPI_Comm_rank(MPI_COMM_WORLD,&n);

   error((ldat[0]!=NPROC0)||(ldat[1]!=NPROC1)||
         (ldat[2]!=NPROC2)||(ldat[3]!=NPROC3)||
         (ldat[4]!=L0)||(ldat[5]!=L1)||(ldat[6]!=L2)||(ldat[7]!=L3)||
         (ldat[8]!=n),1,"read_cnfg [archive.c]","Unexpected lattice data");

   ir+=fread(state,sizeof(int),ns,fin);
   rlxs_reset(state);
   ir+=fread(state,sizeof(int),nd,fin);
   rlxd_reset(state);
   udb=udfld();
   ir+=fread(udb,sizeof(su3_dble),4*VOLUME,fin);

   error_loc(ir!=(9+ns+nd+4*VOLUME),1,"read_cnfg [archive.c]",
             "Incorrect read count");
   error_chk();
   fclose(fin);
   
   error(check_bcd()!=1,1,"read_cnfg [archive.c]",
         "Field does not satisfy open bcd");
   set_flags(UPDATED_UD);
}


static int check_machine(void)
{
   int ie;
   
   error_root(sizeof(stdint_t)!=4,1,"check_machine [archive.c]",
              "Size of a stdint_t integer is not 4");
   error_root(sizeof(double)!=8,1,"check_machine [archive.c]",
              "Size of a double is not 8");   

   ie=endianness();
   error_root(ie==UNKNOWN_ENDIAN,1,"check_machine [archive.c]",
              "Unkown endianness");

   return ie;
}


static void alloc_ubuf(int my_rank)
{
   if (my_rank==0)
   {
      ubuf=amalloc(4*(L3+N3)*sizeof(su3_dble),ALIGN);
      vbuf=ubuf+4*L3;
   }
   else
      ubuf=amalloc(4*L3*sizeof(su3_dble),ALIGN);

   error(ubuf==NULL,1,"alloc_ubuf [archive.c]",
         "Unable to allocate auxiliary array");
}


static void get_links(int iy)
{
   int y3,iz,mu;
   su3_dble *u,*v;

   v=ubuf;
   iy*=L3;

   if (ipt[iy]<(VOLUME/2))
      iy+=1;

   for (y3=0;y3<L3;y3+=2)
   {
      iz=ipt[iy+y3];
      u=udb+8*(iz-(VOLUME/2));

      for (mu=0;mu<8;mu++)
      {
         *v=*u;
         v+=1;
         u+=1;
      }
   }
}


static void set_links(int iy)
{
   int y3,iz,mu;
   su3_dble *u,*v;

   v=ubuf;
   iy*=L3;

   if (ipt[iy]<(VOLUME/2))
      iy+=1;

   for (y3=0;y3<L3;y3+=2)
   {
      iz=ipt[iy+y3];
      u=udb+8*(iz-(VOLUME/2));

      for (mu=0;mu<8;mu++)
      {
         *u=*v;
         v+=1;
         u+=1;
      }
   }
}
/******************************* CERN Format **************************************/
static void export_cnfg_cern(char *out)
{
   int my_rank,np[4],n,iw,ie,tag;
   int x0,x1,x2,x3,y0,y1,y2,ix,iy;
   stdint_t lsize[4];
   double plaq;
   MPI_Status stat;
   FILE *fout=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (ubuf==NULL)
      alloc_ubuf(my_rank);

   ie=check_machine();
   udb=udfld();
   plaq=plaq_sum_dble(1)/((double)(6*N0)*(double)(N1*N2*N3));

   if (my_rank==0)
   {
      fout=fopen(out,"wb");
      error_root(fout==NULL,1,"export_cnfg [archive.c]",
                 "Unable to open output file");

      lsize[0]=N0;
      lsize[1]=N1;
      lsize[2]=N2;
      lsize[3]=N3;

      if (ie==BIG_ENDIAN)
      {
         bswap_int(4,lsize);
         bswap_double(1,&plaq);
      }
      
      iw=fwrite(lsize,sizeof(stdint_t),4,fout);
      iw+=fwrite(&plaq,sizeof(double),1,fout);

      error_root(iw!=5,1,"export_cnfg [archive.c]","Incorrect write count");
   }

   MPI_Barrier(MPI_COMM_WORLD);
   for (ix=0;ix<(N0*N1*N2);ix++)
   {
      x0=(ix/(N1*N2));
      x1=(ix/N2)%N1;
      x2=ix%N2;

      y0=x0%L0;
      y1=x1%L1;
      y2=x2%L2;
      iy=y2+L2*y1+L1*L2*y0;

      np[0]=x0/L0;
      np[1]=x1/L1;
      np[2]=x2/L2;
      iw=0;

      for (x3=0;x3<N3;x3+=L3)
      {
         np[3]=x3/L3;
         n=ipr_global(np);
         if (my_rank==n)
            get_links(iy);

         if (n>0)
         {
            tag=mpi_tag();

            if (my_rank==n)
               MPI_Send(ubuf,4*L3*18,MPI_DOUBLE,0,
                        tag,MPI_COMM_WORLD);

            if (my_rank==0)
               MPI_Recv(ubuf,4*L3*18,MPI_DOUBLE,n,
                        tag,MPI_COMM_WORLD,&stat);
         }

         if (my_rank==0)
         {
            if (ie==BIG_ENDIAN)
               bswap_double(4*L3*18,ubuf);
            iw+=fwrite(ubuf,sizeof(su3_dble),4*L3,fout);
         }
      }

      error_root(iw!=(4*N3),1,"export_cnfg [archive.c]",
                 "Incorrect write count");      
   }
   MPI_Barrier(MPI_COMM_WORLD);

   if (my_rank==0)
      fclose(fout);
}


static void import_cnfg_cern(char *in)
{
   int my_rank,np[4],n,ir,ie,ibc,tag;
   int k,l,x0,x1,x2,y0,y1,y2,y3,c0,c1,c2,ix,iy,ic;
   int n0,n1,n2,n3,nc0,nc1,nc2,nc3;
   stdint_t lsize[4];
   double plaq0,plaq1,eps;
   MPI_Status stat;
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (ubuf==NULL)
      alloc_ubuf(my_rank);

   ie=check_machine();
   udb=udfld();
   
   if (my_rank==0)
   {
      fin=fopen(in,"rb");
      error_root(fin==NULL,1,"import_cnfg [archive.c]",
                 "Unable to open input file");

      ir=fread(lsize,sizeof(stdint_t),4,fin);
      ir+=fread(&plaq0,sizeof(double),1,fin);
      error_root(ir!=5,1,"import_cnfg [archive.c]","Incorrect read count");

      if (ie==BIG_ENDIAN)
      {
         bswap_int(4,lsize);
         bswap_double(1,&plaq0);
      }

      np[0]=(int)(lsize[0]);
      np[1]=(int)(lsize[1]);
      np[2]=(int)(lsize[2]);
      np[3]=(int)(lsize[3]);      
      
      error_root(((N0%np[0])!=0)||((N1%np[1])!=0)||
                 ((N2%np[2])!=0)||((N3%np[3])!=0),1,
                 "import_cnfg [archive.c]","Incompatible lattice sizes");
   }

   MPI_Bcast(np,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&plaq0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   n0=np[0];
   n1=np[1];
   n2=np[2];
   n3=np[3];

   nc0=N0/n0;
   nc1=N1/n1;
   nc2=N2/n2;
   nc3=N3/n3;

   MPI_Barrier(MPI_COMM_WORLD);
   for (ix=0;ix<(n0*n1*n2);ix++)
   {
      x0=(ix/(n1*n2));
      x1=(ix/n2)%n1;
      x2=ix%n2;

      if (my_rank==0)
      {
         n=4*n3;
         ir=fread(vbuf,sizeof(su3_dble),n,fin);
         error_root(ir!=n,1,"import_cnfg [archive.c]",
                    "Incorrect read count");

         if (ie==BIG_ENDIAN)
            bswap_double(n*18,vbuf);
         
         for (k=1;k<nc3;k++)
         {
            for (l=0;l<n;l++)
               vbuf[k*n+l]=vbuf[l];
         }
      }

      for (ic=0;ic<(nc0*nc1*nc2);ic++)
      {
         c0=(ic/(nc1*nc2));
         c1=(ic/nc2)%nc1;
         c2=ic%nc2;

         y0=x0+c0*n0;
         y1=x1+c1*n1;
         y2=x2+c2*n2;
         iy=(y2%L2)+L2*(y1%L1)+L1*L2*(y0%L0);

         np[0]=y0/L0;
         np[1]=y1/L1;
         np[2]=y2/L2;

         for (y3=0;y3<N3;y3+=L3)
         {
            np[3]=y3/L3;
            n=ipr_global(np);

            if (n>0)
            {
               tag=mpi_tag();

               if (my_rank==0)
                  MPI_Send(vbuf+4*y3,4*L3*18,MPI_DOUBLE,n,tag,
                           MPI_COMM_WORLD);

               if (my_rank==n)
                  MPI_Recv(ubuf,4*L3*18,MPI_DOUBLE,0,tag,
                           MPI_COMM_WORLD,&stat);
            }
            else if (my_rank==0)
               for (l=0;l<(4*L3);l++)
                  ubuf[l]=vbuf[4*y3+l];

            if (my_rank==n)
               set_links(iy);
         }
      }
   }
   MPI_Barrier(MPI_COMM_WORLD);

   if (my_rank==0)
      fclose(fin);

   set_flags(UPDATED_UD);
   plaq1=plaq_sum_dble(1)/((double)(6*N0)*(double)(N1*N2*N3));
   eps=sqrt((double)(6*N0)*(double)(N1*N2*N3))*DBL_EPSILON;
   error(fabs(plaq1-plaq0)>eps,1,"import_cnfg [archive.c]",
         "Plaquette test failed");

   ibc=check_bcd();
   error_root((ibc==1)&&(n0!=N0),1,"import_cnfg [archive.c]",
              "Attempt to periodically extend a field with open bcd");

   if (ibc==0)
      openbcd();
}

/*************************** Lexicographic Order **********************************/
/*
#define USE_IMPORT_LEX
#define USE_EXPORT_LEX
#undef  USE_IMPORT_LEX
#undef  USE_EXPORT_LEX
*/

#define USE_COUNT

#if (defined USE_IMPORT_LEX) || (defined USE_EXPORT_LEX )
#ifdef  USE_COUNT
static int cnt0=0;
static int cnt1=0;
static int link_is_zero(su3_dble *u) {
  double *p;
  int i;
  p=(double*)u;
  for(i=0; i<18; i++) if ( *(p+i) != 0.0) return 0;
  return 1;
}

static int link_is_unity(su3_dble *u) {
  double *p;
  int i;
  p=(double*)u;
  for(i=0; i<18; i++) {
    if ( i%8== 0 && *(p+i) != 1.0) return 0;
    if ( i%8!= 0 && *(p+i) != 0.0) return 0;
  }
  return 1;
}

static void count_u(su3_dble *buf, int n) {
  int i;
  for(i=0; i<n; i++) {
    if ( link_is_zero(buf+i) ) {
      cnt0++;
      /* printf("Zero:  rank= %3d index=%4d (%d,%d,%d,%d) mu=%d\n",pdst,ix,x0,x1,x2,x3+i,m); */
    }
    if ( link_is_unity(buf+i) ) {
      cnt1++;
      /* printf("Unity: rank= %3d index=%4d (%d,%d,%d,%d) mu=%d\n",pdst,ix,x0,x1,x2,x3+i,m); */
    }
  }
}

static void print_u(void) {
  message(", %d zero links, %d unity links", cnt0, cnt1);
  cnt0=cnt1=0;
}

#else
#define count_u(x)
#define print_u(x)
#endif


static int check_end(FILE *fp, char *str) {
  long long int nbyte, nend;
  nbyte = ftell(fp);
  fseek(fp,0L,SEEK_END);
  nend=ftell(fp);
  if ( nbyte!=nend )  
    error_root(1,1,str,"File contains more data (%lld B) than expeced (%lld B)\n",nend,nbyte);
  return nbyte;
}
#endif

#ifdef USE_IMPORT_LEX
static void import_cnfg_lex(char *name) {
  int ie, nbyte, rank, pdst;
  int x[4], ix, mu, i, iu[4];
  su3_dble *buf;
  FILE *fp = NULL;
  MPI_Status stat;

  error_root(((L0%2) && (NPROC0>1)),1,"import_cnfg_lex [archive.c]",
	     "Parallelization in 0 direction not supported with odd L0");

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  ie = check_machine();

  udb=udfld();

  /* Open input file */
  if ( rank==0 ) {
    if ( (fp=fopen(name,"r")) == NULL ) 
      error_root(1,1,"import_cnfg_lex [archive.c]","Failed to open input file %s\n",name);
  }

  /* Dummy call to plaq_uidx on all processes, otherwise alloc_idx blocks in call to error */
  plaq_uidx(0,0,iu);  

  /* Allocate read buffer */
  nbyte = L3*4*sizeof(su3_dble);
  buf=amalloc(nbyte,ALIGN);
  error(buf==NULL,1,"import_cnfg_lex [archive.c]","Unable to allocate read buffer");

  MPI_Barrier(MPI_COMM_WORLD); 
  for( x[0]=0; x[0]<L0*NPROC0; x[0]++) {
    for( x[1]=0; x[1]<L1*NPROC1; x[1]++) {
      for( x[2]=0; x[2]<L2*NPROC2; x[2]++) {
	for (x[3]=0; x[3]<L3*NPROC3; x[3]+=L3) {
	  ipt_global(x,&pdst,&ix);

	  /* read on process 0 */
	  if (rank==0) {
	    if ( fread(buf,1,nbyte,fp) != nbyte )
	      error_root(1,1,"import_cnfg_lex [archive.c]","Failed to read links at (%d,%d,%d,%d)\n",x[0],x[1],x[2],x[3]);
	    if (ie==BIG_ENDIAN) bswap_double(L3*72,buf);

	  }

	  /* check on process 0 */
	  if (rank==0) {
	    count_u(buf,4*L3);
	    if ( x[0] == N0-1 ) {
	      for(i=0; i<L3; i++) {
		if ( ! link_is_zero(buf+4*i+0) ) {
#ifdef FORCE_LINKS
		  message("WARNING[imprt_cnfg_lex]: Ignored non-zero link in +0 direction from (%2d,%2d,%2d,%2d)\n",
			  x[0],x[1],x[2],x[3]+i);
		  *(buf+4*i+0) = ud0; 
#else
		  message("WARNING[imprt_cnfg_lex]: Found non-zero link in +0 direction from (%2d,%2d,%2d,%2d)\n",
			  x[0],x[1],x[2],x[3]+i);
#endif
		}
	      }
	    }
	  }

	  /* move to destination process */
	  if (pdst!=0) {
	    if (rank==0) {
	      MPI_Send((double*)buf, L3*4*18, MPI_DOUBLE, pdst, 0, MPI_COMM_WORLD);
	    }
	    if(rank==pdst) {
	      MPI_Recv((double*)buf, L3*4*18, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
	    }
	  }

	  /* store at destination */
	  if (rank==pdst) {
	    for(i=0; i<L3; i++) {

	      if ( x[0] < L0*NPROC0-1 ) {
		plaq_uidx(0,ix,iu);
		*(udb+iu[0]) = *(buf+4*i);
	      }

	      for(mu=1; mu<4; mu++) {
		plaq_uidx(mu-1,ix,iu);
		*(udb+iu[2]) = *(buf+4*i+mu);
	      }

	      ix = iup[ix][3];
	    }
	  }
	}
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD); 

  init_bnd_ud();

  if (rank==0) {
    nbyte = check_end(fp,"import_cnfg_lex [archive.c]");
    message("Gauge field read in lexicographic order from %s (%d B", name, nbyte);
    print_u();
    message(")\n");
    fclose(fp);
  }

  afree(buf);
}
#endif

#ifdef USE_EXPORT_LEX
static void export_cnfg_lex(char *name) {
  int ie, nbyte, rank, psrc;
  int x[4], ix, i, mu, iu[4];
  su3_dble *buf;
  FILE *fp = NULL;
  MPI_Status stat;

  error_root(((L0%2) && (NPROC0>1)),1,"export_cnfg_lex [archive.c]",
	     "Parallelization in 0 direction not supported with odd L0");
  
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); 

  ie = check_machine();

  udb = udfld();

  copy_bnd_ud();

  /* Open output file */
  if ( rank==0 ) {
    if ( (fp=fopen(name,"w")) == NULL ) 
      error_root(1,1,"export_cnfg_lex [archive.c]","Failed to open output file %s\n",name);
  }

  /* Dummy call to plaq_uidx on all processes, otherwise alloc_idx blocks in call to error */
  plaq_uidx(0,0,iu);  

  /* Allocate write buffer */
  nbyte = L3*4*sizeof(su3_dble);
  buf=amalloc(nbyte,ALIGN);
  error(buf==NULL,1,"export_cnfg_lex [archive.c]","Unable to allocate write buffer");

  MPI_Barrier(MPI_COMM_WORLD); 
  for( x[0]=0; x[0]<L0*NPROC0; x[0]++) {
    for( x[1]=0; x[1]<L1*NPROC1; x[1]++) {
      for( x[2]=0; x[2]<L2*NPROC2; x[2]++) {
	for (x[3]=0; x[3]<L3*NPROC3; x[3]+=L3) {
	  ipt_global(x,&psrc,&ix);

	  /* load from destination */
	  if (rank==psrc) {
	    for(i=0; i<L3; i++) {
	      if ( x[0] < L0*NPROC0-1 ) {
		plaq_uidx(0,ix,iu);
		*(buf+4*i+0) = *(udb+iu[0]);
	      }
	      else {
		*(buf+4*i+0) = ud0;
	      }

	      for(mu=1; mu<4; mu++) {
		plaq_uidx(mu-1,ix,iu);
		*(buf+4*i+mu) = *(udb+iu[2]);
	      }

	      ix = iup[ix][3];
	    }
	  }

	  /* move to process 0 */
	  if (psrc!=0) {
	    if (rank==0) {
	      MPI_Recv((double*)buf, L3*4*18, MPI_DOUBLE, psrc, 0, MPI_COMM_WORLD, &stat);
	    }
	    if(rank==psrc) {
	      MPI_Send((double*)buf, L3*4*18, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	    }
	  }

	  /* write on process 0 */
	  if (rank==0) {
	    count_u(buf,4*L3);

	    if (ie==BIG_ENDIAN) bswap_double(L3*72,buf);

	    if ( fwrite(buf,1,nbyte,fp) != nbyte )
	      error_root(1,1,"export_cnfg_lex [archive.c]","Failed to write links at (%d,%d,%d,%d)\n",
			 x[0],x[1],x[2],x[3]);
	  }

	}
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD); 

  if (rank==0) {
    nbyte = check_end(fp,"export_cnfg_lex [archive.c]");
    message("Gauge field written in lexicographic order to %s (%d B", name, nbyte);
    print_u();
    message(")\n");
    fclose(fp);
  }

  afree(buf);
}
#endif
/***************************** Wrapper Functions **********************************/
void export_cnfg(char *out) {
#ifdef USE_EXPORT_LEX
  if ( sf_flg() ) 
    export_cnfg_lex(out);
  else 
#endif
    export_cnfg_cern(out);
}
void import_cnfg(char *in) {
#ifdef USE_IMPORT_LEX
  if ( sf_flg() ) 
    import_cnfg_lex(in);
  else
#endif
    import_cnfg_cern(in);
}
