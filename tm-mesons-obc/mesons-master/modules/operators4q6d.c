/*******************************************************************************
*
* File operators4q6d.c
*
* Copyright (C) 2023 Emanuele Rosi
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Utilities for the evakluation of 3 points correlators with insertion
* of 4 quarks, dimension 6 mixing operators
*
*******************************************************************************
*******************************************************************************
*  EXTERN FUNCTIONS
*  
*  extern void lonfo(void)
*     dumb check function.     
*
*  extern int check_null_spinor(spinor_dble *psi,char *str)
*     checks if the spinor psi has a non null value in the components
*     c1.c1.re and c2.c2.im. It prints the message in *str and the
*     first and last memory allocation of *psi.
*
*  extern char* operator_to_string(int type)
*     it converts the operator type to a string.
*
*  extern char* diractype_to_string(int type)
*     it converts the dirac type to a string.
*
*  extern void mul_type_sd(spinor_dble *psi,int type)
*     multiplies the dirac matrix defined by type for
*     the spinor *psi
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "operators4q6d.h"
#include "su3.h"
#include "utils.h"
#include "block.h"
#include "global.h"
#include "mesons.h"
#include "linalg.h"

static char str_type[5]; /* useful string */

/************************ Check functions ************************/

extern void lonfo(void)
{
   error(1,1,"[operators4qd6.c]","Il lonfo non vaterca, né brigatta.");
}

extern int check_null_spinor(spinor_dble *psi,char *str)
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
         variable = psi[idx].c2.c2.im;
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

/************************ Print functions ************************/

extern char* operator_to_string(int type) /*new function*/
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
      error(1,1,"operator_to_string [operators4qd6.c]","Unknown operator type");
      break;
   }
   return str_type;
}

extern char* diractype_to_string(int type)
{
   switch (type)
   {
   case GAMMA0_TYPE:
      sprintf(str_type,"G0");
      break;
   case GAMMA1_TYPE:
      sprintf(str_type,"G1");
      break;
   case GAMMA2_TYPE:
      sprintf(str_type,"G2");
      break;
   case GAMMA3_TYPE:
      sprintf(str_type,"G3");
      break;
   case GAMMA5_TYPE:
      sprintf(str_type,"G5");
      break;
   case ONE_TYPE:
      sprintf(str_type,"1");
      break;
   case GAMMA0GAMMA1_TYPE:
      sprintf(str_type,"G0G1");
      break;
   case GAMMA0GAMMA2_TYPE:
      sprintf(str_type,"G0G2");
      break;
   case GAMMA0GAMMA3_TYPE:
      sprintf(str_type,"G0G3");
      break;
   case GAMMA0GAMMA5_TYPE:
      sprintf(str_type,"G0G5");
      break;
   case GAMMA1GAMMA2_TYPE:
      sprintf(str_type,"G1G2");
      break;
   case GAMMA1GAMMA3_TYPE:
      sprintf(str_type,"G1G3");
      break;
   case GAMMA1GAMMA5_TYPE:
      sprintf(str_type,"G1G5");
      break;
   case GAMMA2GAMMA3_TYPE:
      sprintf(str_type,"G2G3");
      break;
   case GAMMA2GAMMA5_TYPE:
      sprintf(str_type,"G2G5");
      break;
   case GAMMA3GAMMA5_TYPE:
      sprintf(str_type,"G3G5");
      break;
   default:
      error(1,1,"operator_to_string [operators4qd6.c]","Unknown operator type");
      break;
   }
   return str_type;
}

/************************ Tool functions ************************/

extern void mul_type_sd(spinor_dble *psi,int type) /* new function */
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
      error(1,1,"mul_type_sd [operators4qd6.c]","Invalid Dirac matrix type");
      break;
   }
}
