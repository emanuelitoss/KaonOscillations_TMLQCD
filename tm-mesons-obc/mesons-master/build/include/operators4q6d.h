
/*******************************************************************************
*
* File operators4q6d.h
*
* Copyright (C) 2023 Emanuele Rosi
*
*******************************************************************************/

#ifndef SU3_H
#include "su3.h"
#endif

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

/* operators4q6d.c */
extern void lonfo(void);
extern int check_null_spinor(spinor_dble *psi,char *str);
extern char* operator_to_string(int type);
extern char* diractype_to_string(int type);
extern void mul_type_sd(spinor_dble *psi,int type);
