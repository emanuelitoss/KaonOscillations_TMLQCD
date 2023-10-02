
#ifndef CFCTS_H
#define CFCTS_H

#ifndef SU3_H
#include "su3.h"
#endif

typedef enum
{
   S,P,
   V0,V1,V2,V3,
   A0,A1,A2,A3,
   T01,T02,T03,
   Tt01,Tt02,Tt03,
   T12,T13,T23,
   Tt12,Tt13,Tt23
} dirac_t;


/* CFCTS_C */
extern void cfcts2q(dirac_t A,dirac_t B,spinor_dble **sk,spinor_dble **sl,complex_dble ab[]);
extern void ctrcts2q(dirac_t A,dirac_t B,spinor_dble *s,spinor_dble *r,complex_dble *ab);

/* CFCTS4Q_C */
extern void cfcts4q1(dirac_t A,dirac_t B,dirac_t C,dirac_t D,spinor_dble **sk,spinor_dble **sl,
                     spinor_dble **skp,spinor_dble **slp,complex_dble ab[]);
extern void cfcts4q2(dirac_t A,dirac_t B,dirac_t C,dirac_t D,spinor_dble **sk,spinor_dble **sl,
                     spinor_dble **skp,spinor_dble **slp,complex_dble ab[]);

/* PTSPLIT_C */
extern void ptsplit(int dir,spinor_dble *s,spinor_dble *r);

/* SFCFCTS_C */
extern void pull_slice(int x0,spinor_dble *s,spinor_dble *r);
extern void push_slice(int x0,spinor_dble *s,spinor_dble *r);
extern void sfprop(int x0,int isp,spinor_dble **s,int *status);

/* XSFCFCTS_C */
extern void pull_slice_xsf(int x0,int tau3,spinor_dble *s,spinor_dble *r);
extern void push_slice_xsf(int x0,int tau3,spinor_dble *s,spinor_dble *r);
extern void xsfprop(int x0,int tau3,int isp,spinor_dble **s,int *status);
extern void xsfpropeo(int x0,int tau3,int isp,spinor_dble **s,int *status);

#endif
