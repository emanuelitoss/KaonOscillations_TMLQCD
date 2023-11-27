
/*******************************************************************************
*
* File gauge_tranforms.h
*
* Copyright (C) 2023 Emanuele Rosi
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef GAUGE_TRANSFORMS_H
#define GAUGE_TRANSFORMS_H

#ifndef SU3_h
#include "su3.h"
#endif

/*  GAUGE_TRANSFORMS.C  */
extern void generate_g_trnsfrms(void);
extern void free_g_trnsfrms(void);
extern void g_transform_ud(void);
extern void g_transform_sdble(int volume,spinor_dble *sp);


#endif
