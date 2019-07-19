#ifndef XPARALLELCONTACTDETECTION_DECL_CUH
#define XPARALLELCONTACTDETECTION_DECL_CUH

#include "xdynamics_parallel/xParallelCommon_decl.cuh"

void XDYNAMICS_API cu_calculateHashAndIndex(unsigned int* hash, unsigned int* index, double *pos, unsigned int np);
void XDYNAMICS_API cu_calculateHashAndIndexForPolygonSphere(
	unsigned int* hash, unsigned int* index,
	unsigned int sid, unsigned int nsphere, double *sphere);
void XDYNAMICS_API cu_reorderDataAndFindCellStart(unsigned int* hash, unsigned int* index, unsigned int* cstart, unsigned int* cend, unsigned int* sorted_index, unsigned int np, /*unsigned int nsphere,*/ unsigned int ncell);


#endif