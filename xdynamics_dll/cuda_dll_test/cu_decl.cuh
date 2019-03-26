#ifndef CU_DECL_CUH
#define CU_DECL_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>

__declspec(dllexport) float reductionD3(double3* in, unsigned int np);

#endif