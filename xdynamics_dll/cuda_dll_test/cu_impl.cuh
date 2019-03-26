#include "cu_decl.cuh"
#include <helper_math.h>

__global__ void kernel_function(float3* d, float3* o)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id > 100)
		return;
	o[id] = d[id] + d[id];
}