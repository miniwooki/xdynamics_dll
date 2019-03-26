#include "cu_decl.cuh"
#include "cu_impl.cuh"

float reductionD3(float3* in, float3* out, unsigned int np)
{
	kernel_function<< < 1, 100 >> >(in, out);
	return 0.f;
}