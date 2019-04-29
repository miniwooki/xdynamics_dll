#ifndef XPARALLELCOMMON_DECL_CUH
#define XPARALLELCOMMON_DECL_CUH

#include "xdynamics_decl.h"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>

inline __device__ int sign(float L)
{
	return L < 0 ? -1 : 1;
}

inline __device__ int sign(double L)
{
	return L < 0 ? -1 : 1;
}

inline __device__ double dot(double3& v1, double3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __device__ double3 operator-(double3& v1, double3& v2)
{
	return make_double3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
inline __device__ double3 operator-(double3& v1)
{
	return make_double3(-v1.x, -v1.y, -v1.z);
}

inline __device__ double dot(double3& v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

inline __device__ double3 operator*(double v1, double3 v2)
{
	return make_double3(v1 * v2.x, v1 * v2.y, v1 * v2.z);
}

inline __device__ double3 operator+(double3& v1, double3& v2)
{
	return make_double3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline __host__ __device__ void operator+=(double3& a, double3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

inline __device__ double3 operator/(double3& v1, double v2)
{
	return make_double3(v1.x / v2, v1.y / v2, v1.z / v2);
}

inline __device__ double length(double3& v1)
{
	return sqrt(dot(v1, v1));
}

inline __device__ double3 cross(double3 a, double3 b)
{
	return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __device__ double3 normalize(double3 u)
{
	return u / length(u);
}

struct device_dem_parameters
{
	bool rollingCondition;
	unsigned int np;
	unsigned int nsphere;
	unsigned int ncell;
	uint3 grid_size;
	double dt;
	double half2dt;
	double cell_size;
	double cohesion;
	double3 gravity;
	double3 world_origin;
};

struct device_sph_parameters
{
	bool isShifting;
	xSPHCorrectionType corr;
	int dim;
	int kernel;
	unsigned long long int np;
	unsigned long long int cells;
	unsigned long long int startInnerDummy;
	double3 gridMin;
	double3 gridMax;
	double3 gridSize;
	double3 gravity;
	double rho;
	//double rhop0;
	double kernel_const;
	double kernel_grad_const;
	double kernel_support;
	double kernel_support_sq;
	double smoothing_length;
	double particle_spacing;
	double3 kernel_support_radius;

	//double deltaPKernelInv;
	double gridCellSize;
	int3 gridCellCount;
	double cellsizeInv;
	//double mass;
	//double bmass;
	double viscosity;
	double dist_epsilon;
	double peclet;

	double h;
	double h_sq;
	double h_inv;
	double h_inv_sq;
	double h_inv_2;
	double h_inv_3;
	double h_inv_4;
	double h_inv_5;
	double dt;
	double time;

	double freeSurfaceFactor;
	double shifting_factor;
};

__constant__ device_dem_parameters dcte;
__constant__ device_sph_parameters scte;

void XDYNAMICS_API cu_calculateHashAndIndex(unsigned int* hash, unsigned int* index, double *pos, unsigned int np);
void XDYNAMICS_API cu_calculateHashAndIndexForPolygonSphere(
	unsigned int* hash, unsigned int* index,
	unsigned int sid, unsigned int nsphere, double *sphere);
void XDYNAMICS_API cu_reorderDataAndFindCellStart(unsigned int* hash, unsigned int* index, unsigned int* cstart, unsigned int* cend, unsigned int* sorted_index, unsigned int np, /*unsigned int nsphere,*/ unsigned int ncell);

void XDYNAMICS_API cudaMemoryAlloc(void** data, unsigned int size);
unsigned int XDYNAMICS_API iDivUp(unsigned int a, unsigned int b);
void XDYNAMICS_API computeGridSize(unsigned int n, unsigned int blockSize, unsigned int& numBlocks, unsigned int& numThreads);
void XDYNAMICS_API setDEMSymbolicParameter(device_dem_parameters* h_paras);
void XDYNAMICS_API setSPHSymbolicParameter(device_sph_parameters* h_paras);
void XDYNAMICS_API cuMaxDouble3(double* indata, double* odata, unsigned int np);
double3 XDYNAMICS_API reductionD3(double3* in, unsigned int np);

#endif