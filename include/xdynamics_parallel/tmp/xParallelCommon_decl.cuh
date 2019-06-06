#ifndef XPARALLELCOMMON_DECL_CUH
#define XPARALLELCOMMON_DECL_CUH

#include "xdynamics_decl.h"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>
//#include <helper_math.h>

#pragma warning(disable: 4819)



// sph
struct device_sinusoidal_expression
{
	unsigned int sid;
	unsigned int count;
	double period;
	double freq;
	double c1;
	double c2;
	double stime;
};

struct device_simple_sin_expression
{
	unsigned int sid;
	unsigned int count;
	double freq;
	double amp;
	double stime;
};

struct device_wave_damping
{
	double enable;
	double alpha;
	double start_point;
	double length;
};

struct device_periodic_condition
{
	bool b;
	double3 dir;
	double3 limits;
	double3 velocity;
};

struct device_pointMass_info
{
	double mass;
	double3 pos;
	double3 vel;
	double3 omega;
	double3 force;
	double3 moment;
};

struct device_circle_info
{
	unsigned int sid;
	unsigned int count;
	double r;
};

// dem
struct pair_data
{
	bool enable;
	unsigned int type;
//	unsigned int i;
	unsigned int j;
	double ds;
	double dots;
	//double3 kr;// , kry, krz;
	//double3 ci;// , ciy, ciz;
	//double3 cj;// , cjy, cjx;
	
};

struct rolling_friction_moment
{
	double Tr;
	double3 Tmax;
};

struct device_triangle_info
{
	int id;
	double3 P;
	double3 Q;
	double3 R;
	double3 V;
	double3 W;
	double3 N;
};

struct device_plane_info
{
	double l1, l2;
	double3 u1;
	double3 u2;
	double3 uw;
	double3 xw;
	double3 pa;
	double3 pb;
	double3 w2;
	double3 w3;
	double3 w4;
};

struct device_mesh_mass_info
{
	double px, py, pz;// origin;
	double vx, vy, vz;// double3 vel;
	double ox, oy, oz;// double3 omega;
	double3 force;// double3 force;
	double3 moment;// mx, my, mz;// double3 moment;
	double e0, e1, e2, e3;// double4 ep;
};

struct device_cylinder_info
{
	double len, rbase, rtop;
	double3 pbase;
	double3 ptop;
	double3 origin;
	double3 vel;
	double3 omega;
	double4 ep;
};

__host__ __device__ struct device_contact_property
{
	double Ei, Ej;
	double pri, prj;
	double Gi, Gj;
	double rest;
	double fric;
	double rfric;
	double coh;
	double sratio;
	double rfactor;
};


struct device_force_constant
{
	double kn;
	double vn;
	double ks;
	double vs;
	double mu;
	double ms;
};

struct device_force_constant_d
{
	double kn;
	double vn;
	double ks;
	double vs;
	double mu;
};

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
	uint3 gridCellCount;
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


//__constant__ device_sph_parameters scte;

void XDYNAMICS_API cudaMemoryAlloc(void** data, unsigned int size);
unsigned int XDYNAMICS_API iDivUp(unsigned int a, unsigned int b);
void XDYNAMICS_API computeGridSize(unsigned int n, unsigned int blockSize, unsigned int& numBlocks, unsigned int& numThreads);

//void XDYNAMICS_API setSPHSymbolicParameter(device_sph_parameters* h_paras);
void XDYNAMICS_API cuMaxDouble3(double* indata, double* odata, unsigned int np);
double3 XDYNAMICS_API reductionD3(double3* in, unsigned int np);

#endif