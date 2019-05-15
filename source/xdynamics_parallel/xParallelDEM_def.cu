//#include "xdynamics_parallel/xParallelDEM_decl.cuh"
#include "xdynamics_parallel/xParallelDEM_decl.cuh"
#include <helper_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
//#include <helper_cuda.h>

__constant__ device_dem_parameters dcte;

void setDEMSymbolicParameter(device_dem_parameters *h_paras)
{
	checkCudaErrors(cudaMemcpyToSymbol(dcte, h_paras, sizeof(device_dem_parameters)));
}

__device__ int calcGridHashD(int3 gridPos)
{
	gridPos.x = gridPos.x & (dcte.grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (dcte.grid_size.y - 1);
	gridPos.z = gridPos.z & (dcte.grid_size.z - 1);
	return ((gridPos.z * dcte.grid_size.y) * dcte.grid_size.x) + (gridPos.y * dcte.grid_size.x) + gridPos.x;
}

// calculate position in uniform grid
__device__ int3 calcGridPosD(double3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - dcte.world_origin.x) / dcte.cell_size);
	gridPos.y = floor((p.y - dcte.world_origin.y) / dcte.cell_size);
	gridPos.z = floor((p.z - dcte.world_origin.z) / dcte.cell_size);
	return gridPos;
}

__global__ void calculateHashAndIndex_kernel(
	unsigned int* hash, unsigned int* index, double4* pos, unsigned int np)
{
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= (np)) return;

	volatile double4 p = pos[id];

	int3 gridPos = calcGridPosD(make_double3(p.x, p.y, p.z));
	unsigned _hash = calcGridHashD(gridPos);
	/*if(_hash >= dcte.ncell)
	printf("Over limit - hash number : %d", _hash);*/
	hash[id] = _hash;
	index[id] = id;
}

__global__ void calculateHashAndIndexForPolygonSphere_kernel(
	unsigned int* hash, unsigned int* index,
	unsigned int sid, unsigned int nsphere, double4* sphere)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= nsphere) return;
	volatile double4 p = sphere[id];
	int3 gridPos = calcGridPosD(make_double3(p.x, p.y, p.z));
	unsigned int _hash = calcGridHashD(gridPos);
	hash[sid + id] = _hash;
	index[sid + id] = sid + id;
}

__global__ void reorderDataAndFindCellStart_kernel(
	unsigned int* hash,
	unsigned int* index,
	unsigned int* cstart,
	unsigned int* cend,
	unsigned int* sorted_index,
	unsigned int np)
{
	extern __shared__ uint sharedHash[];	//blockSize + 1 elements
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	unsigned _hash;

	//unsigned int tnp = ;// dcte.np + dcte.nsphere;

	if (id < np)
	{
		_hash = hash[id];

		sharedHash[threadIdx.x + 1] = _hash;

		if (id > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = hash[id - 1];
		}
	}
	__syncthreads();

	if (id < np)
	{
		if (id == 0 || _hash != sharedHash[threadIdx.x])
		{
			cstart[_hash] = id;

			if (id > 0)
				cend[sharedHash[threadIdx.x]] = id;
		}

		if (id == np - 1)
		{
			cend[_hash] = id + 1;
		}

		unsigned int sortedIndex = index[id];
		sorted_index[id] = sortedIndex;
	}
}

void cu_dem_calculateHashAndIndex(
	unsigned int* hash,
	unsigned int* index,
	double *pos,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	calculateHashAndIndex_kernel << < numBlocks, numThreads >> >(hash, index, (double4 *)pos, np);
}

void cu_dem_calculateHashAndIndexForPolygonSphere(
	unsigned int* hash,
	unsigned int* index,
	unsigned int sid,
	unsigned int nsphere,
	double *sphere)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(nsphere, 512, numBlocks, numThreads);
	calculateHashAndIndexForPolygonSphere_kernel << <numBlocks, numThreads >> >(hash, index, sid, nsphere, (double4 *)sphere);
}


void cu_dem_reorderDataAndFindCellStart(
	unsigned int* hash,
	unsigned int* index,
	unsigned int* cstart,
	unsigned int* cend,
	unsigned int* sorted_index,
	unsigned int np,
	//unsigned int nsphere,
	unsigned int ncell)
{
	//std::cout << "step 1" << std::endl;
	//unsigned int tnp = np;// +nsphere;
	thrust::sort_by_key(thrust::device_ptr<unsigned>(hash),
		thrust::device_ptr<unsigned>(hash + np),
		thrust::device_ptr<unsigned>(index));
	//std::cout << "step 2" << std::endl;
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	checkCudaErrors(cudaMemset(cstart, 0xffffffff, ncell*sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(cend, 0, ncell*sizeof(unsigned int)));
	unsigned smemSize = sizeof(unsigned int)*(numThreads + 1);
	//std::cout << "step 3" << std::endl;
	reorderDataAndFindCellStart_kernel << < numBlocks, numThreads, smemSize >> >(
		hash,
		index,
		cstart,
		cend,
		sorted_index,
		np);
}

__global__ void vv_update_position_kernel(
	double4* pos, double3* vel, double3* acc, 
	double4* ep, double4* ev, double4* ea, unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;

	double3 _p = dcte.dt * vel[id] + dcte.half2dt * acc[id];
	double4 _e = dcte.dt * ev[id] + dcte.half2dt * ea[id];
	_e = normalize(_e);
	pos[id].x += _p.x;
	pos[id].y += _p.y;
	pos[id].z += _p.z;
	ep[id] = _e;
}

__device__ double4 calculateInertiaForce(
	double J, double v0, double v1, double v2, double v3,
	double e0, double e1, double e2, double e3)
{
	double f0 = 8 * J*v1*(e0*v1 - e1*v0 - e2*v3 + e3*v2) + 8 * J*v2*(e0*v2 - e2*v0 + e1*v3 - e3*v1) + 8 * J*v3*(e0*v3 - e1*v2 + e2*v1 - e3*v0);
	double f1 = 8 * J*v3*(e0*v2 - e2*v0 + e1*v3 - e3*v1) - 8 * J*v2*(e0*v3 - e1*v2 + e2*v1 - e3*v0) - 8 * J*v0*(e0*v1 - e1*v0 - e2*v3 + e3*v2);
	double f2 = 8 * J*v1*(e0*v3 - e1*v2 + e2*v1 - e3*v0) - 8 * J*v0*(e0*v2 - e2*v0 + e1*v3 - e3*v1) - 8 * J*v3*(e0*v1 - e1*v0 - e2*v3 + e3*v2);
	double f3 = 8 * J*v2*(e0*v1 - e1*v0 - e2*v3 + e3*v2) - 8 * J*v1*(e0*v2 - e2*v0 + e1*v3 - e3*v1) - 8 * J*v0*(e0*v3 - e1*v2 + e2*v1 - e3*v0);
	double J4 = 4.0 * J;
	double e0s = J4 * e0 * e0;
	double e1s = J4 * e1 * e1;
	double e2s = J4 * e2 * e2;
	double e3s = J4 * e3 * e3;
	double a00 = e1s + e2s + e3s;	double a01 = -J4 * e0 * e1;		double a02 = -J4 * e0 * e2;		double a03 = -J4 * e0 * e3;
	double a10 = -J4 * e0 * e1;		double a11 = e0s + e2s + e3s;	double a12 = -J4 * e1 * e2;		double a13 = -J4 * e1 * e3;
	double a20 = -J4 * e0 * e2;		double a21 = -J4 * e1 * e2;		double a22 = e0s + e1s + e3s;	double a23 = -J4 * e2 * e3;
	double a30 = -J4 * e0 * e3;		double a31 = -J4 * e1 * e3;		double a32 = -J4 * e2 * e3;		double a33 = e0s + e1s + e2s;
	double det = 1.0 / (a00*a11*a22*a33 - a00*a11*a23*a32 - a00*a12*a21*a33 + a00*a12*a23*a31 + a00*a13*a21*a32 - a00*a13*a22*a31 - a01*a10*a22*a33 + a01*a10*a23*a32 + a01*a12*a20*a33 - a01*a12*a23*a30 - a01*a13*a20*a32 + a01*a13*a22*a30 + a02*a10*a21*a33 - a02*a10*a23*a31 - a02*a11*a20*a33 + a02*a11*a23*a30 + a02*a13*a20*a31 - a02*a13*a21*a30 - a03*a10*a21*a32 + a03*a10*a22*a31 + a03*a11*a20*a32 - a03*a11*a22*a30 - a03*a12*a20*a31 + a03*a12*a21*a30);
	double ia00 = (a11*a22*a33 - a11*a23*a32 - a11*a23*a32 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) * det;
	double ia01 = -(a01*a22*a33 - a01*a23*a32 - a02*a21*a33 + a02*a23*a31 + a03*a21*a32 - a03*a22*a31) * det;
	double ia02 = (a01*a12*a33 - a01*a13*a32 - a02*a11*a33 + a02*a13*a31 + a03*a11*a32 - a03*a12*a31) * det;
	double ia03 = -(a01*a12*a23 - a01*a13*a22 - a02*a11*a23 + a02*a13*a21 + a03*a11*a22 - a03*a12*a21) * det;
	double ia10 = -(a10*a22*a33 - a10*a23*a32 - a12*a20*a33 + a12*a23*a30 + a13*a20*a32 - a13*a22*a30) * det;
	double ia11 = (a00*a22*a33 - a00*a23*a32 - a02*a20*a33 + a02*a23*a30 + a03*a20*a32 - a03*a22*a30) * det;
	double ia12 = -(a00*a12*a33 - a00*a13*a32 - a02*a10*a33 + a02*a13*a30 + a03*a10*a32 - a03*a12*a30) * det;
	double ia13 = (a00*a12*a23 - a00*a13*a22 - a02*a10*a23 + a02*a13*a20 + a03*a10*a22 - a03*a12*a20) * det;
	double ia20 = (a10*a21*a33 - a10*a23*a31 - a11*a20*a33 + a11*a23*a30 + a13*a20*a31 - a13*a21*a30) * det;
	double ia21 = -(a00*a21*a33 - a00*a23*a31 - a01*a20*a33 + a01*a23*a30 + a03*a20*a31 - a03*a21*a30) * det;
	double ia22 = (a00*a11*a33 - a00*a13*a31 - a01*a10*a33 + a01*a13*a30 + a03*a10*a31 - a03*a11*a30) * det;
	double ia23 = -(a00*a11*a23 - a00*a13*a21 - a01*a10*a23 + a01*a13*a20 + a03*a10*a21 - a03*a11*a20) * det;
	double ia30 = -(a10*a21*a32 - a10*a22*a31 - a11*a20*a32 + a11*a22*a30 + a12*a20*a31 - a12*a21*a30) * det;
	double ia31 = (a00*a21*a32 - a00*a22*a31 - a01*a20*a32 + a01*a22*a30 + a02*a20*a31 - a02*a21*a30) * det;
	double ia32 = -(a00*a11*a32 - a00*a12*a31 - a01*a10*a32 + a01*a12*a30 + a02*a10*a31 - a02*a11*a30) * det;
	double ia33 = (a00*a11*a22 - a00*a12*a21 - a01*a10*a22 + a01*a12*a20 + a02*a10*a21 - a02*a11*a20) * det;
	return make_double4(
		ia00 * f0 + ia01 * f1 + ia02 * f2 + ia03 * f3,
		ia10 * f0 + ia11 * f1 + ia12 * f2 + ia13 * f3,
		ia20 * f0 + ia21 * f1 + ia22 * f2 + ia23 * f3,
		ia30 * f0 + ia31 * f1 + ia32 * f2 + ia33 * f3);
}

__global__ void vv_update_velocity_kernel(
	double3* vel,
	double3* acc,
	double4* ep,
	double4* omega,
	double4* alpha,
	double3* force,
	double3* moment,
	double* mass,
	double* iner,
	unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;
	double m = mass[id];
	double3 v = vel[id];
	//double3 L = acc[id];
	double4 e = ep[id];
	double4 av = omega[id];
	//double3 aa = alpha[id];
	double3 a = (1.0 / m) * force[id];
	double4 in = calculateInertiaForce(iner[id], e.x, e.y, e.z, e.w, av.x, av.y, av.z, av.w);
	//double3 in = (1.0 / iner[id]) * moment[id];

	v += 0.5 * dcte.dt * (acc[id] + a);
	av += 0.5 * dcte.dt * (alpha[id] + in);// aa;

	force[id] = m * dcte.gravity;
	moment[id] = make_double3(0.0, 0.0, 0.0);
	vel[id] = v;
	omega[id] = av;
	acc[id] = a;
	alpha[id] = in;
}

void vv_update_position(double *pos, double *vel, double *acc, double *ep, double *ev, double *ea, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	vv_update_position_kernel << < numBlocks, numThreads >> >(
		(double4 *)pos,
		(double3 *)vel,
		(double3 *)acc,
		(double4 *)ep,
		(double4 *)ev,
		(double4 *)ea,
		np);
}

void vv_update_velocity(
	double *vel, double *acc, double *ep, 
	double *ev, double *ea, double *force, double *moment, 
	double* mass, double* iner, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	vv_update_velocity_kernel << < numBlocks, numThreads >> >(
		(double3 *)vel,
		(double3 *)acc,
		(double4 *)ep,
		(double4 *)ev,
		(double4 *)ea,
		(double3 *)force,
		(double3 *)moment,
		mass,
		iner,
		np);
}

//void cu_calculateHashAndIndex(
//	unsigned int* hash,
//	unsigned int* index,
//	double *pos,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	calculateHashAndIndex_kernel << < numBlocks, numThreads >> >(hash, index, (double4 *)pos, np);
//}
//
//void cu_calculateHashAndIndexForPolygonSphere(
//	unsigned int* hash,
//	unsigned int* index,
//	unsigned int sid,
//	unsigned int nsphere,
//	double *sphere)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(nsphere, 512, numBlocks, numThreads);
//	calculateHashAndIndexForPolygonSphere_kernel << <numBlocks, numThreads >> >(hash, index, sid, nsphere, (double4 *)sphere);
//}
//
//
//void cu_dem_reorderDataAndFindCellStart(
//	unsigned int* hash,
//	unsigned int* index,
//	unsigned int* cstart,
//	unsigned int* cend,
//	unsigned int* sorted_index,
//	unsigned int np,
//	//unsigned int nsphere,
//	unsigned int ncell)
//{
//	//std::cout << "step 1" << std::endl;
//	//unsigned int tnp = np;// +nsphere;
//	thrust::sort_by_key(thrust::device_ptr<unsigned>(hash),
//		thrust::device_ptr<unsigned>(hash + np),
//		thrust::device_ptr<unsigned>(index));
//	//std::cout << "step 2" << std::endl;
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	checkCudaErrors(cudaMemset(cstart, 0xffffffff, ncell*sizeof(unsigned int)));
//	checkCudaErrors(cudaMemset(cend, 0, ncell*sizeof(unsigned int)));
//	unsigned smemSize = sizeof(unsigned int)*(numThreads + 1);
//	//std::cout << "step 3" << std::endl;
//	reorderDataAndFindCellStart_kernel << < numBlocks, numThreads, smemSize >> >(
//		hash,
//		index,
//		cstart,
//		cend,
//		sorted_index,
//		np);
//}

//void cu_calculate_p2p(
//	const int tcm, double* pos, double* vel,
//	double* omega, double* force, double* moment,
//	double* mass, unsigned int* sorted_index, unsigned int* cstart,
//	unsigned int* cend, device_contact_property* cp, unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 256, numBlocks, numThreads);
//	switch (tcm)
//	{
//	case 0:
//		calculate_p2p_kernel<0> << < numBlocks, numThreads >> >(
//			(double4 *)pos, (double3 *)vel,
//			(double3 *)omega, (double3 *)force,
//			(double3 *)moment, mass,
//			sorted_index, cstart,
//			cend, cp, np);
//		break;
//	case 1:
//		calculate_p2p_kernel<1> << < numBlocks, numThreads >> >(
//			(double4 *)pos, (double3 *)vel,
//			(double3 *)omega, (double3 *)force,
//			(double3 *)moment, mass,
//			sorted_index, cstart,
//			cend, cp, np);
//		break;
//	}
//}

//void cu_plane_contact_force(
//	const int tcm, device_plane_info* plan,
//	double* pos, double* vel, double* omega,
//	double* force, double* moment, double* mass,
//	unsigned int np, device_contact_property *cp)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 256, numBlocks, numThreads);
//	switch (tcm)
//	{
//	case 0: plane_contact_force_kernel<0> << < numBlocks, numThreads >> >(
//		plan, (double4 *)pos, (double3 *)vel, (double3 *)omega,
//		(double3 *)force, (double3 *)moment, cp, mass, np);
//		break;
//	case 1: plane_contact_force_kernel<1> << < numBlocks, numThreads >> >(
//		plan, (double4 *)pos, (double3 *)vel, (double3 *)omega,
//		(double3 *)force, (double3 *)moment, cp, mass, np);
//		break;
//	}
//}

//void cu_cube_contact_force(
//	const int tcm, device_plane_info* plan,
//	double* pos, double* vel, double* omega,
//	double* force, double* moment, double* mass,
//	unsigned int np, device_contact_property *cp)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 256, numBlocks, numThreads);
//	for (unsigned int i = 0; i < 6; i++)
//	{
//		switch (tcm)
//		{
//		case 0: plane_contact_force_kernel<0> << < numBlocks, numThreads >> >(
//			plan + i, (double4 *)pos, (double3 *)vel, (double3 *)omega,
//			(double3 *)force, (double3 *)moment, cp, mass, np);
//			break;
//		case 1: plane_contact_force_kernel<1> << < numBlocks, numThreads >> >(
//			plan + i, (double4 *)pos, (double3 *)vel, (double3 *)omega,
//			(double3 *)force, (double3 *)moment, cp, mass, np);
//			break;
//		}
//	}
//}
//
//void cu_cylinder_hertzian_contact_force(
//	const int tcm, device_cylinder_info* cyl,
//	double* pos, double* vel, double* omega,
//	double* force, double* moment,
//	double* mass, unsigned int np, device_contact_property *cp,
//	double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	switch (tcm)
//	{
//	case 0: cylinder_hertzian_contact_force_kernel<0> << < numBlocks, numThreads >> >(
//		cyl, (double4 *)pos, (double3 *)vel, (double3 *)omega,
//		(double3 *)force, (double3 *)moment, cp, mass, mpos, mf, mm, np);
//		break;
//	case 1: cylinder_hertzian_contact_force_kernel<1> << < numBlocks, numThreads >> >(
//		cyl, (double4 *)pos, (double3 *)vel, (double3 *)omega,
//		(double3 *)force, (double3 *)moment, cp, mass, mpos, mf, mm, np);
//		break;
//	}
//
//}
//
//void cu_particle_meshObject_collision(
//	const int tcm, device_triangle_info* dpi, double* dsph, device_mesh_mass_info* dpmi,
//	double* pos, double* vel, double* omega,
//	double* force, double* moment, double* mass,
//	unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, device_contact_property *cp,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	switch (tcm)
//	{
//	case 1:
//		particle_polygonObject_collision_kernel<1> << < numBlocks, numThreads >> >(
//			dpi, (double4 *)dsph, dpmi,
//			(double4 *)pos, (double3 *)vel, (double3 *)omega,
//			(double3 *)force, (double3 *)moment, mass,
//			sorted_index, cstart, cend, cp, np);
//		break;
//	}
//}
//
////double3 reductionD3(double3* in, unsigned int np)
////{
////	unsigned int numBlocks, numThreads;
////	double3 rt = make_double3(0.0, 0.0, 0.0);
////	computeGridSize(np, 512, numBlocks, numThreads);
////	double3* d_out;
////	double3* h_out = new double3[numBlocks];
////	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(double3) * numBlocks));
////	checkCudaErrors(cudaMemset(d_out, 0, sizeof(double3) * numBlocks));
////	//unsigned smemSize = sizeof(double3)*(512);
////	reduce6<double3, 512> << < numBlocks, numThreads/*, smemSize*/ >> >(in, d_out, np);
////	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(double3) * numBlocks, cudaMemcpyDeviceToHost));
////	for (unsigned int i = 0; i < numBlocks; i++){
////		rt.x += h_out[i].x;
////		rt.y += h_out[i].y;
////		rt.z += h_out[i].z;
////	}
////	delete[] h_out;
////	checkCudaErrors(cudaFree(d_out));
////	return rt;
////}
//
//void cu_update_meshObjectData(
//	device_mesh_mass_info *dpmi, double* vList,
//	double* sphere, device_triangle_info* dpi, unsigned int ntriangle)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(ntriangle, 512, numBlocks, numThreads);
//	updatePolygonObjectData_kernel << <numBlocks, numThreads >> >(dpmi, vList, (double4 *)sphere, dpi, ntriangle);
//}
//
////void cu_check_no_collision_pair(
////	double* pos, unsigned int* pinfo,
////	unsigned int* pother, unsigned int np)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	check_no_collision_pair_kernel << <numBlocks, numThreads >> > (
////		(double4 *)pos,
////		(uint2 *)pinfo,
////		pother, np);
////}
////
////void cu_check_new_collision_pair(
////	double* pos, unsigned int* pinfo,
////	unsigned int* pdata, unsigned int* sorted_id,
////	unsigned int* cstart, unsigned int *cend, unsigned int np)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	check_new_collision_pair_kernel << <numBlocks, numThreads >> >(
////		(double4 *)pos,
////		(uint2 *)pinfo,
////		pdata,
////		sorted_id,
////		cstart,
////		cend,
////		np);
////}
////
////
////void cu_calculate_particle_collision_with_pair(
////	double* pos, double* vel, double* omega,
////	double* mass, double* ds, double* force,
////	double* moment, unsigned int* pinfo,
////	unsigned int* pother, device_contact_property* cp, unsigned int np)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	calculate_particle_collision_with_pair_kernel << < numBlocks, numThreads >> >(
////		(double4 *)pos,
////		(double3 *)vel,
////		(double3 *)omega,
////		mass, 
////		(double2 *)ds,
////		(double3 *)force,
////		(double3 *)moment,
////		(uint2 *)pinfo,
////		pother, cp, np);
////}

__device__ device_force_constant getConstant(
	int tcm, double ir, double jr, double im, double jm,
	double iE, double jE, double ip, double jp,
	double iG, double jG, double rest,
	double fric, double rfric, double sratio)
{
	device_force_constant dfc = { 0, 0, 0, 0, 0, 0 };
	double Meq = jm ? (im * jm) / (im + jm) : im;
	double Req = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
	switch (tcm)
	{
	case 0:{
		double Geq = (iG * jG) / (iG*(2 - jp) + jG*(2 - ip));
		double ln_e = log(rest);
		double xi = ln_e / sqrt(ln_e * ln_e + M_PI * M_PI);
		dfc.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
		dfc.vn = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(dfc.kn * Meq);
		dfc.ks = 8.0 * Geq * sqrt(Req);
		dfc.vs = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(dfc.ks * Meq);
		dfc.mu = fric;
		dfc.ms = rfric;
		break;
	}
	case 1:{
		double beta = (M_PI / log(rest));
		dfc.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
		dfc.vn = sqrt((4.0 * Meq * dfc.kn) / (1.0 + beta * beta));
		dfc.ks = dfc.kn * sratio;
		dfc.vs = dfc.vn * sratio;
		dfc.mu = fric;
		dfc.ms = rfric;
		break;
	}
	}

	// 	dfu1.kn = /*(16.f / 15.f)*sqrt(er) * eym * pow((T)((15.f * em * 1.0f) / (16.f * sqrt(er) * eym)), (T)0.2f);*/ (4.0f / 3.0f)*sqrt(er)*eym;
	// 	dfu1.vn = sqrt((4.0f*em * dfu1.kn) / (1 + beta * beta));
	// 	dfu1.ks = dfu1.kn * ratio;
	// 	dfu1.vs = dfu1.vn * ratio;
	// 	dfu1.mu = fric;
	return dfc;
}

__device__ double cohesionForce(
	double ri,
	double rj,
	double Ei,
	double Ej,
	double pri,
	double prj,
	double coh,
	double Fn)
{
	double cf = 0.f;
	if (coh){
		double req = (ri * rj / (ri + rj));
		double Eeq = ((1.0 - pri * pri) / Ei) + ((1.0 - prj * prj) / Ej);
		double rcp = (3.0 * req * (-Fn)) / (4.0 * (1.0 / Eeq));
		double rc = pow(rcp, 1.0 / 3.0);
		double Ac = M_PI * rc * rc;
		cf = coh * Ac;
	}
	return cf;
}

__device__ double3 toGlobal(double3& v, double4& ep)
{
	double3 r0 = make_double3(2.0 * (ep.x*ep.x + ep.y*ep.y - 0.5), 2.0 * (ep.y*ep.z - ep.x*ep.w), 2.0 * (ep.y*ep.w + ep.x*ep.z));
	double3 r1 = make_double3(2.0 * (ep.y*ep.z + ep.x*ep.w), 2.0 * (ep.x*ep.x + ep.z*ep.z - 0.5), 2.0 * (ep.z*ep.w - ep.x*ep.y));
	double3 r2 = make_double3(2.0 * (ep.y*ep.w - ep.x*ep.z), 2.0 * (ep.z*ep.w + ep.x*ep.y), 2.0 * (ep.x*ep.x + ep.w*ep.w - 0.5));
	return make_double3
		(
		r0.x * v.x + r0.y * v.y + r0.z * v.z,
		r1.x * v.x + r1.y * v.y + r1.z * v.z,
		r2.x * v.x + r2.y * v.y + r2.z * v.z
		);
}

__device__ double3 toLocal(double3& v, double4& ep)
{
	double3 r0 = make_double3(2.0 * (ep.x*ep.x + ep.y*ep.y - 0.5), 2.0 * (ep.y*ep.z - ep.x*ep.w), 2.0 * (ep.y*ep.w + ep.x*ep.z));
	double3 r1 = make_double3(2.0 * (ep.y*ep.z + ep.x*ep.w), 2.0 * (ep.x*ep.x + ep.z*ep.z - 0.5), 2.0 * (ep.z*ep.w - ep.x*ep.y));
	double3 r2 = make_double3(2.0 * (ep.y*ep.w - ep.x*ep.z), 2.0 * (ep.z*ep.w + ep.x*ep.y), 2.0 * (ep.x*ep.x + ep.w*ep.w - 0.5));
	return make_double3
		(
		r0.x * v.x + r1.x * v.y + r2.x * v.z,
		r0.y * v.x + r1.y * v.y + r2.y * v.z,
		r0.z * v.x + r1.z * v.y + r2.z * v.z
		);
}

__device__ void HMCModel(
	device_force_constant c, double ir, double jr, double Ei, double Ej, double pri, double prj, double coh,
	double rcon, double cdist, double3 iomega,
	double3 dv, double3 unit, double3& Ft, double3& Fn, double3& M)
{
	// 	if (coh && cdist < 1.0E-8)
	// 		return;

	double fsn = -c.kn * pow(cdist, 1.5);
	double fdn = c.vn * dot(dv, unit);
	double fca = cohesionForce(ir, jr, Ei, Ej, pri, prj, coh, fsn);
	// 	if ((fsn + fca + fdn) < 0 && ir)
	// 		return;
	Fn = (fsn + fca + fdn) * unit;
	double3 e = dv - dot(dv, unit) * unit;
	double mag_e = length(e);
	if (mag_e){
		double3 s_hat = -(e / mag_e);
		double ds = mag_e * dcte.dt;
		double fst = -c.ks * ds;
		double fdt = c.vs * dot(dv, s_hat);
		Ft = (fst + fdt) * s_hat;
		if (length(Ft) >= c.mu * length(Fn))
			Ft = c.mu * fsn * s_hat;
		M = cross(ir * unit, Ft);
		if (length(iomega)){
			double3 on = iomega / length(iomega);
			M += c.ms * fsn * rcon * on;
		}
	}
}

__device__ void DHSModel(
	device_force_constant c, double ir, double jr, double Ei, double Ej, double pri, double prj, double coh,
	double rcon, double cdist, double3 iomega, double& _ds, double& dots,
	double3 dv, double3 unit, double3& Ft, double3& Fn, double3& M)
{
	double fsn = -c.kn * pow(cdist, 1.5);
	double fdn = c.vn * dot(dv, unit);
	double fca = cohesionForce(ir, jr, Ei, Ej, pri, prj, coh, fsn);
	Fn = (fsn + fca + fdn) * unit;
	double3 e = dv - dot(dv, unit) * unit;
	double mag_e = length(e);
	if (mag_e)
	{
		double3 sh = e / mag_e;
		double s_dot = dot(dv, sh);
		double ds = _ds + dcte.dt * (s_dot + dots);
		//double3 rr = d
		_ds = ds;
		dots = s_dot;
		//double ds = mag_e * dcte.dt;
		double ft0 = c.ks * ds + c.vs * (dot(dv, sh));
		double ft1 = c.mu * length(Fn);
		Ft = min(ft0, ft1) * sh;
		M = cross(ir * unit, Ft);
		/*if (length(iomega)){
		double3 on = iomega / length(iomega);
		M += u1.ms * fsn * rcon * on;
		}*/
	}
}

__device__ double particle_plane_contact_detection(
	device_plane_info& pe, double3& xp, double3& wp, double3& u, double r)
{
	double a_l1 = pow(wp.x - pe.l1, 2.0);
	double b_l2 = pow(wp.y - pe.l2, 2.0);
	double sqa = wp.x * wp.x;
	double sqb = wp.y * wp.y;
	double sqc = wp.z * wp.z;
	double sqr = r*r;

	if (abs(wp.z) < r && (wp.x > 0 && wp.x < pe.l1) && (wp.y > 0 && wp.y < pe.l2)){
		double3 dp = xp - pe.xw;
		double3 uu = pe.uw / length(pe.uw);
		int pp = -sign(dot(dp, pe.uw));// dp.dot(pe.UW()));
		u = pp * uu;
		double collid_dist = r - abs(dot(dp, u));// dp.dot(u));
		return collid_dist;
	}

	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr){
		double3 Xsw = xp - pe.xw;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe.l1 && wp.y < 0 && (a_l1 + sqb + sqc) < sqr){
		double3 Xsw = xp - pe.w2;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe.l1 && wp.y > pe.l2 && (a_l1 + b_l2 + sqc) < sqr){
		double3 Xsw = xp - pe.w3;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x < 0 && wp.y > pe.l2 && (sqa + b_l2 + sqc) < sqr){
		double3 Xsw = xp - pe.w4;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	if ((wp.x > 0 && wp.x < pe.l1) && wp.y < 0 && (sqb + sqc) < sqr){
		double3 Xsw = xp - pe.xw;
		double3 wj_wi = pe.w2 - pe.xw;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.x < pe.l1) && wp.y > pe.l2 && (b_l2 + sqc) < sqr){
		double3 Xsw = xp - pe.w4;
		double3 wj_wi = pe.w3 - pe.w4;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;

	}
	else if ((wp.x > 0 && wp.y < pe.l2) && wp.x < 0 && (sqr + sqc) < sqr){
		double3 Xsw = xp - pe.xw;
		double3 wj_wi = pe.w4 - pe.xw;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.y < pe.l2) && wp.x > pe.l1 && (a_l1 + sqc) < sqr){
		double3 Xsw = xp - pe.w2;
		double3 wj_wi = pe.w3 - pe.w2;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	return -1.0;
}

__device__ double3 closestPtPointTriangle(
	device_triangle_info& dpi,
	double3& p,
	double pr,
	int& ct)
{
	double3 a = dpi.P;
	double3 b = dpi.Q;
	double3 c = dpi.R;
	double3 ab = b - a;
	double3 ac = c - a;
	double3 ap = p - a;
	double3 bp = p - b;
	double3 cp = p - c;
	double d1 = dot(ab, ap);
	double d2 = dot(ac, ap);
	double d3 = dot(ab, bp);
	double d4 = dot(ac, bp);
	double d5 = dot(ab, cp);
	double d6 = dot(ac, cp);
	double va = d3 * d6 - d5 * d4;
	double vb = d5 * d2 - d1 * d6;
	double vc = d1 * d4 - d3 * d2;


	if (d1 <= 0.0 && d2 <= 0.0){ ct = 0; return a; }
	if (d3 >= 0.0 && d4 <= d3) { ct = 0; return b; }
	if (d6 >= 0.0 && d5 <= d6) { ct = 0; return c; }

	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){ ct = 1; return a + (d1 / (d1 - d3)) * ab; }
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){ ct = 1; return a + (d2 / (d2 - d6)) * ac; }
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){ ct = 1;	return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b); }
	//ct = 2;
	// P inside face region. Comu0te Q through its barycentric coordinates (u, v, w)
	/*double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;*/
	double denom = 1.0 / (va + vb + vc);
	double3 v = vb * denom * ab;
	double3 w = vc * denom * ac;
	double3 _cpt = a + v + w;
	//double _dist = pr - length(p - _cpt);
	ct = 2;
	//if (_dist > 0) return _cpt;
	return _cpt; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
}

__global__ void calculate_particle_particle_contact_count(
	double4* pos, pair_data* old_pppd, unsigned int *old_count, unsigned int *count,
	unsigned int* sidx, unsigned int *sorted_index, unsigned int *cstart,
	unsigned int* cend, unsigned int _np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	if (id >= np)
		return;
	unsigned int old_sid = 0;
	if (id != 0)
		old_sid = sidx[id - 1];
	unsigned int cnt = 0;
	double4 ipos = pos[id];
	double4 jpos = make_double4(0, 0, 0, 0);
	int3 gridPos = calcGridPosD(make_double3(ipos.x, ipos.y, ipos.z));
	double ir = ipos.w; double jr = 0;
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHashD(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (id == k || k >= np)	continue;
						jpos = pos[k];
						jr = jpos.w;
						double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
						double dist = length(rp);
						double cdist = (ir + jr) - dist;
						if (cdist > 0) cnt++;
						else{
							for (unsigned int j = 0; j < old_count[id]; j++){
								pair_data *pdata = old_pppd + (old_sid + j);
								if (pdata->j == k && pdata->type == 1) 
									pdata->enable = false;
							}
						}
					}
				}
			}
		}
	}
	//old_count[id] = count[id];
	count[id] = cnt;
}

__device__ bool checkOverlab(int3 ctype, double3 p, double3 c, double3 u0, double3 u1)
{
	bool b_over = false;
	if (p.x >= u1.x - 1e-9 && p.x <= u1.x + 1e-9)
		if (p.y >= u1.y - 1e-9 && p.y <= u1.y + 1e-9)
			if (p.z >= u1.z - 1e-9 && p.z <= u1.z + 1e-9)
				b_over = true;

	if (/*(ctype.y || ctype.z) &&*/ !b_over)
	{
		if (u0.x >= u1.x - 1e-9 && u0.x <= u1.x + 1e-9)
			if (u0.y >= u1.y - 1e-9 && u0.y <= u1.y + 1e-9)
				if (u0.z >= u1.z - 1e-9 && u0.z <= u1.z + 1e-9)
					b_over = true;
	}
	return b_over;
}

__device__ double3 ev2omega(double4& e, double4& ev)
{
	return make_double3(
		ev.y*e.x - ev.x*e.y + ev.z*e.w - ev.w*e.z,
		ev.z*e.x - ev.x*e.z - ev.y*e.w + ev.w*e.y,
		ev.y*e.z - ev.x*e.w - ev.z*e.y + ev.w*e.x);
}

__global__ void calculate_particle_triangle_contact_count(
	device_triangle_info* dpi, double4 *pos, pair_data* old_pppd,
	unsigned int* old_count, unsigned int* count,
	unsigned int* sidx, unsigned int* sorted_index,
	unsigned int* cstart, unsigned int* cend, unsigned int _np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	if (id >= np) return;
	double cdist = 0.0;
	double4 ipos4 = pos[id];
	double3 ipos = make_double3(ipos4.x, ipos4.y, ipos4.z);
	double3 unit = make_double3(0.0, 0.0, 0.0);
	int3 gridPos = calcGridPosD(ipos);
	double ir = pos[id].w;
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	unsigned int cnt = 0;
	unsigned int old_sid = 0;
	//	unsigned int overlap_count = 0;
	int3 ctype = make_int3(0, 0, 0);
	double3 previous_cpt = make_double3(0.0, 0.0, 0.0);
	double3 previous_unit = make_double3(0.0, 0.0, 0.0);
	if (id != 0) old_sid = sidx[id - 1];
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHashD(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (k >= np){
							k -= np;
							int t = -1;
							//							unsigned int pidx = dpi[k].id;
							device_triangle_info tri = dpi[k];
							double3 cpt = closestPtPointTriangle(dpi[k], ipos, ir, t);
							double3 unit = -normalize(cross(tri.Q - tri.P, tri.R - tri.P));
							//double angle = dot(pc_unit, unit) / (length(pc_unit) * length(unit));*/
							cdist = ir - length(ipos - cpt);
							if (cdist > 0)
							{
								//double len_cpt = sqrt(dot(cpt, cpt));
								//	cpt = len_cpt ? cpt / sqrt(dot(cpt, cpt)) : cpt;								
								bool overlap = checkOverlab(ctype, previous_cpt, cpt, previous_unit, unit);
								if (overlap)
									continue;
								previous_cpt = cpt;
								previous_unit = unit;
								*(&(ctype.x) + t) += 1;
								cnt++;
							}
							else{
								for (unsigned int j = 0; j < old_count[id]; j++){
									pair_data *pdata = old_pppd + (old_sid + j);
									if (pdata->j == k && pdata->type == 2) 
										pdata->enable = false;
								}
							}
						}
					}
				}
			}
		}
	}
	if (cnt > 1)
		cnt = cnt;
	count[id] += cnt;
}

__global__ void calculate_particle_plane_contact_count(
	device_plane_info *plane, pair_data* old_pppd, unsigned int *old_count, unsigned int *count,
	unsigned int *sidx, double4* pos, unsigned int _nplanes, unsigned int _np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	unsigned int nplanes = _nplanes;
	if (id >= np)
		return;
	//count[id] = 0.0;
	double4 ipos;
	ipos.x = pos[id].x;
	ipos.y = pos[id].y;
	ipos.z = pos[id].z;
	ipos.w = pos[id].w;// (pos[id].x, pos[id].y, pos[id].z, pos[id].w);
	double3 ipos3 = make_double3(ipos.x, ipos.y, ipos.z);
	unsigned int old_sid = 0;
	if (id != 0)
		old_sid = sidx[id - 1];
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < nplanes; i++)
	{
		device_plane_info pl = plane[i];
		double3 unit = make_double3(0, 0, 0);
		double3 dp = make_double3(ipos.x, ipos.y, ipos.z) - pl.xw;
		double3 wp = make_double3(dot(dp, pl.u1), dot(dp, pl.u2), dot(dp, pl.uw));
		double cdist = particle_plane_contact_detection(pl, ipos3, wp, unit, ipos.w);
		if (cdist > 0)
			cnt++;
		else
		{
			for (unsigned int j = 0; j < old_count[id]; j++)
			{
				pair_data *pdata = old_pppd + (old_sid + j);
				if (pdata->j == i && pdata->type == 0)
					pdata->enable = false;
			}
		}
	}
	//old_count[id] = count[id];
	count[id] += cnt;
}

__global__ void copy_old_to_new_pair(
	unsigned int *old_count, unsigned int *new_count, unsigned int* old_sidx, unsigned int* new_sidx,
	pair_data* old_pppd, pair_data* new_pppd, unsigned int _np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.y) + threadIdx.x;
	unsigned int np = _np;
	if (id >= np)
		return;
	unsigned int oc = old_count[id];
	//	unsigned int nc = new_count[id];
	//particle_plane_pair_data new_ppp[10] = { 0, };
	unsigned int old_sid = 0;
	unsigned int new_sid = 0;
	if (id != 0)
	{
		old_sid = old_sidx[id - 1];
		new_sid = new_sidx[id - 1];
	}
	unsigned int new_cnt = 0;
	for (unsigned int i = 0; i < oc; i++)
	{
		if (old_pppd[old_sid + i].enable)
		{
			new_pppd[new_sid + new_cnt++] = old_pppd[old_sid + i];
		}
	}
}

__device__ double3 calculate_rolling_resistance(
	double3& _kr, double3& xc1, double3& xc2, double3& pi, double3& pj, double3& xa, double3& n,
	double Fn, double mu, double kt, double3& M)
{
	double3 dkr = 0.5 * (xc2 + xc1) - xa;
	double3 khat = _kr + dkr;
	double3 kr = khat - dot(khat, n) * n;
	bool limit = length(kr) > 1.0 * mu * (Fn / kt);
	if (limit)
	{
		kr = (1.0 * mu * Fn / kt) * (kr / length(kr));
	}
	double3 _M = cross(xa - pi, kt*kr);
	M += _M;
	return kr;
}

__global__ void new_particle_particle_contact_kernel(
	double4* pos, double4* ep, double3* vel, double4* ev,
	double* mass, double3* force, double3* moment, pair_data *old_pairs, pair_data *pairs,
	unsigned int* old_count, unsigned int* count, unsigned int* old_sidx, unsigned int* sidx, int2* type_count, device_contact_property* cp,
	unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, unsigned int _np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	if (id >= np)
		return;
	double4 ipos = pos[id];
	double3 pos3 = make_double3(ipos.x, ipos.y, ipos.z);
	double4 jpos = make_double4(0, 0, 0, 0);
	double4 epi = ep[id];
	double4 epj = make_double4(0.0, 0.0, 0.0, 0.0);
	double3 ivel = vel[id];
	double3 iomega = ev2omega(ep[id], ev[id]);// [id];
	double3 jvel = make_double3(0.0, 0.0, 0.0);
	double3 jomega = make_double3(0.0, 0.0, 0.0);
	double3 sumF = make_double3(0.0, 0.0, 0.0);
	double3 sumM = make_double3(0.0, 0.0, 0.0);
	double3 Ft = make_double3(0, 0, 0);
	double3 Fn = make_double3(0, 0, 0);// [id] * dcte.gravity;
	double3 M = make_double3(0, 0, 0);
	int3 gridPos = calcGridPosD(pos3);
	double ir = ipos.w; double jr = 0;
	double im = mass[id]; double jm = 0.0;
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	unsigned int old_sid = 0;
	unsigned int sid = 0;
	unsigned int old_cnt = old_count[id];
	//	unsigned int cnt = count[id];
	pair_data pair;
	int tcnt = 0;
	if (id != 0)
	{
		sid = sidx[id - 1];
		old_sid = old_sidx[id - 1];
	}

	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHashD(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (id == k || k >= np)
							continue;
						jpos = pos[k];
						epj = ep[k];
						jr = jpos.w;
						double3 pos3j = make_double3(jpos.x, jpos.y, jpos.z);
						double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
						double dist = length(rp);
						double cdist = (ir + jr) - dist;
						if (cdist > 0)
						{
							double3 unit = rp / dist;
							double3 _c = pos3 - ir * unit;
							double3 Xc0 = toLocal(_c, epi);
							double3 Xc1 = toLocal(_c, epj);
							pair = { true, 1, id, k, 0.0, 0.0, make_double3(0.0, 0.0, 0.0), Xc0, Xc1 };
							for (unsigned int j = 0; j < old_cnt; j++)
							{
								pair_data* pd = old_pairs + (old_sid + j);
								if (pd->enable && pd->j == k)
								{
									pair = *pd;
									break;
								}
							}
							jvel = vel[k];
							jomega = ev2omega(ep[k], ev[k]);
							jm = mass[k];
							double rcon = ipos.w - 0.5 * cdist;
							
							double3 rv = jvel + cross(jomega, -jpos.w * unit) - (ivel + cross(iomega, ipos.w * unit));
							device_force_constant c = getConstant(
								1, ipos.w, jpos.w, im, jm, cp->Ei, cp->Ej,
								cp->pri, cp->prj, cp->Gi, cp->Gj,
								cp->rest, cp->fric, cp->rfric, cp->sratio);
							DHSModel(
								c, ipos.w, jpos.w, cp->Ei, cp->Ej, cp->pri, cp->prj,
								cp->coh, rcon, cdist, iomega, pair.ds, pair.dots,
								rv, unit, Ft, Fn, M);
							calculate_rolling_resistance(
								pair.kr, toGlobal(pair.ci, epi), toGlobal(pair.cj, epj), pos3, 
								pos3j, _c, unit, length(Fn), c.mu, c.ks, M);
							sumF += Fn + Ft;
							sumM += M;
							pairs[sid + tcnt++] = pair;
						}
					}
				}
			}
		}
	}
	force[id] += sumF;
	moment[id] += sumM;
	type_count[id].x = tcnt;
}

__global__ void new_particle_polygon_object_conatct_kernel(
	device_triangle_info* dpi, device_mesh_mass_info* dpmi, pair_data* old_pairs, pair_data* pairs,
	unsigned int* old_count, unsigned int* count, unsigned int* old_sidx, unsigned int* sidx, int2* type_count,
	double4 *pos, double3 *vel, double3 *omega, double3 *force, double3 *moment,
	double* mass, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend,
	device_contact_property *cp, unsigned int _np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	if (id >= np) return;
	unsigned int old_sid = 0;
	unsigned int sid = type_count[id].x + type_count[id].y;
	if (id != 0){
		old_sid = old_sidx[id - 1];
		sid += sidx[id - 1];
	}
	double cdist = 0.0;
	double im = mass[id];
	double4 ipos4 = pos[id];
	double3 ipos = make_double3(ipos4.x, ipos4.y, ipos4.z);
	double3 ivel = vel[id];
	double3 iomega = omega[id];
	double3 unit = make_double3(0.0, 0.0, 0.0);
	int3 gridPos = calcGridPosD(make_double3(ipos.x, ipos.y, ipos.z));
	double ir = ipos4.w;
	double3 M = make_double3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	double3 Fn = make_double3(0, 0, 0);
	double3 Ft = make_double3(0, 0, 0);
	double3 sum_force = make_double3(0, 0, 0);
	double3 sum_moment = make_double3(0, 0, 0);
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	unsigned int old_cnt = old_count[id];
	//	unsigned int cnt = count[id];
	pair_data pair;
	int3 ctype = make_int3(0, 0, 0);
	double3 previous_cpt = make_double3(0.0, 0.0, 0.0);
	double3 previous_unit = make_double3(0.0, 0.0, 0.0);
	//double3 mpos = pmi
	unsigned int cur_cnt = 0;
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHashD(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (k >= dcte.np)
						{
							int t = -1;
							k -= dcte.np;
							unsigned int pidx = dpi[k].id;
							device_contact_property cmp = cp[pidx];
							device_mesh_mass_info pmi = dpmi[pidx];
							double3 mpos = make_double3(pmi.px, pmi.py, pmi.pz);
							double3 cpt = closestPtPointTriangle(dpi[k], ipos, ir, t);
							double3 po2cp = cpt - mpos;
							cdist = ir - length(ipos - cpt);
							Fn = make_double3(0.0, 0.0, 0.0);
							if (cdist > 0)
							{
								//double len_cpt = sqrt(dot(cpt, cpt));
								//cpt = len_cpt ? cpt / sqrt(dot(cpt, cpt)) : cpt;
								device_triangle_info tri = dpi[k];
								double3 qp = tri.Q - tri.P;
								double3 rp = tri.R - tri.P;
								double rcon = ir - 0.5 * cdist;
								unit = -normalize(cross(qp, rp));// unit / length(cross(qp, rp));
								bool overlab = checkOverlab(ctype, previous_cpt, cpt, previous_unit, unit);
								if (overlab)
									continue;
								*(&(ctype.x) + t) += 1;
								//previous_cpt = cpt;
								pair = { true, 2, id, k, 0.0, 0.0, Fn, Fn, Fn };
								for (unsigned int j = 0; j < old_cnt; j++)
								{
									pair_data* pd = old_pairs + old_sid + j;
									if (pd->enable && pd->j == k)
									{
										pair = *pd;
										break;
									}
								}
								previous_cpt = cpt;
								previous_unit = unit;
								*(&(ctype.x) + t) += 1;
								double3 mvel = make_double3(pmi.vx, pmi.vy, pmi.vz);
								double3 momega = make_double3(pmi.ox, pmi.oy, pmi.oz);
								double3 d1 = cross(momega, po2cp);
								double3 dv = mvel + cross(momega, po2cp) - (ivel + cross(iomega, ir * unit));
								device_force_constant c = getConstant(
									1, ir, 0, im, 0, cmp.Ei, cmp.Ej,
									cmp.pri, cmp.prj, cmp.Gi, cmp.Gj,
									cmp.rest, cmp.fric, cmp.rfric, cmp.sratio);
								DHSModel(
									c, ir, 0, cp->Ei, cp->Ej, cp->pri, cp->prj,
									cp->coh, rcon, cdist, iomega, pair.ds, pair.dots,
									dv, unit, Ft, Fn, M);
								sum_force += Fn + Ft;
								sum_moment += M;
								dpmi[pidx].force += -(Fn + Ft);
								dpmi[pidx].moment += -cross(po2cp, Fn + Ft);
								pairs[sid + cur_cnt++] = pair;
							}
						}
					}
				}
			}
		}
	}
	force[id] += sum_force;
	moment[id] += sum_moment;
}

__global__ void new_particle_plane_contact(
	device_plane_info *plane, double4* pos, double3* vel,
	double3* omega, double* mass, double3* force,
	double3* moment, unsigned int* old_count, unsigned int *count,
	unsigned int* old_sidx, unsigned int *sidx, int2 *type_count,
	pair_data *old_pairs, pair_data *pairs, device_contact_property *cp,
	unsigned int _nplanes, unsigned int _np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	unsigned int nplanes = _nplanes;
	if (id >= np)
		return;
	unsigned int old_sid = 0;
	unsigned int sid = type_count[id].x;
	if (id != 0)
	{
		old_sid = old_sidx[id - 1];
		sid = sidx[id - 1] + type_count[id].x;
	}

	double4 ipos = pos[id];
	double3 ivel = vel[id];
	double3 iomega = omega[id];
	double m = mass[id];
	double3 Fn = make_double3(0, 0, 0);
	double3 Ft = make_double3(0, 0, 0);
	double3 M = make_double3(0, 0, 0);
	double3 ipos3 = make_double3(ipos.x, ipos.y, ipos.z);
	unsigned int old_cnt = old_count[id];
	//	unsigned int cnt = count[id];
	pair_data ppp;
	unsigned int cur_cnt = 0;
	for (unsigned int i = 0; i < nplanes; i++)
	{
		ppp = { false, 0, id, i + np, 0.0, 0.0, Fn, Fn, Fn };
		for (unsigned int j = 0; j < old_cnt; j++)
		{
			pair_data* pair = old_pairs + old_sid + j;
			if (pair->enable && pair->j == i + np)
			{
				ppp = *pair;// pairs[sid + j]
				break;
			}
		}

		device_plane_info pl = plane[ppp.j - np];
		double3 unit = make_double3(0, 0, 0);
		double3 dp = make_double3(ipos.x, ipos.y, ipos.z) - pl.xw;
		double3 wp = make_double3(dot(dp, pl.u1), dot(dp, pl.u2), dot(dp, pl.uw));
		double cdist = particle_plane_contact_detection(pl, ipos3, wp, unit, ipos.w);
		if (cdist > 0)
		{
			ppp.enable = true;
			ppp.type = 2;
			double rcon = ipos.w - 0.5 * cdist;
			double3 dv = -(ivel + cross(iomega, ipos.w * unit));
			device_force_constant c = getConstant(
				1, ipos.w, 0.0, m, 0.0, cp->Ei, cp->Ej,
				cp->pri, cp->prj, cp->Gi, cp->Gj,
				cp->rest, cp->fric, cp->rfric, cp->sratio);
			DHSModel(
				c, ipos.w, 0, 0, 0, 0, 0, cp->coh, rcon, cdist,
				iomega, ppp.ds, ppp.dots, dv, unit, Ft, Fn, M);
			force[id] += Fn + Ft;// m_force + shear_force;
			moment[id] += M;
			pairs[sid + cur_cnt++] = ppp;
		}
	}
	type_count[id].y = cur_cnt;
}



__global__ void updatePolygonObjectData_kernel(
	device_mesh_mass_info *dpmi, double* vList,
	double4* sphere, device_triangle_info* dpi, unsigned int ntriangle)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = ntriangle;
	if (id >= np)
		return;
	int s = id * 9;
	int mid = dpi[id].id;
	device_mesh_mass_info pmi = dpmi[mid];
	double3 pos = make_double3(pmi.px, pmi.py, pmi.pz);// dpmi[mid].origin;
	double4 ep = make_double4(pmi.e0, pmi.e1, pmi.e2, pmi.e3);// dpmi[mid].ep;
	double4 sph = sphere[id];
	double3 P = make_double3(vList[s + 0], vList[s + 1], vList[s + 2]);
	double3 Q = make_double3(vList[s + 3], vList[s + 4], vList[s + 5]);
	double3 R = make_double3(vList[s + 6], vList[s + 7], vList[s + 8]);
	P = pos + toGlobal(P, ep);
	Q = pos + toGlobal(Q, ep);
	R = pos + toGlobal(R, ep);
	double3 V = Q - P;
	double3 W = R - P;
	double3 N = cross(V, W);
	N = N / length(N);
	double3 M1 = 0.5 * (Q + P);
	double3 M2 = 0.5 * (R + P);
	double3 D1 = cross(N, V);
	double3 D2 = cross(N, W);
	double t;
	if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
	}
	else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
	}
	else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
	{
		t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
	}
	double3 ctri = M1 + t * D1;
	sphere[id] = make_double4(ctri.x, ctri.y, ctri.z, sph.w);
	dpi[id].P = P;
	dpi[id].Q = Q;
	dpi[id].R = R;
}

void cu_calculate_particle_particle_contact_count(
	double* pos, pair_data* pairs,
	unsigned int* old_pair_count, unsigned int* pair_count,
	unsigned int* pair_start, unsigned int* sorted_id,
	unsigned int* cell_start, unsigned int* cell_end, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	calculate_particle_particle_contact_count << <numBlocks, numThreads >> >(
		(double4*)pos,
		pairs,
		old_pair_count,
		pair_count,
		pair_start,
		sorted_id,
		cell_start,
		cell_end,
		np);
}

void cu_calculate_particle_triangle_contact_count(
	device_triangle_info *dpi, double* pos, pair_data* pairs,
	unsigned int* old_pair_count,
	unsigned int* pair_count,
	unsigned int* pair_start,
	unsigned int* sorted_id,
	unsigned int* cell_start,
	unsigned int* cell_end, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	calculate_particle_triangle_contact_count << <numBlocks, numThreads >> >(
		dpi,
		(double4*)pos,
		pairs,
		old_pair_count,
		pair_count,
		pair_start,
		sorted_id,
		cell_start,
		cell_end,
		np);
}

void cu_calculate_particle_plane_contact_count(
	device_plane_info *plane, pair_data* pairs,
	unsigned int* old_pair_count, unsigned int *count,
	unsigned int *sidx, double* pos, unsigned int nplanes, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	calculate_particle_plane_contact_count << <numBlocks, numThreads >> >(
		plane,
		pairs, 
		old_pair_count,
		count,
		sidx,
		(double4 *)pos,
		nplanes,
		np);
}

unsigned int cu_sumation_contact_count(unsigned int* count, unsigned int np)
{
	unsigned int nc = thrust::reduce(thrust::device_ptr<unsigned int>(count), thrust::device_ptr<unsigned int>(count + np));
	return nc;
}

void cu_copy_old_to_new_pair(
	unsigned int *old_count, unsigned int *new_count,
	unsigned int* old_sidx, unsigned int* new_sidx,
	pair_data* old_pppd, pair_data* new_pppd,
	unsigned int nc, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	thrust::inclusive_scan(
		thrust::device_ptr<unsigned int>(new_count), 
		thrust::device_ptr<unsigned int>(new_count + np), 
		thrust::device_ptr<unsigned int>(new_sidx));
	copy_old_to_new_pair << <numBlocks, numThreads >> >(
		old_count,
		new_count,
		old_sidx,
		new_sidx,
		old_pppd,
		new_pppd,
		np);
}

void cu_new_particle_particle_contact(
	double* pos, double* ep, double* vel, double* ev, 
	double* mass, double* force, double* moment,
	pair_data* old_pairs, pair_data* pairs, 
	unsigned int *old_pair_count, unsigned int* pair_count, 
	unsigned int *old_pair_start, unsigned int *pair_start, int *type_count, 
	device_contact_property* cp, unsigned int *sorted_id, 
	unsigned int* cell_start, unsigned int* cell_end, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	new_particle_particle_contact_kernel << < numBlocks, numThreads >> >(
		(double4 *)pos,
		(double4 *)ep,
		(double3 *)vel,
		(double4 *)ev,
		mass,
		(double3 *)force,
		(double3 *)moment,
		old_pairs, pairs,
		old_pair_count, pair_count,
		old_pair_start, pair_start,
		(int2 *)type_count,
		cp,
		sorted_id,
		cell_start,
		cell_end,
		np);
}

void cu_new_particle_plane_contact(
	device_plane_info *plane, double* pos, double* vel,
	double* omega, double* mass, double* force,	double* moment,
	unsigned int *old_count, unsigned int *count, 
	unsigned int *old_sidx, unsigned int *sidx, int* type_count,
	pair_data *old_pairs, pair_data *pairs, device_contact_property *cp,
	unsigned int nplanes, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	new_particle_plane_contact << < numBlocks, numThreads >> >(
		plane,
		(double4 *)pos,
		(double3 *)vel,
		(double3 *)omega,
		mass,
		(double3 *)force,
		(double3 *)moment,
		old_count, count,
		old_sidx, sidx,
		(int2 *)type_count,
		old_pairs, pairs,
		cp,
		nplanes,
		np);
}

void cu_new_particle_polygon_object_contact(
	device_triangle_info* dpi, device_mesh_mass_info* dpmi,
	pair_data *old_pairs, pair_data *pairs,
	unsigned int* old_count, unsigned int* count,
	unsigned int* old_sidx, unsigned int* sidx, int* type_count,
	double *pos, double *vel, double *omega, double *force, double *moment,
	double* mass, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend,
	device_contact_property *cp, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	new_particle_polygon_object_conatct_kernel << < numBlocks, numThreads >> >(
		dpi, dpmi,
		old_pairs, pairs,
		old_count, count,
		old_sidx, sidx,
		(int2 *)type_count,
		(double4 *)pos,
		(double3 *)vel,
		(double3 *)omega,
		(double3 *)force,
		(double3 *)moment,
		mass,
		sorted_index,
		cstart,
		cend,
		cp,
		np);
}

void cu_update_meshObjectData(
	device_mesh_mass_info *dpmi, double* vList,
	double* sphere, device_triangle_info* dpi, unsigned int ntriangle)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(ntriangle, 512, numBlocks, numThreads);
	updatePolygonObjectData_kernel << <numBlocks, numThreads >> >(dpmi, vList, (double4 *)sphere, dpi, ntriangle);
}
//void cu_calculate_contact_pair_count(
//	double* pos, unsigned int *count,
//	unsigned int* sidx, unsigned int* cstart,
//	unsigned int* cend, unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	calculate_pair_count_kernel << < numBlocks, numThreads >> >(
//		(double4 *)pos,
//		count,
//		sidx,
//		cstart,
//		cend,
//		np);
//}