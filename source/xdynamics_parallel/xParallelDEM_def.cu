#include "xdynamics_parallel/xParallelDEM_decl.cuh"
#include "xdynamics_parallel/xParallelDEM_impl.cuh"
#include "xdynamics_object/xConatct.h"
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
//#include <helper_cuda.h>

void setDEMSymbolicParameter(device_dem_parameters *h_paras)
{
	checkCudaErrors(cudaMemcpyToSymbol(cte, h_paras, sizeof(device_dem_parameters)));
}

unsigned int numThreads, numBlocks;

//Round a / b to nearest higher integer value
unsigned iDivUp(unsigned a, unsigned b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(unsigned n, unsigned blockSize, unsigned &numBlocks, unsigned &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

void vv_update_position(
	double *pos, double* ep, double *vel, 
	double* ev, double *acc, double* ea, unsigned int np)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	vv_update_position_kernel << < numBlocks, numThreads >> > (
		(double4 *)pos,
		(double4 *)ep,
		(double3 *)vel,
		(double4 *)ev,
		(double3 *)acc,
		(double4 *)ea,
		np);
}

void vv_update_velocity(
	double *vel, double *acc, 
	double* ep, double *ev, double *ea, 
	double *force, double *moment, double* mass, 
	double* iner, unsigned int np)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	vv_update_velocity_kernel << < numBlocks, numThreads >> > (
		(double3 *)vel,
		(double3 *)acc,
		(double4 *)ep,
		(double4 *)ev,
		(double4 *)ea,
		(double3 *)force,
		(double3 *)moment,
		mass,
		(double3 *)iner,
		np);
}

void cu_calculateHashAndIndex(
	unsigned int* hash,
	unsigned int* index,
	double *pos,
	unsigned int np)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	calculateHashAndIndex_kernel << < numBlocks, numThreads >> > (hash, index, (double4 *)pos, np);
}

void cu_calculateHashAndIndexForPolygonSphere(
	unsigned int* hash,
	unsigned int* index,
	unsigned int sid,
	unsigned int nsphere,
	double *sphere)
{
	computeGridSize(nsphere, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	calculateHashAndIndexForPolygonSphere_kernel << <numBlocks, numThreads >> > (hash, index, sid, nsphere, (double4 *)sphere);
}


void cu_reorderDataAndFindCellStart(
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
	//std::cout << "begin sortbykey" << std::endl;
	thrust::sort_by_key(thrust::device_ptr<unsigned>(hash),
		thrust::device_ptr<unsigned>(hash + np),
		thrust::device_ptr<unsigned>(index));
	//std::cout << "end sortbykey" << std::endl;
	//std::cout << "step 2" << std::endl;
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	checkCudaErrors(cudaMemset(cstart, 0xffffffff, ncell * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(cend, 0, ncell * sizeof(unsigned int)));
	unsigned smemSize = sizeof(unsigned int)*(numThreads + 1);
	//std::cout << "step 3" << std::endl;
	reorderDataAndFindCellStart_kernel << < numBlocks, numThreads, smemSize >> > (
		hash,
		index,
		cstart,
		cend,
		sorted_index,
		np);
}

void cu_calculate_p2p(
	const int tcm, double* pos, double* ep, double* vel,
	double* ev, double* force, double* moment,
	double* mass, double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd,
	unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, device_contact_property* cp, unsigned int np)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	switch (tcm)
	{
	case 0:
		calculate_p2p_kernel<0> << < numBlocks, numThreads >> > (
			(double4 *)pos, (double4 *)ep, (double3 *)vel,
			(double4 *)ev, (double3 *)force,
			(double3 *)moment, mass, (double3 *)tmax, rres,
			pair_count, pair_id, (double2 *)tsd,
			sorted_index, cstart,
			cend, cp, np);
		break;
	case 1:
		calculate_p2p_kernel<1> << < numBlocks, numThreads >> > (
			(double4 *)pos, (double4 *)ep, (double3 *)vel,
			(double4 *)ev, (double3 *)force,
			(double3 *)moment, mass, (double3 *)tmax, rres,
			pair_count, pair_id, (double2 *)tsd,
			sorted_index, cstart,
			cend, cp, np);
		break;
	}
}

void cu_plane_contact_force(
	const int tcm, device_plane_info* plan, device_body_info* dbi, 
	device_contact_property *cp, device_body_force* dbfm,
	double* pos, double* ep, double* vel, double* ev,
	double* force, double* moment, double* mass,
	double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd,
	unsigned int np, unsigned int nplane)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	memset(dbfm, 0, sizeof(device_body_force) * nplane);

	
	double3* dbf = xContact::deviceBodyForce();
	double3* dbm = xContact::deviceBodyMoment();
	/*checkCudaErrors(cudaMalloc((void**)&dbf, sizeof(double3) * np));
	checkCudaErrors(cudaMalloc((void**)&dbm, sizeof(double3) * np));*/
	for (unsigned int i = 0; i < nplane; i++)
	{
		plane_contact_force_kernel << < numBlocks, numThreads >> > (
			plan, i, dbi, cp, dbf, dbm,
			(double4 *)pos, (double4 *)ep, (double3 *)vel, (double4 *)ev,
			(double3 *)force, (double3 *)moment, mass,
			(double3 *)tmax, rres,
			pair_count, pair_id, (double2 *)tsd, np);

		dbfm[i].force += reductionD3(dbf, np);
		dbfm[i].moment += reductionD3(dbm, np);
	}
	/*checkCudaErrors(cudaFree(dbf));
	checkCudaErrors(cudaFree(dbm));*/
}

void cu_cube_contact_force(
	const int tcm, device_plane_info* plan,
	double* pos, double* ep, double* vel, double* omega,
	double* force, double* moment, double* mass,
	unsigned int np, device_contact_property *cp)
{
	/*computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	for (unsigned int i = 0; i < 6; i++)
	{
	switch (tcm)
	{
	case 0: plane_contact_force_kernel<0> << < numBlocks, numThreads >> > (
	plan + i, (double4 *)pos, (double3 *)vel, (double3 *)omega,
	(double3 *)force, (double3 *)moment, cp, mass, np);
	break;
	case 1: plane_contact_force_kernel<1> << < numBlocks, numThreads >> > (
	plan + i, (double4 *)pos, (double3 *)vel, (double3 *)omega,
	(double3 *)force, (double3 *)moment, cp, mass, np);
	break;
	}
	}*/
}

void cu_cylinder_contact_force(
	const int tcm, device_cylinder_info* cyl, 
	device_body_info* bi, device_body_force* dbfm, device_contact_property *cp,
	double* pos, double* ep, double* vel, double* ev,
	double* force, double* moment,
	double* mass, double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd, 
	unsigned int np, unsigned int ncylinder)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	memset(dbfm, 0, sizeof(device_body_force) * ncylinder);
	double3* dbf = xContact::deviceBodyForce();
	double3* dbm = xContact::deviceBodyMoment();
	//checkCudaErrors(cudaMalloc((void**)&dbf, sizeof(double3) * np));
	//checkCudaErrors(cudaMalloc((void**)&dbm, sizeof(double3) * np));
	/*double3* dbf = xContact::deviceBodyForce();
	double3* dbm = xContact::deviceBodyMoment();*/
	for (unsigned int i = 0; i < ncylinder; i++)
	{
		cylinder_contact_force_kernel << < numBlocks, numThreads >> > (
			cyl, i, bi, dbf, dbm, (double4 *)pos, (double4 *)ep, (double3 *)vel, (double4 *)ev,
			(double3 *)force, (double3 *)moment, cp, mass,
			(double3*)tmax, rres, pair_count, pair_id, (double2 *)tsd, np);
		dbfm[i].force += reductionD3(dbf, np);
		dbfm[i].moment += reductionD3(dbm, np);
	}
	/*checkCudaErrors(cudaFree(dbf));
	checkCudaErrors(cudaFree(dbm));*/
}

void cu_particle_polygonObject_collision(
	device_triangle_info* dpi, device_body_info* dbi,
	device_body_force* dbfm,
	double* pos, double* ep, double* vel, double* ev,
	double* force, double* moment, double* mass,
	double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd, double* dsph,
	unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, device_contact_property *cp,
	unsigned int np, unsigned int bindex, unsigned int eindex, unsigned int nmesh)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	
	double3* dbf = xContact::deviceBodyForce();
	double3* dbm = xContact::deviceBodyMoment();
	particle_polygonObject_collision_kernel<< < numBlocks, numThreads >> > (
		dpi, dbi, dbf, dbm,
		(double4 *)pos, (double4 *)ep, (double3 *)vel, (double4 *)ev,
		(double3 *)force, (double3 *)moment, mass,
		(double3 *)tmax, rres,
		pair_count, pair_id, (double2 *)tsd, (double4 *)dsph,
		sorted_index, cstart, cend, cp, bindex, eindex, np);
}

void cu_decide_rolling_friction_moment(
	double* tmax,
	double* rres,
	double* inertia,
	double* ep,
	double* ev,
	double* moment,
	unsigned int np)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	decide_rolling_friction_moment_kernel << <numBlocks, numThreads >> > (
		(double3 *)tmax,
		rres,
		inertia,
		(double4 *)ep,
		(double4 *)ev,
		(double3 *)moment,
		np);
}

double3 reductionD3(double3* in, unsigned int np)
{
	double3 rt = make_double3(0.0, 0.0, 0.0);
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	double3* d_out;
	double3* h_out = new double3[numBlocks];
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(double3) * numBlocks));
	checkCudaErrors(cudaMemset(d_out, 0, sizeof(double3) * numBlocks));
	unsigned smemSize = sizeof(double3)*(CUDA_THREADS_PER_BLOCK);
	reduce6<double3, CUDA_THREADS_PER_BLOCK> << < numBlocks, numThreads, smemSize >> > (in, d_out, np);
	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(double3) * numBlocks, cudaMemcpyDeviceToHost));
	for (unsigned int i = 0; i < numBlocks; i++) {
		rt.x += h_out[i].x;
		rt.y += h_out[i].y;
		rt.z += h_out[i].z;
	}
	delete[] h_out;
	checkCudaErrors(cudaFree(d_out));
	return rt;
}

void cu_update_meshObjectData(
	double *vList, double* sph, double* dlocal, device_triangle_info* poly,
	device_body_info* dbi, unsigned int ntriangle)
{
	computeGridSize(ntriangle, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	updateMeshObjectData_kernel << <numBlocks, numThreads >> > (
		dbi,
		vList,
		(double4 *)sph,
		(double3 *)dlocal,
		poly,
		ntriangle);
}

void cu_clusters_contact(
	double* pos, double* cpos, double* ep, double* vel,
	double* ev, double* force,
	double* moment, double* mass, double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd,
	unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, device_contact_property* cp, xClusterInformation* xci, unsigned int np)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	calcluate_clusters_contact_kernel << < numBlocks, numThreads >> > (
			(double4 *)pos, (double4* )cpos, (double4 *)ep, (double3 *)vel,
			(double4 *)ev, (double3 *)force,
			(double3 *)moment, mass, (double3 *)tmax, rres,
			pair_count, pair_id, (double2 *)tsd,
			sorted_index, cstart,
			cend, cp, xci, np);
}

void cu_cluster_plane_contact(
	device_plane_info* plan, device_body_info* dbi, device_body_force* dbfm,
	device_contact_property *cp,
	double* pos, double* cpos, double* ep, double* vel, double* ev,
	double* force, double* moment, double* mass,
	double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd, 
	xClusterInformation* xci, unsigned int np, unsigned int nplanes)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	memset(dbfm, 0, sizeof(device_body_force) * nplanes);
	double3* dbf = xContact::deviceBodyForce();
	double3* dbm = xContact::deviceBodyMoment();
	//double3* h_dbf = new double3[np];
	for (unsigned int i = 0; i < nplanes; i++)
	{
		cluster_plane_contact_kernel << < numBlocks, numThreads >> > (
			plan, i, dbi, dbf, dbm,
			(double4 *)pos, 
			(double4*)cpos, 
			(double4 *)ep, 
			(double3 *)vel, 
			(double4 *)ev,
			(double3 *)force, 
			(double3 *)moment,
			cp, mass,
			(double3 *)tmax, rres,
			pair_count, pair_id, 
			(double2 *)tsd, xci, np);
		//cudaMemcpy(h_dbf, dbf, sizeof(double3) * np, cudaMemcpyDeviceToHost);
		dbfm[i].force += reductionD3(dbf, np);
		dbfm[i].moment += reductionD3(dbm, np);
	}
}

void cu_cluster_meshes_contact(
	device_triangle_info * dpi, 
	device_body_info * dbi,
	double * pos,
	double * cpos,
	double * ep,
	double * vel,
	double * ev,
	double * force,
	double * moment,
	device_contact_property * cp,
	double * mass, 
	double * tmax,
	double * rres, 
	unsigned int * pair_count, 
	unsigned int * pair_id, 
	double * tsd,
	unsigned int * sorted_index,
	unsigned int * cstart, 
	unsigned int * cend, 
	xClusterInformation * xci,
	unsigned int bindex,
	unsigned int eindex,
	unsigned int np)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	double3* dbf = xContact::deviceBodyForce();
	double3* dbm = xContact::deviceBodyMoment();
	cluster_meshes_contact_kernel << <numBlocks, numThreads >> > (
		dpi, dbi, dbf, dbm,
		(double4*)pos,
		(double4*)cpos,
		(double4*)ep,
		(double3*)vel,
		(double4*)ev,
		(double3*)force,
		(double3*)moment,
		cp, mass,
		(double3*)tmax,
		rres,
		pair_count, pair_id,
		(double2*)tsd, sorted_index,
		cstart, cend,
		xci, bindex, eindex, np);
}

void vv_update_cluster_position(
	double *pos, double *cpos, double* ep,
	double *rloc, double *vel, double *acc,
	double* ev, double* ea,
	xClusterInformation *xci, unsigned int np)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);

	vv_update_position_cluster_kernel << < numBlocks, numThreads >> > (
		(double4*)pos, 
		(double4*)cpos,
		(double4*)ep, 
		(double3*)rloc,
		(double3*)vel,
		(double3*)acc, 
		(double4*)ev,
		(double4*)ea,
		xci,
		np);
}

void vv_update_cluster_velocity(
	double* cpos, double* ep, double *vel, double *acc, double *ev,
	double *ea, double *force, double *moment, double* rloc,
	double* mass, double* iner, xClusterInformation* xci, unsigned int np)
{
	computeGridSize(np, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	vv_update_cluster_velocity_kernel << <numBlocks, numThreads >> > (
		(double4*)cpos,
		(double4*)ep,
		(double3*)vel,
		(double3*)acc,
		(double4*)ev,
		(double4*)ea,
		(double3*)force,
		(double3*)moment,
		(double3*)rloc,
		mass,
		(double3*)iner,
		xci,
		np
		);
}

void cu_calculate_spring_damper_force(
	double* pos,
	double* vel,
	double* force,
	xSpringDamperConnectionInformation* xsdci,
	xSpringDamperConnectionData* xsdcd,
	xSpringDamperCoefficient* xsdkc,
	unsigned int nc)
{
	computeGridSize(nc, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	calculate_spring_damper_force_kernel << <numBlocks, numThreads >> > (
		(double4*)pos,
		(double3*)vel,
		(double3*)force,
		xsdci,
		xsdcd,
		xsdkc);
}

void cu_calculate_spring_damper_connecting_body_force(
	double* pos,
	double* vel,
	double* ep,
	double* ev,
	double* mass,
	double* force,
	double* moment,
	device_tsda_connection_body_data* xsdbcd,
	xSpringDamperCoefficient* xsdkc,
	unsigned int nc)
{
	computeGridSize(nc, CUDA_THREADS_PER_BLOCK, numBlocks, numThreads);
	calculate_spring_damper_connecting_body_force_kernel << <numBlocks, numThreads >> > (
		(double4*)pos,
		(double3*)vel,
		(double4*)ep,
		(double4*)ev,
		mass,
		(double3*)force,
		(double3*)moment,
		xsdbcd,
		xsdkc);
}