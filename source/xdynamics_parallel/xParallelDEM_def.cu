#include "xdynamics_parallel/xParallelDEM_decl.cuh"
#include "xdynamics_parallel/xParallelDEM_impl.cuh"
#include <helper_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
//#include <helper_cuda.h>

void setSymbolicParameter(device_parameters *h_paras)
{
	checkCudaErrors(cudaMemcpyToSymbol(cte, h_paras, sizeof(device_parameters)));
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

void vv_update_position(double *pos, double *vel, double *acc, unsigned int np)
{
	computeGridSize(np, 512, numBlocks, numThreads);
	vv_update_position_kernel << < numBlocks, numThreads >> >(
		(double4 *)pos,
		(double3 *)vel,
		(double3 *)acc,
		np);
}

void vv_update_velocity(double *vel, double *acc, double *omega, double *alpha, double *force, double *moment, double* mass, double* iner, unsigned int np)
{
	computeGridSize(np, 512, numBlocks, numThreads);
	vv_update_velocity_kernel << < numBlocks, numThreads >> >(
		(double3 *)vel,
		(double3 *)acc,
		(double3 *)omega,
		(double3 *)alpha,
		(double3 *)force,
		(double3 *)moment,
		mass,
		iner,
		np);
}

void cu_calculateHashAndIndex(
	unsigned int* hash,
	unsigned int* index,
	double *pos,
	unsigned int np)
{
	computeGridSize(np, 512, numBlocks, numThreads);
	calculateHashAndIndex_kernel << < numBlocks, numThreads >> >(hash, index, (double4 *)pos, np);
}

void cu_calculateHashAndIndexForPolygonSphere(
	unsigned int* hash,
	unsigned int* index,
	unsigned int sid,
	unsigned int nsphere,
	double *sphere)
{
	computeGridSize(nsphere, 512, numBlocks, numThreads);
	calculateHashAndIndexForPolygonSphere_kernel << <numBlocks, numThreads >> >(hash, index, sid, nsphere, (double4 *)sphere);
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
	thrust::sort_by_key(thrust::device_ptr<unsigned>(hash),
		thrust::device_ptr<unsigned>(hash + np),
		thrust::device_ptr<unsigned>(index));
	//std::cout << "step 2" << std::endl;
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

void cu_calculate_p2p(
	const int tcm, double* pos, double* vel,
	double* omega, double* force, double* moment,
	double* mass, unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, device_contact_property* cp, unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	switch (tcm)
	{
	case 0:
		calculate_p2p_kernel<0> << < numBlocks, numThreads >> >(
			(double4 *)pos, (double3 *)vel,
			(double3 *)omega, (double3 *)force,
			(double3 *)moment, mass,
			sorted_index, cstart,
			cend, cp, np);
		break;
	case 1:
		calculate_p2p_kernel<1> << < numBlocks, numThreads >> >(
			(double4 *)pos, (double3 *)vel,
			(double3 *)omega, (double3 *)force,
			(double3 *)moment, mass,
			sorted_index, cstart,
			cend, cp, np);
		break;
	}
}

void cu_plane_contact_force(
	const int tcm, device_plane_info* plan,
	double* pos, double* vel, double* omega,
	double* force, double* moment, double* mass,
	unsigned int np, device_contact_property *cp)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	switch (tcm)
	{
	case 0: plane_contact_force_kernel<0> << < numBlocks, numThreads >> >(
		plan, (double4 *)pos, (double3 *)vel, (double3 *)omega,
		(double3 *)force, (double3 *)moment, cp, mass, np);
		break;
	case 1: plane_contact_force_kernel<1> << < numBlocks, numThreads >> >(
		plan, (double4 *)pos, (double3 *)vel, (double3 *)omega,
		(double3 *)force, (double3 *)moment, cp, mass, np);
		break;
	}
}

void cu_cube_contact_force(
	const int tcm, device_plane_info* plan,
	double* pos, double* vel, double* omega,
	double* force, double* moment, double* mass,
	unsigned int np, device_contact_property *cp)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	for (unsigned int i = 0; i < 6; i++)
	{
		switch (tcm)
		{
		case 0: plane_contact_force_kernel<0> << < numBlocks, numThreads >> >(
			plan + i, (double4 *)pos, (double3 *)vel, (double3 *)omega,
			(double3 *)force, (double3 *)moment, cp, mass, np);
			break;
		case 1: plane_contact_force_kernel<1> << < numBlocks, numThreads >> >(
			plan + i, (double4 *)pos, (double3 *)vel, (double3 *)omega,
			(double3 *)force, (double3 *)moment, cp, mass, np);
			break;
		}
	}
}

void cu_cylinder_hertzian_contact_force(
	const int tcm, device_cylinder_info* cyl,
	double* pos, double* vel, double* omega,
	double* force, double* moment,
	double* mass, unsigned int np, device_contact_property *cp,
	double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm)
{
	computeGridSize(np, 512, numBlocks, numThreads);
	switch (tcm)
	{
	case 0: cylinder_hertzian_contact_force_kernel<0> << < numBlocks, numThreads >> >(
		cyl, (double4 *)pos, (double3 *)vel, (double3 *)omega,
		(double3 *)force, (double3 *)moment, cp, mass, mpos, mf, mm, np);
		break;
	case 1: cylinder_hertzian_contact_force_kernel<1> << < numBlocks, numThreads >> >(
		cyl, (double4 *)pos, (double3 *)vel, (double3 *)omega,
		(double3 *)force, (double3 *)moment, cp, mass, mpos, mf, mm, np);
		break;
	}

}

void cu_particle_meshObject_collision(
	const int tcm, device_mesh_info* dpi, double* dsph, device_mesh_mass_info* dpmi,
	double* pos, double* vel, double* omega,
	double* force, double* moment, double* mass,
	unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, device_contact_property *cp,
	unsigned int np)
{
	computeGridSize(np, 512, numBlocks, numThreads);
	switch (tcm)
	{
	case 1:
		particle_polygonObject_collision_kernel<1> << < numBlocks, numThreads >> >(
			dpi, (double4 *)dsph, dpmi,
			(double4 *)pos, (double3 *)vel, (double3 *)omega,
			(double3 *)force, (double3 *)moment, mass,
			sorted_index, cstart, cend, cp, np);
		break;
	}
}

double3 reductionD3(double3* in, unsigned int np)
{
	double3 rt = make_double3(0.0, 0.0, 0.0);
	computeGridSize(np, 512, numBlocks, numThreads);
	double3* d_out;
	double3* h_out = new double3[numBlocks];
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(double3) * numBlocks));
	checkCudaErrors(cudaMemset(d_out, 0, sizeof(double3) * numBlocks));
	//unsigned smemSize = sizeof(double3)*(512);
	reduce6<double3, 512> << < numBlocks, numThreads/*, smemSize*/ >> >(in, d_out, np);
	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(double3) * numBlocks, cudaMemcpyDeviceToHost));
	for (unsigned int i = 0; i < numBlocks; i++){
		rt.x += h_out[i].x;
		rt.y += h_out[i].y;
		rt.z += h_out[i].z;
	}
	delete[] h_out;
	checkCudaErrors(cudaFree(d_out));
	return rt;
}

void cu_update_meshObjectData(
	device_mesh_mass_info *dpmi, double* vList,
	double* sphere, device_mesh_info* dpi, unsigned int ntriangle)
{
	computeGridSize(ntriangle, 512, numBlocks, numThreads);
	updatePolygonObjectData_kernel << <numBlocks, numThreads >> >(dpmi, vList, (double4 *)sphere, dpi, ntriangle);
}

//void cu_check_no_collision_pair(
//	double* pos, unsigned int* pinfo,
//	unsigned int* pother, unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	check_no_collision_pair_kernel << <numBlocks, numThreads >> > (
//		(double4 *)pos,
//		(uint2 *)pinfo,
//		pother, np);
//}
//
//void cu_check_new_collision_pair(
//	double* pos, unsigned int* pinfo,
//	unsigned int* pdata, unsigned int* sorted_id,
//	unsigned int* cstart, unsigned int *cend, unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	check_new_collision_pair_kernel << <numBlocks, numThreads >> >(
//		(double4 *)pos,
//		(uint2 *)pinfo,
//		pdata,
//		sorted_id,
//		cstart,
//		cend,
//		np);
//}
//
//
//void cu_calculate_particle_collision_with_pair(
//	double* pos, double* vel, double* omega,
//	double* mass, double* ds, double* force,
//	double* moment, unsigned int* pinfo,
//	unsigned int* pother, device_contact_property* cp, unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	calculate_particle_collision_with_pair_kernel << < numBlocks, numThreads >> >(
//		(double4 *)pos,
//		(double3 *)vel,
//		(double3 *)omega,
//		mass, 
//		(double2 *)ds,
//		(double3 *)force,
//		(double3 *)moment,
//		(uint2 *)pinfo,
//		pother, cp, np);
//}

void cu_calculate_particle_particle_contact_count(
	double* pos, pair_data* pairs,
	unsigned int* old_pair_count, unsigned int* pair_count,
	unsigned int* pair_start, unsigned int* sorted_id,
	unsigned int* cell_start, unsigned int* cell_end, unsigned int np)
{
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

unsigned int cu_calculate_particle_plane_contact_count(
	device_plane_info *plane, pair_data* pairs,
	unsigned int* old_pair_count, unsigned int *count,
	unsigned int *sidx, double* pos, unsigned int nplanes, unsigned int np)
{
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
	unsigned int nc = thrust::reduce(thrust::device_ptr<unsigned int>(count), thrust::device_ptr<unsigned int> (count + np));
	return nc;
}
void cu_copy_old_to_new_pair(
	unsigned int *old_count, unsigned int *new_count,
	unsigned int* old_sidx, unsigned int* new_sidx,
	pair_data* old_pppd, pair_data* new_pppd,
	unsigned int nc, unsigned int np)
{
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
	double* pos, double* vel, double* omega, 
	double* mass, double* force, double* moment,
	pair_data* old_pairs, pair_data* pairs, 
	unsigned int *old_pair_count, unsigned int* pair_count, 
	unsigned int *old_pair_start, unsigned int *pair_start, int *type_count, 
	device_contact_property* cp, unsigned int *sorted_id, 
	unsigned int* cell_start, unsigned int* cell_end, unsigned int np)
{
	computeGridSize(np, 512, numBlocks, numThreads);
	new_particle_particle_contact_kernel << < numBlocks, numThreads >> >(
		(double4 *)pos,
		(double3 *)vel,
		(double3 *)omega,
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