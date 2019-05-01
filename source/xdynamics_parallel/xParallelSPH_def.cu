#include "xdynamics_parallel/xParallelSPH_decl.cuh"
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

__constant__ device_sph_parameters scte;

void setSPHSymbolicParameter(device_sph_parameters *h_paras)
{
	checkCudaErrors(cudaMemcpyToSymbol(scte, h_paras, sizeof(device_sph_parameters)));
}

//void cuBoundaryMoving(
//	unsigned long long int sid,
//	unsigned long long int pcount,
//	double stime,
//	double* pos,
//	double* pos0,
//	double* vel,
//	double* auxVel,
//	unsigned long long int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	cuBoundaryMoving_kernel << < numBlocks, numThreads >> >(
//		sid,
//		pcount,
//		stime,
//		(double3 *)pos,
//		(double3 *)pos0,
//		(double3 *)vel,
//		(double3 *)auxVel);
//}
//

__device__ unsigned int calcGridHashS(int3 cell)
{
	if (scte.dim == 3){
		return cell.x + (cell.y * scte.gridCellCount.x) + (cell.z * scte.gridCellCount.x * scte.gridCellCount.y);
	}
	return cell.y * scte.gridCellCount.x + cell.x;
}

// calculate position in uniform grid
__device__ int3 calcGridPosS(double3 p)
{
	int3 gridPos;
	gridPos.x = (int)floor((p.x - scte.gridMin.x) * scte.cellsizeInv);
	gridPos.y = (int)floor((p.y - scte.gridMin.y) * scte.cellsizeInv);
	gridPos.z = 0;
	if (scte.dim == 3)
		gridPos.z = (int)floor((p.z - scte.gridMin.z) * scte.cellsizeInv);

	return gridPos;
}

__device__ int3 LoopStart(double3 pos)
{
	if (scte.dim == 3){
		// 		int3 cell = calcGridPos(pos);
		// 		return make_int3(
		// 			//max(cell.x - 1, 0),
		// 			//max(cell.y - 1, 0),
		// 			//max(cell.z - 1, 0));
	}
	return calcGridPosS(pos - scte.kernel_support_radius);
}

__device__ int3 LoopEnd(double3 pos)
{
	if (scte.dim == 3){
		// 		int3 cell = calcGridPos(pos);
		// 		return make_int3(
		// 			//min(cell.x + 1, scte.gridCellCount.x - 1),
		// 			//min(cell.y + 1, scte.gridCellCount.y - 1),
		// 			//min(cell.z + 1, scte.gridCellCount.z - 1));
	}
	return calcGridPosS(pos + scte.kernel_support_radius);
}

__device__ double3 sphKernelGrad_Quintic(double QSq, double3 posDif)
{
	double Q = sqrt(QSq);
	if (Q < 1.0)
		return (scte.kernel_grad_const / Q) * (pow(3.0 - Q, 4.0) - 6 * pow(2.0 - Q, 4.0) + 15 * pow(1.0 - Q, 4.0)) * posDif;
	else if (Q < 2.0)
		return (scte.kernel_grad_const / Q) * (pow(3.0 - Q, 4.0) - 6 * pow(2.0 - Q, 4.0)) * posDif;
	else if (Q < 3.0)
		return (scte.kernel_grad_const / Q) * (pow(3.0 - Q, 4.0)) * posDif;
	return make_double3(0.0, 0.0, 0.0);
}

__device__ double sphKernel_Quintic(double QSq)
{
	double Q = sqrt(QSq);
	if (Q < 1.0)
		return scte.kernel_const * (pow(3.0 - Q, 5.0) - 6 * pow(2.0 - Q, 5.0) + 15 * pow(1.0 - Q, 5.0));
	else if (Q < 2.0)
		return scte.kernel_const * (pow(3.0 - Q, 5.0) - 6 * pow(2.0 - Q, 5.0));
	else if (Q < 3.0)
		return scte.kernel_const * pow(3.0 - Q, 5.0);

	return 0.0;
}

__device__ double3 sphKernelGrad_Quadratic(double QSq, double3 posDif)
{
	double Q = sqrt(QSq);
	if (Q < 0.5)
		return (scte.kernel_grad_const / Q) * (pow(2.5 - Q, 3.0) - 5.0 * pow(1.5 - Q, 3.0) + 10 * pow(0.5 - Q, 3.0)) * posDif;
	else if (Q < 1.5)
		return (scte.kernel_grad_const / Q) * (pow(2.5 - Q, 3.0) - 5.0 * pow(1.5 - Q, 3.0)) * posDif;
	else if (Q < 2.5)
		return (scte.kernel_grad_const / Q) * pow(2.5 - Q, 3.0) * posDif;

	return make_double3(0.0, 0.0, 0.0);
}

__device__ double3 sphKernelGrad_Cubic(double QSq, double3 posDif)
{
	double Q = sqrt(QSq);
	if (Q < 1.0)
		return scte.kernel_grad_const * (4.0 - 3.0 * Q) * (posDif);
	else{
		double dif = 2 - Q;
		return scte.kernel_grad_const * dif * dif * (posDif / Q);
	}
	//return make_double3(0.f);
}

__device__ double3 sphKernelGrad_Wendland(double QSq, double3 posDif)
{
	double Q = sqrt(QSq);
	if (Q <= 2.0)
		return (scte.kernel_grad_const / Q) * Q * pow((1 - 0.5 * Q), 3) * posDif;
	return make_double3(0.0, 0.0, 0.0);
}

__device__ double sphKernel(double QSq)
{
	double W;
	switch (scte.kernel){
	case QUINTIC_KERNEL: W = sphKernel_Quintic(QSq); break;
		// 	case CUBIC_SPLINE: gradW = sphKernel_Cubic(QSq); break;
		// 	case QUADRATIC: gradW = sphKernel_Quadratic(QSq); break;
		// 	case WENDLAND: gradW = sphKernel_Wendland(QSq); break;
	}
	return W;
}

__device__ double3 sphKernelGrad(double QSq, double3 posDif)
{
	double3 gradW;
	switch (scte.kernel){
	case QUINTIC_KERNEL: gradW = sphKernelGrad_Quintic(QSq, posDif); break;
	case CUBIC_SPLINE_KERNEL: gradW = sphKernelGrad_Cubic(QSq, posDif); break;
	case QUADRATIC_KERNEL: gradW = sphKernelGrad_Quadratic(QSq, posDif); break;
	case WENDLAND_KERNEL: gradW = sphKernelGrad_Wendland(QSq, posDif); break;
	}
	return gradW;
}

__global__ void calculateHashAndIndex_kernel(
	uint2* hash,
	unsigned int* index,
	double3* pos,
	unsigned int _np/* = scte.np*/)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= _np) return;
	double3 p = pos[id];

	int3 gridPos = calcGridPosS(p);
	unsigned int _hash = calcGridHashS(gridPos);

	hash[id] = make_uint2(_hash, id);
	index[id] = hash[id].x;
}

__global__ void reorderDataAndFindCellStart_kernel(
	uint2 *hashes, unsigned int *cell_start)
{
	extern __shared__ unsigned int sharedHash[];
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int _hash;
	if (id < scte.np){

		_hash = hashes[id].x;
		if (_hash > scte.cells)
			return;
		if (id == 0)
			cell_start[_hash] = 0;
		sharedHash[threadIdx.x + 1] = _hash;
		if (id>0 && threadIdx.x == 0)
			sharedHash[0] = hashes[id - 1].x;
	}
	__syncthreads();
	if (id < scte.np){
		if (id > 0 && _hash != sharedHash[threadIdx.x]){
			if (_hash > scte.cells)
				return;
			cell_start[_hash] = id;
		}
	}
}

//__global__ void kernel_correction_kernel(
//	double3* pos,
//	double6* corr,
//	double* mass,
//	xMaterialType* type,
//	uint2* hashes,
//	unsigned int* cell_start)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	xMaterialType tp = type[id];
//	if (tp == DUMMY)
//		return;
//	double3 p = pos[id];
//	double xx = 0.0;
//	double yy = 0.0;
//	double xy = 0.0;
//	double xz = 0.0;
//	double yz = 0.0;
//	double zz = 0.0;
//	double QSq = 0.0;
//	double m = 0.0;
//	double3 dp = make_double3(0.0, 0.0, 0.0);
//	double3 gradW = make_double3(0.0, 0.0, 0.0);
//	int3 cell_j = make_int3(0, 0, 0);
//	int3 cell = calcGridPos(p);
//	int3 loopStart = LoopStart(p);
//	int3 loopEnd = LoopEnd(p);
//
//	for (cell_j.z = loopStart.z; cell_j.z <= loopEnd.z; cell_j.z++){
//		for (cell_j.y = loopStart.y; cell_j.y <= loopEnd.y; cell_j.y++){
//			for (cell_j.x = loopStart.x; cell_j.x <= loopEnd.x; cell_j.x++){
//
//				unsigned int hash = calcGridHash(cell_j);
//				unsigned int j = cell_start[hash];
//				if (j != 0xffffffff){
//					for (uint2 hash2 = hashes[j]; hash == hash2.x; hash2 = hashes[j]){
//						if (id != hash2.y){
//							m = mass[hash2.y];
//							dp = p - pos[hash2.y];
//							QSq = dot(dp, dp) * scte.h_inv_sq;
//							if (QSq < scte.kernel_support_sq)
//							{
//								gradW = sphKernelGrad(QSq, dp);
//								xx -= dp.x * gradW.x * (m / scte.rho);
//								yy -= dp.y * gradW.y * (m / scte.rho);
//								xy -= dp.x * gradW.y * (m / scte.rho);
//								if (scte.dim == 3)
//								{
//									zz -= dp.z * gradW.z;
//									xz -= dp.x * gradW.z;
//									yz -= dp.y * gradW.z;
//								}
//							}
//						}
//						if ((++j) == scte.np)
//							break;
//					}
//				}
//			}
//		}
//	}
//	if (scte.dim == 3)
//	{
//		double det = 1.0 / (xx * (zz * yy - yz * yz) - xy * (zz * xy - yz * xz) + xz * (yz * xy - yy * xz));
//		if (abs(det) > 0.01)
//		{
//			corr[id] = make_sym_tensor(det*(zz*yy - yz*yz), det*(yz*xz - zz*xy), det*(yz*xy - yy*xz), det*(zz*xx - xz*xz), det*(xy*xz - yz*xx), det*(xx*yy - xy*xy));
//		}
//		else
//		{
//			corr[id] = make_sym_tensor(1, 0, 0, 1, 0, 1);
//		}
//	}
//	else
//	{
//		double det = 1.0 / (xx * yy - xy * xy);
//		if (abs(det) > 0.01)
//		{
//			corr[id] = make_sym_tensor(det*yy, det*(-xy), det*xx, 0.0, 0.0, 0.0);
//		}
//		else
//		{
//			corr[id] = make_sym_tensor(1, 0, 1, 0, 0, 0);
//		}
//	}
//}
//
//__device__ double3 correctGradientW(double3 gradW, double6 c)
//{
//	if (scte.dim == 3)
//	{
//		return make_double3(
//			gradW.x * c.s0 + gradW.y * c.s1 + gradW.z * c.s2,
//			gradW.x * c.s1 + gradW.y * c.s3 + gradW.z * c.s4,
//			gradW.x * c.s2 + gradW.y * c.s4 + gradW.z * c.s5
//			);
//	}
//	return make_double3(gradW.x * c.s0 + gradW.y * c.s1, gradW.x * c.s1 + gradW.y * c.s2, 0.0);
//}
//
//__global__ void setViscosityFreeSurfaceParticles_kernel(
//	double3* pos,
//	double* tbVisc,
//	bool* isf,
//	xMaterialType* type,
//	double* maxVel,
//	uint2* hashes,
//	unsigned int* cell_start)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	//if (!isf[id]) return;
//	xMaterialType tp = type[id];
//	//*maxVel = 2.0;
//	if (tp == DUMMY)
//		return;
//	if (isf[id])
//	{
//		tbVisc[id] += scte.peclet;
//		return;
//	}
//
//	double visc_td = 0.0;
//	double3 p = pos[id];
//	double3 dp = make_double3(0.0, 0.0, 0.0);
//	double QSq = 0;
//	//tbVisc[id] += scte.peclet;
//	int3 cell_j = make_int3(0, 0, 0);
//	int3 cell = calcGridPos(p);
//	int3 loopStart = LoopStart(p);
//	int3 loopEnd = LoopEnd(p);
//	for (cell_j.z = loopStart.z; cell_j.z <= loopEnd.z; cell_j.z++){
//		for (cell_j.y = loopStart.y; cell_j.y <= loopEnd.y; cell_j.y++){
//			for (cell_j.x = loopStart.x; cell_j.x <= loopEnd.x; cell_j.x++){
//				unsigned int hash = calcGridHash(cell_j);
//				unsigned int j = cell_start[hash];
//				if (j != 0xffffffff){
//					for (uint2 hash2 = hashes[j]; hash == hash2.x; hash2 = hashes[j]){
//						if (id != hash2.y){
//							dp = p - pos[hash2.y];
//							QSq = dot(dp, dp) * scte.h_inv_sq;
//							if (QSq < scte.kernel_support_sq)
//							{
//								if (isf[id] != true && isf[hash2.y] == true)
//								{
//									double dist = abs(dp.y);// sqrt(dot(dp, dp));
//									if (dist < 2.0*scte.particle_spacing)
//									{
//										visc_td = scte.peclet;
//									}
//									//visc_td = v < visc_td ? v : visc_td;
//								}
//							}
//						}
//						if ((++j) == scte.np)
//							break;
//					}
//				}
//			}
//		}
//	}
//	tbVisc[id] += visc_td;
//}
//
//__global__ void predict_the_acceleration_kernel(
//	double3* pos,
//	double3* vel,
//	double3* acc,
//	double6* corr,
//	double* tbVisc,
//	double* mass,
//	double* rho,
//	xMaterialType* type,
//	bool* isf,
//	uint2* hashes,
//	unsigned int* cell_start,
//	device_periodic_condition* dpc = NULL)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	xMaterialType tp = type[id];
//	if (tp == BOUNDARY || tp == DUMMY)
//		return;
//	double3 p = pos[id];
//	double3 dp = make_double3(0.0, 0.0, 0.0);
//	double3 v = vel[id];
//	double3 vj = make_double3(0.0, 0.0, 0.0);
//	double3 dv = make_double3(0.0, 0.0, 0.0);
//	double3 a = scte.gravity;
//	double3 gradW = make_double3(0.0, 0.0, 0.0);
//	double mj = 0.0;// mass[id];
//	double rho_i = rho[id];
//	double rho_j = 0.0;
//	double visc_ta = tbVisc[id];//
//	//if (isf[id] == false)
//	visc_ta = scte.viscosity;
//	//else
//	//	visc_ta = scte.peclet;
//	double visc_tb = 0.0;
//
//	double p1 = 0;
//	double p2 = 0;
//	double QSq = 0;
//	int3 cell_j = make_int3(0, 0, 0);
//	int3 cell = calcGridPos(p);
//	int3 loopStart = LoopStart(p);
//	int3 loopEnd = LoopEnd(p);
//	bool peri = false;
//	bool peri2 = false;
//	for (cell_j.z = loopStart.z; cell_j.z <= loopEnd.z; cell_j.z++){
//		for (cell_j.y = loopStart.y; cell_j.y <= loopEnd.y; cell_j.y++){
//			for (cell_j.x = loopStart.x; cell_j.x <= loopEnd.x; cell_j.x++){
//				if (dpc){
//					if (peri && cell_j.x == 1)
//						peri2 = true;
//
//					if (cell_j.x == scte.gridCellCount.x)
//					{
//						cell_j.x = 0;
//						peri = true;
//					}
//					else if (cell_j.x == 0)
//					{
//						cell_j.x = scte.gridCellCount.x - 1;
//						peri = true;
//					}
//				}
//
//				unsigned int hash = calcGridHash(cell_j);
//				unsigned int j = cell_start[hash];
//				if (j != 0xffffffff){
//					for (uint2 hash2 = hashes[j]; hash == hash2.x; hash2 = hashes[j]){
//						if (id != hash2.y){
//							dp = p - pos[hash2.y];
//							if (dpc && peri)
//								dp.x = p.x < pos[hash2.y].x ? dpc->limits.x + dp.x : -dpc->limits.x + dp.x;
//							QSq = dot(dp, dp) * scte.h_inv_sq;
//							if (QSq < scte.kernel_support_sq)
//							{
//								mj = mass[hash2.y];
//								gradW = sphKernelGrad(QSq, dp);
//								if (scte.corr == GRADIENT_CORRECTION)
//								{
//									gradW = correctGradientW(gradW, corr[id]);
//								}
//								rho_j = rho[hash2.y];
//								visc_tb = tbVisc[hash2.y];///*(tbVisc[hash2.y] && type[hash2.y] == FLUID) ? tbVisc[hash2.y] : */scte.viscosity;
//								/*	if (dpc && peri)
//								vj = (p.x < pos[hash2.y].x && type[hash2.y] == FLUID) ? dpc->velocity : vel[hash2.y];
//								else*/
//								vj = vel[hash2.y];
//								/*			if (isf[hash2.y])
//								visc_tb = scte.peclet;*/
//								dv = v - vj;
//								p1 = 8.0 * ((scte.viscosity + visc_ta) + (scte.viscosity + visc_tb)) / (rho_i + rho_j);
//								//p1 = (rho_i * (scte.viscosity + visc_ta) + rho_j * (scte.viscosity + visc_tb)) / (rho_i * rho_j);
//								p2 = dot(dv, dp) / (dot(dp, dp) + scte.dist_epsilon);
//								//p2 = dot(dp, gradW) / (dot(dp, dp) + scte.dist_epsilon);
//								a += mj * (p1 * p2) * gradW;
//								//a += mj * (p1 * p2) * dv;
//							}
//						}
//						if ((++j) == scte.np)
//							break;
//					}
//				}
//				if (dpc)
//				{
//					if (peri2)
//					{
//						cell_j.x = loopEnd.x;
//						peri = false;
//						peri2 = false;
//					}
//					if (peri && cell_j.x == scte.gridCellCount.x - 1)
//					{
//						cell_j.x = 0;
//						peri = false;
//					}
//				}
//			}
//		}
//	}
//	acc[id] = a;
//}
//
//__global__ void predict_the_temporal_position_kernel(
//	double3* pos,
//	double3* auxPos,
//	double3* vel,
//	xMaterialType* type)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	//tParticle tp = type[id];
//	/*if (tp == BOUNDARY || tp == DUMMY){
//	auxPos[id] =
//	return;
//	}*/
//
//	auxPos[id] = pos[id] + 0.5 * scte.dt * vel[id];
//}
//
//__global__ void predict_the_temporal_velocity_kernel(
//	double3* vel,
//	double3* auxVel,
//	double3* acc,
//	xMaterialType* type)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	xMaterialType tp = type[id];
//	if (tp == BOUNDARY || tp == DUMMY){
//		return;
//	}
//
//
//	auxVel[id] = vel[id] + scte.dt * acc[id];
//}
//
//__global__ void calculation_free_surface_kernel(
//	double3* pos,
//	double* press,
//	double* mass,
//	double* rho,
//	bool* isf,
//	double3* ufs,
//	bool* nearfs,
//	double* div_r,
//	xMaterialType* type,
//	uint2* hashes,
//	unsigned int* cell_start,
//	device_periodic_condition* dpc = NULL)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	xMaterialType tp = type[id];
//	if (tp == DUMMY)
//		return;
//	double3 p = pos[id];
//	double3 dp = make_double3(0.0, 0.0, 0.0);
//	double3 gradW = make_double3(0.0, 0.0, 0.0);
//	double mj = 0;
//	double QSq = 0;
//	double gp = 0;
//	double mdiv_r = 0;
//	double dr = 0;
//	double jrho = 0;
//	int3 cell_j = make_int3(0, 0, 0);
//	int3 cell = calcGridPos(p);
//	int3 loopStart = LoopStart(p);
//	int3 loopEnd = LoopEnd(p);
//	bool peri = false;
//	bool peri2 = false;
//	double3 gradC = make_double3(0, 0, 0);
//	unsigned int cnt = 0;
//	for (cell_j.z = loopStart.z; cell_j.z <= loopEnd.z; cell_j.z++){
//		for (cell_j.y = loopStart.y; cell_j.y <= loopEnd.y; cell_j.y++){
//			for (cell_j.x = loopStart.x; cell_j.x <= loopEnd.x; cell_j.x++){
//				if (dpc){
//					if (peri && cell_j.x == 1)
//						peri2 = true;
//
//					if (cell_j.x == scte.gridCellCount.x)
//					{
//						cell_j.x = 0;
//						peri = true;
//					}
//					else if (cell_j.x == 0)
//					{
//						cell_j.x = scte.gridCellCount.x - 1;
//						peri = true;
//					}
//				}
//				unsigned int hash = calcGridHash(cell_j);
//				unsigned int j = cell_start[hash];
//				if (j != 0xffffffff){
//					for (uint2 hash2 = hashes[j]; hash == hash2.x; hash2 = hashes[j]){
//						if (id != hash2.y){
//							dp = p - pos[hash2.y];
//							jrho = rho[hash2.y];
//							if (dpc && peri)
//								dp.x = p.x < pos[hash2.y].x ? dpc->limits.x + dp.x : -dpc->limits.x + dp.x;
//							mj = mass[hash2.y];
//							QSq = dot(dp, dp) * scte.h_inv_sq;
//							if (QSq < scte.kernel_support_sq)
//							{
//								gradW = sphKernelGrad(QSq, dp);
//								gp = dot(gradW, dp);
//								mdiv_r += dot(dp, dp);
//								dr -= (mj / jrho) * gp;
//								gradC += (mj / jrho) * gradW;
//								cnt++;
//							}
//						}
//						if ((++j) == scte.np)
//							break;
//					}
//				}
//				if (dpc){
//					if (peri2)
//					{
//						cell_j.x = loopEnd.x;
//						peri = false;
//						peri2 = false;
//					}
//					if (peri && cell_j.x == scte.gridCellCount.x - 1)
//					{
//						cell_j.x = 0;
//						peri = false;
//					}
//				}
//			}
//		}
//	}
//	div_r[id] = dr;//;mdiv_r / cnt;
//	if (dr < scte.freeSurfaceFactor){
//		isf[id] = true;
//		press[id] = 0.0;
//		if (tp == FLUID)
//			ufs[id] = gradC / length(gradC);
//	}
//	else{
//		isf[id] = false;
//	}
//	//if (tp == BOUNDARY){
//	//	double pr = press[id];
//	//	unsigned int j = id + 1;
//	//	while (j < scte.np && type[j] == DUMMY){
//	//		press[j] = pr;
//	//		j++;
//	//	}
//	//}
//}
//
//__global__ void ppe_right_hand_side_kernel(
//	double3* pos,
//	double3* auxVel,
//	double6* corr,
//	double* mass,
//	double* rho,
//	bool* fs,
//	xMaterialType* type,
//	uint2* hashes,
//	unsigned int* cell_start,
//	double* out,
//	device_periodic_condition* dpc = NULL)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	xMaterialType tp = type[id];
//	if (/*fs[id] || */tp == DUMMY){
//		out[id] = 0.0;
//		return;
//	}
//	if (fs[id] || tp == BOUNDARY)
//	{
//		out[id] = 0.0;
//		return;
//	}
//	double3 p = pos[id];
//	double3 v = auxVel[id];
//	double3 vj = make_double3(0.0, 0.0, 0.0);
//	double3 dp = make_double3(0.0, 0.0, 0.0);
//	double3 dv = make_double3(0.0, 0.0, 0.0);
//	double3 gradW = make_double3(0.0, 0.0, 0.0);
//	double mj = 0;
//	double QSq = 0;
//	double div_u = 0;
//	//double rhoi =
//	int3 cell_j = make_int3(0, 0, 0);
//	int3 cell = calcGridPos(p);
//	int3 loopStart = LoopStart(p);
//	int3 loopEnd = LoopEnd(p);
//	bool peri = false;
//	bool peri2 = false;
//	for (cell_j.z = loopStart.z; cell_j.z <= loopEnd.z; cell_j.z++){
//		for (cell_j.y = loopStart.y; cell_j.y <= loopEnd.y; cell_j.y++){
//			for (cell_j.x = loopStart.x; cell_j.x <= loopEnd.x; cell_j.x++){
//				if (dpc){
//					if (peri && cell_j.x == 1)
//						peri2 = true;
//
//					if (cell_j.x == scte.gridCellCount.x)
//					{
//						cell_j.x = 0;
//						peri = true;
//					}
//					else if (cell_j.x == 0)
//					{
//						cell_j.x = scte.gridCellCount.x - 1;
//						peri = true;
//					}
//				}
//				unsigned int hash = calcGridHash(cell_j);
//				unsigned int j = cell_start[hash];
//				if (j != 0xffffffff){
//					for (uint2 hash2 = hashes[j]; hash == hash2.x; hash2 = hashes[j]){
//						if (id != hash2.y){
//							/*	if (type[id] == INNER_DUMMY || type[hash2.y] == DUMMY){
//							if ((++j) == scte.np)
//							break;
//							continue;
//							}*/
//							/*	tParticle tp_j = type[hash2.y];
//							if ((tp == BOUNDARY && tp_j == FLOATING) || (tp == FLOATING && tp_j == BOUNDARY))
//							return;*/
//							dp = p - pos[hash2.y];
//							if (dpc && peri)
//								dp.x = p.x < pos[hash2.y].x ? dpc->limits.x + dp.x : -dpc->limits.x + dp.x;
//							/*			if (dpc && peri)
//							vj = (p.x < pos[hash2.y].x && type[hash2.y] == FLUID) ? dpc->velocity : auxVel[hash2.y];
//							else*/
//							vj = auxVel[hash2.y];
//							dv = v - vj;
//							mj = mass[hash2.y];
//							QSq = dot(dp, dp) * scte.h_inv_sq;
//							if (QSq < scte.kernel_support_sq)
//							{
//								gradW = sphKernelGrad(QSq, dp);
//								if (scte.corr == GRADIENT_CORRECTION)
//								{
//									gradW = correctGradientW(gradW, corr[id]);
//								}
//								div_u += mj * dot(dv, gradW);
//							}
//						}
//						if ((++j) == scte.np)
//							break;
//					}
//				}
//				if (dpc){
//					if (peri2)
//					{
//						cell_j.x = loopEnd.x;
//						peri = false;
//						peri2 = false;
//					}
//					if (peri && cell_j.x == scte.gridCellCount.x - 1)
//					{
//						cell_j.x = 0;
//						peri = false;
//					}
//				}
//			}
//		}
//	}
//	double rhoi = rho[id];
//	div_u *= -(1.0 / rhoi);
//	out[id] = (rhoi / scte.dt) * div_u;
//}
//
//__global__ void pressure_poisson_equation_kernel(
//	double3* pos,
//	double* press,
//	double6* corr,
//	double* mass,
//	double* rho,
//	bool* isf,
//	xMaterialType* type,
//	uint2* hashes,
//	unsigned int* cell_start,
//	double* out,
//	device_periodic_condition* dpc = NULL)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	xMaterialType tp = type[id];
//	if (tp == DUMMY || isf[id] == true){
//		out[id] = 0.0;
//		return;
//	}
//	//if (isf[id] && (tp == BOUNDARY/* || tp == FLOATING*/))
//	//{
//	//	out[id] = 0.0;
//	//	return;
//	//}
//	//if (isf[id] && (tp == BOUNDARY))
//	//{
//	//	out[id] = 0.0;
//	//	return;
//	//}
//	double3 p = pos[id];
//	double3 dp = make_double3(0.0, 0.0, 0.0);
//	double3 dv = make_double3(0.0, 0.0, 0.0);
//	double3 gradW = make_double3(0.0, 0.0, 0.0);
//	double mj = 0;
//	double ipress = press[id];
//	double jpress = 0.0;
//	double QSq = 0;
//	double dpress = 0;
//	double m_press = 0;
//
//	int3 cell_j = make_int3(0, 0, 0);
//	int3 cell = calcGridPos(p);
//	int3 loopStart = LoopStart(p);
//	int3 loopEnd = LoopEnd(p);
//	bool peri = false;
//	bool peri2 = false;
//	if ((tp == FLUID || tp == FLOATING) && isf[id])
//		ipress *= 2.0;
//
//	for (cell_j.z = loopStart.z; cell_j.z <= loopEnd.z; cell_j.z++){
//		for (cell_j.y = loopStart.y; cell_j.y <= loopEnd.y; cell_j.y++){
//			for (cell_j.x = loopStart.x; cell_j.x <= loopEnd.x; cell_j.x++){
//				if (dpc){
//					if (peri && cell_j.x == 1)
//						peri2 = true;
//
//					if (cell_j.x == scte.gridCellCount.x)
//					{
//						cell_j.x = 0;
//						peri = true;
//					}
//					else if (cell_j.x == 0)
//					{
//						cell_j.x = scte.gridCellCount.x - 1;
//						peri = true;
//					}
//				}
//				unsigned int hash = calcGridHash(cell_j);
//				unsigned int j = cell_start[hash];
//				if (j != 0xffffffff){
//					for (uint2 hash2 = hashes[j]; hash == hash2.x; hash2 = hashes[j]){
//						if (id != hash2.y){
//							/*						if (type[id] == BOUNDARY && (type[hash2.y] == INNER_DUMMY || type[hash2.y] == DUMMY)){
//							if ((++j) == scte.np)
//							break;
//							continue;
//							}*/
//							/*				tParticle tp_j = type[hash2.y];
//							if ((tp == BOUNDARY && tp_j == FLOATING) || (tp == FLOATING && tp_j == BOUNDARY))
//							return;*/
//							jpress = press[hash2.y];
//							// 							if(isf[hash2.y])
//							// 								jpress = 0.0;
//							dp = p - pos[hash2.y];
//							if (dpc && peri)
//								dp.x = p.x < pos[hash2.y].x ? dpc->limits.x + dp.x : -dpc->limits.x + dp.x;
//							dpress = ipress - jpress;
//							mj = mass[hash2.y];
//							QSq = dot(dp, dp) * scte.h_inv_sq;
//							if (QSq < scte.kernel_support_sq)
//							{
//								gradW = sphKernelGrad(QSq, dp);
//								if (scte.corr == GRADIENT_CORRECTION)
//								{
//									gradW = correctGradientW(gradW, corr[id]);
//								}
//								double mp = mj * (dpress * dot(dp, gradW)) / (dot(dp, dp) + scte.dist_epsilon);
//								m_press += mp;
//							}
//						}
//						if ((++j) == scte.np)
//							break;
//					}
//				}
//				if (dpc){
//					if (peri2)
//					{
//						cell_j.x = loopEnd.x;
//						peri = false;
//						peri2 = false;
//					}
//					if (peri && cell_j.x == scte.gridCellCount.x - 1)
//					{
//						cell_j.x = 0;
//						peri = false;
//					}
//				}
//			}
//		}
//	}
//	m_press *= 2.0 / rho[id];
//	out[id] = m_press;
//}
//
//__global__ void update_pressure_residual_kernel(
//	double* press,
//	double alpha,
//	double* conj0,
//	double omega,
//	double* conj1,
//	double* tmp1,
//	double* resi,
//	xMaterialType* type,
//	bool* isf)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	double p = press[id];
//	p = p + (alpha * conj0[id] + omega * conj1[id]);
//	press[id] = p;
//	resi[id] = conj1[id] - omega * tmp1[id];
//}
//
//__global__ void update_conjugate_kernel(
//	double* conj0,
//	double* resi,
//	double beta,
//	double omega,
//	double* tmp0,
//	xMaterialType* type)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	// 	if (type[id] == DUMMY)
//	// 	{
//	// 		conj0[id] = 0.0;
//	// 		return;
//	// 	}
//	conj0[id] = resi[id] + beta*(conj0[id] - omega * tmp0[id]);
//}
//
//__global__ void update_dummy_pressure_from_boundary_kernel(
//	double* press,
//	uint4* idn,
//	xMaterialType* type,
//	bool* isf)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//
//	double pr = press[id];
//	xMaterialType tp = type[id];
//	// 	if (isf[id])
//	// 	{
//	// 		press[id] = 0.0;
//	// 	}
//	//if (tp == INNER_DUMMY){
//	//	//if (scte.startInnerDummy)
//	//	//  		if (isf[id] && pr < 0)
//	//	//  			pr = press[id] = 0.0;
//
//	//	unsigned int idx = id - scte.startInnerDummy;
//	//	uint4 neigh = idn[idx];
//	//	double _pr = (press[neigh.x] + press[neigh.y] + press[neigh.z] + press[neigh.w]) * 0.25;
//	//	press[id] = _pr;
//	//}
//	if (tp == BOUNDARY)
//	{
//		unsigned int j = id + 1;
//		while (j < scte.np && type[j] == DUMMY){
//			press[j] = pr;
//			j++;
//		}
//	}
//	//else if (tp == FLOATING)
//	//{
//	//	unsigned int j = id + 1;
//	//	while (j < scte.np && type[j] == FLOATING_DUMMY){
//	//		press[j] = pr;
//	//		j++;
//	//	}
//	//}
//}
//
//__global__ void correct_by_adding_the_pressure_gradient_term_kernel(
//	double3* pos,
//	double3* auxPos,
//	double3* vel,
//	double3* auxVel,
//	double3* acc,
//	double3* ufs,
//	double6* corr,
//	bool* isf,
//	double* mass,
//	double* rho,
//	double* press,
//	xMaterialType* type,
//	uint2* hashes,
//	unsigned int* cell_start,
//	device_periodic_condition* dpc = NULL)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	xMaterialType tp = type[id];
//	if (tp == BOUNDARY || tp == DUMMY)
//		return;
//	//if (tp == MOVING_BOUNDARY)
//	//{
//	//	type[id] = BOUNDARY;
//	//	return;
//	//}
//	// 	if (isf[id] == true)
//	// 		isf[id] == true;
//	double3 p = pos[id];
//	double3 jp = make_double3(0.0, 0.0, 0.0);
//	double3 dp = make_double3(0.0, 0.0, 0.0);
//	double3 gradW = make_double3(0.0, 0.0, 0.0);
//	double mj = 0;
//	double ipress = press[id];
//	double jpress = 0.0;
//	double irho = rho[id];
//	double jrho = 0.0;
//	double QSq = 0;
//	double pij = 0;
//	//	bool fs = isf[id];
//	//	double3 uf = ufs[id];
//	double3 gradp = make_double3(0.0, 0.0, 0.0);
//
//	int3 cell_j = make_int3(0, 0, 0);
//	int3 cell = calcGridPos(p);
//	int3 loopStart = LoopStart(p);
//	int3 loopEnd = LoopEnd(p);
//	bool peri = false;
//	bool peri2 = false;
//	for (cell_j.z = loopStart.z; cell_j.z <= loopEnd.z; cell_j.z++){
//		for (cell_j.y = loopStart.y; cell_j.y <= loopEnd.y; cell_j.y++){
//			for (cell_j.x = loopStart.x; cell_j.x <= loopEnd.x; cell_j.x++){
//				if (dpc){
//					if (peri && cell_j.x == 1)
//						peri2 = true;
//
//					if (cell_j.x == scte.gridCellCount.x)
//					{
//						cell_j.x = 0;
//						peri = true;
//					}
//					else if (cell_j.x == 0)
//					{
//						cell_j.x = scte.gridCellCount.x - 1;
//						peri = true;
//					}
//				}
//				unsigned int hash = calcGridHash(cell_j);
//				unsigned int j = cell_start[hash];
//				if (j != 0xffffffff){
//					for (uint2 hash2 = hashes[j]; hash == hash2.x; hash2 = hashes[j]){
//						if (id != hash2.y)
//						{
//							//if (tp >= FLOATING && tp != type[hash2.y])
//							//{
//							//	++j;
//							//	continue;
//							//}
//							/*	tParticle tp_j = type[hash2.y];
//							if ((tp == FLOATING && tp_j == DUMMY) || (tp == FLOATING && tp_j == BOUNDARY))
//							{
//							if ((++j) == scte.np)
//							break;
//							continue;
//							}
//							*/
//							jpress = press[hash2.y];
//							// 							if (tp == FLOATING && type[hash2.y] == FLOATING)
//							// 							{
//							// 								++j;
//							// 								continue;
//							// 							}
//							jp = pos[hash2.y];
//							jrho = rho[hash2.y];
//							dp = p - jp;
//							if (dpc && peri)
//								dp.x = p.x < pos[hash2.y].x ? dpc->limits.x + dp.x : -dpc->limits.x + dp.x;
//							pij = (jpress + ipress) / (irho * jrho);
//							mj = /*type[hash2.y] == FLOATING ? 0.004 : */mass[hash2.y];
//							QSq = dot(dp, dp) * scte.h_inv_sq;
//							if (QSq < scte.kernel_support_sq)
//							{
//								gradW = sphKernelGrad(QSq, dp);
//								if (scte.corr == GRADIENT_CORRECTION)
//								{
//									gradW = correctGradientW(gradW, corr[id]);
//								}
//								gradp += mj * pij * gradW;
//							}
//						}
//						if ((++j) == scte.np)
//							break;
//					}
//				}
//				if (dpc){
//					if (peri2)
//					{
//						cell_j.x = loopEnd.x;
//						peri = false;
//						peri2 = false;
//					}
//					if (peri && cell_j.x == scte.gridCellCount.x - 1)
//					{
//						cell_j.x = 0;
//						peri = false;
//					}
//				}
//			}
//		}
//	}
//	acc[id] = acc[id] - gradp - scte.gravity;
//	double3 nv = auxVel[id] - (scte.dt) * gradp;
//	if (tp == FLUID)
//		p = p + scte.dt * nv;//;/ +(scte.isShifting ? shiftedPos[id] : make_double3(0.f));
//	//p = p + 0.5 * scte.dt * (nv + vel[id]);
//	if (dpc)
//	{
//		if (p.x > dpc->limits.x)
//		{
//			p.x -= dpc->limits.x;
//
//		}
//		if (p.x < 0.5)
//		{
//			nv.x = (6.0 / pow(5.0, 2.0)) * dpc->velocity.x * p.y * (5.0 - p.y);
//		}
//
//		///*if (p.x < 0.5)
//		//	nv.x =*/ dpc->velocity.x;
//	}
//	pos[id] = p;
//	vel[id] = nv;
//}
//
//__global__ void sinusoidal_expression_kernel(
//	device_sinusoidal_expression* dse,
//	double3* initpos,
//	double3* pos,
//	double3* vel,
//	double3* auxVel,
//	xMaterialType* tp,
//	double time)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id < dse->sid || id >= (dse->sid + dse->count)) return;
//	//	double3 ini_p = initpos[id];
//	//double3 old_p = pos[id];
//	double3 new_p = make_double3(0.0, 0.0, 0.0);
//	double3 new_v = make_double3(0.0, 0.0, 0.0);
//	//double dt = time - dse->stime;
//	//	double fq = dse->freq;
//	double mv = 2.0 * M_PI * 0.75;
//	//double dx = dse->c1 * sin(fq * dse->period * 0.75) + dse->c2*sin(fq * dse->period * 0.75);
//	new_p.x = 0.5 * dse->c1 * sin(dse->freq * (time - dse->stime) + mv) + dse->c1 * 0.5;//dse->c1 * sin(fq * dt + fq * dse->period * 0.75) + dse->c1;// +dse->c2 * sin(2.0*fq*dt + fq * dse->period * 0.75) - dx;// 0.5 * dse->stroke * sin(fq * (time - dse->stime) + fq * dse->period * 0.75)/* + dse->stroke * 0.5*/;
//	new_v.x = 0.5 * dse->c1 * dse->freq * cos(dse->freq * (time - dse->stime) + mv);// dse->c1 * fq * cos(fq * dt + dse->period * 0.75);// +2.0*dse->c2 * fq * cos(2.0 * fq * dt);
//	pos[id] += new_p;
//	vel[id] += new_v;
//	auxVel[id] += new_v;
//	//tp[id] = MOVING_BOUNDARY;
//}
//
////__global__ void sinusoidal_expressionbydata_kernel(
////	unsigned int sid,
////	unsigned int count,
////	tExpression *dexps,
////	double3 *initpos,
////	double3 *pos,
////	double3 *vel,
////	double3 *auxVel,
////	unsigned int step)
////{
////	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
////	if (id < sid || id >= (sid + count)) return;
////	double3 ini_p = initpos[id];
////	//double3 old_p = pos[id];
////	double3 new_p = make_double3(0.0, 0.0, 0.0);
////	double3 new_v = make_double3(0.0, 0.0, 0.0);
////
////	new_p.x = dexps[step].p;
////	new_v.x = dexps[step].v;
////	pos[id] = ini_p + new_p;
////	vel[id] = new_v;
////	auxVel[id] = new_v;
////}
//
////__global__ void linear_expression_kernel(
////	unsigned int sid,
////	unsigned int count,
////	double3 *initPos,
////	double3 *pos,
////	double3 *vel,
////	double3 *auxVel,
////	double time)
////{
////	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
////	if (id < sid || id >= (sid + count)) return;
////	double3 ini_p = initPos[id];
////	//double3 old_p = pos[id];
////	double3 new_p = make_double3(0.0, 0.0, 0.0);
////	double3 new_v = make_double3(0.0, 0.0, 0.0);
////	double gradient = 0.01; // 기울기 입력 *******************************************
////	new_p.x = gradient * time;
////	new_v.x = gradient;
////	pos[id] = ini_p + new_p;
////	vel[id] = new_v;
////	auxVel[id] = new_v;
////}
//
//__global__ void simple_sin_expression_kernel(
//	device_simple_sin_expression* dse,
//	double3* initpos,
//	double3* pos,
//	double3* vel,
//	double3* auxVel,
//	double time)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id < dse->sid || id >= (dse->sid + dse->count)) return;
//	double3 ini_p = initpos[id];
//	double3 new_p = make_double3(0.0, 0.0, 0.0);
//	double3 new_v = make_double3(0.0, 0.0, 0.0);
//	new_p.x = dse->amp * sin(dse->freq * (time - dse->stime));
//	new_v.x = dse->amp * dse->freq * cos(dse->freq * (time - dse->stime));
//	pos[id] = ini_p + new_p;
//	vel[id] = new_v;
//	auxVel[id] = new_v;
//}
//
//
//
//__global__ void wave_damping_formula_kernel(
//	device_damping_condition* ddc,
//	double3* pos,
//	double3* vel,
//	double3* auxVel)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= scte.np) return;
//	double3 p = pos[id];
//	if (p.x < ddc->start_point) return;
//	double sq = -ddc->alpha * (ddc->length - (p.x - ddc->start_point));
//	double fx = 1 - exp(sq);
//	double3 new_v = fx * vel[id];
//	vel[id] = new_v;
//	//auxVel[id] = new_v;
//}
//
//__global__ void particle_spacing_average_kernel(
//	double3* pos, xMaterialType* type, bool *isf,
//	unsigned int *cell_start, uint2* hashes, double* avr, device_periodic_condition* dpc = NULL)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= scte.np) return;
//	if (type[id] != FLUID)
//		return;
//	if (isf[id] == true)
//		return;
//	double sum = 0.0;
//	int cnt = 0;
//	double3 p = pos[id];
//	int3 cell_j = make_int3(0, 0, 0);
//	int3 cell = calcGridPos(p);
//	int3 loopStart = LoopStart(p);
//	int3 loopEnd = LoopEnd(p);
//	double QSq = 0.0;
//	bool peri = false;
//	bool peri2 = false;
//	for (cell_j.z = loopStart.z; cell_j.z <= loopEnd.z; cell_j.z++){
//		for (cell_j.y = loopStart.y; cell_j.y <= loopEnd.y; cell_j.y++){
//			for (cell_j.x = loopStart.x; cell_j.x <= loopEnd.x; cell_j.x++){
//				if (dpc){
//					if (peri && cell_j.x == 1)
//						peri2 = true;
//
//					if (cell_j.x == scte.gridCellCount.x)
//					{
//						cell_j.x = 0;
//						peri = true;
//					}
//					else if (cell_j.x == 0)
//					{
//						cell_j.x = scte.gridCellCount.x - 1;
//						peri = true;
//					}
//				}
//				unsigned int hash = calcGridHash(cell_j);
//				unsigned int j = cell_start[hash];
//				if (j != 0xffffffff){
//					for (uint2 hash2 = hashes[j]; hash == hash2.x; hash2 = hashes[j]){
//						if (id != hash2.y){
//							double3 dp = p - pos[hash2.y];
//							if (dpc && peri)
//								dp.x = p.x < pos[hash2.y].x ? dpc->limits.x + dp.x : -dpc->limits.x + dp.x;
//							QSq = dot(dp, dp) * scte.h_inv_sq;
//							if (QSq < scte.kernel_support_sq)
//							{
//								sum += length(dp);
//								cnt++;
//							}
//						}
//						if ((++j) == scte.np)
//							break;
//					}
//				}
//				if (dpc){
//					if (peri2)
//					{
//						cell_j.x = loopEnd.x;
//						peri = false;
//						peri2 = false;
//					}
//					if (peri && cell_j.x == scte.gridCellCount.x - 1)
//					{
//						cell_j.x = 0;
//						peri = false;
//					}
//				}
//			}
//		}
//	}
//	avr[id] = sum / cnt;
//}
//
////__global__ void set_particle_rearrange_without_dummy_free_surface(
////	double3* pos,
////	double3
////	unsigned int* idx)
////{
////
////}
//
//__global__ void particle_shifting_kernel(
//	double3 *shiftedPos,
//	double3 *pos,
//	double3 *shift,
//	double *avr,
//	double *maxVel,
//	double *mass,
//	double *press,
//	double *rho,
//	xMaterialType *type,
//	double *div_r,
//	bool *isf,
//	device_periodic_condition* dpc,
//	uint2 *hashes,
//	unsigned int *cell_start)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= scte.np) return;
//	if (type[id] != FLUID)
//		return;
//	if (isf[id] == true)
//		return;
//	double3 dp = make_double3(0.0, 0.0, 0.0);
//	double3 gradW = make_double3(0.0, 0.0, 0.0);
//	double QSq = 0.0;
//	double3 p = pos[id];
//	int3 cell_j = make_int3(0, 0, 0);
//	int3 cell = calcGridPos(p);
//	int3 loopStart = LoopStart(p);
//	int3 loopEnd = LoopEnd(p);
//	double alpha = *maxVel * scte.dt;
//	double3 R = make_double3(0.0, 0.0, 0.0);
//	bool peri = false;
//	bool peri2 = false;
//	//double avr = particle_spacing_average(id, pos, loopStart, loopEnd, cell_start, hashes);
//	for (cell_j.z = loopStart.z; cell_j.z <= loopEnd.z; cell_j.z++){
//		for (cell_j.y = loopStart.y; cell_j.y <= loopEnd.y; cell_j.y++){
//			for (cell_j.x = loopStart.x; cell_j.x <= loopEnd.x; cell_j.x++){
//				if (dpc){
//					if (peri && cell_j.x == 1)
//						peri2 = true;
//
//					if (cell_j.x == scte.gridCellCount.x)
//					{
//						cell_j.x = 0;
//						peri = true;
//					}
//					else if (cell_j.x == 0)
//					{
//						cell_j.x = scte.gridCellCount.x - 1;
//						peri = true;
//					}
//				}
//				unsigned int hash = calcGridHash(cell_j);
//				unsigned int j = cell_start[hash];
//				if (j != 0xffffffff){
//					for (uint2 hash2 = hashes[j]; hash == hash2.x; hash2 = hashes[j]){
//						if (id != hash2.y){
//							dp = p - pos[hash2.y];
//							if (dpc && peri)
//								dp.x = p.x < pos[hash2.y].x ? dpc->limits.x + dp.x : -dpc->limits.x + dp.x;
//							QSq = dot(dp, dp) * scte.h_inv_sq;
//							if (QSq < scte.kernel_support_sq){
//								double r = pow(avr[id], 2.0) / pow(length(dp), 2.0);
//								double3 nij = dp / length(dp);
//								R += r * nij;
//							}
//						}
//						if ((++j) == scte.np)
//							break;
//					}
//				}
//				if (dpc){
//					if (peri2)
//					{
//						cell_j.x = loopEnd.x;
//						peri = false;
//						peri2 = false;
//					}
//					if (peri && cell_j.x == scte.gridCellCount.x - 1)
//					{
//						cell_j.x = 0;
//						peri = false;
//					}
//				}
//			}
//		}
//	}
//	double3 shift_r = scte.shifting_factor * alpha * R;
//	shiftedPos[id] = p + shift_r;
//	shift[id] = shift_r;
//}
//
//__global__ void particle_shifting_update_kernel(
//	double3* pos,
//	double3* new_vel,
//	double* new_press,
//	double3* old_vel,
//	double* old_press,
//	double3* shift,
//	double* mass,
//	double* rho,
//	xMaterialType* type,
//	bool* isf,
//	device_periodic_condition* dpc,
//	uint2* hashes,
//	unsigned int *cell_start)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= scte.np) return;
//	if (type[id] != FLUID)
//		return;
//	if (isf[id] == true)
//		return;
//	double3 dp = make_double3(0.0, 0.0, 0.0);
//	double3 gradW = make_double3(0.0, 0.0, 0.0);
//	double QSq = 0.0;
//	double ipress = old_press[id];
//	double3 ipos = pos[id];
//	double3 ivel = old_vel[id];
//	double3 gp = make_double3(0.0, 0.0, 0.0);
//	double3 gvx = make_double3(0.0, 0.0, 0.0);
//	double3 gvy = make_double3(0.0, 0.0, 0.0);
//	double3 gvz = make_double3(0.0, 0.0, 0.0);
//	int3 cell_j = make_int3(0, 0, 0);
//	int3 cell = calcGridPos(ipos);
//	int3 loopStart = LoopStart(ipos);
//	int3 loopEnd = LoopEnd(ipos);
//	bool peri = false;
//	bool peri2 = false;
//	for (cell_j.z = loopStart.z; cell_j.z <= loopEnd.z; cell_j.z++){
//		for (cell_j.y = loopStart.y; cell_j.y <= loopEnd.y; cell_j.y++){
//			for (cell_j.x = loopStart.x; cell_j.x <= loopEnd.x; cell_j.x++){
//				if (dpc){
//					if (peri && cell_j.x == 1)
//						peri2 = true;
//
//					if (cell_j.x == scte.gridCellCount.x)
//					{
//						cell_j.x = 0;
//						peri = true;
//					}
//					else if (cell_j.x == 0)
//					{
//						cell_j.x = scte.gridCellCount.x - 1;
//						peri = true;
//					}
//				}
//				unsigned int hash = calcGridHash(cell_j);
//				unsigned int j = cell_start[hash];
//				if (j != 0xffffffff){
//					for (uint2 hash2 = hashes[j]; hash == hash2.x; hash2 = hashes[j]){
//						if (id != hash2.y){
//							dp = ipos - pos[hash2.y];
//							if (dpc && peri)
//								dp.x = ipos.x < pos[hash2.y].x ? dpc->limits.x + dp.x : -dpc->limits.x + dp.x;
//							QSq = dot(dp, dp) * scte.h_inv_sq;
//							if (QSq < scte.kernel_support_sq)
//							{
//								double3 jvel = old_vel[hash2.y];
//								/*					if (dpc && peri)
//								jvel = (ipos.x < pos[hash2.y].x && type[hash2.y] == FLUID) ? dpc->velocity : old_vel[hash2.y];
//								*/
//								double jpress = old_press[hash2.y];
//								double jmass = mass[hash2.y];
//								double jrho = rho[hash2.y];
//								gradW = sphKernelGrad(QSq, dp);
//								gradW = (jmass / jrho) * gradW;
//								gp += (ipress + jpress) * gradW;
//								gvx += (jvel.x - ivel.x) * gradW;
//								gvy += (jvel.y - ivel.y) * gradW;
//								gvz += (jvel.z - ivel.z) * gradW;
//							}
//						}
//						if ((++j) == scte.np)
//							break;
//					}
//				}
//				if (dpc){
//					if (peri2)
//					{
//						cell_j.x = loopEnd.x;
//						peri = false;
//						peri2 = false;
//					}
//					if (peri && cell_j.x == scte.gridCellCount.x - 1)
//					{
//						cell_j.x = 0;
//						peri = false;
//					}
//				}
//			}
//		}
//	}
//	double3 dr = shift[id];
//	new_press[id] += dot(dr, gp);
//	double3 n_vel = make_double3(dot(gvx, dr), dot(gvy, dr), dot(gvz, dr));
//	new_vel[id] += n_vel;// make_double3(dot(gvx, dr), dot(gvy, dr), dot(gvz, dr));
//}
//
//
//
////template <int BLOCKSIZE>
////void __global__ findMaxWithVector3(double3* inputvals, double* outputvals, int N)
////{
////	__shared__ volatile double data[BLOCKSIZE];
////	double maxval = sqrt(dot(inputvals[threadIdx.x]));
////	for (int i = blockDim.x + threadIdx.x; i < N; i += blockDim.x)
////	{
////		maxval = maxfunc(maxval, sqrt(dot(inputvals[i])));
////	}
////	data[threadIdx.x] = maxval;
////	__syncthreads();
////	if (threadIdx.x < 32) {
////		for (int i = 32 + threadIdx.x; i < BLOCKSIZE; i += 32){
////			data[threadIdx.x] = maxfunc(data[threadIdx.x], data[i]);
////		}
////		if (threadIdx.x < 16) data[threadIdx.x] = maxfunc(data[threadIdx.x], data[threadIdx.x + 16]);
////		if (threadIdx.x < 8) data[threadIdx.x] = maxfunc(data[threadIdx.x], data[threadIdx.x + 8]);
////		if (threadIdx.x < 4) data[threadIdx.x] = maxfunc(data[threadIdx.x], data[threadIdx.x + 4]);
////		if (threadIdx.x < 2) data[threadIdx.x] = maxfunc(data[threadIdx.x], data[threadIdx.x + 2]);
////		if (threadIdx.x == 0){
////			data[0] = maxfunc(data[0], data[1]);
////			outputvals[threadIdx.x] = data[0];
////		}
////	}
////}
//
//void __global__ mixingLengthTurbulence_kernel(
//	double3* pos,
//	double3* vel,
//	double6* corr,
//	double* tbVisc,
//	xMaterialType* type,
//	uint2* hashes,
//	unsigned int *cell_start,
//	device_periodic_condition* dpc)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	xMaterialType tp = type[id];
//	if (tp == DUMMY){
//		tbVisc[id] = 0.0;
//		return;
//	}
//
//	double3 p = pos[id];
//	double3 v = vel[id];
//	double3 dp = make_double3(0.0, 0.0, 0.0);
//	double3 dv = make_double3(0.0, 0.0, 0.0);
//	double3 gradW = make_double3(0.0, 0.0, 0.0);
//	double QSq = 0;
//	double Sa = 0.0;
//	int3 cell_j = make_int3(0, 0, 0);
//	int3 cell = calcGridPos(p);
//	int3 loopStart = LoopStart(p);
//	int3 loopEnd = LoopEnd(p);
//	bool peri = false;
//	bool peri2 = false;
//	for (cell_j.z = loopStart.z; cell_j.z <= loopEnd.z; cell_j.z++){
//		for (cell_j.y = loopStart.y; cell_j.y <= loopEnd.y; cell_j.y++){
//			for (cell_j.x = loopStart.x; cell_j.x <= loopEnd.x; cell_j.x++){
//				if (dpc){
//					if (peri && cell_j.x == 1)
//						peri2 = true;
//
//					if (cell_j.x == scte.gridCellCount.x)
//					{
//						cell_j.x = 0;
//						peri = true;
//					}
//					else if (cell_j.x == 0)
//					{
//						cell_j.x = scte.gridCellCount.x - 1;
//						peri = true;
//					}
//				}
//				unsigned int hash = calcGridHash(cell_j);
//				unsigned int j = cell_start[hash];
//				if (j != 0xffffffff){
//					for (uint2 hash2 = hashes[j]; hash == hash2.x; hash2 = hashes[j]){
//						if (id != hash2.y){
//							// 							if (type[hash2.y] == DUMMY)
//							// 							{
//							// 								++j;
//							// 								continue;
//							// 							}
//
//							//jpress = press[hash2.y];
//							dp = p - pos[hash2.y];
//							dv = v - vel[hash2.y];
//							if (dpc && peri)
//								dp.x = p.x < pos[hash2.y].x ? dpc->limits.x + dp.x : -dpc->limits.x + dp.x;
//							//pij = (jpress + ipress) / (scte.rho * scte.rho);
//							//pij = (ipress / (scte.rho * scte.rho)) + (jpress / (scte.rho * scte.rho));
//							//mj = mass[hash2.y];
//							QSq = dot(dp, dp) * scte.h_inv_sq;
//							if (QSq < scte.kernel_support_sq)
//							{
//								gradW = sphKernelGrad(QSq, dp);
//								if (scte.corr == GRADIENT_CORRECTION)
//								{
//									gradW = correctGradientW(gradW, corr[id]);
//								}
//								double v1 = (scte.rho + scte.rho) / (scte.rho * scte.rho);
//								double v2 = dot(dv, dv) / (dot(dp, dp) + scte.dist_epsilon);
//								Sa -= 0.5 * v1 * v2 * dot(dp, gradW);
//							}
//						}
//						if ((++j) == scte.np)
//							break;
//					}
//				}
//				if (dpc){
//					if (peri2)
//					{
//						cell_j.x = loopEnd.x;
//						peri = false;
//						peri2 = false;
//					}
//					if (peri && cell_j.x == scte.gridCellCount.x - 1)
//					{
//						cell_j.x = 0;
//						peri = false;
//					}
//				}
//			}
//		}
//	}
//	tbVisc[id] = pow(scte.particle_spacing, 2.0) * sqrt(Sa);
//}
//
////template <typename T, unsigned int blockSize>
////__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n)
////{
////	/*extern*/ __shared__ T sdata[512];
////	unsigned int tid = threadIdx.x;
////	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
////	unsigned int gridSize = blockSize * 2 * gridDim.x;
////
////	T mySum = make_double3(0, 0, 0);;
////	//sdata[tid] = make_double3(0, 0, 0);
////
////	while (i < n)
////	{
////		//sdata[tid] += g_idata[i] + g_idata[i + blockSize]; 
////		mySum += g_idata[i];
////		if (i + blockSize < n)
////			mySum += g_idata[i + blockSize];
////		i += gridSize;
////	}
////	sdata[tid] = mySum;
////	__syncthreads();
////	if ((blockSize >= 512) && (tid < 256)) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads();
////	if ((blockSize >= 256) && (tid < 128)) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();
////	if ((blockSize >= 128) && (tid < 64)) { sdata[tid] = mySum = mySum + sdata[tid + 64]; } __syncthreads();
////	if ((blockSize >= 64) && (tid < 32)){ sdata[tid] = mySum = mySum + sdata[tid + 32]; } __syncthreads();
////	if ((blockSize >= 32) && (tid < 16)){ sdata[tid] = mySum = mySum + sdata[tid + 16]; } __syncthreads();
////
////	if ((blockSize >= 16) && (tid < 8))
////	{
////		sdata[tid] = mySum = mySum + sdata[tid + 8];
////	}
////
////	__syncthreads();
////
////	if ((blockSize >= 8) && (tid < 4))
////	{
////		sdata[tid] = mySum = mySum + sdata[tid + 4];
////	}
////
////	__syncthreads();
////
////	if ((blockSize >= 4) && (tid < 2))
////	{
////		sdata[tid] = mySum = mySum + sdata[tid + 2];
////	}
////
////	__syncthreads();
////
////	if ((blockSize >= 2) && (tid < 1))
////	{
////		sdata[tid] = mySum = mySum + sdata[tid + 1];
////	}
////
////	__syncthreads();
////
////	if (tid == 0) g_odata[blockIdx.x] = mySum;
////}
//
//__global__ void cuReplaceDataByID_kernel(
//	double3 *opos,
//	double3 *oavel,
//	double *omass,
//	double *opress,
//	double3 *ipos,
//	double3 *iavel,
//	double *imass,
//	double *ipress,
//	unsigned int *m_id)
//{
//	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= (scte.np)) return;
//	unsigned int rid = m_id[id];
//	if (rid == 0 && id != 0) return;
//	opos[rid] = ipos[id];
//	oavel[rid] = iavel[id];
//	omass[rid] = imass[id];
//	opress[rid] = ipress[id];
//}
//

void cu_sph_calculateHashAndIndex(unsigned int *hashes, unsigned int *cell_id, double *pos, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	//ulonglong2* _hashes = (ulonglong2 *)hashes;
	calculateHashAndIndex_kernel << < numBlocks, numThreads >> >(
		(uint2 *)hashes,
		cell_id,
		(double3 *)pos,
		np);

	thrust::sort_by_key(thrust::device_ptr<unsigned int>(cell_id),
		thrust::device_ptr<unsigned int>(cell_id + np),
		thrust::device_ptr<uint2>((uint2 *)hashes));
}
//
void cu_sph_reorderDataAndFindCellStart(unsigned int *hashes, unsigned int* cell_start, unsigned int np, unsigned int nc)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	checkCudaErrors(cudaMemset(cell_start, 0xffffffff, nc * sizeof(unsigned int)));
	unsigned int smemSize = sizeof(unsigned int) * (numThreads + 1);
	reorderDataAndFindCellStart_kernel << < numBlocks, numThreads, smemSize >> >(
		(uint2 *)hashes,
		cell_start);
}

//void cuKernelCorrection(
//	double* pos,
//	double* corr,
//	double* mass,
//	xMaterialType* type,
//	unsigned int* hashes,
//	unsigned int* cell_start,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	kernel_correction_kernel << < numBlocks, numThreads >> >(
//		(double3 *)pos,
//		(double6 *)corr,
//		mass,
//		type,
//		(uint2 *)hashes,
//		cell_start);
//}
//
//void cuSetViscosityFreeSurfaceParticles(
//	double* pos,
//	double* tbVisc,
//	bool* isf,
//	xMaterialType* type,
//	double* maxVel,
//	unsigned int* hashes,
//	unsigned int* cell_start,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	setViscosityFreeSurfaceParticles_kernel << <numBlocks, numThreads >> >(
//		(double3 *)pos,
//		tbVisc,
//		isf,
//		type,
//		maxVel,
//		(uint2 *)hashes,
//		cell_start);
//}
//
//void cuPredict_the_acceleration(
//	double* pos,
//	double* vel,
//	double* acc,
//	double* mass,
//	double* rho,
//	xMaterialType* type,
//	bool* isf,
//	double* corr,
//	double* tbVisc,
//	unsigned int* hashes,
//	unsigned int* cell_start,
//	unsigned int np,
//	device_periodic_condition* dpc)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	predict_the_acceleration_kernel << <numBlocks, numThreads >> >(
//		(double3 *)pos,
//		(double3 *)vel,
//		(double3 *)acc,
//		(double6 *)corr,
//		tbVisc,
//		mass,
//		rho,
//		type,
//		isf,
//		(uint2 *)hashes,
//		cell_start,
//		dpc);
//}
//
//void cuPredict_the_position(double *pos, double *auxPos, double *vel, xMaterialType* type, unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	predict_the_temporal_position_kernel << < numBlocks, numThreads >> >(
//		(double3 *)pos,
//		(double3 *)auxPos,
//		(double3 *)vel,
//		type);
//}
//
//void cuPredict_the_temporal_velocity(
//	double* vel,
//	double* auxVel,
//	double* acc,
//	xMaterialType* type,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	predict_the_temporal_velocity_kernel << < numBlocks, numThreads >> >(
//		(double3 *)vel,
//		(double3 *)auxVel,
//		(double3 *)acc,
//		type);
//}
//
//void cuCalculation_free_surface(
//	double* pos,
//	double* press,
//	double* mass,
//	double* rho,
//	bool* isf,
//	double* ufs,
//	bool* nearfs,
//	double* div_r,
//	xMaterialType* tp,
//	unsigned int* hashes,
//	unsigned int* cell_start,
//	unsigned int np,
//	device_periodic_condition* dpc)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	calculation_free_surface_kernel << < numBlocks, numThreads >> >(
//		(double3 *)pos,
//		press,
//		mass,
//		rho,
//		isf,
//		(double3 *)ufs,
//		nearfs,
//		div_r,
//		//NULL,
//		tp,
//		(uint2 *)hashes,
//		cell_start,
//		dpc);
//}
//
//void cuCalculation_free_surface_with_shifting(
//	double* pos,
//	double* press,
//	double* mass,
//	bool* isf,
//	double* div_r,
//	double* shiftedPos,
//	xMaterialType* tp,
//	unsigned int* hashes,
//	unsigned int* cell_start,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	// 	calculation_free_surface_kernel << < numBlocks, numThreads >> >(
//	// 		(double3 *)pos,
//	// 		press,
//	// 		mass,
//	// 		isf,
//	// 		div_r,
//	// 		//(double3 *)shiftedPos,
//	// 		tp,
//	// 		(ulonglong2 *)hashes,
//	// 		cell_start);
//}
//
//void cuPPE_right_hand_side(
//	double* pos,
//	double* auxVel,
//	double* corr,
//	double* mass,
//	double* rho,
//	bool* fs,
//	xMaterialType* type,
//	unsigned int* hashes,
//	unsigned int* cell_start,
//	double* out,
//	unsigned int np,
//	device_periodic_condition* dpc)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	ppe_right_hand_side_kernel << < numBlocks, numThreads >> >(
//		(double3 *)pos,
//		(double3 *)auxVel,
//		(double6 *)corr,
//		mass,
//		rho,
//		fs,
//		type,
//		(uint2 *)hashes,
//		cell_start,
//		out,
//		dpc);
//	// 	double* h_lhs = new double[np];
//	// 	cudaMemcpy(h_lhs, out, sizeof(double) * np, cudaMemcpyDeviceToHost);
//	// 	delete[] h_lhs;
//}
//
//void cuPressure_poisson_equation(
//	double* pos,
//	double* press,
//	double* corr,
//	double* mass,
//	double* rho,
//	bool* isf,
//	xMaterialType* type,
//	unsigned int* hashes,
//	unsigned int* cell_start,
//	double* out,
//	unsigned int np,
//	device_periodic_condition* dpc)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	pressure_poisson_equation_kernel << < numBlocks, numThreads >> >(
//		(double3 *)pos,
//		press,
//		(double6 *)corr,
//		mass,
//		rho,
//		isf,
//		type,
//		(uint2 *)hashes,
//		cell_start,
//		out,
//		dpc);
//}
//
//void cuUpdate_pressure_residual(
//	double* press,
//	double alpha,
//	double* conj0,
//	double omega,
//	double* conj1,
//	double* tmp1,
//	double* resi,
//	xMaterialType* type,
//	bool* isf,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	update_pressure_residual_kernel << < numBlocks, numThreads >> >(
//		press,
//		alpha,
//		conj0,
//		omega,
//		conj1,
//		tmp1,
//		resi,
//		type,
//		isf);
//}
//
//void cuUpdate_conjugate(double* conj0, double* resi, double beta, double omega, double* tmp0, xMaterialType* type, unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	update_conjugate_kernel << < numBlocks, numThreads >> >(
//		conj0,
//		resi,
//		beta,
//		omega,
//		tmp0,
//		type);
//}
//
//void cuUpdate_dummy_pressure_from_boundary(double* press, unsigned int* innerDummyNeighbors, xMaterialType* type, bool* isf, unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	update_dummy_pressure_from_boundary_kernel << < numBlocks, numThreads >> >(
//		press,
//		(uint4 *)innerDummyNeighbors,
//		type,
//		isf);
//}
//
//void cuCorrect_by_adding_the_pressure_gradient_term(
//	double* pos,
//	double* auxPos,
//	double* vel,
//	double* auxVel,
//	double* acc,
//	double* ufs,
//	double* corr,
//	double* mass,
//	double* rho,
//	double* press,
//	bool* isf,
//	xMaterialType* type,
//	unsigned int* hashes,
//	unsigned int* cell_start,
//	unsigned int np,
//	device_periodic_condition* dpc)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	correct_by_adding_the_pressure_gradient_term_kernel << < numBlocks, numThreads >> >(
//		(double3 *)pos,
//		(double3 *)auxPos,
//		(double3 *)vel,
//		(double3 *)auxVel,
//		(double3 *)acc,
//		(double3 *)ufs,
//		(double6 *)corr,
//		isf,
//		mass,
//		rho,
//		press,
//		type,
//		(uint2 *)hashes,
//		cell_start,
//		dpc);
//}
//
//void cuSinusoidalExpression(
//	device_sinusoidal_expression *dse,
//	double* initpos,
//	double* pos,
//	double* vel,
//	double* auxVel,
//	xMaterialType* type,
//	double time,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	sinusoidal_expression_kernel << < numBlocks, numThreads >> >(
//		dse,
//		(double3 *)initpos,
//		(double3 *)pos,
//		(double3 *)vel,
//		(double3 *)auxVel,
//		type,
//		time);
//}
//
////void cuSinusoidalExpressionByData(
////	unsigned int sid,
////	unsigned int count,
////	tExpression* dexps,
////	double* initPos,
////	double* pos,
////	double *vel,
////	double* auxVel,
////	unsigned int step, unsigned int np)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	sinusoidal_expressionbydata_kernel << < numBlocks, numThreads >> >(
////		sid,
////		count,
////		dexps,
////		(double3 *)initPos,
////		(double3 *)pos,
////		(double3 *)vel,
////		(double3 *)auxVel,
////		step);
////}
//
////void cuLinearExpression(
////	unsigned int sid,
////	unsigned int count,
////	double* initPos,
////	double* pos,
////	double *vel,
////	double* auxVel,
////	double time,
////	unsigned int np)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	linear_expression_kernel << < numBlocks, numThreads >> >(
////		sid,
////		count,
////		(double3 *)initPos,
////		(double3 *)pos,
////		(double3 *)vel,
////		(double3 *)auxVel,
////		time);
////}
//
////void cuSimpleSinExpression(
////	device_simple_sin_expression *dse,
////	double* initpos,
////	double* pos,
////	double* vel,
////	double* auxVel,
////	double time, unsigned int np)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	simple_sin_expression_kernel << < numBlocks, numThreads >> >(
////		dse,
////		(double3 *)initpos,
////		(double3 *)pos,
////		(double3 *)vel,
////		(double3 *)auxVel,
////		time);
////}
//
//void cuWave_damping_formula(
//	device_damping_condition* ddc,
//	double* pos,
//	double* vel,
//	double* auxVel,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	wave_damping_formula_kernel << <numBlocks, numThreads >> >(
//		ddc,
//		(double3 *)pos,
//		(double3 *)vel,
//		(double3 *)auxVel);
//}
//
//void cuParticleSpacingAverage(
//	double* pos,
//	xMaterialType* type,
//	bool *isf,
//	unsigned int *cell_start,
//	unsigned int *hashes,
//	double *avr,
//	device_periodic_condition* dpc,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//
//	particle_spacing_average_kernel << <numBlocks, numThreads >> >(
//		(double3 *)pos,
//		type,
//		isf,
//		cell_start,
//		(uint2 *)hashes,
//		avr, dpc);
//}
//
//void cuParticle_shifting(
//	double* shiftedPos,
//	double* pos,
//	double* shift,
//	double* avr,
//	double* maxVel,
//	double* mass,
//	double* press,
//	double* rho,
//	xMaterialType *type,
//	double* div_r,
//	bool* isf,
//	device_periodic_condition* dpc,
//	unsigned int* hashes,
//	unsigned int* cell_start,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//
//	particle_shifting_kernel << <numBlocks, numThreads >> >(
//		(double3 *)shiftedPos,
//		(double3 *)pos,
//		(double3 *)shift,
//		avr,
//		maxVel,
//		mass,
//		press,
//		rho,
//		type,
//		div_r,
//		isf,
//		dpc,
//		(uint2 *)hashes,
//		cell_start);
//}
//
//void cuParticle_shifting_update(
//	double* pos,
//	double* new_vel,
//	double* new_press,
//	double* old_vel,
//	double* old_press,
//	double* shift,
//	double* mass,
//	double* rho,
//	xMaterialType* type,
//	bool* isf,
//	device_periodic_condition* dpc,
//	unsigned int *hashes,
//	unsigned int *cell_start,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	particle_shifting_update_kernel << < numBlocks, numThreads >> >(
//		(double3 *)pos,
//		(double3 *)new_vel,
//		new_press,
//		(double3 *)old_vel,
//		old_press,
//		(double3 *)shift,
//		mass,
//		rho,
//		type,
//		isf,
//		dpc,
//		(uint2 *)hashes,
//		cell_start);
//}
//
//
//
//void cuMixingLengthTurbulence(double *pos, double *vel, double* corr, double *tbVisc, xMaterialType* type, unsigned int* hashes, unsigned int* cell_start, unsigned int np, device_periodic_condition* dpc)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	mixingLengthTurbulence_kernel << < numBlocks, numThreads >> >(
//		(double3 *)pos,
//		(double3 *)vel,
//		(double6 *)corr,
//		tbVisc,
//		type,
//		(uint2 *)hashes,
//		cell_start,
//		dpc);
//
//}
//
//
//
//
//void cuReplaceDataByID(
//	double* m_pos,
//	double* m_avel,
//	double* m_mass,
//	double* m_press,
//	double* pos,
//	double* avel,
//	double* mass,
//	double* press,
//	unsigned int* m_id,
//	unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	cuReplaceDataByID_kernel << < numBlocks, numThreads >> >
//		((double3*)m_pos
//		, (double3*)m_avel
//		, m_mass
//		, m_press
//		, (double3*)pos
//		, (double3*)avel
//		, mass
//		, press
//		, m_id
//		);
//}

////void cuPPE_right_hand_side2(double* pos, double* auxVel, double* mass, unsigned int* hashes, unsigned int* cell_start, double* out, unsigned int np)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	ppe_right_hand_side_kernel2 << < numBlocks, numThreads >> >(
////		(double3 *)pos,
////		(double3 *)auxVel,
////		mass,
////		(uint2 *)hashes,
////		cell_start,
////		out,
////		np);
////}
////
////void cuPressure_poisson_equation2(double* pos, double* press, double* mass, unsigned int* hashes, unsigned int* cell_start, double* out, unsigned int np)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	pressure_poisson_equation_kernel2 << < numBlocks, numThreads >> >(
////		(double3 *)pos,
////		press,
////		mass,
////		(uint2 *)hashes,
////		cell_start,
////		out,
////		np);
////}
////
////void cuUpdate_pressure_residual2(
////	double* press,
////	double alpha,
////	double* conj0,
////	double omega,
////	double* conj1,
////	double* tmp1,
////	double* resi,
////	unsigned int np)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	update_pressure_residual_kernel2 << < numBlocks, numThreads >> >(
////		press,
////		alpha,
////		conj0,
////		omega,
////		conj1,
////		tmp1,
////		resi,
////		np);
////}
////
////void cuUpdate_conjugate2(double* conj0, double* resi, double beta, double omega, double* tmp0, unsigned int np)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	update_conjugate_kernel2 << < numBlocks, numThreads >> >(
////		conj0,
////		resi,
////		beta,
////		omega,
////		tmp0,
////		np);
////}
////
////void cuUpdate_dummy_pressure_from_boundary2(double* m_press, double* press, unsigned int* innerDummyNeighbors, xMaterialType* type, bool* isf, unsigned int* m_id, unsigned int np)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	update_dummy_pressure_from_boundary_kernel2 << < numBlocks, numThreads >> >(
////		m_press,
////		press,
////		(uint4 *)innerDummyNeighbors,
////		type,
////		isf,
////		m_id);
////}
////
////void cuContact_force_circle_boundary(
////	double* pos, xMaterialType* type, device_pointMass_info* dpmi,
////	device_circle_info* dci, device_contact_parameters* dcp,
////	unsigned int* hashes, unsigned int* cell_start, unsigned int np)
////{
////	//computeGridSize(np, 512, numBlocks, numThreads);
////	//contact_force_circle_boundary_kernel << < numBlocks, numThreads>> >(
////	//	(double3 *)pos,
////	//	type
////	//	dpmi,
////	//	dci,
////	//	dcp,
////	//	(uint2 *)hashes,
////	//	cell_start);
////}
////
////void cuContactDistance(double* pos, xMaterialType* type,
////	device_circle_info* dci, unsigned int* hashes, unsigned int* cell_start,
////	unsigned int* cid, double *dist, unsigned int np, unsigned int nc)
////{
////	computeGridSize(np, 512, numBlocks, numThreads);
////	contact_distance_kernel << < numBlocks, numThreads >> >(
////		(double3 *)pos,
////		type,
////		dci,
////		(uint2 *)hashes,
////		cell_start,
////		cid,
////		dist,
////		nc);
////	thrust::sort_by_key(thrust::device_ptr<unsigned int>(cid),
////		thrust::device_ptr<unsigned int>(cid + nc),
////		thrust::device_ptr<double>(dist));
////}
////
