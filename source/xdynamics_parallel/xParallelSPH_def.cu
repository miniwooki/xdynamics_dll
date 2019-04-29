#include "xdynamics_parallel/xParallelSPH_impl.cuh"
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>



//Round a / b to nearest higher integer value


// compute grid and thread block size for a given number of elements


void cuBoundaryMoving(
	unsigned long long int sid,
	unsigned long long int pcount,
	double stime,
	double* pos,
	double* pos0,
	double* vel,
	double* auxVel,
	unsigned long long int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	cuBoundaryMoving_kernel << < numBlocks, numThreads >> >(
		sid,
		pcount,
		stime,
		(double3 *)pos,
		(double3 *)pos0,
		(double3 *)vel,
		(double3 *)auxVel);
}

void cuCalcHashValue(unsigned int *hashes, unsigned int *cell_id, double *pos, unsigned int np)
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

void cuReorderDataAndFindCellStart(unsigned int *hashes, unsigned int* cell_start, unsigned int np, unsigned int nc)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	checkCudaErrors(cudaMemset(cell_start, 0xffffffff, nc * sizeof(unsigned int)));
	unsigned int smemSize = sizeof(unsigned int) * (numThreads + 1);
	reorderDataAndFindCellStart_kernel << < numBlocks, numThreads, smemSize >> >(
		(uint2 *)hashes,
		cell_start);
}

void cuKernelCorrection(
	double* pos,
	double* corr,
	double* mass,
	xMaterialType* type,
	unsigned int* hashes,
	unsigned int* cell_start,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	kernel_correction_kernel << < numBlocks, numThreads >> >(
		(double3 *)pos,
		(double6 *)corr,
		mass,
		type,
		(uint2 *)hashes,
		cell_start);
}

void cuSetViscosityFreeSurfaceParticles(
	double* pos,
	double* tbVisc,
	bool* isf,
	xMaterialType* type,
	double* maxVel,
	unsigned int* hashes,
	unsigned int* cell_start,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	setViscosityFreeSurfaceParticles_kernel << <numBlocks, numThreads >> >(
		(double3 *)pos,
		tbVisc,
		isf,
		type,
		maxVel,
		(uint2 *)hashes,
		cell_start);
}

void cuPredict_the_acceleration(
	double* pos,
	double* vel,
	double* acc,
	double* mass,
	double* rho,
	xMaterialType* type,
	bool* isf,
	double* corr,
	double* tbVisc,
	unsigned int* hashes,
	unsigned int* cell_start,
	unsigned int np,
	device_periodic_condition* dpc)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	predict_the_acceleration_kernel << <numBlocks, numThreads >> >(
		(double3 *)pos,
		(double3 *)vel,
		(double3 *)acc,
		(double6 *)corr,
		tbVisc,
		mass,
		rho,
		type,
		isf,
		(uint2 *)hashes,
		cell_start,
		dpc);
}

void cuPredict_the_position(double *pos, double *auxPos, double *vel, xMaterialType* type, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	predict_the_temporal_position_kernel << < numBlocks, numThreads >> >(
		(double3 *)pos,
		(double3 *)auxPos,
		(double3 *)vel,
		type);
}

void cuPredict_the_temporal_velocity(
	double* vel,
	double* auxVel,
	double* acc,
	xMaterialType* type,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	predict_the_temporal_velocity_kernel << < numBlocks, numThreads >> >(
		(double3 *)vel,
		(double3 *)auxVel,
		(double3 *)acc,
		type);
}

void cuCalculation_free_surface(
	double* pos,
	double* press,
	double* mass,
	double* rho,
	bool* isf,
	double* ufs,
	bool* nearfs,
	double* div_r,
	xMaterialType* tp,
	unsigned int* hashes,
	unsigned int* cell_start,
	unsigned int np,
	device_periodic_condition* dpc)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	calculation_free_surface_kernel << < numBlocks, numThreads >> >(
		(double3 *)pos,
		press,
		mass,
		rho,
		isf,
		(double3 *)ufs,
		nearfs,
		div_r,
		//NULL,
		tp,
		(uint2 *)hashes,
		cell_start,
		dpc);
}

void cuCalculation_free_surface_with_shifting(
	double* pos,
	double* press,
	double* mass,
	bool* isf,
	double* div_r,
	double* shiftedPos,
	xMaterialType* tp,
	unsigned int* hashes,
	unsigned int* cell_start,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	// 	calculation_free_surface_kernel << < numBlocks, numThreads >> >(
	// 		(double3 *)pos,
	// 		press,
	// 		mass,
	// 		isf,
	// 		div_r,
	// 		//(double3 *)shiftedPos,
	// 		tp,
	// 		(ulonglong2 *)hashes,
	// 		cell_start);
}

void cuPPE_right_hand_side(
	double* pos,
	double* auxVel,
	double* corr,
	double* mass,
	double* rho,
	bool* fs,
	xMaterialType* type,
	unsigned int* hashes,
	unsigned int* cell_start,
	double* out,
	unsigned int np,
	device_periodic_condition* dpc)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	ppe_right_hand_side_kernel << < numBlocks, numThreads >> >(
		(double3 *)pos,
		(double3 *)auxVel,
		(double6 *)corr,
		mass,
		rho,
		fs,
		type,
		(uint2 *)hashes,
		cell_start,
		out,
		dpc);
	// 	double* h_lhs = new double[np];
	// 	cudaMemcpy(h_lhs, out, sizeof(double) * np, cudaMemcpyDeviceToHost);
	// 	delete[] h_lhs;
}

void cuPressure_poisson_equation(
	double* pos,
	double* press,
	double* corr,
	double* mass,
	double* rho,
	bool* isf,
	xMaterialType* type,
	unsigned int* hashes,
	unsigned int* cell_start,
	double* out,
	unsigned int np,
	device_periodic_condition* dpc)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	pressure_poisson_equation_kernel << < numBlocks, numThreads >> >(
		(double3 *)pos,
		press,
		(double6 *)corr,
		mass,
		rho,
		isf,
		type,
		(uint2 *)hashes,
		cell_start,
		out,
		dpc);
}

void cuUpdate_pressure_residual(
	double* press,
	double alpha,
	double* conj0,
	double omega,
	double* conj1,
	double* tmp1,
	double* resi,
	xMaterialType* type,
	bool* isf,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	update_pressure_residual_kernel << < numBlocks, numThreads >> >(
		press,
		alpha,
		conj0,
		omega,
		conj1,
		tmp1,
		resi,
		type,
		isf);
}

void cuUpdate_conjugate(double* conj0, double* resi, double beta, double omega, double* tmp0, xMaterialType* type, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	update_conjugate_kernel << < numBlocks, numThreads >> >(
		conj0,
		resi,
		beta,
		omega,
		tmp0,
		type);
}

void cuUpdate_dummy_pressure_from_boundary(double* press, unsigned int* innerDummyNeighbors, xMaterialType* type, bool* isf, unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	update_dummy_pressure_from_boundary_kernel << < numBlocks, numThreads >> >(
		press,
		(uint4 *)innerDummyNeighbors,
		type,
		isf);
}

void cuCorrect_by_adding_the_pressure_gradient_term(
	double* pos,
	double* auxPos,
	double* vel,
	double* auxVel,
	double* acc,
	double* ufs,
	double* corr,
	double* mass,
	double* rho,
	double* press,
	bool* isf,
	xMaterialType* type,
	unsigned int* hashes,
	unsigned int* cell_start,
	unsigned int np,
	device_periodic_condition* dpc)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	correct_by_adding_the_pressure_gradient_term_kernel << < numBlocks, numThreads >> >(
		(double3 *)pos,
		(double3 *)auxPos,
		(double3 *)vel,
		(double3 *)auxVel,
		(double3 *)acc,
		(double3 *)ufs,
		(double6 *)corr,
		isf,
		mass,
		rho,
		press,
		type,
		(uint2 *)hashes,
		cell_start,
		dpc);
}

void cuSinusoidalExpression(
	device_sinusoidal_expression *dse,
	double* initpos,
	double* pos,
	double* vel,
	double* auxVel,
	xMaterialType* type,
	double time,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	sinusoidal_expression_kernel << < numBlocks, numThreads >> >(
		dse,
		(double3 *)initpos,
		(double3 *)pos,
		(double3 *)vel,
		(double3 *)auxVel,
		type,
		time);
}

//void cuSinusoidalExpressionByData(
//	unsigned int sid,
//	unsigned int count,
//	tExpression* dexps,
//	double* initPos,
//	double* pos,
//	double *vel,
//	double* auxVel,
//	unsigned int step, unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	sinusoidal_expressionbydata_kernel << < numBlocks, numThreads >> >(
//		sid,
//		count,
//		dexps,
//		(double3 *)initPos,
//		(double3 *)pos,
//		(double3 *)vel,
//		(double3 *)auxVel,
//		step);
//}

//void cuLinearExpression(
//	unsigned int sid,
//	unsigned int count,
//	double* initPos,
//	double* pos,
//	double *vel,
//	double* auxVel,
//	double time,
//	unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	linear_expression_kernel << < numBlocks, numThreads >> >(
//		sid,
//		count,
//		(double3 *)initPos,
//		(double3 *)pos,
//		(double3 *)vel,
//		(double3 *)auxVel,
//		time);
//}

//void cuSimpleSinExpression(
//	device_simple_sin_expression *dse,
//	double* initpos,
//	double* pos,
//	double* vel,
//	double* auxVel,
//	double time, unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	simple_sin_expression_kernel << < numBlocks, numThreads >> >(
//		dse,
//		(double3 *)initpos,
//		(double3 *)pos,
//		(double3 *)vel,
//		(double3 *)auxVel,
//		time);
//}

void cuWave_damping_formula(
	device_damping_condition* ddc,
	double* pos,
	double* vel,
	double* auxVel,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	wave_damping_formula_kernel << <numBlocks, numThreads >> >(
		ddc,
		(double3 *)pos,
		(double3 *)vel,
		(double3 *)auxVel);
}

void cuParticleSpacingAverage(
	double* pos,
	xMaterialType* type,
	bool *isf,
	unsigned int *cell_start,
	unsigned int *hashes,
	double *avr,
	device_periodic_condition* dpc,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);

	particle_spacing_average_kernel << <numBlocks, numThreads >> >(
		(double3 *)pos,
		type,
		isf,
		cell_start,
		(uint2 *)hashes,
		avr, dpc);
}

void cuParticle_shifting(
	double* shiftedPos,
	double* pos,
	double* shift,
	double* avr,
	double* maxVel,
	double* mass,
	double* press,
	double* rho,
	xMaterialType *type,
	double* div_r,
	bool* isf,
	device_periodic_condition* dpc,
	unsigned int* hashes,
	unsigned int* cell_start,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);

	particle_shifting_kernel << <numBlocks, numThreads >> >(
		(double3 *)shiftedPos,
		(double3 *)pos,
		(double3 *)shift,
		avr,
		maxVel,
		mass,
		press,
		rho,
		type,
		div_r,
		isf,
		dpc,
		(uint2 *)hashes,
		cell_start);
}

void cuParticle_shifting_update(
	double* pos,
	double* new_vel,
	double* new_press,
	double* old_vel,
	double* old_press,
	double* shift,
	double* mass,
	double* rho,
	xMaterialType* type,
	bool* isf,
	device_periodic_condition* dpc,
	unsigned int *hashes,
	unsigned int *cell_start,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	particle_shifting_update_kernel << < numBlocks, numThreads >> >(
		(double3 *)pos,
		(double3 *)new_vel,
		new_press,
		(double3 *)old_vel,
		old_press,
		(double3 *)shift,
		mass,
		rho,
		type,
		isf,
		dpc,
		(uint2 *)hashes,
		cell_start);
}



void cuMixingLengthTurbulence(double *pos, double *vel, double* corr, double *tbVisc, xMaterialType* type, unsigned int* hashes, unsigned int* cell_start, unsigned int np, device_periodic_condition* dpc)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	mixingLengthTurbulence_kernel << < numBlocks, numThreads >> >(
		(double3 *)pos,
		(double3 *)vel,
		(double6 *)corr,
		tbVisc,
		type,
		(uint2 *)hashes,
		cell_start,
		dpc);

}




void cuReplaceDataByID(
	double* m_pos,
	double* m_avel,
	double* m_mass,
	double* m_press,
	double* pos,
	double* avel,
	double* mass,
	double* press,
	unsigned int* m_id,
	unsigned int np)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(np, 512, numBlocks, numThreads);
	cuReplaceDataByID_kernel << < numBlocks, numThreads >> >
		((double3*)m_pos
		, (double3*)m_avel
		, m_mass
		, m_press
		, (double3*)pos
		, (double3*)avel
		, mass
		, press
		, m_id
		);
}

//void cuPPE_right_hand_side2(double* pos, double* auxVel, double* mass, unsigned int* hashes, unsigned int* cell_start, double* out, unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	ppe_right_hand_side_kernel2 << < numBlocks, numThreads >> >(
//		(double3 *)pos,
//		(double3 *)auxVel,
//		mass,
//		(uint2 *)hashes,
//		cell_start,
//		out,
//		np);
//}
//
//void cuPressure_poisson_equation2(double* pos, double* press, double* mass, unsigned int* hashes, unsigned int* cell_start, double* out, unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	pressure_poisson_equation_kernel2 << < numBlocks, numThreads >> >(
//		(double3 *)pos,
//		press,
//		mass,
//		(uint2 *)hashes,
//		cell_start,
//		out,
//		np);
//}
//
//void cuUpdate_pressure_residual2(
//	double* press,
//	double alpha,
//	double* conj0,
//	double omega,
//	double* conj1,
//	double* tmp1,
//	double* resi,
//	unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	update_pressure_residual_kernel2 << < numBlocks, numThreads >> >(
//		press,
//		alpha,
//		conj0,
//		omega,
//		conj1,
//		tmp1,
//		resi,
//		np);
//}
//
//void cuUpdate_conjugate2(double* conj0, double* resi, double beta, double omega, double* tmp0, unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	update_conjugate_kernel2 << < numBlocks, numThreads >> >(
//		conj0,
//		resi,
//		beta,
//		omega,
//		tmp0,
//		np);
//}
//
//void cuUpdate_dummy_pressure_from_boundary2(double* m_press, double* press, unsigned int* innerDummyNeighbors, xMaterialType* type, bool* isf, unsigned int* m_id, unsigned int np)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	update_dummy_pressure_from_boundary_kernel2 << < numBlocks, numThreads >> >(
//		m_press,
//		press,
//		(uint4 *)innerDummyNeighbors,
//		type,
//		isf,
//		m_id);
//}
//
//void cuContact_force_circle_boundary(
//	double* pos, xMaterialType* type, device_pointMass_info* dpmi,
//	device_circle_info* dci, device_contact_parameters* dcp,
//	unsigned int* hashes, unsigned int* cell_start, unsigned int np)
//{
//	//computeGridSize(np, 512, numBlocks, numThreads);
//	//contact_force_circle_boundary_kernel << < numBlocks, numThreads>> >(
//	//	(double3 *)pos,
//	//	type
//	//	dpmi,
//	//	dci,
//	//	dcp,
//	//	(uint2 *)hashes,
//	//	cell_start);
//}
//
//void cuContactDistance(double* pos, xMaterialType* type,
//	device_circle_info* dci, unsigned int* hashes, unsigned int* cell_start,
//	unsigned int* cid, double *dist, unsigned int np, unsigned int nc)
//{
//	computeGridSize(np, 512, numBlocks, numThreads);
//	contact_distance_kernel << < numBlocks, numThreads >> >(
//		(double3 *)pos,
//		type,
//		dci,
//		(uint2 *)hashes,
//		cell_start,
//		cid,
//		dist,
//		nc);
//	thrust::sort_by_key(thrust::device_ptr<unsigned int>(cid),
//		thrust::device_ptr<unsigned int>(cid + nc),
//		thrust::device_ptr<double>(dist));
//}
//
