#include "xdynamics_simulation/xSmoothedParticleHydrodynamicsSimulation.h"
#include <fstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "xdynamics_parallel/xParallelSPH_decl.cuh"
#include "xdynamics_object/xCubicSplineKernel.h"
#include "xdynamics_object/xQuadraticKernel.h"
#include "xdynamics_object/xQuinticKernel.h"
#include "xdynamics_object/xWendlandKernel.h"

xSmoothedParticleHydrodynamicsSimulation::xSmoothedParticleHydrodynamicsSimulation()
{
	min_world = new_vector3d(FLT_MAX, FLT_MAX, FLT_MAX);
	max_world = new_vector3d(FLT_MIN, FLT_MIN, FLT_MIN);
}

xSmoothedParticleHydrodynamicsSimulation::~xSmoothedParticleHydrodynamicsSimulation()
{
	clearMemory();
}

void xSmoothedParticleHydrodynamicsSimulation::clearMemory()
{
	if (pos) delete[] pos; pos = NULL;
	if (vel) delete[] vel; vel = NULL;
	if (acc) delete[] acc; acc = NULL;
	if (aux_vel) delete[] aux_vel; aux_vel = NULL;
	if (press) delete[] press; press = NULL;
	if (rho) delete[] rho; rho = NULL;
	if (isFreeSurface) delete[] isFreeSurface; isFreeSurface = NULL;
	if (ptype) delete[] ptype; ptype = NULL;
	if (mass) delete[] mass; mass = NULL;
	if (xSimulation::Gpu())
	{
		if (d_pos) cudaFree(d_pos); d_pos = NULL;
		if (d_vel) cudaFree(d_vel); d_vel = NULL;
		if (d_acc) cudaFree(d_acc); d_acc = NULL;
		if (d_aux_vel) cudaFree(d_aux_vel); d_aux_vel = NULL;
		if (d_press) cudaFree(d_press); d_press = NULL;
		if (d_rho) cudaFree(d_rho); d_rho = NULL;
		if (d_isFreeSurface) cudaFree(d_isFreeSurface); d_isFreeSurface = NULL;
		if (d_ptype) cudaFree(d_ptype); ptype = NULL;
		if (d_lhs) cudaFree(d_lhs); d_lhs = NULL;
		if (d_rhs) cudaFree(d_rhs); d_rhs = NULL;
		if (d_conj0) cudaFree(d_conj0); d_conj0 = NULL;
		if (d_conj1) cudaFree(d_conj1); d_conj1 = NULL;
		if (d_tmp0) cudaFree(d_tmp0); d_tmp0 = NULL;
		if (d_tmp1) cudaFree(d_tmp1); d_tmp1 = NULL;
		if (d_residual) cudaFree(d_residual); d_residual = NULL;
		if (d_mass) cudaFree(d_mass); d_mass = NULL;
	}
}

void xSmoothedParticleHydrodynamicsSimulation::allocationMemory(unsigned int _np)
{
	np = _np;
	pos = new vector3d[np * 3];
	vel = new vector3d[np * 3];
	acc = new vector3d[np * 3];
	aux_vel = new vector3d[np * 3];
	press = new double[np];
	rho = new double[np];
	isFreeSurface = new bool[np];
	ptype = new xMaterialType[np];
	mass = new double[np];
	alloc_size = sizeof(double) * np * 15 + sizeof(xMaterialType) * np + sizeof(bool) * np;
	memset(pos, 0, alloc_size);
}

int xSmoothedParticleHydrodynamicsSimulation::Initialize(xSmoothedParticleHydrodynamicsModel* _xsph)
{
	xsph = _xsph;
	switch (xsph->KernelData().type)
	{
	case CUBIC_SPLINE_KERNEL: ker = new xCubicSplineKernel; break;
	case QUADRATIC_KERNEL: ker = new xQuadraticKernel; break;
	case QUINTIC_KERNEL: ker = new xQuinticKernel; break;
	case WENDLAND_KERNEL: ker = new xWendlandKernel; break;
	}
	particle_space = xsph->ParticleSpacing();
	ker->setupKernel(particle_space, xsph->KernelData());
	support_radius = ker->KernelSupport() * ker->h;
	depsilon = ker->h * ker->h * 0.01;
	allocationMemory(xsph->NumTotalParticle());
	memcpy(pos, xsph->Position(), sizeof(double) * np * 3);
	memcpy(vel, xsph->Velocity(), sizeof(double) * np * 3);
	for (unsigned int i = 0; i < np; i++)
	{
		acc[i] = xModel::gravity;
		mass[i] = xsph->ParticleMass();
		rho[i] = xsph->ReferenceDensity();
		if (min_world >= (pos[i] - new_vector3d(1e-9, 1e-9, 1e-9)))	min_world = pos[i];
		if (max_world >= (pos[i] + new_vector3d(1e-9, 1e-9, 1e-9)))	max_world = pos[i];
	}
	xsc = new xSmoothCell;
	xsc->setCellSize(support_radius);
	vector3d rw = new_vector3d(particle_space, particle_space, particle_space);
	xsc->setWorldBoundary(min_world - rw, max_world + rw);
	xsc->initialize(np);
	
	if (xSimulation::Gpu())
	{
		setupCudaData();
	}
	return xDynamicsError::xdynamicsSuccess;
}

void xSmoothedParticleHydrodynamicsSimulation::setupCudaData()
{
	cudaMemoryAlloc((void**)&d_pos, sizeof(double) * np * 3);
	cudaMemoryAlloc((void**)&d_vel, sizeof(double) * np * 3);
	cudaMemoryAlloc((void**)&d_aux_vel, sizeof(double) * np * 3);
	cudaMemoryAlloc((void**)&d_acc, sizeof(double) * np * 3);
	cudaMemoryAlloc((void**)&d_press, sizeof(double) * np);
	cudaMemoryAlloc((void**)&d_rho, sizeof(double) * np);
	cudaMemoryAlloc((void**)&d_mass, sizeof(double) * np);
	cudaMemoryAlloc((void**)&d_isFreeSurface, sizeof(bool) * np);
	cudaMemoryAlloc((void**)&d_ptype, sizeof(xMaterialType) * np);

	if (xsph->WaveDampingData().enable)
	{
		device_wave_damping d = { 0, };
		d.alpha = xsph->WaveDampingData().alpha;
		d.start_point = xsph->WaveDampingData().start_point;
		d.length = xsph->WaveDampingData().length;
		cudaMemoryAlloc((void**)&d_wdamp, sizeof(device_wave_damping));
		checkCudaErrors(cudaMemcpy(d_wdamp, &d, sizeof(device_wave_damping), cudaMemcpyHostToDevice));
	}
	checkCudaErrors(cudaMemcpy(d_mass, mass, sizeof(double) * np, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_pos, pos, sizeof(double3) * np, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_rho, rho, sizeof(double) * np, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vel, vel, sizeof(double3) * np, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aux_vel, vel, sizeof(double3) * np, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_isFreeSurface, isFreeSurface, sizeof(bool) * np, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_press, press, sizeof(double) * np, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ptype, ptype, sizeof(xMaterialType) * np, cudaMemcpyHostToDevice));
	device_sph_parameters dp;
	dp.isShifting = false;// md->IsShifting();
	dp.shifting_factor = 0.01;// 0.1 * md->KernelLength() * md->KernelLength() / dt;
	dp.corr = xsph->CorrectionType();// CORRECTION;
	dp.dim = xsph->Dimension();
	dp.kernel = xsph->KernelData().type;
	dp.np = np;
	dp.cells = xsc->nCell();// fd->getNumCell();
	dp.startInnerDummy = 0;// _isInnerBoundary ? md->overlappingCornerStartIndex : 0;
	dp.gridMin = make_double3(xsc->MinimumGridPosition().x, xsc->MinimumGridPosition().y, xsc->MinimumGridPosition().z);// make_double3(fd->gridMin().x, fd->gridMin().y, fd->gridMin().z);
	dp.gridMax = make_double3(xsc->MaximumGridPosition().x, xsc->MaximumGridPosition().y, xsc->MaximumGridPosition().z);// make_double3(fd->gridMax().x, fd->gridMax().y, fd->gridMax().z);
	dp.gridSize = make_double3(xsc->GridSize().x, xsc->GridSize().y, xsc->GridSize().z);// make_double3(fd->gridSize().x, fd->gridSize().y, fd->gridSize().z);
	dp.gravity = make_double3(xModel::gravity.x, xModel::gravity.y, xModel::gravity.z);
	dp.rho = xsph->ReferenceDensity();
	dp.peclet = 0;// 1.98 * md->ParticleSpacing() / 30.0;// 2.0 * sqrt(9.80665 * 0.3) * pspace / 30.0;
	dp.kernel_const = ker->KernelConst();
	dp.kernel_grad_const = ker->KernelGradConst();
	dp.kernel_support = ker->KernelSupport();
	dp.kernel_support_sq = ker->KernelSupprotSq();
	dp.smoothing_length = 0.0;// md->KernelLength();
	dp.particle_spacing = particle_space;// ->ParticleSpacing();
	dp.kernel_support_radius = make_double3(support_radius, support_radius, support_radius);
	dp.gridCellSize = xsc->cs;// fd->gridCellSize();
	dp.gridCellCount = make_uint3(xsc->gs.x, xsc->gs.y, xsc->gs.z);
	dp.cellsizeInv = 1.0 / dp.gridCellSize;
	dp.viscosity = xsph->KinematicViscosity();
	dp.dist_epsilon = depsilon;
	dp.h = ker->h;
	dp.h_sq = ker->h_sq;
	dp.h_inv = ker->h_inv;
	dp.h_inv_sq = ker->h_inv_sq;
	dp.h_inv_2 = ker->h_inv_2;
	dp.h_inv_3 = ker->h_inv_3;
	dp.h_inv_4 = ker->h_inv_4;
	dp.h_inv_5 = ker->h_inv_5;
	dp.dt = xSimulation::dt;
	dp.time = 0;
	dp.freeSurfaceFactor = xsph->FreeSurfaceFactor();
	setSPHSymbolicParameter(&dp);
}