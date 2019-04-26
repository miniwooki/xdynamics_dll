#include "xdynamics_simulation/xSmoothedParticleHydrodynamicsSimulation.h"
#include <fstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>

xSmoothedParticleHydrodynamicsSimulation::xSmoothedParticleHydrodynamicsSimulation()
{

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
	}
}

void xSmoothedParticleHydrodynamicsSimulation::allocationMemory()
{

}

int xSmoothedParticleHydrodynamicsSimulation::Initialize(xSmoothedParticleHydrodynamicsModel* _xsph)
{
	xsph = _xsph;
	
}