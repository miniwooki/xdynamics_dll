#include "xdynamics_simulation/xIncompressibleSPH.h"

xIncompressibleSPH::xIncompressibleSPH()
	: xSmoothedParticleHydrodynamicsSimulation()
{

}

xIncompressibleSPH::~xIncompressibleSPH()
{
	if (xSimulation::Gpu())
	{
		if (d_lhs) cudaFree(d_lhs); d_lhs = NULL;
		if (d_rhs) cudaFree(d_rhs); d_rhs = NULL;
		if (d_conj0) cudaFree(d_conj0); d_conj0 = NULL;
		if (d_conj1) cudaFree(d_conj1); d_conj1 = NULL;
		if (d_tmp0) cudaFree(d_tmp0); d_tmp0 = NULL;
		if (d_tmp1) cudaFree(d_tmp1); d_tmp1 = NULL;
		if (d_residual) cudaFree(d_residual); d_residual = NULL;
	}
}

int xIncompressibleSPH::Initialize(xSmoothedParticleHydrodynamicsModel* _xsph)
{
	if (xSmoothedParticleHydrodynamicsSimulation::Initialize(_xsph))
	{
		return xDynamicsError::xdynamicsErrorIncompressibleSPHInitialization;
	}
	if (xSimulation::Gpu())
	{
		xIncompressibleSPH::setupCudaDataISPH();
	}
	return xDynamicsError::xdynamicsSuccess;
}

int xIncompressibleSPH::OneStepSimulation(double ct, unsigned int cstep)
{
	return xDynamicsError::xdynamicsSuccess;
}

void xIncompressibleSPH::setupCudaDataISPH()
{
	cudaMemoryAlloc((void**)&d_lhs, sizeof(double) * np);
	cudaMemoryAlloc((void**)&d_rhs, sizeof(double) * np);
	cudaMemoryAlloc((void**)&d_conj0, sizeof(double) * np);
	cudaMemoryAlloc((void**)&d_conj1, sizeof(double) * np);
	cudaMemoryAlloc((void**)&d_tmp0, sizeof(double) * np);
	cudaMemoryAlloc((void**)&d_tmp1, sizeof(double) * np);
	cudaMemoryAlloc((void**)&d_residual, sizeof(double) * np);
}
