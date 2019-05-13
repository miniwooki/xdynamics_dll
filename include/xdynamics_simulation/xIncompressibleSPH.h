#ifndef XINCOMPRESSIBLESPH_H
#define XINCOMPRESSIBLESPH_H

#include "xdynamics_decl.h"
#include "xSmoothedParticleHydrodynamicsSimulation.h"

class XDYNAMICS_API xIncompressibleSPH : public xSmoothedParticleHydrodynamicsSimulation
{
public:
	xIncompressibleSPH();
	~xIncompressibleSPH();

	virtual int Initialize(xSmoothedParticleHydrodynamicsModel* _xsph);
	virtual int OneStepSimulation(double ct, unsigned int cstep);

private:
	void setupCudaDataISPH();

private:
	double* d_lhs;
	double* d_rhs;
	double* d_conj0;
	double* d_conj1;
	double* d_tmp0;
	double* d_tmp1;
	double* d_residual;
};

#endif