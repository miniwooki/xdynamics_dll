#ifndef XSMOOTHEDPARTICLEHYDRODYNAMICSSIMULATION_H
#define XSMOOTHEDPARTICLEHYDRODYNAMICSSIMULATION_H

#include "xdynamics_decl.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_manager/xSmoothedParticleHydrodynamicsModel.h"

class XDYNAMICS_API xSmoothedParticleHydrodynamicsSimulation : public xSimulation
{
public:
	xSmoothedParticleHydrodynamicsSimulation();
	virtual ~xSmoothedParticleHydrodynamicsSimulation();

	virtual int Initialize(xSmoothedParticleHydrodynamicsModel* _xsph);
	bool Initialized();
	virtual int OneStepSimulation(double ct, unsigned int cstep) = 0;
	QString SaveStepResult(std::fstream& of);

protected:
	void clearMemory();
	void allocationMemory();
	bool isInitialize;
	unsigned int np;
	xSmoothedParticleHydrodynamicsModel* xsph;

	double *pos;
	double *vel;
	double *acc;
	double *aux_vel;
	double *press;
	double *rho;
	bool *isFreeSurface;
	xMaterialType *ptype;

	double *d_pos;
	double *d_vel;
	double *d_acc;
	double *d_aux_vel;
	double *d_press;
	double *d_rho;
	bool *d_isFreeSurface;
	xMaterialType *d_ptype;

	double* d_lhs;
	double* d_rhs;
	double* d_conj0;
	double* d_conj1;
	double* d_tmp0;
	double* d_tmp1;
	double* d_residual;
};

#endif