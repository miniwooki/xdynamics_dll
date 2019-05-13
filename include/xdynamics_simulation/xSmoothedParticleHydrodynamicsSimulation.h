#ifndef XSMOOTHEDPARTICLEHYDRODYNAMICSSIMULATION_H
#define XSMOOTHEDPARTICLEHYDRODYNAMICSSIMULATION_H

#include "xdynamics_decl.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_algebra/xSmoothCell.h"
#include "xdynamics_object/xKernelFunction.h"
#include "xdynamics_manager/xSmoothedParticleHydrodynamicsModel.h"
#include "xdynamics_parallel/xParallelCommon_decl.cuh"

class XDYNAMICS_API xSmoothedParticleHydrodynamicsSimulation : public xSimulation
{
public:
	xSmoothedParticleHydrodynamicsSimulation();
	virtual ~xSmoothedParticleHydrodynamicsSimulation();

	virtual int Initialize(xSmoothedParticleHydrodynamicsModel* _xsph);
	bool Initialized();
	virtual int OneStepSimulation(double ct, unsigned int cstep) = 0;
	QString SaveStepResult(unsigned int pt, double ct);

protected:
	void clearMemory();
	void allocationMemory(unsigned int np);
	void setupCudaData();
	bool isInitialize;
	unsigned int np;
	unsigned int alloc_size;
	double particle_space;
	double support_radius;
	double depsilon;
	xSmoothedParticleHydrodynamicsModel* xsph;
	xSmoothCell* xsc;
	xKernelFunction* ker;
	device_wave_damping *d_wdamp;
	vector3d min_world;
	vector3d max_world;

	vector3d *pos;
	vector3d *vel;
	vector3d *acc;
	vector3d *aux_vel;
	double *press;
	double *rho;
	double *mass;
	bool *isFreeSurface;
	xMaterialType *ptype;

	double *d_pos;
	double *d_vel;
	double *d_acc;
	double *d_aux_vel;
	double *d_press;
	double *d_rho;
	double *d_mass;
	bool *d_isFreeSurface;
	xMaterialType *d_ptype;
};

#endif