#ifndef XSMOOTHEDPARTICLEHYDRODYNAMICS_H
#define XSMOOTHEDPARTICLEHYDRODYNAMICS_H

#include "xdynamics_decl.h"
#include "xModel.h"
#include "xdynamics_manager/xParticleMananger.h"

class xKernelFunction;

class XDYNAMICS_API xSmoothedParticleHydrodynamicsModel
{
public:
	xSmoothedParticleHydrodynamicsModel();
	xSmoothedParticleHydrodynamicsModel(std::string _name);
	~xSmoothedParticleHydrodynamicsModel();

	xParticleManager* XParticleManager();
	void setKernelFunctionData(xKernelFunctionData& d);

private:
	QString name;
	double k_viscosity;
	double ref_rho;
	double fs_factor;
	double pspace;
	double water_depth;
	xSPHCorrectionType corr;
	xTurbulenceType turb;
	xBoundaryTreatmentType bound;
	//xKenelType ker;
	xKernelFunctionData ker_data;
	xKernelFunction *xker;
	xParticleManager *xpmgr;
};

#endif