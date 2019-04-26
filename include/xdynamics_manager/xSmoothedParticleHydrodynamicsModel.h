#ifndef XSMOOTHEDPARTICLEHYDRODYNAMICS_H
#define XSMOOTHEDPARTICLEHYDRODYNAMICS_H

#include "xdynamics_decl.h"
#include "xModel.h"
#include "xdynamics_manager/xParticleMananger.h"
#include <QtCore/QVector>

class xKernelFunction;
class xObjectManager;

class XDYNAMICS_API xSmoothedParticleHydrodynamicsModel
{
public:
	xSmoothedParticleHydrodynamicsModel();
	xSmoothedParticleHydrodynamicsModel(std::string _name);
	~xSmoothedParticleHydrodynamicsModel();

	static xSmoothedParticleHydrodynamicsModel* XSPH();

	xParticleManager* XParticleManager();
	void setParticleSpacing(double ps);
	void setFreeSurfaceFactor(double fs);
	void setReferenceDensity(double d);
	void setKinematicViscosity(double v);
	void setKernelFunctionData(xKernelFunctionData& d);

	xBoundaryTreatmentType BoundaryTreatmentType();

	bool CheckCorner(vector3d p);
	void DefineCorners(xObjectManager* xobj);
	void CreateParticles(xObjectManager* xobj);

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

	QVector<xOverlapCorner> overlappingCorners;
};

#endif