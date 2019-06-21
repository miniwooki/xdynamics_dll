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
	xSmoothedParticleHydrodynamicsModel(std::wstring _name);
	~xSmoothedParticleHydrodynamicsModel();

	static xSmoothedParticleHydrodynamicsModel* XSPH();

	//xParticleManager* XParticleManager();
	void setParticleSpacing(double ps);
	void setFreeSurfaceFactor(double fs);
	void setReferenceDensity(double d);
	void setKinematicViscosity(double v);
	void setKernelFunctionData(xKernelFunctionData& d);

	xBoundaryTreatmentType BoundaryTreatmentType();
	xSPHCorrectionType CorrectionType();

	bool CheckCorner(vector3d p);
	void DefineCorners(xObjectManager* xobj);
	void CreateParticles(xObjectManager* xobj);
	void ExportParticleDataForView(std::wstring& path);

	unsigned int NumTotalParticle();
	unsigned int NumFluid();
	unsigned int NumBoundary();
	unsigned int NumDummy();
	unsigned int Dimension();

	double ParticleMass();
	double ParticleVolume();
	double ReferenceDensity();
	double ParticleSpacing();
	double KinematicViscosity();
	double FreeSurfaceFactor();
	vector3d* Position();
	vector3d* Velocity();
	xKernelFunctionData& KernelData();
	xWaveDampingData& WaveDampingData();

private:
	unsigned int CreateOverlapCornerDummyParticles(unsigned int sid, bool isOnlyCount);
	unsigned int CreateOverlapCornerDummyParticle(unsigned int id, vector3d& p, vector3d& n1, vector3d& n2, bool isOnlyCount);

private:
	QString name;

	unsigned int dim;
	unsigned int np;
	unsigned int nfluid;
	unsigned int nbound;
	unsigned int ndummy;
	unsigned int nlayers;

	double p_mass;
	double p_volume;
	double k_viscosity;
	double ref_rho;
	double fs_factor;
	double pspace;
	double water_depth;

	vector3d* pos;
	vector3d* vel;
	xMaterialType* type;

	xSPHCorrectionType corr;
	xTurbulenceType turb;
	xBoundaryTreatmentType bound;
	//xKenelType ker;
	xKernelFunctionData ker_data;
	xWaveDampingData wave_damping_data;
	xKernelFunction *xker;
	//xParticleManager *xpmgr;

	QVector<xOverlapCorner> overlappingCorners;
};

#endif