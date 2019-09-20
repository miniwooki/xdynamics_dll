#ifndef XDYNAMICS_MANAGER_H
#define XDYNAMICS_MANAGER_H

#include "xdynamics_decl.h"
#include "xdynamics_manager/xModel.h"
#include "xdynamics_manager/xMultiBodyModel.h"
#include "xdynamics_manager/XDiscreteElementMethodModel.h"
#include "xdynamics_manager/xSmoothedParticleHydrodynamicsModel.h"
#include "xdynamics_manager/xObjectManager.h"
#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_manager/xResultManager.h"

class xDynamicsSimulator;

class XDYNAMICS_API xDynamicsManager : public xModel
{
public:
	enum modelType{ MBD = 0, DEM, SPH, OBJECT, CONTACT };
	enum solverType{ NO_DEFINED_SOLVER_TYPE = -1, ONLY_MBD = 0, ONLY_DEM, ONLY_SPH, COUPLED_MBD_DEM, COUPLED_MBD_SPH };

	xDynamicsManager();
	~xDynamicsManager();

	static xDynamicsManager* This();

	bool getSimulatorFromCommand(int argc, char* argv[]);
	void CreateModel(std::string str, modelType t, bool isOnAir = true);
	void initialize_result_manager(unsigned int npt);
	void release_result_manager();

	xMultiBodyModel* XMBDModel();
	xDiscreteElementMethodModel* XDEMModel();
	xSmoothedParticleHydrodynamicsModel* XSPHModel();
	xObjectManager* XObject();
	xContactManager* XContact();
	xResultManager* XResult();

	xMultiBodyModel* XMBDModel(std::string& n);
	xDiscreteElementMethodModel* XDEMModel(std::string& n);
	xSmoothedParticleHydrodynamicsModel* XSPHModel(std::string& n);
	xObjectManager* XObject(std::string& n);
	xContactManager* XContact(std::string& n);

	int OpenModelXLS(const char* n);
	bool upload_model_results(std::string path);

private:
	void setOnAirModel(modelType t, std::string n);
	xmap<xstring, xMultiBodyModel*> xmbds;
	xmap<xstring, xDiscreteElementMethodModel*> xdems;
	xmap<xstring, xSmoothedParticleHydrodynamicsModel*> xsphs;
	xmap<xstring, xObjectManager*> xoms;
	xmap<xstring, xContactManager*> xcms;
	xMultiBodyModel* xmbd;
	xDiscreteElementMethodModel* xdem;
	xSmoothedParticleHydrodynamicsModel* xsph;
	xObjectManager *xom;
	xContactManager *xcm;
	xResultManager *xrm;
// 
// private:
// 	void xDeleteMXMBD();
// 	void xDeleteMXDEM();
};

#endif