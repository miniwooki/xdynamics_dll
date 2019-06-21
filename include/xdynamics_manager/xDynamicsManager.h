#ifndef XDYNAMICS_MANAGER_H
#define XDYNAMICS_MANAGER_H

#include "xdynamics_decl.h"
#include "xdynamics_manager/xModel.h"
#include "xdynamics_manager/xMultiBodyModel.h"
#include "xdynamics_manager/XDiscreteElementMethodModel.h"
#include "xdynamics_manager/xSmoothedParticleHydrodynamicsModel.h"
#include "xdynamics_manager/xObjectManager.h"
#include "xdynamics_manager/xContactManager.h"

class xDynamicsSimulator;

class XDYNAMICS_API xDynamicsManager : public xModel
{
public:
	enum modelType{ MBD = 0, DEM, SPH, OBJECT, CONTACT };
	enum solverType{ ONLY_MBD = 0, ONLY_DEM, ONLY_SPH, COUPLED_MBD_DEM, COUPLED_MBD_SPH };

	xDynamicsManager();
	~xDynamicsManager();

	bool getSimulatorFromCommand(int argc, wchar_t* argv[]);
	void CreateModel(std::wstring str, modelType t, bool isOnAir = true);

	xMultiBodyModel* XMBDModel();
	xDiscreteElementMethodModel* XDEMModel();
	xSmoothedParticleHydrodynamicsModel* XSPHModel();
	xObjectManager* XObject();
	xContactManager* XContact();

	xMultiBodyModel* XMBDModel(std::wstring& n);
	xDiscreteElementMethodModel* XDEMModel(std::wstring& n);
	xSmoothedParticleHydrodynamicsModel* XSPHModel(std::wstring& n);
	xObjectManager* XObject(std::wstring& n);
	xContactManager* XContact(std::wstring& n);

	solverType OpenModelXLS(const wchar_t* n);

private:
	void setOnAirModel(modelType t, std::wstring n);
	QMap<QString, xMultiBodyModel*> xmbds;
	QMap<QString, xDiscreteElementMethodModel*> xdems;
	QMap<QString, xSmoothedParticleHydrodynamicsModel*> xsphs;
	QMap<QString, xObjectManager*> xoms;
	QMap<QString, xContactManager*> xcms;
	xMultiBodyModel* xmbd;
	xDiscreteElementMethodModel* xdem;
	xSmoothedParticleHydrodynamicsModel* xsph;
	xObjectManager *xom;
	xContactManager *xcm;
// 
// private:
// 	void xDeleteMXMBD();
// 	void xDeleteMXDEM();
};

#endif