#ifndef XDYNAMICS_MANAGER_H
#define XDYNAMICS_MANAGER_H

#include "xdynamics_decl.h"
#include "xdynamics_manager/xModel.h"
#include "xdynamics_manager/xMultiBodyModel.h"
#include "xdynamics_manager/XDiscreteElementMethodModel.h"
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
	void CreateModel(std::string str, modelType t, bool isOnAir = true);

	xMultiBodyModel* XMBDModel();
	xDiscreteElementMethodModel* XDEMModel();
	xObjectManager* XObject();
	xContactManager* XContact();

	xMultiBodyModel* XMBDModel(std::string& n);
	xDiscreteElementMethodModel* XDEMModel(std::string& n);
	xObjectManager* XObject(std::string& n);
	xContactManager* XContact(std::string& n);

	solverType OpenModelXLS(const wchar_t* n);

private:
	void setOnAirModel(modelType t, std::string n);
	QMap<QString, xMultiBodyModel*> xmbds;
	QMap<QString, xDiscreteElementMethodModel*> xdems;
	QMap<QString, xObjectManager*> xoms;
	QMap<QString, xContactManager*> xcms;
	xMultiBodyModel* xmbd;
	xDiscreteElementMethodModel* xdem;
	xObjectManager *xom;
	xContactManager *xcm;
// 
// private:
// 	void xDeleteMXMBD();
// 	void xDeleteMXDEM();
};

#endif