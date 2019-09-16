#include "xdynamics_manager/XDiscreteElementMethodModel.h"
#include "xdynamics_manager/xObjectManager.h"

xDiscreteElementMethodModel::xDiscreteElementMethodModel()
	: xpmgr(NULL)
	, tsda(NULL)
{
	xpmgr = new xParticleManager;
}

xDiscreteElementMethodModel::xDiscreteElementMethodModel(std::string _name)
	: name(_name)
	, xpmgr(NULL)
	, tsda(NULL)
{
	xpmgr = new xParticleManager;
}

xDiscreteElementMethodModel::~xDiscreteElementMethodModel()
{
	if (xpmgr) delete xpmgr; xpmgr = NULL;
	if (tsda) delete tsda; tsda = NULL;
}

xParticleManager* xDiscreteElementMethodModel::XParticleManager()
{
	return xpmgr;
}

xSpringDamperForce* xDiscreteElementMethodModel::XSpringDamperForce()
{
	return tsda;
}

xSpringDamperForce* xDiscreteElementMethodModel::CreateForceElement(std::string _name, xForce::fType _type, std::string bn, std::string an)
{
//	xForce* xf = NULL;
	switch (_type)
	{
	case xForce::TSDA_LIST_DATA:
		tsda = new xSpringDamperForce(_name);
		xLog::log("Create Translational Spring Damper Element From List Data : " + _name);
		break;
	}
	if (tsda)
	{
		tsda->setBaseBodyName(bn);
		tsda->setActionBodyName(an);
	}
	//forces[name] = xf;
	return tsda;
}
