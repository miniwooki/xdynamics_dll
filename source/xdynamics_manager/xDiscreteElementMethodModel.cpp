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

xRotationSpringDamperForce * xDiscreteElementMethodModel::XRotationalSpringDamperForce()
{
	return rsda;
}

xForce* xDiscreteElementMethodModel::CreateForceElement(std::string _name, xForce::fType _type, std::string bn, std::string an)
{
	xForce* xf = nullptr;
//	xForce* xf = NULL;
	switch (_type)
	{
	case xForce::TSDA_LIST_DATA:
		xf = new xSpringDamperForce(_name);
		xLog::log("Create Translational Spring Damper Element From List Data : " + _name);
		break;
	case xForce::RSDA_LIST_DATA:
		xf = new xRotationSpringDamperForce(_name);
		xLog::log("Create Rotational Spring Damper Element From List Data : " + _name);
		break;
	}
	if (xf) {
		xf->setBaseBodyName(bn);
		xf->setActionBodyName(an);
	}
	return xf;
}
