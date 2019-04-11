#include "xdynamics_manager/XDiscreteElementMethodModel.h"
#include "xdynamics_manager/xObjectManager.h"

xDiscreteElementMethodModel::xDiscreteElementMethodModel()
	: xpmgr(NULL)
{
	xpmgr = new xParticleManager;
}

xDiscreteElementMethodModel::xDiscreteElementMethodModel(std::string _name)
	: name(QString::fromStdString(_name))
	, xpmgr(NULL)
{
	xpmgr = new xParticleManager;
}

xDiscreteElementMethodModel::~xDiscreteElementMethodModel()
{
	if (xpmgr) delete xpmgr; xpmgr = NULL;
}

xParticleManager* xDiscreteElementMethodModel::XParticleManager()
{
	return xpmgr;
}