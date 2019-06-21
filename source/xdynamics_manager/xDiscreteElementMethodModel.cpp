#include "xdynamics_manager/XDiscreteElementMethodModel.h"
#include "xdynamics_manager/xObjectManager.h"

xDiscreteElementMethodModel::xDiscreteElementMethodModel()
	: xpmgr(NULL)
{
	xpmgr = new xParticleManager;
}

xDiscreteElementMethodModel::xDiscreteElementMethodModel(std::wstring _name)
	: name(QString::fromStdWString(_name))
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
