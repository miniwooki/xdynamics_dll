#include "xdynamics_manager/xSmoothedParticleHydrodynamicsModel.h"
#include "xdynamics_manager/xObjectManager.h"

xSmoothedParticleHydrodynamicsModel::xSmoothedParticleHydrodynamicsModel()
	: xpmgr(NULL)
{
	xpmgr = new xParticleManager;
}

xSmoothedParticleHydrodynamicsModel::xSmoothedParticleHydrodynamicsModel(std::string _name)
	: name(QString::fromStdString(_name))
	, xpmgr(NULL)
{
	xpmgr = new xParticleManager;
}

xSmoothedParticleHydrodynamicsModel::~xSmoothedParticleHydrodynamicsModel()
{
	if (xpmgr) delete xpmgr; xpmgr = NULL;
}

xParticleManager* xSmoothedParticleHydrodynamicsModel::XParticleManager()
{
	return xpmgr;
}

void xSmoothedParticleHydrodynamicsModel::setKernelFunctionData(xKernelFunctionData& d)
{
	ker_data = d;
}
