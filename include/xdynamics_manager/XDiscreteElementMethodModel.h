#ifndef XDISCRETEELEMENTMETHODMODEL_H
#define XDISCRETEELEMENTMETHODMODEL_H

#include "xdynamics_decl.h"
#include "xModel.h"
#include "xdynamics_manager/xParticleMananger.h"

class XDYNAMICS_API xDiscreteElementMethodModel
{
public:
	xDiscreteElementMethodModel();
	xDiscreteElementMethodModel(std::string _name);
	~xDiscreteElementMethodModel();

	xParticleManager* XParticleManager();
private:
	QString name;
	xParticleManager *xpmgr;
};

#endif