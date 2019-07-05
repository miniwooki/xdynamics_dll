#ifndef XDISCRETEELEMENTMETHODMODEL_H
#define XDISCRETEELEMENTMETHODMODEL_H

#include "xdynamics_decl.h"
#include "xModel.h"
#include "xdynamics_object/xSpringDamperForce.h"
#include "xdynamics_manager/xParticleMananger.h"

class XDYNAMICS_API xDiscreteElementMethodModel
{
public:
	xDiscreteElementMethodModel();
	xDiscreteElementMethodModel(std::string _name);
	~xDiscreteElementMethodModel();

	xParticleManager* XParticleManager();
	xSpringDamperForce* XSpringDamperForce();
	xSpringDamperForce* CreateForceElement(std::string _name, xForce::fType _type, std::string bn, std::string an);

private:
	QString name;
	xParticleManager *xpmgr;
	xSpringDamperForce* tsda;
};

#endif