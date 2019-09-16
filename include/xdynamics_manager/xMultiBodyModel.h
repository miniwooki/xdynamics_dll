#ifndef XMULTIBODYMODEL_H
#define XMULTIBODYMODEL_H

#include "xdynamics_manager/xModel.h"
#include "xdynamics_object/xPointMass.h"
#include "xdynamics_object/xSpringDamperForce.h"
#include "xdynamics_object/xRevoluteConstraint.h"
#include "xdynamics_object/xTranslationConstraint.h"
#include "xdynamics_object/xSphericalConstraint.h"
#include "xdynamics_object/xUniversalConstraint.h"
#include "xdynamics_object/xDrivingConstraint.h"
#include "xdynamics_object/xRotationalAxialForce.h"
#include "xmap.hpp"

class XDYNAMICS_API xMultiBodyModel 
{
public:
	xMultiBodyModel();
	xMultiBodyModel(std::string _name);
	~xMultiBodyModel();

	unsigned int NumMass();
	unsigned int NumConstraint();
	unsigned int NumDrivingConstraint();

	xmap<xstring, xPointMass*>& Masses();
	xmap<xstring, xPointMass*>* Masses_ptr();
	xmap<xstring, xKinematicConstraint*>& Joints();
	xmap<xstring, xForce*>& Forces();
	xmap<xstring, xDrivingConstraint*>& Drivings();
	xPointMass* XMass(std::string& ws);
	xKinematicConstraint* XJoint(std::string& ws);
	xForce* XForce(std::string& ws);
	xDrivingConstraint* xDriving(std::string& ws);

	xPointMass* CreatePointMass(std::string _name);
	xKinematicConstraint* CreateKinematicConstraint(std::string _name, xKinematicConstraint::cType _type, std::string _i, std::string _j);
	xForce* CreateForceElement(std::string _name, xForce::fType _type, std::string bn, std::string an);
	xDrivingConstraint* CreateDrivingConstraint(std::string _name, xKinematicConstraint* _kc);

	//void InsertPointMassFromShape(xPointMass* pm);

private:
	xmap<xstring, xPointMass*> masses;
	xmap<xstring, xForce*> forces;
	xmap<xstring, xKinematicConstraint*> constraints;
	xmap<xstring, xDrivingConstraint*> dconstraints;
};

#endif