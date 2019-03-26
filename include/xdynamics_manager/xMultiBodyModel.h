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

class XDYNAMICS_API xMultiBodyModel 
{
public:
	xMultiBodyModel();
	xMultiBodyModel(std::string _name);
	~xMultiBodyModel();

	unsigned int NumMass();
	unsigned int NumConstraint();
	unsigned int NumDrivingConstraint();

	QMap<QString, xPointMass*>& Masses();
	QMap<QString, xKinematicConstraint*>& Joints();
	QMap<QString, xForce*>& Forces();
	QMap<QString, xDrivingConstraint*>& Drivings();
	xPointMass* XMass(std::string& ws);
	xKinematicConstraint* XJoint(std::string& ws);
	xForce* XForce(std::string& ws);
	xDrivingConstraint* xDriving(std::string& ws);

	xPointMass* CreatePointMass(std::string _name);
	xKinematicConstraint* CreateKinematicConstraint(std::string _name, xKinematicConstraint::cType _type, std::string _i, std::string _j);
	xForce* CreateForceElement(std::string _name, xForce::fType _type, std::string bn, std::string an);
	xDrivingConstraint* CreateDrivingConstraint(std::string _name, xKinematicConstraint* _kc);

private:
	QMap<QString, xPointMass*> masses;
	QMap<QString, xForce*> forces;
	QMap<QString, xKinematicConstraint*> constraints;
	QMap<QString, xDrivingConstraint*> dconstraints;
};

#endif