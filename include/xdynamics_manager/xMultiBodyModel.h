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

class XDYNAMICS_API xMultiBodyModel 
{
public:
	xMultiBodyModel();
	xMultiBodyModel(std::wstring _name);
	~xMultiBodyModel();

	unsigned int NumMass();
	unsigned int NumConstraint();
	unsigned int NumDrivingConstraint();

	QMap<QString, xPointMass*>& Masses();
	QMap<QString, xKinematicConstraint*>& Joints();
	QMap<QString, xForce*>& Forces();
	QMap<QString, xDrivingConstraint*>& Drivings();
	xPointMass* XMass(std::wstring& ws);
	xKinematicConstraint* XJoint(std::wstring& ws);
	xForce* XForce(std::wstring& ws);
	xDrivingConstraint* xDriving(std::wstring& ws);

	xPointMass* CreatePointMass(std::wstring _name);
	xKinematicConstraint* CreateKinematicConstraint(std::wstring _name, xKinematicConstraint::cType _type, std::wstring _i, std::wstring _j);
	xForce* CreateForceElement(std::wstring _name, xForce::fType _type, std::wstring bn, std::wstring an);
	xDrivingConstraint* CreateDrivingConstraint(std::wstring _name, xKinematicConstraint* _kc);

	//void InsertPointMassFromShape(xPointMass* pm);

private:
	QMap<QString, xPointMass*> masses;
	QMap<QString, xForce*> forces;
	QMap<QString, xKinematicConstraint*> constraints;
	QMap<QString, xDrivingConstraint*> dconstraints;
};

#endif