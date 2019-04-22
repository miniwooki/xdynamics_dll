#include "xdynamics_manager/xMultiBodyModel.h"
#include "xdynamics_manager/xObjectManager.h"


xMultiBodyModel::xMultiBodyModel()
{

}

xMultiBodyModel::xMultiBodyModel(std::string _name)
{

}

xMultiBodyModel::~xMultiBodyModel()
{
	//qDeleteAll(masses);
	qDeleteAll(forces);
	qDeleteAll(dconstraints);
	qDeleteAll(constraints);
}

unsigned int xMultiBodyModel::NumMass()
{
	return masses.size();
}

unsigned int xMultiBodyModel::NumConstraint()
{
	return constraints.size();
}

unsigned int xMultiBodyModel::NumDrivingConstraint()
{
	return dconstraints.size();
}

QMap<QString, xPointMass*>& xMultiBodyModel::Masses()
{
	return masses;
}

QMap<QString, xKinematicConstraint*>& xMultiBodyModel::Joints()
{
	return constraints;
}

QMap<QString, xForce*>& xMultiBodyModel::Forces()
{
	return forces;
}

QMap<QString, xDrivingConstraint*>& xMultiBodyModel::Drivings()
{
	return dconstraints;
}

xPointMass* xMultiBodyModel::XMass(std::string& _ws)
{
	if (_ws == "ground")
		return xModel::Ground();
	QString ws = QString::fromStdString(_ws);
	QStringList keys = masses.keys();
	QStringList::const_iterator it = qFind(keys, ws);
	if (it == keys.end())
		return NULL;
	return masses[ws];
}

xKinematicConstraint* xMultiBodyModel::XJoint(std::string& _ws)
{
	QString ws = QString::fromStdString(_ws);
	QStringList keys = constraints.keys();
	QStringList::const_iterator it = qFind(keys, ws);
	if (it == keys.end())
		return NULL;
	return constraints[ws];
}

xForce* xMultiBodyModel::XForce(std::string& _ws)
{
	QString ws = QString::fromStdString(_ws);
	QStringList keys = forces.keys();
	QStringList::const_iterator it = qFind(keys, ws);
	if (it == keys.end())
		return NULL;
	return forces[ws];
}

xDrivingConstraint* xMultiBodyModel::xDriving(std::string& _ws)
{
	QString ws = QString::fromStdString(_ws);
	QStringList keys = dconstraints.keys();
	QStringList::const_iterator it = qFind(keys, ws);
	if (it == keys.end())
		return NULL;
	return dconstraints[ws];
}

xPointMass* xMultiBodyModel::CreatePointMass(std::string _name)
{
	QString name = QString::fromStdString(_name);
	xPointMass* xpm = NULL;
	if (xObjectManager::XOM()->XObject(_name))
	{
		xpm = (dynamic_cast<xPointMass*>(xObjectManager::XOM()->XObject(_name)));
		masses[name] = xpm;
		return xpm;
	}
	xpm = new xPointMass(_name);
	masses[name] = xpm;
	xObjectManager::XOM()->addObject(xpm);
	xLog::log("Create PointMass : " + _name);
	//std::wcout << "Create point mass - " << _name.c_str() << ", Num. mass - " << masses.size() << std::endl;
	return xpm;
}

xKinematicConstraint* xMultiBodyModel::CreateKinematicConstraint(
	std::string _name, xKinematicConstraint::cType _type, std::string _i, std::string _j)
{
	QString name = QString::fromStdString(_name);
	xKinematicConstraint* xkc = NULL;
	switch (_type)
	{
	case xKinematicConstraint::REVOLUTE:
		xkc = new xRevoluteConstraint(_name, _i, _j);
		xLog::log("Create Revolute Joint : " + _name);
		//std::wcout << "Create revolute constraint - " << _name.c_str() << ", Num. constraint - " << constraints.size() << std::endl;
		break;
	case xKinematicConstraint::TRANSLATIONAL:
		xkc = new xTranslationConstraint(_name, _i, _j);
		xLog::log("Create Translation Joint : " + _name);
		//std::wcout << "Create translation constraint - " << _name.c_str() << ", Num. constraint - " << constraints.size() << std::endl;
		break;
	case xKinematicConstraint::SPHERICAL:
		xkc = new xSphericalConstraint(_name, _i, _j);
		xLog::log("Create Spherical Joint : " + _name);
		//std::wcout << "Create spherical constraint - " << _name.c_str() << ", Num. constraint - " << constraints.size() << std::endl;
		break;
	case xKinematicConstraint::UNIVERSAL:
		xkc = new xUniversalConstraint(_name, _i, _j);
		xLog::log("Create Universal Joint : " + _name);
		//std::wcout << "Create universal constraint - " << _name.c_str() << ", Num. constraint - " << constraints.size() << std::endl;
		break;
	}
	constraints[name] = xkc;
	return xkc;
}

xForce* xMultiBodyModel::CreateForceElement(std::string _name, xForce::fType _type, std::string bn, std::string an)
{
	QString name = QString::fromStdString(_name);
	xForce* xf = NULL;
	switch (_type)
	{
	case xForce::TSDA:
		xf = new xSpringDamperForce(_name);
		xLog::log("Create Translational Spring Damper Element : " + _name);
		break;
	case xForce::RSDA:
		break;
	case xForce::RAXIAL:
		xf = new xRotationalAxialForce(_name);
		xLog::log("Create Rotational Axial Force Element : " + _name);
		break;
	}
	if (xf)
	{
		xf->setBaseBodyName(bn);
		xf->setActionBodyName(an);
	}
	forces[name] = xf;
	return xf;
}

xDrivingConstraint* xMultiBodyModel::CreateDrivingConstraint(std::string _name, xKinematicConstraint* _kc)
{
	QString name = QString::fromStdString(_name);
	xDrivingConstraint* xdc = new xDrivingConstraint(_name, _kc);
	dconstraints[name] = xdc;
	xLog::log("Create Driving : " + _name);
	//std::wcout << "Create driving constraint - " << _name.c_str() << ", Num. driving constraint - " << dconstraints.size() << std::endl;
	return xdc;
}
