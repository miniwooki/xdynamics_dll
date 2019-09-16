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
	if(forces.size()) forces.delete_all();
	if(dconstraints.size()) dconstraints.delete_all();
	if(constraints.size()) constraints.delete_all();
	/*qDeleteAll(forces);
	qDeleteAll(dconstraints);
	qDeleteAll(constraints);*/
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

xmap<xstring, xPointMass*>& xMultiBodyModel::Masses()
{
	return masses;
}

xmap<xstring, xPointMass*>* xMultiBodyModel::Masses_ptr()
{
	return &masses;
}

xmap<xstring, xKinematicConstraint*>& xMultiBodyModel::Joints()
{
	return constraints;
}

xmap<xstring, xForce*>& xMultiBodyModel::Forces()
{
	return forces;
}

xmap<xstring, xDrivingConstraint*>& xMultiBodyModel::Drivings()
{
	return dconstraints;
}

xPointMass* xMultiBodyModel::XMass(std::string& _ws)
{
	if (_ws == "ground")
		return xModel::Ground();
	xstring ws = _ws;
	//QString ws = QString::fromStdString(_ws);
	//QStringList keys = masses.keys();
	xmap<xstring, xPointMass*>::iterator it = masses.find(ws);
	//QStringList::const_iterator it = qFind(keys, ws);
	if (it == masses.end())
		return NULL;
	return it.value();// masses[ws];
}

xKinematicConstraint* xMultiBodyModel::XJoint(std::string& _ws)
{
	xstring ws = _ws;// QString ws = QString::fromStdString(_ws);
	xmap<xstring, xKinematicConstraint*>::iterator it = constraints.find(ws);
	//QStringList::const_iterator it = qFind(keys, ws);
	if (it == constraints.end())
		return NULL;
	return it.value();// constraints[ws];
}

xForce* xMultiBodyModel::XForce(std::string& _ws)
{
	xstring ws = _ws;// QString ws = QString::fromStdString(_ws);
	//QStringList keys = forces.keys();
	xmap<xstring, xForce*>::iterator it = forces.find(ws);//QStringList::const_iterator it = qFind(keys, ws);
	if (it == forces.end())
		return NULL;
	return it.value();// forces[ws];
}

xDrivingConstraint* xMultiBodyModel::xDriving(std::string& _ws)
{
	xstring ws = _ws;// QString ws = QString::fromStdString(_ws);
	//QStringList keys = dconstraints.keys();
	xmap<xstring, xDrivingConstraint*>::iterator it = dconstraints.find(ws);//QStringList::const_iterator it = qFind(keys, ws);
	if (it == dconstraints.end())
		return NULL;
	return it.value();
}

xPointMass* xMultiBodyModel::CreatePointMass(std::string _name)
{
	xstring name = _name;
	xPointMass* xpm = NULL;
	if (xObjectManager::XOM()->XObject(_name))
	{
		xpm = (dynamic_cast<xPointMass*>(xObjectManager::XOM()->XObject(_name)));
		masses.insert(name, xpm);// = xpm;
// 		if (xpm->Shape() == MESH_SHAPE)
// 			dynamic_cast<xMeshObject*>(xObjectManager::XOM()->XObject(_name))->translation()
		return xpm;
	}
	xpm = new xPointMass(_name);
	masses.insert(name, xpm);
	xObjectManager::XOM()->addObject(xpm);
	xLog::log("Create PointMass : " + _name);
	//std::cout << "Create point mass - " << _name.c_str() << ", Num. mass - " << masses.size() << std::endl;
	return xpm;
}

xKinematicConstraint* xMultiBodyModel::CreateKinematicConstraint(
	std::string _name, xKinematicConstraint::cType _type, std::string _i, std::string _j)
{
	xstring name = _name;
	xKinematicConstraint* xkc = NULL;
	switch (_type)
	{
	case xKinematicConstraint::REVOLUTE:
		xkc = new xRevoluteConstraint(_name, _i, _j);
		xLog::log("Create Revolute Joint : " + _name);
		//std::cout << "Create revolute constraint - " << _name.c_str() << ", Num. constraint - " << constraints.size() << std::endl;
		break;
	case xKinematicConstraint::TRANSLATIONAL:
		xkc = new xTranslationConstraint(_name, _i, _j);
		xLog::log("Create Translation Joint : " + _name);
		//std::cout << "Create translation constraint - " << _name.c_str() << ", Num. constraint - " << constraints.size() << std::endl;
		break;
	case xKinematicConstraint::SPHERICAL:
		xkc = new xSphericalConstraint(_name, _i, _j);
		xLog::log("Create Spherical Joint : " + _name);
		//std::cout << "Create spherical constraint - " << _name.c_str() << ", Num. constraint - " << constraints.size() << std::endl;
		break;
	case xKinematicConstraint::UNIVERSAL:
		xkc = new xUniversalConstraint(_name, _i, _j);
		xLog::log("Create Universal Joint : " + _name);
		//std::cout << "Create universal constraint - " << _name.c_str() << ", Num. constraint - " << constraints.size() << std::endl;
		break;
	}
	constraints.insert(name, xkc);// = xkc;
	return xkc;
}

xForce* xMultiBodyModel::CreateForceElement(std::string _name, xForce::fType _type, std::string bn, std::string an)
{
	xstring name = _name;
	xForce* xf = NULL;
	switch (_type)
	{
	case xForce::TSDA:
		xf = new xSpringDamperForce(_name);
		xLog::log("Create Translational Spring Damper Element : " + _name);
		break;
	case xForce::TSDA_LIST_DATA:
		xf = new xSpringDamperForce(_name);
		xLog::log("Create Translational Spring Damper Element From List Data : " + _name);
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
	forces.insert(name, xf);
	return xf;
}

xDrivingConstraint* xMultiBodyModel::CreateDrivingConstraint(std::string _name, xKinematicConstraint* _kc)
{
	xstring name = _name;
	xDrivingConstraint* xdc = new xDrivingConstraint(_name, _kc);
	dconstraints.insert(name, xdc);
	xLog::log("Create Driving : " + _name);
	//std::cout << "Create driving constraint - " << _name.c_str() << ", Num. driving constraint - " << dconstraints.size() << std::endl;
	return xdc;
}

// void xMultiBodyModel::InsertPointMassFromShape(xPointMass* pm)
// {
// 
// }
