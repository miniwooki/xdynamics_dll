#include "xdynamics_manager/xMultiBodyModel.h"
#include "xdynamics_manager/xObjectManager.h"
#include <map>


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

void xMultiBodyModel::AppendPointMass(xPointMass* xpm)
{
	xstring name = xpm->Name();
	masses.insert(name, xpm);
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

xDummyMass * xMultiBodyModel::CreateDummyMass(std::string _name)
{
	xstring name = _name;
	xDummyMass* xpm = NULL;
	if (xObjectManager::XOM()->XObject(_name))
	{
		xpm = (dynamic_cast<xDummyMass*>(xObjectManager::XOM()->XObject(_name)));
		masses.insert(name, xpm);// = xpm;
// 		if (xpm->Shape() == MESH_SHAPE)
// 			dynamic_cast<xMeshObject*>(xObjectManager::XOM()->XObject(_name))->translation()
		return xpm;
	}
	xpm = new xDummyMass(_name);
	masses.insert(name, xpm);
	xObjectManager::XOM()->addObject(xpm);
	xLog::log("Create dummy mass : " + _name);
	//std::cout << "Create point mass - " << _name.c_str() << ", Num. mass - " << masses.size() << std::endl;
	return xpm;
}

void xMultiBodyModel::CreatePointMassesFromFile(std::string _name)
{
	std::fstream sfs;
	sfs.open(_name, std::ios_base::in);
	if (sfs.is_open())
	{
		while (!sfs.eof())
		{
			xPointMassData xpm = { 0, };
			std::string nm;
			sfs >> nm >> xpm.mass >> xpm.ixx >> xpm.iyy >> xpm.izz
				>> xpm.ixy >> xpm.iyz >> xpm.ixz
				>> xpm.px >> xpm.py >> xpm.pz
				>> xpm.e0 >> xpm.e1 >> xpm.e2 >> xpm.e3
				>> xpm.vx >> xpm.vy >> xpm.vz;
			xPointMass* pm = CreatePointMass(nm);
			pm->SetDataFromStructure(masses.size(), xpm);
			
		}
	}
	sfs.close();
}

void xMultiBodyModel::CreateKinematicConstraintsFromFile(std::string _name)
{
	std::fstream sfs;
	sfs.open(_name, std::ios_base::in);
	if (sfs.is_open())
	{
		while (!sfs.eof())
		{
			std::string nm;
			std::string base;
			std::string action;
			int type_id = -1;
			xKinematicConstraint::cType type;
			xJointData xjd = { 0, };
			bool isdriving = false;
			sfs >> nm >> type_id >> base >> action
				>> xjd.lx >> xjd.ly >> xjd.lz
				>> xjd.fix >> xjd.fiy >> xjd.fiz
				>> xjd.gix >> xjd.giy >> xjd.giz
				>> xjd.fjx >> xjd.fjy >> xjd.fjz
				>> xjd.gjx >> xjd.gjy >> xjd.gjz >> isdriving;
			type = (xKinematicConstraint::cType)type_id;
			xKinematicConstraint* xkc = CreateKinematicConstraint(nm, type, base, action);
			xkc->SetupDataFromStructure(XMass(base), XMass(action), xjd);
			if (isdriving)
			{
				std::string d_name;
				double st;
				double cv;
				sfs >> d_name >> st >> cv;
				xDrivingConstraint* xdc = CreateDrivingConstraint(d_name, xkc);
				xdc->setStartTime(st);
				xdc->setConstantVelocity(cv);
			}
		}
	}
	sfs.close();
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
	case xKinematicConstraint::FIXED:
		xkc = new xFixConstraint(_name, _i, _j);
		xLog::log("Creat Fixed Joint : " + _name);
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
		xf = new xRotationSpringDamperForce(_name);
		xLog::log("Create Rotational Spring Damper Element From List Data : " + _name);
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

void xMultiBodyModel::set_driving_rotation_data(unsigned int i, xDrivingRotationResultData xdrr)
{
	xmap<xstring, xDrivingConstraint*>::iterator it = dconstraints.begin();
	for (unsigned int k = 0; k < i; k++)
		it.next();
	xDrivingConstraint* xdc = it.value();
	xdc->setRevolutionCount(xdrr.rev_count);
	xdc->setRotationAngle(xdrr.theta);
	xdc->setDerivativeRevolutionCount(xdrr.drev_count);
}

// void xMultiBodyModel::InsertPointMassFromShape(xPointMass* pm)
// {
// 
// }
