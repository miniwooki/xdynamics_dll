#include "xdynamics_manager/xModel.h"
#include "xdynamics_object/xPointMass.h"
//#include "boost/filesystem.hpp"
#include <windows.h>

//using namespace boost::filesystem;

xstring xModel::name = "Model1";
xstring xModel::path = xstring(getenv("USERPROFILE")) + xstring("/Documents/xdynamics/");
xPointMass* xModel::ground = NULL;
xModel::angle_type xModel::angle = xModel::EULER_PARAMETERS;
vector3d xModel::gravity = new_vector3d(0.0, -9.80665, 0.0);
//resultStorage* model::rs = NULL;// new resultStorage;
//int model::count = -1;
xModel::unit_type xModel::unit = MKS;
//bool model::isSinglePrecision = false;

xModel::xModel()
{

}

xModel::xModel(const std::string _name)
{
	name = _name;
	std::cout << _name << std::endl;
	xUtilityFunctions::CreateDirectory(path.toStdString().c_str());
	if (!ground)
	{
		ground = new xPointMass("ground");
		ground->setXpmIndex(-1);
	}
	gps = gps;
}

xModel::~xModel()
{
	if (ground) delete ground; ground = NULL;
	xLog::releaseLogSystem();
	//if (lg) delete lg; lg = NULL;
	xModel::initialize();
}

void xModel::initialize()
{
	name = "Model1";
	path = xstring(getenv("USERPROFILE")) + xstring("/Documents/xdynamics/");
	ground = NULL;
	angle = xModel::EULER_PARAMETERS;
	gravity = new_vector3d(0.0, -9.80665, 0.0);
	//resultStorage* model::rs = NULL;// new resultStorage;
	//int model::count = -1;
	unit = MKS;
}

int xModel::OneDOF()
{
	return (int)angle;
}

xPointMass* xModel::Ground()
{
	return ground;
}

void xModel::setModelName(const std::string n)
{
	if (name != n)
	{
		name = n;// wsprintfW(name, TEXT("%s"), n);
		std::string full_path = path.toStdString() + n + "/";
		xUtilityFunctions::CreateDirectory(full_path.c_str());
		launchLogSystem(full_path);
		xLog::log("Change Directory : " + full_path);
	}
	else
	{
		xLog::log("Not change : You have attempted to change to the currently configured path");
	}
}

void xModel::setModelPath(const std::string p)
{
	path = p;// wsprintfW(path, TEXT("%s"), p);
}

void xModel::setGravity(double g, int d)
{
	vector3d u;
	switch (d)
	{
	case 1: u.x = 1.0; break;
	case 2: u.y = 1.0; break;
	case 3: u.z = 1.0; break;
	case -1: u.x = -1.0; break;
	case -2: u.y = -1.0; break;
	case -3: u.z = -1.0; break;
	}
	gravity = g * u;
}

void xModel::setGravity(double x, double y, double z)
{
	gravity.x = x;
	gravity.y = y;
	gravity.z = z;
}

void xModel::launchLogSystem(std::string lpath)
{
	xLog::launchLogSystem(lpath);
}

std::string xModel::getModelName()
{
	return name.toStdString();
}

std::string xModel::makeFilePath(std::string file_name)
{
	std::string f = path.toStdString() + name.toStdString() + "/" + file_name;
	return f;
}
