#include "xdynamics_manager/xModel.h"
#include "xdynamics_object/xPointMass.h"
//#include "boost/filesystem.hpp"
#include <windows.h>

//using namespace boost::filesystem;

QString xModel::name = "Model1";
QString xModel::path = kor(getenv("USERPROFILE")) + "/Documents/xdynamics/";
xPointMass* xModel::ground = NULL;
xModel::angle_type xModel::angle = xModel::EULER_PARAMETERS;
vector3d xModel::gravity = new_vector3d(0.0, -9.80665, 0.0);
//resultStorage* model::rs = NULL;// new resultStorage;
//int model::count = -1;
xModel::unit_type xModel::unit = MKS;
//bool model::isSinglePrecision = false;

xModel::xModel()
{
	//xLog();
// 	wchar_t* pv = NULL;
// 	size_t len = 0;
// 	_wdupenv_s(&pv, &len, L"USERPROFILE");
// 	path = pv;// wsprintfW(path, TEXT("%s"), pv);
// 	//wchar_t wcarr[11] = { 0, };
// 	//MultiByteToWideChar(CP_ACP, NULL, "/Documents/", -1, wcarr, 11 * 2);
// 	path = path + L"/Documents/xdynamics/";
// 		//lstrcatW(path, L"/Documents/xdynamics/");
// 	delete pv;
}

xModel::xModel(const QString _name)
{
	name = _name;
	std::cout << _name.toStdString() << std::endl;
	xUtilityFunctions::CreateDirectory(path.toStdWString().c_str());
	if (!ground)
	{
		ground = new xPointMass("ground");
		ground->setXpmIndex(-1);
	}
}

xModel::~xModel()
{
	if (ground) delete ground; ground = NULL;
	xLog::releaseLogSystem();
	//if (lg) delete lg; lg = NULL;
}

int xModel::OneDOF()
{
	return (int)angle;
}

xPointMass* xModel::Ground()
{
	return ground;
}

void xModel::setModelName(const QString n)
{
	if (name != n)
	{
		name = n;// wsprintfW(name, TEXT("%s"), n);
		QString full_path = path + name + "/";
		xUtilityFunctions::CreateDirectory(full_path.toStdWString().c_str());
		launchLogSystem(full_path.toStdString());
		xLog::log("Change Directory : " + full_path.toStdString());
	}
	else
	{
		xLog::log("Not change : You have attempted to change to the currently configured path");
	}
}

void xModel::setModelPath(const QString p)
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

void xModel::launchLogSystem(std::string lpath)
{
	xLog::launchLogSystem(lpath);
}

std::string xModel::makeFilePath(std::string file_name)
{
	std::string f = path.toStdString() + name.toStdString() + "/" + file_name;
	return f;
}