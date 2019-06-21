#include "xdynamics_object/xForce.h"


xForce::xForce()
	: type(NO_TYPE)
	, i(0)
	, j(0)
	, spi(new_vector3d(0.0, 0.0, 0.0))
	, spj(new_vector3d(0.0, 0.0, 0.0))
{

}

xForce::xForce(std::wstring _name, fType _type)
	: name(QString::fromStdWString(_name))
	, type(_type)
	, i(0)
	, j(0)
	, spi(new_vector3d(0.0, 0.0, 0.0))
	, spj(new_vector3d(0.0, 0.0, 0.0))
{

}

xForce::~xForce()
{

}

void xForce::setBaseBodyName(std::wstring bn) { base = QString::fromStdWString(bn); }
void xForce::setActionBodyName(std::wstring an) { action = QString::fromStdWString(an); }
void xForce::setBaseBodyIndex(int _i) { i = _i; }
void xForce::setActionBodyIndex(int _j){ j = _j; }
// void xForce::setBaseLocalCoordinate() 
// { 
// 	if (i_ptr)
// 	{
// 
// 	}
// }
// void xForce::setActionLocalCoordinate() 
// { 
// 	spj = _spj; 
// }

std::wstring xForce::Name()
{
	return name.toStdWString();
}

xForce::fType xForce::Type()
{
	return type;
}

std::wstring xForce::BaseBodyName()
{
	return base.toStdWString();
}

std::wstring xForce::ActionBodyName()
{
	return action.toStdWString();
}
