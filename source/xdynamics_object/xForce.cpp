#include "xdynamics_object/xForce.h"


xForce::xForce()
	: type(NO_TYPE)
	, i(0)
	, j(0)
	, spi(new_vector3d(0.0, 0.0, 0.0))
	, spj(new_vector3d(0.0, 0.0, 0.0))
{

}

xForce::xForce(std::string _name, fType _type)
	: name(_name)
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

void xForce::setBaseBodyName(std::string bn) { base = bn; }
void xForce::setActionBodyName(std::string an) { action = an; }
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

std::string xForce::Name()
{
	return name.toStdString();
}

xForce::fType xForce::Type()
{
	return type;
}

std::string xForce::BaseBodyName()
{
	return base.toStdString();
}

std::string xForce::ActionBodyName()
{
	return action.toStdString();
}

xObject * xForce::GeneralObject()
{
	return g_ptr;
}
