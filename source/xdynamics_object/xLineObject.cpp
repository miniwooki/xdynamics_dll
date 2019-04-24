#include "xdynamics_object/xLineObject.h"

xLineObject::xLineObject()
	: xPointMass()
{
	memset(&len, 0, sizeof(double) * 10);
}

xLineObject::xLineObject(std::string _name)
	: xPointMass(_name)
{
	memset(&len, 0, sizeof(double) * 10);
	xObject::shape = LINE_SHAPE;
}

xLineObject::~xLineObject()
{

}

bool xLineObject::define(vector3d p0, vector3d p1, vector3d n)
{
	return true;
}

vector3d xLineObject::Normal() const
{
	return normal;
}

vector3d xLineObject::StartPoint() const
{
	return spoint;
}

vector3d xLineObject::EndPoint() const
{
	return epoint;
}

void xLineObject::SetupDataFromStructure(xLineObjectData& d)
{
	this->define(
		new_vector3d(d.p0x, d.p0y, d.p0z),
		new_vector3d(d.p1x, d.p1y, d.p1z),
		new_vector3d(d.nx, d.ny, d.nz));
}

