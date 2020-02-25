#include "..\..\include\xdynamics_object\xDummyMass.h"

xDummyMass::xDummyMass()
	: xPointMass()
	, dependency_mass(nullptr)
{
}

xDummyMass::xDummyMass(std::string name)
	: xPointMass(name, DUMMY_SHAPE)
	, dependency_mass(nullptr)
{
}

xDummyMass::~xDummyMass()
{
}

void xDummyMass::setDependencyBody(xPointMass * body)
{
	dependency_mass = body;
}

void xDummyMass::setRelativeLocation(double x, double y, double z)
{
	relative_loc.x = x;
	relative_loc.y = y;
	relative_loc.z = z;
}
