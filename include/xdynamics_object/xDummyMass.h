#pragma once

#include "xdynamics_object/xPointMass.h"

class xDummyMass : public xPointMass
{
public:
	xDummyMass();
	xDummyMass(std::string name);
	~xDummyMass();

	void setDependencyBody(xPointMass* body);
	void setRelativeLocation(double x, double y, double z);

private:
	xPointMass* dependency_mass;
	vector3d relative_loc;
};
