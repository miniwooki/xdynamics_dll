#pragma once

#include "xdynamics_object/xPointMass.h"

class xParticle : public xPointMass
{
public:
	xParticle();
	xParticle(std::string name);
	~xParticle();

	void setIndex(unsigned int _id);
	void setRadius(double v);

private:
	long int id;
	double radius;
};
