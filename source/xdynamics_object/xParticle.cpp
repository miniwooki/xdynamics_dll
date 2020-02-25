#include "xdynamics_object/xParticle.h"

xParticle::xParticle()
	: id(-1)
	, radius(0)
	, xPointMass()
{

}

xParticle::xParticle(std::string name)
	: id(-1)
	, radius(0)
	, xPointMass(name, PARTICLE)
{
}

xParticle::~xParticle()
{
}

void xParticle::setIndex(unsigned int _id)
{
	id = _id;
}

void xParticle::setRadius(double v)
{
	radius = v;
}
