#pragma once

#include "xmap.hpp"
#include "xdynamics_object/xPoint2Spring.h"

class xGeneralSpringDamper
{
public:
	xGeneralSpringDamper();
	xGeneralSpringDamper(std::string _name);
	~xGeneralSpringDamper();

	void allocAttachPoint(unsigned int n);
	void allocRotationalSpring(unsigned int n);
	void appendPoint2Spring(xPoint2Spring* _p2s);
	void appendAttachPoint(unsigned int id, xAttachPoint& _xap);
	void appendRotationalSpring(unsigned int id, xRotationalSpringData& _xrd);

	void setPointData(double* mass, double* pos, double* ep, double* vel, double* avel, double *force, double* moment);

private:
	xstring name;
	xmap<xstring, xPoint2Spring*> p2s;
	unsigned int nattach;
	unsigned int nrsda;
	xAttachPoint* aps;
	xRotationalSpringData* rsd;
};