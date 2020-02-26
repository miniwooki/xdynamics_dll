#pragma once

#include "xmap.hpp"
#include "xdynamics_object/xPoint2Spring.h"

//class xPoint2Spring;
class xSpringDamperForce;
class xRotationSpringDamperForce;

class xGeneralSpringDamper
{
public:
	xGeneralSpringDamper();
	xGeneralSpringDamper(std::string _name);
	~xGeneralSpringDamper();

	void allocAttachPoint(unsigned int n);
	//void allocRotationalSpring(unsigned int n);
	void appendPoint2Spring(xPoint2Spring* _p2s);
	void appendAttachPoint(unsigned int id, xAttachPoint& _xap);
	void appendRotationalSpring(xRotationSpringDamperForce* _xrd);
	void calculateForce();
//	xPoint2Spring* Point2Spring(std::string n);
	//void setPointData(double* mass, double* pos, double* ep, double* vel, double* avel, double *force, double* moment);

private:
	xstring name;
	unsigned int nattach;
	unsigned int ntsda;
	unsigned int nrsda;
	xAttachPoint* aps;
	xmap<xstring, xSpringDamperForce*> tsda;
	xmap<xstring, xRotationSpringDamperForce*> rsda;
};