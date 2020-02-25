#pragma once

#include "xdynamics_algebra/xAlgebraMath.h"
#include "xdynamics_object/xObject.h"

class xSpringDamperForce;

class xPoint2Spring : public xObject
{
public:
	xPoint2Spring();
	xPoint2Spring(std::string name);
	virtual ~xPoint2Spring();

	void SetSpringDamperData(
		unsigned int i0, unsigned int i1,
		vector3d sp0, vector3d sp1,
		double _k, double _c, double _len);

	void ConnectFirstPoint(
		double mass,
		vector4d* p, vector3d* v,
		euler_parameters* ep, euler_parameters* ev, 
		vector3d* force, vector3d* moment);
	void ConnectSecondPoint(
		double mass,
		vector4d* p, vector3d* v,
		euler_parameters* ep, euler_parameters* ev, 
		vector3d* force, vector3d* moment);
	void CalculateSpringForce();

	unsigned int FirstIndex();
	unsigned int SecondIndex();

public:
	vector3d spi, spj;
	unsigned int id0, id1;
	double k, c;
	double init_l;
	double mass0, mass1;
	vector4d* p0, *p1;
	vector3d* v0, *v1;
	euler_parameters* ep0, *ep1;
	euler_parameters* ev0, *ev1;
	vector3d* f0, *f1;
	vector3d* m0, *m1;
};