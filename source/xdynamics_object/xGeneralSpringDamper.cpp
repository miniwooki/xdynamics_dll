#include "..\..\include\xdynamics_object\xGeneralSpringDamper.h"
#pragma once

xGeneralSpringDamper::xGeneralSpringDamper()
{
}

xGeneralSpringDamper::xGeneralSpringDamper(std::string _name)
	: name(_name)
{
}

xGeneralSpringDamper::~xGeneralSpringDamper()
{

}

void xGeneralSpringDamper::allocAttachPoint(unsigned int n)
{
	aps = new xAttachPoint[n];
}

void xGeneralSpringDamper::allocRotationalSpring(unsigned int n)
{
	rsd = new xRotationalSpringData[n];
}

void xGeneralSpringDamper::appendPoint2Spring(xPoint2Spring * _p2s)
{
	p2s.insert(_p2s->Name(), _p2s);
}

void xGeneralSpringDamper::appendAttachPoint(unsigned int id, xAttachPoint & _xap)
{
	aps[id] = _xap;
}

void xGeneralSpringDamper::appendRotationalSpring(unsigned int id, xRotationalSpringData & _xrd)
{
	rsd[id] = _xrd;
}

void xGeneralSpringDamper::setPointData(
	double * mass, double * pos,
	double * ep, double * vel, double * avel,
	double * force, double * moment)
{
	xmap<xstring, xPoint2Spring*>::iterator it = p2s.begin();
	vector3d* p = (vector3d*)pos;
	vector3d* v = (vector3d*)vel;
	euler_parameters* e = (euler_parameters*)ep;
	euler_parameters* ed = (euler_parameters*)avel;
	vector3d* f = (vector3d*)force;
	vector3d* m = (vector3d*)moment;
	for (it; it != p2s.end(); it.next()) {
		xPoint2Spring* ps = it.value();
		unsigned int id0 = ps->FirstIndex();
		unsigned int id1 = ps->SecondIndex();
		double mass0 = mass[id0];
		double mass1 = mass[id1];
		vector3d* p0 = p + id0;
		vector3d* p1 = p + id1;
		vector3d* v0 = v + id0;
		vector3d* v1 = v + id1;
		euler_parameters *e0 = e + id0;
		euler_parameters *e1 = e + id1;
		euler_parameters *ed0 = ed + id0;
		euler_parameters *ed1 = ed + id1;
		vector3d* f0 = f + id0;
		vector3d* f1 = f + id1;
		vector3d* m0 = m + id0;
		vector3d* m1 = m + id1;
		ps->ConnectFirstPoint(mass0, p0, v0, e0, ed0, f0, m0);
		ps->ConnectSecondPoint(mass1, p1, v1, e1, ed1, f1, m1);
	}
}
