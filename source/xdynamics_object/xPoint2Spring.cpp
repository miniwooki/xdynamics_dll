#include "xdynamics_object/xPoint2Spring.h"
#include "xdynamics_object/xSpringDamperForce.h"

xPoint2Spring::xPoint2Spring()
	: xObject()
	, id0(0)
	, id1(0)
	, p0(nullptr), p1(nullptr)
	, v0(nullptr), v1(nullptr)
	, ep0(nullptr), ep1(nullptr)
	, ev0(nullptr), ev1(nullptr)
	, f0(nullptr), f1(nullptr)
	, m0(nullptr), m1(nullptr)
	, mass0(0), mass1(0)
	, k(0), c(0), init_l(0)
{

}

xPoint2Spring::xPoint2Spring(std::string name)
	: xObject(name)
	, id0(0)
	, id1(0)
	, p0(nullptr), p1(nullptr)
	, v0(nullptr), v1(nullptr)
	, ep0(nullptr), ep1(nullptr)
	, ev0(nullptr), ev1(nullptr)
	, f0(nullptr), f1(nullptr)
	, m0(nullptr), m1(nullptr)
	, mass0(0), mass1(0)
	, k(0), c(0), init_l(0)
{

}

xPoint2Spring::~xPoint2Spring()
{

}

void xPoint2Spring::SetSpringDamperData(
	unsigned int i0, unsigned int i1,
	vector3d sp0, vector3d sp1,
	double _k, double _c, double _len)
{
	id0 = i0;
	id1 = i1;
	spi = sp0;
	spj = sp1;
	k = _k;
	c = _c;
	init_l = _len;
}

void xPoint2Spring::ConnectFirstPoint(
	double mass, vector3d sp, vector3d* p, vector3d* v,
	euler_parameters* ep, euler_parameters* ev,
	vector3d* force, vector3d* moment)
{
	p0 = p;
	v0 = v;
	ep0 = ep;
	ev0 = ev;
	f0 = force;
	m0 = moment;
	mass0 = mass;
}

void xPoint2Spring::ConnectSecondPoint(
	double mass, vector3d sp, vector3d* p, vector3d* v,
	euler_parameters* ep, euler_parameters* ev,
	vector3d* force, vector3d* moment)
{
	p0 = p;
	v0 = v;
	ep0 = ep;
	ev0 = ev;
	f0 = force;
	m0 = moment;
	mass0 = mass;
}

void xPoint2Spring::CalculateSpringForce()
{
	vector3d fi, fj;
	vector3d mi, mj;
	xSpringDamperForce::xCalculateForce(
		spi, spj, k, c, init_l, *p0, *p1,
		*v0, *v1, *ep0, *ep1, *ev0, *ev1,
		fi, fj, mi, mj);
	f0->x = fi.x; f0->y = fi.y; f0->z = fi.z;
	f1->x = fj.x; f1->y = fj.y; f1->z = fj.z;
	m0->x = mi.x; m0->y = mi.y; m0->z = mi.z;
	m1->x = mj.x; m1->y = mj.y; m1->z = mj.z;
}

unsigned int xPoint2Spring::FirstIndex()
{
	return id0;
}

unsigned int xPoint2Spring::SecondIndex()
{
	return id1;
}
