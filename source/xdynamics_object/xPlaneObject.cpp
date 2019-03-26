#include "xdynamics_object/xPlaneObject.h"

xPlaneObject::xPlaneObject()
	: xPointMass()
{
	memset(&l1, 0, sizeof(double) * 2 + sizeof(vector3d) * 11);
	xObject::shape = PLANE_SHAPE;
}

xPlaneObject::xPlaneObject(std::string _name)
	: xPointMass(_name)
{
	memset(&l1, 0, sizeof(double) * 2 + sizeof(vector3d) * 11);
	xObject::shape = PLANE_SHAPE;
}

xPlaneObject::~xPlaneObject()
{

}

bool xPlaneObject::define(double dx, double dy, vector3d& d, vector3d& _xw, vector3d& _pc)
{
	xw = _xw;
	minp.x = xw.x < minp.x ? xw.x : minp.x; minp.y = xw.y < minp.y ? xw.y : minp.y; minp.z = xw.z < minp.z ? xw.z : minp.z;
	maxp.x = xw.x > maxp.x ? xw.x : maxp.x; maxp.y = xw.y > maxp.y ? xw.y : maxp.y; maxp.z = xw.z > maxp.z ? xw.z : maxp.z;
	w3 = _pc;
	minp.x = w3.x < minp.x ? w3.x : minp.x; minp.y = w3.y < minp.y ? w3.y : minp.y; minp.z = w3.z < minp.z ? w3.z : minp.z;
	maxp.x = w3.x > maxp.x ? w3.x : maxp.x; maxp.y = w3.y > maxp.y ? w3.y : maxp.y; maxp.z = w3.z > maxp.z ? w3.z : maxp.z;
	w2 = cross(d, _xw);
	w2 = w2 /length(w2);
	double hl = sqrt(0.25 * dx * dx + 0.25 * dy * dy);
	w2 = hl * w2;
	minp.x = w2.x < minp.x ? w2.x : minp.x; minp.y = w2.y < minp.y ? w2.y : minp.y; minp.z = w2.z < minp.z ? w2.z : minp.z;
	maxp.x = w2.x > maxp.x ? w2.x : maxp.x; maxp.y = w2.y > maxp.y ? w2.y : maxp.y; maxp.z = w2.z > maxp.z ? w2.z : maxp.z;
	w4 = cross(d, _pc);
	w4 = w4 / length(w4);
	w4 = hl * w4;
	minp.x = w4.x < minp.x ? w4.x : minp.x; minp.y = w4.y < minp.y ? w4.y : minp.y; minp.z = w4.z < minp.z ? w4.z : minp.z;
	maxp.x = w4.x > maxp.x ? w4.x : maxp.x; maxp.y = w4.y > maxp.y ? w4.y : maxp.y; maxp.z = w4.z > maxp.z ? w4.z : maxp.z;
	xPointMass::pos = 0.5 * (_xw + _pc);
	pa = w2 - xw;
	pb = w4 - xw;
	l1 = length(pa);// .length();
	l2 = length(pb);// .length();
	u1 = pa / l1;
	u2 = pb / l2;
	uw = cross(u1, u2);
	return true;
}

bool xPlaneObject::define(vector3d& _xw, vector3d& _pa, vector3d& _pb)
{
	w2 = _pa;
	//w3 = _pc;
	w4 = _pb;
	xw = _xw;

	pa = _pa;
	pb = _pb;

	pa -= xw;
	pb -= xw;
	l1 = length(pa);// .length();
	l2 = length(pb);// .length();
	u1 = pa / l1;
	u2 = pb / l2;
	uw = cross(u1, u2);
	xPointMass::pos = 0.5 * (_pa + _pb);
	w3 = xw + w2 + l2 * u2;
	return true;
}

double xPlaneObject::L1() const { return l1; }
double xPlaneObject::L2() const { return l2; }
vector3d xPlaneObject::U1() const { return u1; }
vector3d xPlaneObject::U2() const { return u2; }
vector3d xPlaneObject::UW() const { return uw; }
vector3d xPlaneObject::XW() const { return xw; }
vector3d xPlaneObject::PA() const { return pa; }
vector3d xPlaneObject::PB() const { return pb; }
vector3d xPlaneObject::W2() const { return w2; }
vector3d xPlaneObject::W3() const { return w3; }
vector3d xPlaneObject::W4() const { return w4; }
vector3d xPlaneObject::MinPoint() const { return minp; }
vector3d xPlaneObject::MaxPoint() const { return maxp; }

void xPlaneObject::SetupDataFromStructure(xPlaneObjectData& d)
{
	this->define(
		d.dx, d.dy,
		new_vector3d(d.drx, d.dry, d.drz),
		new_vector3d(d.pox, d.poy, d.poz),
		new_vector3d(d.p1x, d.p1y, d.p1z));
}
