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

bool xPlaneObject::define(vector3d& p0, vector3d& p1, vector3d& p2, vector3d& p3)
{
	xw = p0;
	w2 = p1;
	w3 = p2;
	w4 = p3;
	minp.x = xw.x < minp.x ? xw.x : minp.x; minp.y = xw.y < minp.y ? xw.y : minp.y; minp.z = xw.z < minp.z ? xw.z : minp.z;
	maxp.x = xw.x > maxp.x ? xw.x : maxp.x; maxp.y = xw.y > maxp.y ? xw.y : maxp.y; maxp.z = xw.z > maxp.z ? xw.z : maxp.z;
	minp.x = w3.x < minp.x ? w3.x : minp.x; minp.y = w3.y < minp.y ? w3.y : minp.y; minp.z = w3.z < minp.z ? w3.z : minp.z;
	maxp.x = w3.x > maxp.x ? w3.x : maxp.x; maxp.y = w3.y > maxp.y ? w3.y : maxp.y; maxp.z = w3.z > maxp.z ? w3.z : maxp.z;
	minp.x = w2.x < minp.x ? w2.x : minp.x; minp.y = w2.y < minp.y ? w2.y : minp.y; minp.z = w2.z < minp.z ? w2.z : minp.z;
	maxp.x = w2.x > maxp.x ? w2.x : maxp.x; maxp.y = w2.y > maxp.y ? w2.y : maxp.y; maxp.z = w2.z > maxp.z ? w2.z : maxp.z;
	minp.x = w4.x < minp.x ? w4.x : minp.x; minp.y = w4.y < minp.y ? w4.y : minp.y; minp.z = w4.z < minp.z ? w4.z : minp.z;
	maxp.x = w4.x > maxp.x ? w4.x : maxp.x; maxp.y = w4.y > maxp.y ? w4.y : maxp.y; maxp.z = w4.z > maxp.z ? w4.z : maxp.z;
	xPointMass::pos = 0.5 * (xw + w3);
	pa = w2 - xw;
	pb = w4 - xw;
	l1 = length(pa);// .length();
	l2 = length(pb);// .length();
	u1 = pa / l1;
	u2 = pb / l2;
	uw = cross(u1, u2);
	local_point[0] = toLocal(xw - xPointMass::pos);
	local_point[1] = toLocal(w2 - xPointMass::pos);
	local_point[2] = toLocal(w3 - xPointMass::pos);
	local_point[3] = toLocal(w4 - xPointMass::pos);
	return true;
}

bool xPlaneObject::define(bool isMovingObject, vector3d& _xw, vector3d& _pa, vector3d& _pb)
{
	this->setDynamicsBody(isMovingObject);
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
	w3 = xw + pa + pb;
	xPointMass::pos = 0.5 * (xw + w3);
	local_point[0] = toLocal(xw - xPointMass::pos);
	local_point[1] = toLocal(w2 - xPointMass::pos);
	local_point[2] = toLocal(w3 - xPointMass::pos);
	local_point[3] = toLocal(w4 - xPointMass::pos);
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

vector3d xPlaneObject::LocalPoint(unsigned int i)
{
	return local_point[i];
}

void xPlaneObject::SetupDataFromStructure(xPlaneObjectData& d)
{
	data = d;
	this->define(
		new_vector3d(d.p0x, d.p0y, d.p0z),
		new_vector3d(d.p1x, d.p1y, d.p1z),
		new_vector3d(d.p2x, d.p2y, d.p2z),
		new_vector3d(d.p3x, d.p3y, d.p3z));
}

unsigned int xPlaneObject::create_sph_particles(double ps, unsigned int nlayers, vector3d* p, xMaterialType* t)
{
	unsigned int nx = static_cast<unsigned int>((l1 / ps) + 1e-9) - 1;
	unsigned int ny = static_cast<unsigned int>((l2 / ps) + 1e-9) - 1;
	unsigned int count = 0;
	if (material == FLUID){
		for (unsigned int x = 0; x < nx; x++){
			vector3d px = xw + (x * ps) * pa;
			for (unsigned int y = 0; y < ny; y++){
				if (p){
					vector3d _p = px + (xw + (y * ps) * pb);
					p[count] = new_vector3d(_p.x, _p.y, _p.z);
					t[count] = FLUID;
				}
				count++;
			}
		}
	}
	
	return count;
}
