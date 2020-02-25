#include "xdynamics_object/xGeneralSpringDamper.h"
#include "xdynamics_object/xPoint2Spring.h"
#include "xdynamics_object/xRotationSpringDamperForce.h"
#include "xdynamics_manager/xObjectManager.h"

xGeneralSpringDamper::xGeneralSpringDamper()
	: nattach(0)
	, ntsda(0)
	, nrsda(0)
	, aps(nullptr)
{

}

xGeneralSpringDamper::xGeneralSpringDamper(std::string _name)
	: name(_name)
	, nattach(0)
	, ntsda(0)
	, nrsda(0)
	, aps(nullptr)
{
}

xGeneralSpringDamper::~xGeneralSpringDamper()
{
	if (aps) delete[] aps; aps = nullptr;
	rsda.delete_all;
}

void xGeneralSpringDamper::allocAttachPoint(unsigned int n)
{
	aps = new xAttachPoint[n];
}
//
//void xGeneralSpringDamper::allocRotationalSpring(unsigned int n)
//{
//	rsd = new xRotationalSpringData[n];
//}

void xGeneralSpringDamper::appendPoint2Spring(xPoint2Spring * _p2s)
{
	//p2s.insert(_p2s->Name(), _p2s);
}

void xGeneralSpringDamper::appendAttachPoint(unsigned int id, xAttachPoint & _xap)
{
	aps[id] = _xap;
}

void xGeneralSpringDamper::appendRotationalSpring(xRotationSpringDamperForce* _xrd)
{
	rsda.insert(_xrd->Name(), _xrd);
}

void xGeneralSpringDamper::calculateForce()
{
	/*xmap<xstring, xPoint2Spring*>::iterator it = p2s.begin();
	for (it; it != p2s.end(); it.next()) {
		it.value()->CalculateSpringForce();
	}*/
	xmap<xstring, xRotationSpringDamperForce*>::iterator rit = rsda.begin();
	while (rit.has_next()) {
		xRotationSpringDamperForce* rs = rit.value();
		//xPointMass* body = dynamic_cast<xPointMass*>(obj);
		//xPoint2Spring* ps = p2s.find(rd.p2s).value();
		vector3d mi = new_vector3d(0, 0, 0);
		vector3d mj = new_vector3d(0, 0, 0);
		//vector3d rj = new_vector3d(ps->p0->x, ps->p0->y, ps->p0->z);
		rs->xCalculateForceBodyAndP2S(mi, mj);
		xPoint2Spring* ps = dynamic_cast<xPoint2Spring*>(rs->GeneralObject());
		ps->m0->x += mj.x;
		ps->m0->y += mj.y;
		ps->m0->z += mj.z;
	}
}

xPoint2Spring * xGeneralSpringDamper::Point2Spring(std::string n)
{
	return p2s.find(n).value();
}

void xGeneralSpringDamper::setPointData(
	double * mass, double * pos,
	double * ep, double * vel, double * avel,
	double * force, double * moment)
{
	xmap<xstring, xPoint2Spring*>::iterator it = p2s.begin();
	vector4d* p = (vector4d*)pos;
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
		vector4d* p0 = p + id0;
		vector4d* p1 = p + id1;
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
