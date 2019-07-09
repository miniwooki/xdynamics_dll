#include "xdynamics_object/xParticleCylinderContact.h"
#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_object/xCylinderObject.h"
#include "xdynamics_simulation/xSimulation.h"

xParticleCylinderContact::xParticleCylinderContact()
	: xContact()
	, p_ptr(NULL)
	, c_ptr(NULL)
	, dci(NULL)
{

}

xParticleCylinderContact::xParticleCylinderContact(std::string _name, xObject* o1, xObject* o2)
	: xContact(_name, PARTICLE_CYLINDER)
	, p_ptr(NULL)
	, c_ptr(NULL)
	, dci(NULL)
{
	if (o1->Shape() == CYLINDER_SHAPE)
	{
		c_ptr = dynamic_cast<xCylinderObject*>(o1);
		p_ptr = dynamic_cast<xParticleObject*>(o2);
	}
	else
	{
		c_ptr = dynamic_cast<xCylinderObject*>(o2);
		p_ptr = dynamic_cast<xParticleObject*>(o1);
	}
}

xParticleCylinderContact::~xParticleCylinderContact()
{

}

// void xParticleCylinderContact::allocHostMemory(unsigned int n)
// {
// 
// }

bool xParticleCylinderContact::pcylCollision(xContactPairList* pairs, unsigned int i, double r, double m, vector3d& p, vector3d& v, vector3d& o, double &R, vector3d& T, vector3d& F, vector3d& M, unsigned int nco, xClusterInformation* xci, vector4d* cpos)
{
	return true;
}

void xParticleCylinderContact::updateCollisionPair(
	xContactPairList& xcpl, double r, vector3d pos)
{
	vector3d cpt = new_vector3d(0, 0, 0);
	vector3d unit = new_vector3d(0, 0, 0);
	double cdist = particle_cylinder_contact_detection(pos, unit, cpt, r);
	if (cdist > 0)
	{
		//if(xcpl.IsNewCylinderContactPair(id))
	}
}

void xParticleCylinderContact::cudaMemoryAlloc(unsigned int np)
{
	xContact::cudaMemoryAlloc(np);
	if (xSimulation::Gpu())
	{

	}
}

void xParticleCylinderContact::cuda_collision(double *pos, double *vel, double *omega, double *mass, double *force, double *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{

}

device_cylinder_info* xParticleCylinderContact::deviceCylinderInfo()
{
	return dci;
}

double xParticleCylinderContact::particle_cylinder_contact_detection(
	vector3d& pt, vector3d& u, vector3d& cp, double r)
{
	double dist = -1.0;
	vector3d ab = new_vector3d(hci.ptop.x - hci.pbase.x, hci.ptop.y - hci.pbase.y, hci.ptop.z - hci.pbase.z);
	vector3d p = new_vector3d(pt.x, pt.y, pt.z);
	double t = dot(p - hci.pbase, ab) / dot(ab, ab);
	vector3d _cp = new_vector3d(0.0, 0.0, 0.0);
	if (t >= 0 && t <= 1) {
		_cp = hci.pbase + t * ab;
		dist = length(p - _cp);
		u = (_cp - p) / dist;
		cp = _cp - hci.len_rr.z * u;
		return hci.len_rr.y + r - dist;
	}
	else {

		_cp = hci.pbase + t * ab;
		dist = length(p - _cp);
		if (dist < hci.len_rr.z) {
			vector3d OtoCp = c_ptr->Position() - _cp;
			double OtoCp_ = length(OtoCp);
			u = OtoCp / OtoCp_;
			cp = _cp - hci.len_rr.z * u;
			return hci.len_rr.x * 0.5 + r - OtoCp_;
		}
		/*vector3d A_1 = makeTFM_1(hci.ep);
		vector3d A_2 = makeTFM_2(hci.ep);
		vector3d A_3 = makeTFM_3(hci.ep);*/
		vector3d _at = p - hci.ptop;
		vector3d at = c_ptr->toLocal(_at);
		//double r = length(at);
		cp = hci.ptop;
		if (abs(at.y) > hci.len_rr.x) {
			_at = p - hci.pbase;
			at = c_ptr->toLocal(_at);
			cp = hci.pbase;
		}
		double pi = atan(at.x / at.z);
		if (pi < 0 && at.z < 0) {
			_cp.x = hci.len_rr.z * sin(-pi);
		}
		else if (pi > 0 && at.x < 0 && at.z < 0) {
			_cp.x = hci.len_rr.z * sin(-pi);
		}
		else {
			_cp.x = hci.len_rr.z * sin(pi);
		}
		_cp.z = hci.len_rr.z * cos(pi);
		if (at.z < 0 && _cp.z > 0) {
			_cp.z = -_cp.z;
		}
		else if (at.z > 0 && _cp.z < 0) {
			_cp.z = -_cp.z;
		}
		_cp.y = 0.;
		cp = cp + c_ptr->toGlobal(_cp);

		vector3d disVec = cp - p;
		dist = length(disVec);
		u = disVec / dist;
		if (dist < r) {
			return r - dist;
		}
	}
	return -1.0;
}
