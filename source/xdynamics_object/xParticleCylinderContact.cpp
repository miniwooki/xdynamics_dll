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
	hci = { c_ptr->cylinder_length(), c_ptr->cylinder_bottom_radius(), c_ptr->cylinder_top_radius(), c_ptr->bottom_position(), c_ptr->top_position() };
	empty_cylinder_part = c_ptr->empty_part_type();
}

xParticleCylinderContact::~xParticleCylinderContact()
{

}

// void xParticleCylinderContact::allocHostMemory(unsigned int n)
// {
// 
// }

bool xParticleCylinderContact::pcylCollision(
	xContactPairList* pairs, unsigned int i, double r, 
	double m, vector3d& p, vector3d& v, vector3d& o, 
	double &R, vector3d& T, vector3d& F, vector3d& M, 
	unsigned int nco, xClusterInformation* xci, vector4d* cpos)
{
	unsigned int ci = 0;
	unsigned int neach = 1;
	vector3d cp = p;
	if (nco && cpos)
	{
		for (unsigned int j = 0; j < nco; j++)
			if (i >= xci[j].sid && i < xci[j].sid + xci[j].count * xci[j].neach)
				neach = xci[j].neach;
		//ck = j / neach;
		ci = i / neach;
		cp = new_vector3d(cpos[ci].x, cpos[ci].y, cpos[ci].z);
	}
	foreach(xPairData* d, pairs->CylinderPair())
	{

		vector3d m_fn = new_vector3d(0.0, 0.0, 0.0);
		vector3d m_m = new_vector3d(0.0, 0.0, 0.0);
		vector3d m_ft = new_vector3d(0.0, 0.0, 0.0);
		double rcon = r - 0.5 * d->gab;
		vector3d u = new_vector3d(d->nx, d->ny, d->nz);
		vector3d cpt = new_vector3d(d->cpx, d->cpy, d->cpz);
		vector3d dcpr = cpt - cp;
		//vector3d cp = r * u;
		vector3d dv = -v - cross(o, dcpr);
		//unsigned int jjjj = d->id;
		//xContactMaterialParameters cmp = hcmp[d->id];
		xContactParameters c = getContactParameters(
			r, 0.0,
			m, 0.0,
			mpp.Ei, mpp.Ej,
			mpp.Pri, mpp.Prj,
			mpp.Gi, mpp.Gj,
			restitution, stiffnessRatio,
			friction, rolling_factor, cohesion);
		if (d->gab < 0 && abs(d->gab) < abs(c.coh_s))
		{
			double f = JKRSeperationForce(c, cohesion);
			double cf = cohesionForce(cohesion, d->gab, c.coh_r, c.coh_e, c.coh_s, f);
			F -= cf * u;
			continue;
		}
		else if (d->isc && d->gab < 0 && abs(d->gab) > abs(c.coh_s))
		{
			d->isc = false;
			continue;
		}
		switch (force_model)
		{
		case DHS: DHSModel(c, d->gab, d->delta_s, d->dot_s, cohesion, dv, u, m_fn, m_ft); break;
		}
		/*if (cmp.cohesion && c.coh_s > d->gab)
			d->isc = false;*/
		RollingResistanceForce(c.rfric, r, 0.0, dcpr, m_fn, m_ft, R, T);
		F += m_fn + m_ft;
		M += cross(dcpr, m_fn + m_ft);
	}
	return true;
}

void xParticleCylinderContact::updateCollisionPair(
	xContactPairList& xcpl, double r, vector3d pos)
{
	vector3d cpt = new_vector3d(0, 0, 0);
	vector3d unit = new_vector3d(0, 0, 0);
	bool isInnerContact = false;
	unsigned int cpart = 0;
	double cdist = particle_cylinder_contact_detection(pos, unit, cpt, r, isInnerContact);
	if (cdist > 0) {
		vector3d cpt = pos + r * unit;
		if (xcpl.IsNewCylinderContactPair(0))
		{
			xPairData *pd = new xPairData;
			*pd = { CYLINDER_SHAPE, true, 0, 0, 0, 0, cpt.x, cpt.y, cpt.z, cdist, unit.x, unit.y, unit.z };
			xcpl.insertCylinderContactPair(pd);
		}
		else
		{
			xPairData *pd = xcpl.CylinderPair(0);
			pd->gab = cdist;
			pd->cpx = cpt.x;
			pd->cpy = cpt.y;
			pd->cpz = cpt.z;
			pd->nx = unit.x;
			pd->ny = unit.y;
			pd->nz = unit.z;
		}
	}
	else
	{
		xPairData *pd = xcpl.CylinderPair(0);
		if (pd)
		{
			bool isc = pd->isc;
			if (!isc)
				xcpl.deleteCylinderPairData(0);
			else
			{
				vector3d cpt = pos + r * unit;
				pd->gab = cdist;
				pd->cpx = cpt.x;
				pd->cpy = cpt.y;
				pd->cpz = cpt.z;
				pd->nx = unit.x;
				pd->ny = unit.y;
				pd->nz = unit.z;
			}
		}
	}
	if (isInnerContact)
	{
		double cdist = particle_cylinder_inner_base_or_top_contact_detection(pos, unit, cpt, r);
		if (cdist > 0) {
			vector3d cpt = pos + r * unit;
			if (xcpl.IsNewCylinderContactPair(1))
			{
				xPairData *pd = new xPairData;
				*pd = { CYLINDER_SHAPE, true, 0, 1, 0, 0, cpt.x, cpt.y, cpt.z, cdist, unit.x, unit.y, unit.z };
				xcpl.insertCylinderContactPair(pd);
			}
			else
			{
				xPairData *pd = xcpl.CylinderPair(1);
				pd->gab = cdist;
				pd->cpx = cpt.x;
				pd->cpy = cpt.y;
				pd->cpz = cpt.z;
				pd->nx = unit.x;
				pd->ny = unit.y;
				pd->nz = unit.z;
			}
		}
		else
		{
			xPairData *pd = xcpl.CylinderPair(1);
			if (pd)
			{
				bool isc = pd->isc;
				if (!isc)
					xcpl.deleteCylinderPairData(1);
				else
				{
					vector3d cpt = pos + r * unit;
					pd->gab = cdist;
					pd->cpx = cpt.x;
					pd->cpy = cpt.y;
					pd->cpz = cpt.z;
					pd->nx = unit.x;
					pd->ny = unit.y;
					pd->nz = unit.z;
				}
			}
		}
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
	vector3d& pt, vector3d& u, vector3d& cp, double r, bool& isInnerContact)
{
	//isInnerContact = false;
	double dist = -1.0;
	vector3d cyl_pos = c_ptr->Position();
	euler_parameters cyl_ep = c_ptr->EulerParameters();
	vector3d cyl_base = cyl_pos + c_ptr->toGlobal(hci.pbase);
	vector3d cyl_top = cyl_pos + c_ptr->toGlobal(hci.ptop);
	vector3d ab = cyl_top - cyl_base;
	vector3d p = new_vector3d(pt.x, pt.y, pt.z);
	double t = dot(p - cyl_base, ab) / dot(ab, ab);
	vector3d _cp = new_vector3d(0.0, 0.0, 0.0);
	// radial contact
	if (t >= 0 && t <= 1) {
		_cp = cyl_base + t * ab;
		dist = length(p - _cp);
		if (dist == 0)
		{
			isInnerContact = true;
			return 0;
		}
		double gab = 0;
		u = (_cp - p) / dist;
		cp = _cp - hci.len_rr.z * u;
		// inner radial contact
		if (dist < hci.len_rr.z)
		{
			isInnerContact = true;
			u = -u;
			gab = dist + r - hci.len_rr.y;
		}
		else//outer radial contact		
			gab = hci.len_rr.y + r - dist;
		return gab;
	}
	else {//external top or bottom contact

		_cp = cyl_base + t * ab;
		dist = length(p - _cp);
		double thick = c_ptr->cylinder_thickness();
		int one = thick ? 1 : 0;
		if (dist < hci.len_rr.z + thick && dist > one * hci.len_rr.z) {
			vector3d OtoCp = c_ptr->Position() - _cp;
			double OtoCp_ = length(OtoCp);
			u = OtoCp / OtoCp_;
			cp = _cp - hci.len_rr.z * u;
			return hci.len_rr.x * 0.5 + r - OtoCp_;
		}
		vector3d _at = p - cyl_top;
		vector3d at = c_ptr->toLocal(_at);
		//double r = length(at);
		cp = cyl_top;
		if (abs(at.y) > hci.len_rr.x) {
			_at = p - cyl_base;
			at = c_ptr->toLocal(_at);
			cp = cyl_base;
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

double xParticleCylinderContact::particle_cylinder_inner_base_or_top_contact_detection(
	vector3d & pt, vector3d & u, vector3d & cp, double r)
{
	//isInnerContact = false;
	double dist = -1.0;
	vector3d cyl_pos = c_ptr->Position();
	euler_parameters cyl_ep = c_ptr->EulerParameters();
	vector3d cyl_base = cyl_pos + c_ptr->toGlobal(hci.pbase);
	vector3d cyl_top = cyl_pos + c_ptr->toGlobal(hci.ptop);
	vector3d ab = cyl_top - cyl_base;
	vector3d p = new_vector3d(pt.x, pt.y, pt.z);
	double t = dot(p - cyl_base, ab) / dot(ab, ab);
	vector3d _cp = new_vector3d(0.0, 0.0, 0.0);
	double gab = -1.0;
	_cp = cyl_base + t * ab;
	dist = length(p - _cp);
	if (empty_cylinder_part == xCylinderObject::TOP_CIRCLE && t > 0.6)
		return 0.0;
	if (empty_cylinder_part == xCylinderObject::BOTTOM_CIRCLE && t < 0.4)
		return 0.0;
	if (dist < hci.len_rr.z) {
		vector3d OtoCp = cyl_pos - _cp;
		double OtoCp_ = length(OtoCp);
		u = -OtoCp / OtoCp_;
		cp = pt + r * u;// _cp - hci.len_rr.z * u;
		gab = r + OtoCp_ - hci.len_rr.x * 0.5;// +r - OtoCp_;
	}
	
	return gab;
}
