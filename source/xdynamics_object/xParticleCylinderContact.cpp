#include "xdynamics_object/xParticleCylinderContact.h"
#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_object/xCylinderObject.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_manager/xDynamicsManager.h"

unsigned int xParticleCylinderContact::defined_count = 0;
bool xParticleCylinderContact::allocated_static = false;

double* xParticleCylinderContact::d_tsd_pcyl = nullptr;
unsigned int* xParticleCylinderContact::d_pair_count_pcyl = nullptr;
unsigned int* xParticleCylinderContact::d_pair_id_pcyl = nullptr;

double* xParticleCylinderContact::tsd_pcyl = nullptr;
unsigned int* xParticleCylinderContact::pair_count_pcyl = nullptr;
unsigned int* xParticleCylinderContact::pair_id_pcyl = nullptr;

xParticleCylinderContact::xParticleCylinderContact()
	: xContact()
	, id(0)
	, p_ptr(NULL)
	, c_ptr(NULL)
{

}

xParticleCylinderContact::xParticleCylinderContact(std::string _name, xObject* o1, xObject* o2)
	: xContact(_name, PARTICLE_CYLINDER)
	, id(0)
	, p_ptr(NULL)
	, c_ptr(NULL)
{
	if (o1 && o2)
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
		mpp = { o1->Youngs(), o2->Youngs(), o1->Poisson(), o2->Poisson(), o1->Shear(), o2->Shear() };
		empty_cylinder_part = c_ptr->empty_part_type();
	}
}

xParticleCylinderContact::~xParticleCylinderContact()
{
	if (d_pair_count_pcyl) checkCudaErrors(cudaFree(d_pair_count_pcyl)); d_pair_count_pcyl = NULL;
	if (d_pair_id_pcyl) checkCudaErrors(cudaFree(d_pair_id_pcyl)); d_pair_id_pcyl = NULL;
	if (d_tsd_pcyl) checkCudaErrors(cudaFree(d_tsd_pcyl)); d_tsd_pcyl = NULL;
	if (dci) checkXerror(cudaFree(dci)); dci = NULL;
	if (dbi) checkXerror(cudaFree(dbi)); dbi = NULL;
	//if (dbf) checkXerror(cudaFree(dci)); dci = NULL;

	if (pair_count_pcyl) delete[] pair_count_pcyl; pair_count_pcyl = NULL;
	if (pair_id_pcyl) delete[] pair_id_pcyl; pair_id_pcyl = NULL;
	if (tsd_pcyl) delete[] tsd_pcyl; tsd_pcyl = NULL;
}

void xParticleCylinderContact::define(unsigned int idx, unsigned int np)
{
	id = defined_count;
	xContact::define(idx, np);
	hci =
	{
		id,
		(unsigned int)c_ptr->empty_part_type(),
		c_ptr->cylinder_thickness(),
		c_ptr->cylinder_length(),
		c_ptr->cylinder_bottom_radius(),
		c_ptr->cylinder_top_radius(),
		c_ptr->bottom_position(),
		c_ptr->top_position()
	};
	if (xSimulation::Gpu())
	{		
		if (!allocated_static)
		{
			pair_count_pcyl = new unsigned int[np];
			pair_id_pcyl = new unsigned int[np * MAX_P2CY_COUNT];
			tsd_pcyl = new double[2 * np * MAX_P2CY_COUNT];
			checkXerror(cudaMalloc((void**)&d_pair_count_pcyl, sizeof(unsigned int) * np));
			checkXerror(cudaMalloc((void**)&d_pair_id_pcyl, sizeof(unsigned int) * np * MAX_P2CY_COUNT));
			checkXerror(cudaMalloc((void**)&d_tsd_pcyl, sizeof(double2) * np * MAX_P2CY_COUNT));
			checkXerror(cudaMemset(d_pair_count_pcyl, 0, sizeof(unsigned int) * np));
			checkXerror(cudaMemset(d_pair_id_pcyl, 0, sizeof(unsigned int) * np * MAX_P2CY_COUNT));
			checkXerror(cudaMemset(d_tsd_pcyl, 0, sizeof(double2) * np * MAX_P2CY_COUNT));
		}
		
		checkXerror(cudaMalloc((void**)&dci, sizeof(device_cylinder_info)));
		checkXerror(cudaMalloc((void**)&dbi, sizeof(device_body_info)));
		checkXerror(cudaMemset(dbi, 0, sizeof(device_body_info)));
		checkXerror(cudaMemcpy(dci, &hci, sizeof(device_cylinder_info), cudaMemcpyHostToDevice));
		xDynamicsManager::This()->XResult()->set_p2cyl_contact_data((int)MAX_P2CY_COUNT);
		update();
	}
	defined_count++;
}

void xParticleCylinderContact::local_initialize()
{
	defined_count = 0;
}

void xParticleCylinderContact::update()
{
	//device_body_info *bi = NULL;
	//if (nContactObject || is_first_set_up)
	//	bi = new device_body_info[nContactObject];


	if (xSimulation::Gpu())
	{
		//unsigned int mcnt = 0;
		//for (xmap<unsigned int, xContact*>::iterator it = pair_contact.begin(); it != pair_contact.end(); it.next())
		//{
			//xPointMass* pm = NULL;
			//xContact* xc = it.value();
			//xCylinderObject *c = dynamic_cast<xParticleCylinderContact*>(xc)->CylinderObject();
		euler_parameters ep = c_ptr->EulerParameters();
		euler_parameters ed = c_ptr->DEulerParameters();
		host_body_info hbi = {
			c_ptr->Mass(),
			c_ptr->Position().x, c_ptr->Position().y, c_ptr->Position().z,
			c_ptr->Velocity().x, c_ptr->Velocity().y, c_ptr->Velocity().z,
			ep.e0, ep.e1, ep.e2, ep.e3,
			ed.e0, ed.e1, ed.e2, ed.e3
		};
		
		//checkCudaErrors(cudaMemset(db_force, 0, sizeof(double3) * ncylinders));
		//checkCudaErrors(cudaMemset(db_moment, 0, sizeof(double3) * ncylinders));
		checkXerror(cudaMemcpy(dbi, &hbi, sizeof(device_body_info), cudaMemcpyHostToDevice));
	}
}

// void xParticleCylinderContact::initialize()
// {
// 	xParticleCylinderContact::local_initialize();
// }

void xParticleCylinderContact::savePartData(unsigned int np)
{
	checkXerror(cudaMemcpy(pair_count_pcyl, d_pair_count_pcyl, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
	checkXerror(cudaMemcpy(pair_id_pcyl, d_pair_id_pcyl, sizeof(unsigned int) * np * MAX_P2CY_COUNT, cudaMemcpyDeviceToHost));
	checkXerror(cudaMemcpy(tsd_pcyl, d_tsd_pcyl, sizeof(double2) * np * MAX_P2CY_COUNT, cudaMemcpyDeviceToHost));
	xDynamicsManager::This()->XResult()->save_p2cyl_contact_data(pair_count_pcyl, pair_id_pcyl, tsd_pcyl);
}


xCylinderObject * xParticleCylinderContact::CylinderObject()
{
	return c_ptr;
}



void xParticleCylinderContact::collision_gpu(
	double *pos, double* cpos, xClusterInformation* xci,
	double *ep, double *vel, double *ev,
	double *mass, double* inertia,
	double *force, double *moment,
	double *tmax, double* rres,
	unsigned int *sorted_id,
	unsigned int *cell_start,
	unsigned int *cell_end,
	unsigned int np)
{
	if (xSimulation::Gpu())
	{
		double fm[6] = { 0, };
		if(!cpos) cu_cylinder_contact_force(dci, dbi, dcp, pos, ep, vel, ev, force, moment, mass,	tmax, rres, d_pair_count_pcyl, d_pair_id_pcyl, d_tsd_pcyl, np);
		else if (cpos) cu_cluster_cylinder_contact(dci, dbi, xci, dcp, pos, cpos, ep, vel, ev, force, moment, mass, tmax, rres, d_pair_count_pcyl, d_pair_id_pcyl, d_tsd_pcyl, np);

		if (c_ptr->isDynamicsBody())
		{
			fm[0] = reduction(xContact::deviceBodyForceX(), np);
			fm[1] = reduction(xContact::deviceBodyForceY(), np);
			fm[2] = reduction(xContact::deviceBodyForceZ(), np);
			fm[3] = reduction(xContact::deviceBodyMomentX(), np);
			fm[4] = reduction(xContact::deviceBodyMomentY(), np);
			fm[5] = reduction(xContact::deviceBodyMomentZ(), np);
			c_ptr->addAxialForce(fm[0], fm[1], fm[2]);
			c_ptr->addAxialMoment(fm[3], fm[4], fm[5]);
		}
	}
	
}

void xParticleCylinderContact::collision_cpu(
	vector4d * pos, euler_parameters * ep, vector3d * vel,
	euler_parameters * ev, double* mass, double & rres, vector3d & tmax,
	vector3d & force, vector3d & moment, unsigned int nco,
	xClusterInformation * xci, vector4d * cpos)
{
	for (xmap<unsigned int, xPairData*>::iterator it = c_pairs.begin(); it != c_pairs.end(); it.next())
	{
		unsigned int id = it.value()->id;
		if (id >= 1000)
			id -= 1000;
		unsigned int ci = id;
		unsigned int neach = 1;
		vector3d cp = new_vector3d(pos[id].x, pos[id].y, pos[id].z);
		double r = pos[id].w;
		double m = mass[id];
		if (nco)
		{
			for (unsigned int j = 0; j < nco; j++)
				if (id >= xci[j].sid && id < xci[j].sid + xci[j].count * xci[j].neach)
				{
					neach = xci[j].neach; ci = id / neach;
				}
			cp = new_vector3d(cpos[ci].x, cpos[ci].y, cpos[ci].z);
			r = cpos[ci].w;
		}
		//double m = mass[ci];
		vector3d v = vel[ci];
		vector3d o = ToAngularVelocity(ep[ci], ev[ci]);
		xPairData* d = it.value();

		vector3d m_fn = new_vector3d(0.0, 0.0, 0.0);
		vector3d m_m = new_vector3d(0.0, 0.0, 0.0);
		vector3d m_ft = new_vector3d(0.0, 0.0, 0.0);
		double rcon = r - 0.5 * d->gab;
		vector3d u = new_vector3d(d->nx, d->ny, d->nz);
		vector3d cpt = new_vector3d(d->cpx, d->cpy, d->cpz);
		vector3d dcpr = cpt - cp;
		vector3d dcpr_j = cpt - c_ptr->Position();
		vector3d oj = 2.0 * GMatrix(c_ptr->EulerParameters()) * c_ptr->DEulerParameters();
		vector3d dv = c_ptr->Velocity() + cross(oj, dcpr_j) - (vel[ci] + cross(o, dcpr));

		xContactParameters c = getContactParameters(
			r, 0.0,
			m, c_ptr->Mass(),
			mpp.Ei, mpp.Ej,
			mpp.Pri, mpp.Prj,
			mpp.Gi, mpp.Gj,
			restitution, stiffnessRatio, s_friction,
			friction, rolling_factor, cohesion);

		if (d->gab < 0 && abs(d->gab) < abs(c.coh_s))
		{
			double f = JKRSeperationForce(c, cohesion);
			double cf = cohesionForce(cohesion, d->gab, c.coh_r, c.coh_e, c.coh_s, f);
			force -= cf * u;
			continue;
		}
		else if (d->isc && d->gab < 0 && abs(d->gab) > abs(c.coh_s))
		{
			d->isc = false;
			continue;
		}
		switch (xContact::ContactForceModel())
		{
		case DHS: DHSModel(c, d->gab, d->delta_s, d->dot_s, cohesion, dv, u, m_fn, m_ft); break;
		case HERTZ_MINDLIN_NO_SLIP: Hertz_Mindlin(c, d->gab, d->delta_s, d->dot_s, cohesion, dv, u, m_fn, m_ft); break;
		}
		/*if (cmp.cohesion && c.coh_s > d->gab)
			d->isc = false;*/
		RollingResistanceForce(c.rfric, r, 0.0, dcpr, m_fn, m_ft, rres, tmax);
		vector3d sum_force = m_fn + m_ft;
		force += sum_force;
		moment += cross(dcpr, sum_force);
		/*	if(i==0)
				printf("cyl force : [%e, %e, %e]\n", sum_force.x, sum_force.y, sum_force.z);*/
		c_ptr->addContactForce(-sum_force.x, -sum_force.y, -sum_force.z);
		vector3d body_moment = -cross(dcpr_j, sum_force);
		c_ptr->addContactMoment(-body_moment.x, -body_moment.y, -body_moment.z);
	}
}

double xParticleCylinderContact::particle_cylinder_contact_detection(vector3d& pt, vector3d& u, vector3d& cp, double r, bool& isInnerContact)
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
		//cp = _cp - hci.len_rr.z * u;
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
			//cp = _cp - hci.len_rr.z * u;
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
			//cct = CIRCLE_LINE_CONTACT;
			return r - dist;
		}
	}
	return -1.0;
}

double xParticleCylinderContact::particle_cylinder_inner_base_or_top_contact_detection(vector3d & pt, vector3d & u, vector3d & cp, double r)
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
	if (c_ptr->empty_part_type() == xCylinderObject::TOP_CIRCLE && t > 0.6)
		return 0.0;
	if (c_ptr->empty_part_type() == xCylinderObject::BOTTOM_CIRCLE && t < 0.4)
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

void xParticleCylinderContact::updateCollisionPair(unsigned int id, double r, vector3d pos)
{
	unsigned int cnt = 0;
	//unsigned int id = hci.id;
	vector3d cpt = new_vector3d(0, 0, 0);
	vector3d unit = new_vector3d(0, 0, 0);
	bool isInnerContact = false;
	unsigned int cpart = 0;
	double cdist = particle_cylinder_contact_detection(pos, unit, cpt, r, isInnerContact);

	if (cdist > 0) {
		vector3d cpt = pos + r * unit;
		if (c_pairs.find(id) == c_pairs.end())
		{
			xPairData *pd = new xPairData;
			*pd = { CYLINDER_SHAPE, true, 0, id, 0, 0, cpt.x, cpt.y, cpt.z, cdist, unit.x, unit.y, unit.z };
			c_pairs.insert(id, pd);// xcpl.insertCylinderContactPair(pd);
		}
		else
		{
			xPairData *pd = c_pairs.find(id).value();// xcpl.CylinderPair(id);
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
		xmap<unsigned int, xPairData*>::iterator it = c_pairs.find(id);
		if (it != c_pairs.end())
		{
			if (it.value())
			{
				xPairData* pd = it.value();
				bool isc = pd->isc;
				if (!isc)
				{
					delete c_pairs.take(id);
				}
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
	if (isInnerContact)
	{
		double cdist = particle_cylinder_inner_base_or_top_contact_detection(pos, unit, cpt, r);
		if (cdist > 0) {
			vector3d cpt = pos + r * unit;
			if (c_pairs.find(id + 1000) == c_pairs.end())
			{
				xPairData *pd = new xPairData;
				*pd = { CYLINDER_SHAPE, true, 0, id + 1000, 0, 0, cpt.x, cpt.y, cpt.z, cdist, unit.x, unit.y, unit.z };
				c_pairs.insert(id + 1000, pd);// xcpl.insertCylinderContactPair(pd);
			}
			else
			{
				xPairData *pd = c_pairs.find(id + 1000).value();// xcpl.CylinderPair(id + 1000);
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
			xmap<unsigned int, xPairData*>::iterator it = c_pairs.find(id + 1000);
			if (it != c_pairs.end())
			{
				xPairData* pd = it.value();
				bool isc = pd->isc;
				if (!isc)
				{
					delete c_pairs.take(id + 1000);
				}
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

//device_cylinder_info* xParticleCylinderContact::deviceCylinderInfo()
//{
//	return dci;
//}


