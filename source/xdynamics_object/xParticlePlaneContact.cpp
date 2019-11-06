#include "xdynamics_object/xParticlePlaneContact.h"
#include "xdynamics_object/xPlaneObject.h"
#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_manager/xDynamicsManager.h"

unsigned int xParticlePlaneContact::defined_count = 0;
bool xParticlePlaneContact::allocated_static = false;

double* xParticlePlaneContact::d_tsd_ppl = nullptr;
unsigned int* xParticlePlaneContact::d_pair_count_ppl = nullptr;
unsigned int* xParticlePlaneContact::d_pair_id_ppl = nullptr;

double* xParticlePlaneContact::tsd_ppl = nullptr;
unsigned int* xParticlePlaneContact::pair_count_ppl = nullptr;
unsigned int* xParticlePlaneContact::pair_id_ppl = nullptr;

xParticlePlaneContact::xParticlePlaneContact()
	: xContact()
	, p(nullptr)
	, pe(nullptr)
{

}

xParticlePlaneContact::xParticlePlaneContact(std::string _name, xObject* o1, xObject* o2)
	: xContact(_name, PARTICLE_PANE)
	, p(nullptr)
	, pe(nullptr)
{
	if (o1 && o2)
	{
		if (o1->Shape() == PLANE_SHAPE)
		{
			pe = dynamic_cast<xPlaneObject*>(o1);
			p = dynamic_cast<xParticleObject*>(o2);
		}
		else
		{
			pe = dynamic_cast<xPlaneObject*>(o2);
			p = dynamic_cast<xParticleObject*>(o1);
		}
		mpp = { o1->Youngs(), o2->Youngs(), o1->Poisson(), o2->Poisson(), o1->Shear(), o2->Shear() };
	}	
}

xParticlePlaneContact::~xParticlePlaneContact()
{
	if (d_pair_count_ppl) checkCudaErrors(cudaFree(d_pair_count_ppl)); d_pair_count_ppl = NULL;
	if (d_pair_id_ppl) checkCudaErrors(cudaFree(d_pair_id_ppl)); d_pair_id_ppl = NULL;
	if (d_tsd_ppl) checkCudaErrors(cudaFree(d_tsd_ppl)); d_tsd_ppl = NULL;
	if (pair_count_ppl) delete[] pair_count_ppl; pair_count_ppl = NULL;
	if (pair_id_ppl) delete[] pair_id_ppl; pair_id_ppl = NULL;
	if (tsd_ppl) delete[] tsd_ppl; tsd_ppl = NULL;
	if (dpi) checkCudaErrors(cudaFree(dpi)); dpi = NULL;
	if (dbi) checkCudaErrors(cudaFree(dbi)); dbi = NULL;

	c_pairs.delete_all();
}

void xParticlePlaneContact::define(unsigned int idx, unsigned int np)
{
	id = defined_count;
	xContact::define(idx, np);

	if (xSimulation::Gpu())
	{
		if (!allocated_static)
		{
			pair_count_ppl = new unsigned int[np];
			pair_id_ppl = new unsigned int[np * MAX_P2MS_COUNT];
			tsd_ppl = new double[2 * np * MAX_P2MS_COUNT];
			checkXerror(cudaMalloc((void**)&d_pair_count_ppl, sizeof(unsigned int) * np));
			checkXerror(cudaMalloc((void**)&d_pair_id_ppl, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
			checkXerror(cudaMalloc((void**)&d_tsd_ppl, sizeof(double2) * np * MAX_P2MS_COUNT));
			checkXerror(cudaMemset(d_pair_count_ppl, 0, sizeof(unsigned int) * np));
			checkXerror(cudaMemset(d_pair_id_ppl, 0, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
			checkXerror(cudaMemset(d_tsd_ppl, 0, sizeof(double2) * np * MAX_P2MS_COUNT));
			
			allocated_static = true;
		}
		checkXerror(cudaMalloc((void**)&dpi, sizeof(device_plane_info)));		
		checkXerror(cudaMalloc((void**)&dbi, sizeof(device_body_info)));
		//checkXerror(cudaMemcpy(dpi, &hpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
		xDynamicsManager::This()->XResult()->set_p2pl_contact_data((int)MAX_P2PL_COUNT);
		
	}
	update();
	defined_count++;
}

void xParticlePlaneContact::local_initialize()
{
	defined_count = 0;
}

void xParticlePlaneContact::update()
{
	hpi.id = id;
	hpi.xw = pe->Position() + pe->toGlobal(pe->LocalPoint(0));
	hpi.w2 = pe->Position() + pe->toGlobal(pe->LocalPoint(1));
	hpi.w3 = pe->Position() + pe->toGlobal(pe->LocalPoint(2));
	hpi.w4 = pe->Position() + pe->toGlobal(pe->LocalPoint(3));
	hpi.pa = hpi.w2 - hpi.xw;
	hpi.pb = hpi.w4 - hpi.xw;
	hpi.l1 = length(hpi.pa);
	hpi.l2 = length(hpi.pb);
	hpi.u1 = hpi.pa / hpi.l1;
	hpi.u2 = hpi.pb / hpi.l2;
	hpi.uw = cross(hpi.u1, hpi.u2);
	
	

	if (xSimulation::Gpu())
	{
		unsigned int mcnt = 0;
		euler_parameters ep = pe->EulerParameters();
		euler_parameters ed = pe->DEulerParameters();
		host_body_info hbi = {
			pe->Mass(),
			pe->Position().x, pe->Position().y, pe->Position().z,
			pe->Velocity().x, pe->Velocity().y, pe->Velocity().z,
			ep.e0, ep.e1, ep.e2, ep.e3,
			ed.e0, ed.e1, ed.e2, ed.e3
		};
		//}
		checkXerror(cudaMemcpy(dpi, &hpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
		checkXerror(cudaMemcpy(dbi, &hbi, sizeof(device_body_info), cudaMemcpyHostToDevice));
	}

}

xPlaneObject* xParticlePlaneContact::PlaneObject()
{
	return iobj->Shape() == PLANE_SHAPE ? dynamic_cast<xPlaneObject*>(iobj) : dynamic_cast<xPlaneObject*>(jobj);
}

void xParticlePlaneContact::setPlane(xPlaneObject* _pe)
{
	//pe = _pe;
}

void xParticlePlaneContact::collision_gpu(
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
		if(!cpos) cu_plane_contact_force(dpi, dbi, dcp, pos, ep, vel, ev, force, moment, mass, tmax, rres, d_pair_count_ppl, d_pair_id_ppl, d_tsd_ppl, np);
		else if (cpos) cu_cluster_plane_contact(dpi, dbi, dcp, pos, cpos, ep, vel, ev, force, moment, mass, tmax, rres, d_pair_count_ppl, d_pair_id_ppl, d_tsd_ppl, xci, np);
		if (pe->isDynamicsBody())
		{
			fm[0] = reduction(xContact::deviceBodyForceX(), np);
			fm[1] = reduction(xContact::deviceBodyForceY(), np);
			fm[2] = reduction(xContact::deviceBodyForceZ(), np);
			fm[3] = reduction(xContact::deviceBodyMomentX(), np);
			fm[4] = reduction(xContact::deviceBodyMomentY(), np);
			fm[5] = reduction(xContact::deviceBodyMomentZ(), np);
			pe->addAxialForce(fm[0], fm[1], fm[2]);
			pe->addAxialMoment(fm[3], fm[4], fm[5]);
		}
	}	
}

void xParticlePlaneContact::collision_cpu(
	vector4d * pos, euler_parameters * ep, vector3d * vel, 
	euler_parameters * ev, double* mass, double & rres, vector3d & tmax, 
	vector3d & force, vector3d & moment, unsigned int nco, 
	xClusterInformation * xci, vector4d * cpos)
{
	for (xmap<unsigned int, xPairData*>::iterator it = c_pairs.begin(); it != c_pairs.end(); it.next())
	{
		unsigned int id = it.key();
		xPairData* d = it.value();
		unsigned int ci = id;
		unsigned int neach = 1;
		vector3d cp = new_vector3d(pos[id].x, pos[id].y, pos[id].z);
		double r = pos[id].w;
		if (nco)
		{
			for (unsigned int j = 0; j < nco; j++)
				if (id >= xci[j].sid && id < xci[j].sid + xci[j].count * xci[j].neach)
				{
					neach = xci[j].neach; ci = id / neach;
				}
			cp = new_vector3d(cpos[id].x, cpos[id].y, cpos[id].z);
			r = cpos[id].w;
		}			
		double m = mass[ci];
		//xPlaneObject* pl = pair_ip[d->id];
		vector3d m_fn = new_vector3d(0.0, 0.0, 0.0);
		vector3d m_m = new_vector3d(0.0, 0.0, 0.0);
		vector3d m_ft = new_vector3d(0.0, 0.0, 0.0);
		double rcon = r - 0.5 * d->gab;
		vector3d u = new_vector3d(d->nx, d->ny, d->nz);
		vector3d cpt = new_vector3d(d->cpx, d->cpy, d->cpz);
		vector3d dcpr = cpt - cp;

		vector3d dcpr_j = cpt - pe->Position();
		vector3d oi = 2.0 * GMatrix(ep[ci]) * ev[ci];
		vector3d oj = 2.0 * GMatrix(pe->EulerParameters()) * pe->DEulerParameters();
		vector3d dv = pe->Velocity() + cross(oj, dcpr_j) - (vel[ci] + cross(oi, dcpr));

		//xContactMaterialParameters cmp = hcmp[d->id];
		xContactParameters c = getContactParameters(
			r, 0.0,
			m, pe->Mass(),
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

		RollingResistanceForce(c.rfric, r, 0.0, dcpr, m_fn, m_ft, rres, tmax);
		vector3d sum_force = m_fn + m_ft;
		force += sum_force;
		moment += cross(dcpr, sum_force);
		pe->addContactForce(-sum_force.x, -sum_force.y, -sum_force.z);
		vector3d body_moment = -cross(dcpr_j, sum_force);
		pe->addContactMoment(-body_moment.x, -body_moment.y, -body_moment.z);

	}
}

void xParticlePlaneContact::updateCollisionPair(unsigned int pid, double r, vector3d pos)
{
	//host_plane_info *pe = hpi;
	vector3d dp = pos - hpi.xw;
	vector3d wp = new_vector3d(dot(dp, hpi.u1), dot(dp, hpi.u2), dot(dp, hpi.uw));
	vector3d u;
	//xContactMaterialParameters cmp = hcmp[i];

	double cdist = particle_plane_contact_detection(pe, u, pos, wp, r);
	if (cdist > 0) {
		vector3d cpt = pos + r * u;
		if (c_pairs.find(pid) == c_pairs.end()/*xcpl.IsNewPlaneContactPair(hpi.id)*/)
		{
			xPairData *pd = new xPairData;
			*pd = { PLANE_SHAPE, true, 0, pid, 0, 0, cpt.x, cpt.y, cpt.z, cdist, u.x, u.y, u.z };
			c_pairs.insert(pid, pd);// insertPlaneContactPair(pd);
		}
		else
		{
			xmap<unsigned int, xPairData*>::iterator it = c_pairs.find(pid);
			if (it != c_pairs.end())
			{
				xPairData *pd = it.value();// .PlanePair(hpi.id);
				pd->gab = cdist;
				pd->cpx = cpt.x;
				pd->cpy = cpt.y;
				pd->cpz = cpt.z;
				pd->nx = u.x;
				pd->ny = u.y;
				pd->nz = u.z;
			}			
		}
	}
	else
	{
		xmap<unsigned int, xPairData*>::iterator it = c_pairs.find(pid);
		if (it != c_pairs.end())
		{
			xPairData *pd = it.value();
			if (pd)
			{
				bool isc = pd->isc;
				if (!isc)
				{
					if (c_pairs.find(pid) != c_pairs.end())
						delete c_pairs.take(pid);
				}
				else
				{
					vector3d cpt = pos + r * u;
					pd->gab = cdist;
					pd->cpx = cpt.x;
					pd->cpy = cpt.y;
					pd->cpz = cpt.z;
					pd->nx = u.x;
					pd->ny = u.y;
					pd->nz = u.z;
				}
			}
		}
		
	}
	
	
}

void xParticlePlaneContact::savePartData(unsigned int np)
{
	checkXerror(cudaMemcpy(pair_count_ppl, d_pair_count_ppl, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
	checkXerror(cudaMemcpy(pair_id_ppl, d_pair_id_ppl, sizeof(unsigned int) * np * MAX_P2PL_COUNT, cudaMemcpyDeviceToHost));
	checkXerror(cudaMemcpy(tsd_ppl, d_tsd_ppl, sizeof(double2) * np * MAX_P2PL_COUNT, cudaMemcpyDeviceToHost));
	xDynamicsManager::This()->XResult()->save_p2pl_contact_data(pair_count_ppl, pair_id_ppl, tsd_ppl);
}

bool xParticlePlaneContact::detect_contact(vector4f& p, xPlaneObject& pe, vector3f& cpoint)
{
	vector3d pos =
	{
		static_cast<double>(p.x),
		static_cast<double>(p.y),
		static_cast<double>(p.z)
	};
	double r = static_cast<double>(p.w);
	vector3d dp = pos - pe.XW();// xw;
	vector3d wp = new_vector3d(dot(dp, pe.U1()), dot(dp, pe.U2()), dot(dp, pe.UW()));
	vector3d u;
	double a_l1 = pow(wp.x - pe.L1(), 2.0);
	double b_l2 = pow(wp.y - pe.L2(), 2.0);
	double sqa = wp.x * wp.x;
	double sqb = wp.y * wp.y;
	double sqc = wp.z * wp.z;
	double sqr = r * r;

//	vector3d dp = pos - pe.XW();
	vector3d uu = pe.UW() / length(pe.UW());
	int pp = -xsign(dot(dp, pe.UW()));
	u = -uu;
	double collid_dist = r - abs(dot(dp, u));
	bool is_contact = collid_dist > 0;
	vector3d _cp = pos + r * u;
	cpoint = 
	{
	static_cast<float>(_cp.x),
	static_cast<float>(_cp.y),
	static_cast<float>(_cp.z)
	};
	return is_contact;
}

double xParticlePlaneContact::particle_plane_contact_detection(	xPlaneObject* _pe, vector3d& u, vector3d& xp, vector3d& wp, double r)
{
// 	double a_l1 = pow(wp.x - _pe->L1(), 2.0);
// 	double b_l2 = pow(wp.y - _pe->L2(), 2.0);
// 	double sqa = wp.x * wp.x;
// 	double sqb = wp.y * wp.y;
// 	double sqc = wp.z * wp.z;
// 	double sqr = r*r;
// 
// 	// The sphere contacts with the wall face
// 	if (abs(wp.z) < r && (wp.x > 0 && wp.x < _pe->L1()) && (wp.y > 0 && wp.y < _pe->L2())){
// 		vector3d dp = xp - _pe->XW();
// 		vector3d uu = _pe->UW() / length(_pe->UW());
// 		int pp = 0;// -sign(dot(dp, _pe->UW()));
// 		u = pp * uu;
// 		double collid_dist = r - abs(dot(dp, u));
// 		return collid_dist;
// 	}
// 
// 	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->XW();
// 		double h = length(Xsw);
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	else if (wp.x > _pe->L1() && wp.y < 0 && (a_l1 + sqb + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->W2();
// 		double h = length(Xsw);
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	else if (wp.x > _pe->L1() && wp.y > _pe->L2() && (a_l1 + b_l2 + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->W3();
// 		double h = length(Xsw);
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	else if (wp.x < 0 && wp.y > _pe->L2() && (sqa + b_l2 + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->W4();
// 		double h = length(Xsw);
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	if ((wp.x > 0 && wp.x < _pe->L1()) && wp.y < 0 && (sqb + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->XW();
// 		vector3d wj_wi = _pe->W2() - _pe->XW();
// 		vector3d us = wj_wi / length(wj_wi);
// 		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
// 		double h = length(h_star);
// 		u = -h_star / h;
// 		return r - h;
// 	}
// 	else if ((wp.x > 0 && wp.x < _pe->L1()) && wp.y > _pe->L2() && (b_l2 + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->W4();
// 		vector3d wj_wi = _pe->W3() - _pe->W4();
// 		vector3d us = wj_wi / length(wj_wi);
// 		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
// 		double h = length(h_star);
// 		u = -h_star / h;
// 		return r - h;
// 
// 	}
// 	else if ((wp.x > 0 && wp.y < _pe->L2()) && wp.x < 0 && (sqr + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->XW();
// 		vector3d wj_wi = _pe->W4() - _pe->XW();
// 		vector3d us = wj_wi / length(wj_wi);
// 		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
// 		double h = length(h_star);
// 		u = -h_star / h;
// 		return r - h;
// 	}
// 	else if ((wp.x > 0 && wp.y < _pe->L2()) && wp.x > _pe->L1() && (a_l1 + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->W2();
// 		vector3d wj_wi = _pe->W3() - _pe->W2();
// 		vector3d us = wj_wi / length(wj_wi);
// 		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
// 		double h = length(h_star);
// 		u = -h_star / h;
// 		return r - h;
// 	}
// 
// 
 	return -1.0f;
}
