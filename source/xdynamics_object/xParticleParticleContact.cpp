#include "xdynamics_object/xParticleParticleContact.h"
#include "xdynamics_simulation/xSimulation.h"

xParticleParticleContact::xParticleParticleContact()
	: xContact()
	, d_pair_count_pp(nullptr)
	, d_pair_id_pp(nullptr)
	, d_tsd_pp(nullptr)
	, pair_count_pp(nullptr)
	, pair_id_pp(nullptr)
	, tsd_pp(nullptr)
{

}

xParticleParticleContact::xParticleParticleContact(std::string _name)
	: xContact(_name, PARTICLE_PARTICLE)
	, d_pair_count_pp(nullptr)
	, d_pair_id_pp(nullptr)
	, d_tsd_pp(nullptr)
	, pair_count_pp(nullptr)
	, pair_id_pp(nullptr)
	, tsd_pp(nullptr)
{

}

xParticleParticleContact::~xParticleParticleContact()
{
	if (d_pair_count_pp) checkCudaErrors(cudaFree(d_pair_count_pp)); d_pair_count_pp = NULL;
	if (d_pair_id_pp) checkCudaErrors(cudaFree(d_pair_id_pp)); d_pair_id_pp = NULL;
	if (d_tsd_pp) checkCudaErrors(cudaFree(d_tsd_pp)); d_tsd_pp = NULL;
	if (pair_count_pp) delete[] pair_count_pp; pair_count_pp = NULL;
	if (pair_id_pp) delete[] pair_id_pp; pair_id_pp = NULL;
	if (tsd_pp) delete[] tsd_pp; tsd_pp = NULL;
}

void xParticleParticleContact::define(unsigned int idx, unsigned int np)
{

	xContact::define(idx, np);
	if (xSimulation::Gpu())
	{
		pair_count_pp = new unsigned int[np];
		pair_id_pp = new unsigned int[np * MAX_P2P_COUNT];
		tsd_pp = new double[2 * np * MAX_P2P_COUNT];
		checkXerror(cudaMalloc((void**)&d_pair_count_pp, sizeof(unsigned int) * np));
		checkXerror(cudaMalloc((void**)&d_pair_id_pp, sizeof(unsigned int) * np * MAX_P2P_COUNT));
		checkXerror(cudaMalloc((void**)&d_tsd_pp, sizeof(double2) * np * MAX_P2P_COUNT));
		checkXerror(cudaMemset(d_pair_count_pp, 0, sizeof(unsigned int) * np));
		checkXerror(cudaMemset(d_pair_id_pp, 0, sizeof(unsigned int) * np * MAX_P2P_COUNT));
		checkXerror(cudaMemset(d_tsd_pp, 0, sizeof(double2) * np * MAX_P2P_COUNT));
	}
	
	xDynamicsManager::This()->XResult()->set_p2p_contact_data((int)MAX_P2P_COUNT);
}

void xParticleParticleContact::update()
{

}

void xParticleParticleContact::savePartData(unsigned int np)
{
	checkXerror(cudaMemcpy(pair_count_pp, d_pair_count_pp, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
	checkXerror(cudaMemcpy(pair_id_pp, d_pair_id_pp, sizeof(unsigned int) * np * MAX_P2P_COUNT, cudaMemcpyDeviceToHost));
	checkXerror(cudaMemcpy(tsd_pp, d_tsd_pp, sizeof(double2) * np * MAX_P2P_COUNT, cudaMemcpyDeviceToHost));
	xDynamicsManager::This()->XResult()->save_p2p_contact_data(pair_count_pp, pair_id_pp, tsd_pp);
}

void xParticleParticleContact::deviceContactCount(double* pos, unsigned int *sorted_id, unsigned int *cstart, unsigned int *cend, unsigned int np)
{

}

// void xParticleParticleContact::cppCollision(
// 	double ir, double jr,
// 	double im, double jm,
// 	vector3d& ip, vector3d& jp,
// 	vector3d& iv, vector3d& jv,
// 	vector3d& io, vector3d& jo,
// 	vector3d& F, vector3d& M)
// {
// 	vector3d m_f, m_m;
// 	vector3d rp = jp - ip;
// 	double dist = length(rp);
// 	double cdist = (ir + jr) - dist;
// 	//double rcon = pos[i].w - cdist;
// 	unsigned int rid = 0;
// 	if (cdist > 0){
// 		vector3d u = rp / dist;
// 		double rcon = ir - 0.5 * cdist;
// 		vector3d cp = rcon * u;
// 		vector3d rv = jv + cross(jo, -jr * u) - (iv + cross(io, ir * u));
// 		xContactParameters c = getContactParameters(
// 			ir, jr,
// 			im, jm,
// 			mpp.Ei, mpp.Ej,
// 			mpp.Pri, mpp.Prj,
// 			mpp.Gi, mpp.Gj);
// 		switch (force_model)
// 		{
// 		case DHS: DHSModel(c, cdist, cp, rv, u, m_f, m_m); break;
// 		}
// 
// 		F += m_f;
// 		M += m_m;
// 	}
// }

void xParticleParticleContact::cppCollision(
	xContactPairList* pairs,
	unsigned int i, 
	vector4d *pos,
	vector4d *cpos,
	vector3d *vel, 
	euler_parameters *ep,
	euler_parameters *ev,
	double *mass, 
	double &res, 
	vector3d &tmax, 
	vector3d& F, 
	vector3d& M,
	xClusterInformation* xci,
	unsigned int nco)
{
	for(xmap<unsigned int, xPairData*>::iterator it = pairs->ParticlePair().begin(); it != pairs->ParticlePair().end(); it.next())
 	//foreach(xPairData* d, pairs->ParticlePair())
	{
		xPairData* d = it.value();
		vector3d m_fn = new_vector3d(0, 0, 0);
		vector3d m_m = new_vector3d(0, 0, 0);
		vector3d m_ft = new_vector3d(0, 0, 0);
		unsigned int j = d->id;
		unsigned int neach = 0;
		unsigned int ck = j;
		unsigned int ci = i;
		vector3d cpi = new_vector3d(pos[i].x, pos[i].y, pos[i].z);
		vector3d cpj = new_vector3d(pos[j].x, pos[j].y, pos[j].z);
		if (nco && cpos)
		{
			for (unsigned int j = 0; j < nco; j++)
				if (i >= xci[j].sid && i < xci[j].sid + xci[j].count * xci[j].neach)
					neach = xci[j].neach;
			ck = j / neach;
			ci = i / neach;
			cpi = new_vector3d(cpos[ci].x, cpos[ci].y, cpos[ci].z);
			cpj = new_vector3d(cpos[ck].x, cpos[ck].y, cpos[ck].z);
		}
		double rcon = pos[i].w - 0.5 * d->gab;
		vector3d u = new_vector3d(d->nx, d->ny, d->nz);
		vector3d cpt = new_vector3d(d->cpx, d->cpy, d->cpz);
		vector3d dcpr = cpt - cpi;
		vector3d dcpr_j = cpt - cpj;
		//vector3d cp = rcon * u;
		vector3d wi = ToAngularVelocity(ep[ci], ev[ci]);
		vector3d wj = ToAngularVelocity(ep[ck], ev[ck]);
		vector3d rv = vel[ck] + cross(wj, dcpr_j) - (vel[ci] + cross(wi, dcpr));
		xContactParameters c = getContactParameters(
			pos[i].w, pos[j].w,
			mass[ci], mass[ck],
			mpp.Ei, mpp.Ej,
			mpp.Pri, mpp.Prj,
			mpp.Gi, mpp.Gj,
			restitution, stiffnessRatio, s_friction,
			friction, rolling_factor, cohesion);
		if (d->gab < 0 && abs(d->gab) < abs(c.coh_s))
		{
			double f = JKRSeperationForce(c, cohesion);
			double cf = cohesionForce(cohesion, d->gab, c.coh_r, c.coh_e, c.coh_s, f);
			F -= cf * u;
			continue; 
		}
		//else if (d->isc && d->gab < 0 && abs(d->gab) > abs(c.coh_s))
		//{
		//	d->isc = false;
		//	continue;
		//}
		switch (xContact::ContactForceModel())
		{
		case DHS: DHSModel(c, d->gab, d->delta_s, d->dot_s, cohesion, rv, u, m_fn, m_ft); break;
		case HERTZ_MINDLIN_NO_SLIP: Hertz_Mindlin(c, d->gab, d->delta_s, d->dot_s, cohesion, rv, u, m_fn, m_ft); break;
		}
		RollingResistanceForce(rolling_factor, pos[i].w, pos[j].w, dcpr, m_fn, m_ft, res, tmax);
		F += m_fn + m_ft;
		M += cross(dcpr, m_fn + m_ft);
	}
}

void xParticleParticleContact::updateCollisionPair(
	unsigned int id, bool isc, xContactPairList& xcpl, 
	double ri, double rj, vector3d& posi, vector3d& posj)
{
	vector3d rp = posj - posi;
	double dist = length(rp);
	double cdist = (ri + rj) - dist;
	//double rcon = pos[i].w - cdist;
	unsigned int rid = 0;
	vector3d u = rp / dist;
	if (cdist > 0){
		
		vector3d cpt = posi + ri * u;
		if (xcpl.IsNewParticleContactPair(id))
		{
			xPairData *pd = new xPairData;
			*pd = { PARTICLES, true, 0, id, 0, 0, cpt.x, cpt.y, cpt.z, cdist, u.x, u.y, u.z };
			xcpl.insertParticleContactPair(pd);
		}		
		else
		{
			xPairData *pd = xcpl.ParticlePair(id);
			
			pd->gab = cdist;
			pd->cpx = cpt.x;
			pd->cpy = cpt.y;
			pd->cpz = cpt.z;
			pd->nx = u.x;
			pd->ny = u.y;
			pd->nz = u.z;
		}
	}
	else
	{
		double coh_s = cohesionSeperationDepth(cohesion, ri, rj, mpp.Pri, mpp.Prj, mpp.Ei, mpp.Ej);
		if (abs(cdist) < abs(coh_s))
		{
			xPairData *pd = xcpl.ParticlePair(id);
			vector3d cpt = posi + (ri + coh_s * 0.5) * u;
			if (pd)
			{
				xPairData *pd = xcpl.ParticlePair(id);
				
				pd->gab = cdist;
				pd->cpx = cpt.x;
				pd->cpy = cpt.y;
				pd->cpz = cpt.z;
				pd->nx = u.x;
				pd->ny = u.y;
				pd->nz = u.z;
			}
			else
			{
				xPairData *pd = new xPairData;
				*pd = { PARTICLES, true, 0, id, 0, 0, cpt.x, cpt.y, cpt.z, cdist, u.x, u.y, u.z };
				xcpl.insertParticleContactPair(pd);
			}
		}
		else
		{
			xPairData *pd = xcpl.ParticlePair(id);
			if (pd)
			{
				//bool isc = pd->isc;
				//if (!isc)
				xcpl.deleteParticlePairData(id);
			}
		}
	}
}

void xParticleParticleContact::collision(
	double *pos, double *ep, double *vel, double *ev,
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
		cu_calculate_p2p(pos, ep, vel, ev, force, moment, mass,
			tmax, rres, d_pair_count_pp, d_pair_id_pp, d_tsd_pp, sorted_id,
			cell_start, cell_end, dcp, np);
	}
// 	cu_check_no_collision_pair(pos, d_pair_idx, d_pair_other, np);
// 	cu_check_new_collision_pair(pos, d_pair_idx, d_pair_other, sorted_id, cell_start, cell_end, np);
// 	cu_calculate_particle_collision_with_pair(pos, vel , omega, mass, d_tan, force, moment, d_pair_idx, d_pair_other, dcp, np);
// 	//cu_calculate_p2p(1, pos, vel, omega, force, moment, mass, sorted_id, cell_start, cell_end, dcp, np);
}