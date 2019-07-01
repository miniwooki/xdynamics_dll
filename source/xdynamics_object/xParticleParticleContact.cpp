#include "xdynamics_object/xParticleParticleContact.h"
#include "xdynamics_simulation/xSimulation.h"

xParticleParticleContact::xParticleParticleContact()
	: xContact()
// 	, d_pair_idx(NULL)
// 	, d_pair_other(NULL)
// 	, d_tan(NULL)
{

}

xParticleParticleContact::xParticleParticleContact(std::string _name)
	: xContact(_name, PARTICLE_PARTICLE)
{

}

xParticleParticleContact::~xParticleParticleContact()
{
// 	if (d_pair_idx) checkCudaErrors(cudaFree(d_pair_idx)); d_pair_idx = NULL;
// 	if (d_pair_other) checkCudaErrors(cudaFree(d_pair_other)); d_pair_other = NULL;
// 	if (d_tan) checkCudaErrors(cudaFree(d_tan)); d_tan = NULL;
}

void xParticleParticleContact::cudaMemoryAlloc(unsigned int np)
{
	xContact::cudaMemoryAlloc(np);
	if (xSimulation::Gpu())
	{
// 		checkCudaErrors(cudaMalloc((void**)&d_pair_idx, sizeof(unsigned int) * np * 2));
// 		checkCudaErrors(cudaMalloc((void**)&d_pair_other, sizeof(unsigned int) * np * MAXIMUM_PAIR_NUMBER));
// 		checkCudaErrors(cudaMalloc((void**)&d_tan, sizeof(double) * np * MAXIMUM_PAIR_NUMBER));
// 		checkCudaErrors(cudaMemset(d_pair_idx, 0, sizeof(unsigned int) * np * 2));
// 		checkCudaErrors(cudaMemset(d_pair_other, 0, sizeof(unsigned int) * np * MAXIMUM_PAIR_NUMBER));
// 		checkCudaErrors(cudaMemset(d_tan, 0, sizeof(unsigned int) * np * MAXIMUM_PAIR_NUMBER));
	}
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
	vector3d *vel, 
	vector3d *omega,
	double *mass, 
	double &res, 
	vector3d &tmax, 
	vector3d& F, 
	vector3d& M,
	xClusterInformation* xci,
	unsigned int nco)
{
 	foreach(xPairData* d, pairs->ParticlePair())
	{
		vector3d m_fn = new_vector3d(0, 0, 0);
		vector3d m_m = new_vector3d(0, 0, 0);
		vector3d m_ft = new_vector3d(0, 0, 0);
		unsigned int j = d->id;
		unsigned int neach = 0;
		unsigned int ck = 0;
		unsigned int ci = i;
		if (nco)
		{
			for (unsigned int j = 0; j < nco; j++)
				if (i >= xci[j].sid && i < xci[j].sid + xci[j].count * xci[j].neach)
					neach = xci[j].neach;
			ck = j / neach;
			ci = i / neach;
		}
		double rcon = pos[i].w - 0.5 * d->gab;
		vector3d u = new_vector3d(d->nx, d->ny, d->nz);
		vector3d cp = rcon * u;
		vector3d rv = vel[ck] + cross(omega[ck], -pos[j].w * u) - (vel[ci] + cross(omega[ci], pos[i].w * u));
		xContactParameters c = getContactParameters(
			pos[i].w, pos[j].w,
			mass[ci], mass[ck],
			mpp.Ei, mpp.Ej,
			mpp.Pri, mpp.Prj,
			mpp.Gi, mpp.Gj,
			restitution, stiffnessRatio,
			friction, rolling_factor, cohesion);
		switch (force_model)
		{
		case DHS: DHSModel(c, d->gab, d->delta_s, d->dot_s, cp, rv, u, m_fn, m_ft, m_m); break;
		}
		RollingResistanceForce(rolling_factor, pos[i].w, pos[j].w, cp, m_fn, m_ft, res, tmax);
		F += m_fn + m_ft;
		M += m_m;
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
	if (cdist > 0){
		vector3d u = rp / dist;
		if (xcpl.IsNewParticleContactPair(id))
		{
			xPairData *pd = new xPairData;
			*pd = { PARTICLES, isc, 0, id, 0, 0, cdist, u.x, u.y, u.z };
			xcpl.insertParticleContactPair(pd);
		}		
		else
		{
			xPairData *pd = xcpl.ParticlePair(id);
			pd->gab = cdist;
			pd->nx = u.x;
			pd->ny = u.y;
			pd->nz = u.z;
		}
	}
	else
	{
		xcpl.deleteParticlePairData(id);
	}
}

void xParticleParticleContact::cuda_collision(
	double *pos, double *vel, double *omega,
	double *mass, double *force, double *moment,
	unsigned int *sorted_id, unsigned int *cell_start,
	unsigned int *cell_end, unsigned int np)
{
// 	cu_check_no_collision_pair(pos, d_pair_idx, d_pair_other, np);
// 	cu_check_new_collision_pair(pos, d_pair_idx, d_pair_other, sorted_id, cell_start, cell_end, np);
// 	cu_calculate_particle_collision_with_pair(pos, vel , omega, mass, d_tan, force, moment, d_pair_idx, d_pair_other, dcp, np);
// 	//cu_calculate_p2p(1, pos, vel, omega, force, moment, mass, sorted_id, cell_start, cell_end, dcp, np);
}