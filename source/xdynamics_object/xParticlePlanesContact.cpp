#include "xdynamics_object/xParticlePlanesContact.h"
#include "xdynamics_object/xParticleCubeContact.h"
#include "xdynamics_object/xPlaneObject.h"
#include "xdynamics_object/xCubeObject.h"
// #include <thrust/scan.h>
// #include <thrust/reduce.h>
// #include <thrust/execution_policy.h>
// #include <thrust/device_ptr.h>

xParticlePlanesContact::xParticlePlanesContact()
	: xContact()
	, hpi(NULL)
	, dpi(NULL)
	, nplanes(0)
	, xmps(NULL)
	, hcmp(NULL)
// 	, d_pair_count(NULL)
// 	, d_old_pair_start(NULL)
// 	, d_pair_start(NULL)
// 	, d_pppd(NULL)
{

}

xParticlePlanesContact::~xParticlePlanesContact()
{
	if (hcmp) delete[] hcmp; hcmp = NULL;
	if (xmps) delete[] xmps; xmps = NULL;
	if (hpi) delete[] hpi; hpi = NULL;
	if (dpi) checkCudaErrors(cudaFree(dpi)); dpi = NULL;
// 	if (d_pair_count) checkCudaErrors(cudaFree(d_pair_count)); d_pair_count = NULL;
// 	if (d_old_pair_start) checkCudaErrors(cudaFree(d_old_pair_start)); d_old_pair_start = NULL;
// 	if (d_pair_start) checkCudaErrors(cudaFree(d_pair_start)); d_pair_start = NULL;
// 	if (d_pppd) checkCudaErrors(cudaFree(d_pppd)); d_pppd = NULL;
	//qDeleteAll(pair_ip);
}

void xParticlePlanesContact::define(unsigned int id, xParticlePlaneContact* d)
{
	xPlaneObject *p = d->PlaneObject();
	xContactMaterialParameters cp = { 0, };
	xMaterialPair xmp = { 0, };
	cp.restitution = d->Restitution();
	cp.stiffness_ratio = d->StiffnessRatio();
	cp.friction = d->Friction();
	cp.rolling_resistance = d->RollingFactor();
	xmp = d->MaterialPropertyPair();
	hcmp[id] = cp;
	xmps[id] = xmp;
	pair_ip[id] = p;
	hpi[id] = {
		p->L1(), p->L2(),
		p->U1(), p->U2(),
		p->UW(), p->XW(),
		p->PA(), p->PB(),
		p->W2(), p->W3(), p->W4()
	};
}

void xParticlePlanesContact::define(unsigned int id, xParticleCubeContact* d)
{
	xPlaneObject *ps = d->CubeObject()->Planes();
	xContactMaterialParameters cp = { 0, };
	xMaterialPair xmp = { 0, };
	for (unsigned int i = 0; i < 6; i++)
	{
		xPlaneObject *p = &ps[i];
		cp.restitution = d->Restitution();
		cp.stiffness_ratio = d->StiffnessRatio();
		cp.friction = d->Friction();
		cp.rolling_resistance = d->RollingFactor();
		xmp = d->MaterialPropertyPair();
		hcmp[id + i] = cp;
		xmps[id + i] = xmp;
		pair_ip[id + i] = p;
		hpi[id + i] = {
			p->L1(), p->L2(),
			p->U1(), p->U2(),
			p->UW(), p->XW(),
			p->PA(), p->PB(),
			p->W2(), p->W3(), p->W4()
		};
	}
}

void xParticlePlanesContact::allocHostMemory(unsigned int n)
{
	nplanes = n;
	if(!hpi) hpi = new host_plane_info[nplanes];
	if(!hcmp) hcmp = new xContactMaterialParameters[nplanes];
	if(!xmps) xmps = new xMaterialPair[nplanes];
}

device_plane_info* xParticlePlanesContact::devicePlaneInfo()
{
	return dpi;
}

unsigned int xParticlePlanesContact::NumPlanes()
{
	return nplanes;
}

double xParticlePlanesContact::particle_plane_contact_detection(host_plane_info* _pe, vector3d& u, vector3d& xp, vector3d& wp, double r)
{
	double a_l1 = pow(wp.x - _pe->l1, 2.0);
	double b_l2 = pow(wp.y - _pe->l2, 2.0);
	double sqa = wp.x * wp.x;
	double sqb = wp.y * wp.y;
	double sqc = wp.z * wp.z;
	double sqr = r*r;

	// The sphere contacts with the wall face
	if (abs(wp.z) < r && (wp.x > 0 && wp.x < _pe->l1) && (wp.y > 0 && wp.y < _pe->l2)){
		vector3d dp = xp - _pe->xw;
		vector3d uu = _pe->uw / length(_pe->uw);
		int pp = -xsign(dot(dp, _pe->uw));
		u = pp * uu;
		double collid_dist = r - abs(dot(dp, u));
		return collid_dist;
	}

	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr){
		vector3d Xsw = xp - _pe->xw;
		double h = length(Xsw);
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > _pe->l1 && wp.y < 0 && (a_l1 + sqb + sqc) < sqr){
		vector3d Xsw = xp - _pe->w2;
		double h = length(Xsw);
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > _pe->l1 && wp.y > _pe->l2 && (a_l1 + b_l2 + sqc) < sqr){
		vector3d Xsw = xp - _pe->w3;
		double h = length(Xsw);
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x < 0 && wp.y > _pe->l2 && (sqa + b_l2 + sqc) < sqr){
		vector3d Xsw = xp - _pe->w4;
		double h = length(Xsw);
		u = Xsw / h;
		return r - h;
	}
	if ((wp.x > 0 && wp.x < _pe->l1) && wp.y < 0 && (sqb + sqc) < sqr){
		vector3d Xsw = xp - _pe->xw;
		vector3d wj_wi = _pe->w2 - _pe->xw;
		vector3d us = wj_wi / length(wj_wi);
		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.x < _pe->l1) && wp.y > _pe->l2 && (b_l2 + sqc) < sqr){
		vector3d Xsw = xp - _pe->w4;
		vector3d wj_wi = _pe->w3 - _pe->w4;
		vector3d us = wj_wi / length(wj_wi);
		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);
		u = -h_star / h;
		return r - h;

	}
	else if ((wp.x > 0 && wp.y < _pe->l2) && wp.x < 0 && (sqr + sqc) < sqr){
		vector3d Xsw = xp - _pe->xw;
		vector3d wj_wi = _pe->w4 - _pe->xw;
		vector3d us = wj_wi / length(wj_wi);
		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.y < _pe->l2) && wp.x > _pe->l1 && (a_l1 + sqc) < sqr){
		vector3d Xsw = xp - _pe->w2;
		vector3d wj_wi = _pe->w3 - _pe->w2;
		vector3d us = wj_wi / length(wj_wi);
		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);
		u = -h_star / h;
		return r - h;
	}


	return -1.0f;
}


bool xParticlePlanesContact::cpplCollision(
	xContactPairList* pairs, double r, double m, 
	vector3d& p, vector3d& v, vector3d& o,
	double &res, vector3d& tmax, vector3d& F, vector3d& M)
{
	foreach(xPairData* d, pairs->PlanePair())
	{
		vector3d m_fn = new_vector3d(0.0, 0.0, 0.0);
		vector3d m_m = new_vector3d(0.0, 0.0, 0.0);
		vector3d m_ft = new_vector3d(0.0, 0.0, 0.0);
		double rcon = r - 0.5 * d->gab;
		vector3d u = new_vector3d(d->nx, d->ny, d->nz);
		vector3d cp = r * u;
		vector3d dv = -v - cross(o, cp);
		xContactMaterialParameters cmp = hcmp[d->id];
		xContactParameters c = getContactParameters(
			r, 0.0,
			m, 0.0,
			xmps[d->id].Ei, xmps[d->id].Ej,
			xmps[d->id].Pri, xmps[d->id].Prj,
			xmps[d->id].Gi, xmps[d->id].Gj, 
			cmp.restitution, cmp.stiffness_ratio,
			cmp.friction, cmp.rolling_friction, cmp.cohesion);
		switch (force_model)
		{
		case DHS: DHSModel(c, d->gab, d->delta_s, d->dot_s, cp, dv, u, m_fn, m_ft, m_m); break;
		}
		RollingResistanceForce(c.rfric, r, 0.0, cp, m_fn, m_ft, res, tmax);
		F += m_fn + m_ft;
		M += m_m;
	}
	return true;
}

unsigned int xParticlePlanesContact::NumContact()
{
	return nplanes;
}

void xParticlePlanesContact::updateCollisionPair(
	xContactPairList& xcpl, double r, vector3d pos)
{
	for (unsigned int i = 0; i < nplanes; i++)
	{
		host_plane_info *pe = &hpi[i];
		vector3d dp = pos - pe->xw;
		vector3d wp = new_vector3d(dot(dp, pe->u1), dot(dp, pe->u2), dot(dp, pe->uw));
		vector3d u;

		double cdist = particle_plane_contact_detection(pe, u, pos, wp, r);
		if (cdist > 0){		
			if (xcpl.IsNewPlaneContactPair(i))
			{
				xPairData *pd = new xPairData;
				*pd = { PLANE_SHAPE, 0, i, 0, 0, cdist, u.x, u.y, u.z };
				xcpl.insertPlaneContactPair(pd);
			}
			else
			{
				xPairData *pd = xcpl.PlanePair(i);
				pd->gab = cdist;
				pd->nx = u.x;
				pd->ny = u.y;
				pd->nz = u.z;
			}
		}
		else
		{
			xcpl.deletePlanePairData(i);
		}
	}
}

void xParticlePlanesContact::cudaMemoryAlloc(unsigned int np)
{
	device_contact_property *_hcp = new device_contact_property[nplanes];
	for (unsigned int i = 0; i < nplanes; i++)
	{
		_hcp[i] = { xmps[i].Ei, xmps[i].Ej, xmps[i].Pri, xmps[i].Prj, xmps[i].Gi, xmps[i].Gj,
			hcmp[i].restitution, hcmp[i].friction, hcmp[i].rolling_friction, hcmp[i].cohesion, hcmp[i].stiffness_ratio, hcmp[i].rolling_resistance };
	}
	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_plane_info) * nplanes));
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property) * nplanes));
// 	checkCudaErrors(cudaMalloc((void**)&d_pair_count, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMalloc((void**)&d_old_pair_count, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMalloc((void**)&d_pair_start, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMalloc((void**)&d_old_pair_start, sizeof(unsigned int) * np));
	checkCudaErrors(cudaMemcpy(dpi, hpi, sizeof(device_plane_info) * nplanes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dcp, _hcp, sizeof(device_contact_property) * nplanes, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMemset(d_pair_count, 0, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMemset(d_old_pair_count, 0, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMemset(d_pair_start, 0, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMemset(d_old_pair_start, 0, sizeof(unsigned int) * np));
	delete[] _hcp;
}

void xParticlePlanesContact::cuda_collision(double *pos, double *vel, double *omega, double *mass, double *force, double *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{
// 	unsigned int nc = cu_calculate_particle_plane_contact_count(
// 		dpi, d_pppd, d_old_pair_count, d_pair_count, d_pair_start, pos, nplanes, np);
// 	if (nc)
// 		bool pause = true;
// // 	thrust::constant_iterator<unsigned int> first(10);
// //  	unsigned int ncount = thrust::reduce(d_pair_count, d_pair_count + np);
//  	checkCudaErrors(cudaMemcpy(d_old_pair_start, d_pair_start, sizeof(unsigned int) * np, cudaMemcpyDeviceToDevice));
// // 	thrust::inclusive_scan(d_pair_count, d_pair_count + ncount, d_pair_start);
// 	particle_plane_pair_data* d_old_pppd;
// 	checkCudaErrors(cudaMalloc((void**)&d_old_pppd, sizeof(particle_plane_pair_data) * ncontact));
// 	checkCudaErrors(cudaMemcpy(d_old_pppd, d_pppd, sizeof(particle_plane_pair_data) * ncontact, cudaMemcpyDeviceToDevice));
// 	if (d_pppd)
// 		checkCudaErrors(cudaFree(d_pppd));
// 	checkCudaErrors(cudaMalloc((void**)&d_pppd, sizeof(particle_plane_pair_data) * nc));
// 	checkCudaErrors(cudaMemset(d_pppd, 0, sizeof(particle_plane_pair_data) * nc));
// 	checkCudaErrors(cudaMemcpy(d_old_pair_start, d_pair_start, sizeof(unsigned int) * np, cudaMemcpyDeviceToDevice));
//  	cu_copy_old_to_new_memory(d_old_pair_count, d_pair_count, d_old_pair_start, d_pair_start, d_old_pppd, d_pppd, nc, np);
// 	checkCudaErrors(cudaFree(d_old_pppd));
// 	cu_update_particle_plane_contact(dpi, pos, vel, omega, mass, force, moment, d_pair_count, d_pair_start, d_pppd, dcp, nplanes, np);
// 	ncontact = nc;
}

