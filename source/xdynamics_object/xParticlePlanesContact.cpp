#include "xdynamics_object/xParticlePlanesContact.h"
#include "xdynamics_object/xParticleCubeContact.h"
#include "xdynamics_object/xPlaneObject.h"
#include "xdynamics_object/xCubeObject.h"
#include "xdynamics_simulation/xSimulation.h"
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
	, dbi(NULL)
	, dbf(NULL)
	, nContactObject(0)

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
	if (dbf) delete[] dbf; dbf = NULL;
	if (dpi) checkCudaErrors(cudaFree(dpi)); dpi = NULL;
	if (dbi) checkCudaErrors(cudaFree(dbi)); dbi = NULL;
	if (db_force) checkCudaErrors(cudaFree(db_force)); db_force = NULL;
	if (db_moment) checkCudaErrors(cudaFree(db_moment)); db_moment = NULL;
// 	if (d_pair_count) checkCudaErrors(cudaFree(d_pair_count)); d_pair_count = NULL;
// 	if (d_old_pair_start) checkCudaErrors(cudaFree(d_old_pair_start)); d_old_pair_start = NULL;
// 	if (d_pair_start) checkCudaErrors(cudaFree(d_pair_start)); d_pair_start = NULL;
// 	if (d_pppd) checkCudaErrors(cudaFree(d_pppd)); d_pppd = NULL;
	//qDeleteAll(pair_ip);
}

void xParticlePlanesContact::define(unsigned int id, xParticlePlaneContact* d)
{
	xPlaneObject *p = d->PlaneObject();
	if (p->MovingObject())
		nmoving++;
	xContactMaterialParameters cp = { 0, };
	xMaterialPair xmp = { 0, };
	cp.restitution = d->Restitution();
	cp.stiffness_ratio = d->StiffnessRatio();
	cp.friction = d->Friction();
	cp.rolling_friction = d->RollingFactor();
	cp.cohesion = d->Cohesion();
	cp.stiffness_multiplyer = d->StiffMultiplyer();
	xmp = d->MaterialPropertyPair();
	hcmp[id] = cp;
	xmps[id] = xmp;
	pair_ip[id] = p;
	hpi[id] = {
		p->MovingObject(),
		nContactObject,// p->MovingObject() ? nmoving - 1 : 0,
		p->L1(), p->L2(),
		p->U1(), p->U2(),
		p->UW(), p->XW(),
		p->PA(), p->PB(),
		p->W2(), p->W3(), p->W4()
	};
	nContactObject++;
	pair_contact[id] = d;
}

void xParticlePlanesContact::define(unsigned int id, xParticleCubeContact* d)
{
	xPlaneObject *ps = d->CubeObject()->Planes();
	xContactMaterialParameters cp = { 0, };
	xMaterialPair xmp = { 0, };
	if (d->CubeObject()->MovingObject())
		nmoving++;
	for (unsigned int i = 0; i < 6; i++)
	{
		xPlaneObject *p = &ps[i];
		cp.restitution = d->Restitution();
		cp.stiffness_ratio = d->StiffnessRatio();
		cp.friction = d->Friction();
		cp.cohesion = d->Cohesion();
		cp.rolling_friction = d->RollingFactor();
		cp.stiffness_multiplyer = d->StiffMultiplyer();
		xmp = d->MaterialPropertyPair();
		hcmp[id + i] = cp;
		xmps[id + i] = xmp;
		pair_ip[id + i] = p;
		hpi[id + i] = {
			d->CubeObject()->MovingObject(),
			nContactObject,// d->CubeObject()->MovingObject() ? nmoving - 1 : 0,
			p->L1(), p->L2(),
			p->U1(), p->U2(),
			p->UW(), p->XW(),
			p->PA(), p->PB(),
			p->W2(), p->W3(), p->W4()
		};
	}
	nContactObject++;
	pair_contact[id] = d;
}

void xParticlePlanesContact::allocHostMemory(unsigned int n)
{
	nplanes = n;
	if(!hpi) hpi = new host_plane_info[nplanes];
	if(!hcmp) hcmp = new xContactMaterialParameters[nplanes];
	if(!xmps) xmps = new xMaterialPair[nplanes];
}

void xParticlePlanesContact::updataPlaneObjectData(bool is_first_set_up)
{
	device_body_info *bi = NULL;
	if (nmoving || is_first_set_up)
		bi = new device_body_info[nContactObject];
	//else
	//	return;
	QMapIterator<unsigned int, xPlaneObject*> it(pair_ip);
	while (it.hasNext())
	{
		it.next();
		unsigned int id = it.key();
		xPlaneObject* p = it.value();
		if (p->MovingObject())
		{
			host_plane_info new_hpi = { 0, };
			new_hpi.ismoving = hpi[id].ismoving;
			new_hpi.mid = hpi[id].mid;
			new_hpi.xw = p->Position() + p->toGlobal(p->LocalPoint(0));
			new_hpi.w2 = p->Position() + p->toGlobal(p->LocalPoint(1));
			new_hpi.w3 = p->Position() + p->toGlobal(p->LocalPoint(2));
			new_hpi.w4 = p->Position() + p->toGlobal(p->LocalPoint(3));
			new_hpi.pa = new_hpi.w2 - new_hpi.xw;
			new_hpi.pb = new_hpi.w4 - new_hpi.xw;
			new_hpi.l1 = length(new_hpi.pa);// .length();
			new_hpi.l2 = length(new_hpi.pb);// .length();
			new_hpi.u1 = new_hpi.pa / new_hpi.l1;
			new_hpi.u2 = new_hpi.pb / new_hpi.l2;
			new_hpi.uw = cross(new_hpi.u1, new_hpi.u2);
			hpi[id] = new_hpi;

			if (xSimulation::Gpu())
				checkCudaErrors(cudaMemcpy(dpi + id, &new_hpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
		}
		//if ((xSimulation::Gpu() && p->MovingObject()) || (is_first_set_up && p->MovingObject()))
		//{
		//	if (mcnt >= nContactObject)
		//		continue;
		//	xContact* xc = pair_contact[hpi[id].mid];
		//	xPointMass* pm = NULL;
		//	if (xc->PairType() == PARTICLE_CUBE)
		//	{
		//		pm = dynamic_cast<xParticleCubeContact*>(xc)->CubeObject();
		//	}
		//	else
		//	{
		//		pm = p;
		//	}
		//	euler_parameters ep = pm->EulerParameters(), ed = pm->DEulerParameters();
		//	bi[hpi[id].mid] = {
		//		pm->Mass(),
		//		pm->Position().x, pm->Position().y, pm->Position().z,
		//		pm->Velocity().x, pm->Velocity().y, pm->Velocity().z,
		//		ep.e0, ep.e1, ep.e2, ep.e3,
		//		ed.e0, ed.e1, ed.e2, ed.e3
		//	};
		//	//checkCudaErrors(cudaMemset(db_force, 0, sizeof(double3) * nplanes));
		//	//checkCudaErrors(cudaMemset(db_moment, 0, sizeof(double3) * nplanes));
		//	//std::cout << "plane_velocity : [" << p->Velocity().x << ", " << p->Velocity().y << ", " << p->Velocity().z << std::endl;
		//	//std::cout << "plane_velocity : [" << p->Velocity().x << ", " << p->Velocity().y << ", " << p->Velocity().z << std::endl;
		//	checkCudaErrors(cudaMemcpy(dbi, bi, sizeof(device_body_info) * nContactObject, cudaMemcpyHostToDevice));
		//	mcnt++;
		//}
	}
	if (xSimulation::Gpu())
	{
		unsigned int mcnt = 0;
		foreach(xContact* xc, pair_contact)
		{
			xPointMass* pm = NULL;
			if (xc->PairType() == PARTICLE_CUBE)
				pm = dynamic_cast<xParticleCubeContact*>(xc)->CubeObject();
			else if (xc->PairType() == PARTICLE_PANE)
				pm = dynamic_cast<xParticlePlaneContact*>(xc)->PlaneObject();
			if (pm->MovingObject() == false && !is_first_set_up) continue;
			euler_parameters ep = pm->EulerParameters(), ed = pm->DEulerParameters();
			bi[mcnt++] = {
				pm->Mass(),
				pm->Position().x, pm->Position().y, pm->Position().z,
				pm->Velocity().x, pm->Velocity().y, pm->Velocity().z,
				ep.e0, ep.e1, ep.e2, ep.e3,
				ed.e0, ed.e1, ed.e2, ed.e3
			};
			//checkCudaErrors(cudaMemset(db_force, 0, sizeof(double3) * nplanes));
			//checkCudaErrors(cudaMemset(db_moment, 0, sizeof(double3) * nplanes));
			//std::cout << "plane_velocity : [" << p->Velocity().x << ", " << p->Velocity().y << ", " << p->Velocity().z << std::endl;
			//std::cout << "plane_velocity : [" << p->Velocity().x << ", " << p->Velocity().y << ", " << p->Velocity().z << std::endl;			
		}
		if(bi)
			checkCudaErrors(cudaMemcpy(dbi, bi, sizeof(device_body_info) * nContactObject, cudaMemcpyHostToDevice));
	}
	
	//if (nmoving)
	if(bi)
		delete[] bi; 
	
}

device_plane_info* xParticlePlanesContact::devicePlaneInfo()
{
	return dpi;
}

device_body_info * xParticlePlanesContact::devicePlaneBodyInfo()
{
	return dbi;
}

device_body_force * xParticlePlanesContact::deviceBodyForceAndMoment()
{
	return dbf;
}

unsigned int xParticlePlanesContact::NumPlanes()
{
	return nplanes;
}

void xParticlePlanesContact::getPlaneContactForce()
{
	if (nmoving)
	{
		//double3 *hbf = new double3[nplanes];// device_body_force *hbf = new device_body_force[nmoving];
		//double3 *hbm = new double3[nplanes];
		//device_body_info *hbi = new device_body_info[nmoving];
		//checkCudaErrors(cudaMemcpy(hbf, db_force, sizeof(double3) * nplanes, cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpy(hbm, db_moment, sizeof(double3) * nplanes, cudaMemcpyDeviceToHost));

		QMapIterator<unsigned int, xPlaneObject*> xpl(pair_ip);
		while (xpl.hasNext())
		{
			xpl.next();
			unsigned int id = xpl.key();
			xPlaneObject* o = xpl.value();
			if (o->MovingObject())
			{
				//std::cout << "plane force : [" << dbf[id].force.x << ", " << dbf[id].force.y << ", " << dbf[id].force.z << "]" << std::endl;
				/*if (hbi[id].force.y != 0.0)
					bool ddd = true;*/
				o->addContactForce(dbf[id].force.x, dbf[id].force.y, dbf[id].force.z);
				o->addContactMoment(dbf[id].moment.x, dbf[id].moment.y, dbf[id].moment.z);
			}

		}
	}
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
	//if (abs(wp.z) < r && (wp.x > 0 && wp.x < _pe->l1) && (wp.y > 0 && wp.y < _pe->l2)){
	//	vector3d dp = xp - _pe->xw;
	//	vector3d uu = _pe->uw / length(_pe->uw);
	//	int pp = -xsign(dot(dp, _pe->uw));
	//	u = pp * uu;
	//	double collid_dist = r - abs(dot(dp, u));
	//	return collid_dist;
	//}

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
	vector3d dp = xp - _pe->xw;
	vector3d uu = _pe->uw / length(_pe->uw);
	int pp = -xsign(dot(dp, _pe->uw));
	u = -uu;
	double collid_dist = r - abs(dot(dp, u));
	return collid_dist;
}

bool xParticlePlanesContact::cpplCollision(
	xContactPairList* pairs, unsigned int i, double r, double m, 
	vector3d& p, vector3d& v, vector3d& o,
	double &res, vector3d& tmax, vector3d& F, vector3d& M, 
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
	foreach(xPairData* d, pairs->PlanePair())
	{
		xPlaneObject* pl = pair_ip[d->id];
		vector3d m_fn = new_vector3d(0.0, 0.0, 0.0);
		vector3d m_m = new_vector3d(0.0, 0.0, 0.0);
		vector3d m_ft = new_vector3d(0.0, 0.0, 0.0);
		double rcon = r - 0.5 * d->gab;
		vector3d u = new_vector3d(d->nx, d->ny, d->nz);
		vector3d cpt = new_vector3d(d->cpx, d->cpy, d->cpz);
		vector3d dcpr = cpt - cp;
		//vector3d cp = r * u;
		vector3d dcpr_j = cpt - pl->Position();
		vector3d oj = 2.0 * GMatrix(pl->EulerParameters()) * pl->DEulerParameters();
		vector3d dv = pl->Velocity() + cross(oj, dcpr_j) - (v + cross(o, dcpr));
		//unsigned int jjjj = d->id;
		xContactMaterialParameters cmp = hcmp[d->id];
		xContactParameters c = getContactParameters(
			r, 0.0,
			m, 0.0,
			xmps[d->id].Ei, xmps[d->id].Ej,
			xmps[d->id].Pri, xmps[d->id].Prj,
			xmps[d->id].Gi, xmps[d->id].Gj, 
			cmp.restitution, cmp.stiffness_ratio,
			cmp.friction, cmp.rolling_friction, cmp.cohesion);
		if (d->gab < 0 && abs(d->gab) < abs(c.coh_s))
		{
			double f = JKRSeperationForce(c, cmp.cohesion);
			double cf = cohesionForce(cmp.cohesion, d->gab, c.coh_r, c.coh_e, c.coh_s, f);
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
		case DHS: DHSModel(c, d->gab, d->delta_s, d->dot_s, cmp.cohesion, dv, u, m_fn, m_ft); break;
		}
		/*if (cmp.cohesion && c.coh_s > d->gab)
			d->isc = false;*/
		RollingResistanceForce(c.rfric, r, 0.0, dcpr, m_fn, m_ft, res, tmax);
		vector3d sum_force = m_fn + m_ft;
		F += sum_force;
		M += cross(dcpr, sum_force);
		pl->addContactForce(-sum_force.x, -sum_force.y, -sum_force.z);
		vector3d body_moment = -cross(dcpr_j, sum_force);
		pl->addContactMoment(-body_moment.x, -body_moment.y, -body_moment.z);

	}
	return true;
}

unsigned int xParticlePlanesContact::NumContact()
{
	return nplanes;
}

xContactMaterialParameters & xParticlePlanesContact::ContactMaterialParameters(unsigned int id)
{
	return hcmp[id];
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
		xContactMaterialParameters cmp = hcmp[i];

		double cdist = particle_plane_contact_detection(pe, u, pos, wp, r);
		if (cdist > 0){		
			vector3d cpt = pos + r * u;
			if (xcpl.IsNewPlaneContactPair(i))
			{
				xPairData *pd = new xPairData;
				*pd = { PLANE_SHAPE, true, 0, i, 0, 0, cpt.x, cpt.y, cpt.z, cdist, u.x, u.y, u.z };
				xcpl.insertPlaneContactPair(pd);
			}
			else
			{
				xPairData *pd = xcpl.PlanePair(i);
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
			xPairData *pd = xcpl.PlanePair(i);
			if (pd)
			{
				bool isc = pd->isc;
				if (!isc)
					xcpl.deletePlanePairData(i);
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

void xParticlePlanesContact::cudaMemoryAlloc(unsigned int np)
{
	device_contact_property *_hcp = new device_contact_property[nplanes];
	for (unsigned int i = 0; i < nplanes; i++)
	{
		_hcp[i] = { xmps[i].Ei, xmps[i].Ej, xmps[i].Pri, xmps[i].Prj, xmps[i].Gi, xmps[i].Gj,
			hcmp[i].restitution, hcmp[i].friction, hcmp[i].rolling_friction, hcmp[i].cohesion, hcmp[i].stiffness_ratio, hcmp[i].stiffness_multiplyer };
	}
	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_plane_info) * nplanes));
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property) * nplanes));
	checkCudaErrors(cudaMalloc((void**)&dbi, sizeof(device_body_info) * nplanes));
	/*checkCudaErrors(cudaMalloc((void**)&db_force, sizeof(double3) * nplanes));
	checkCudaErrors(cudaMalloc((void**)&db_moment, sizeof(double3) * nplanes));*/
// 	checkCudaErrors(cudaMalloc((void**)&d_pair_count, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMalloc((void**)&d_old_pair_count, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMalloc((void**)&d_pair_start, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMalloc((void**)&d_old_pair_start, sizeof(unsigned int) * np));
	checkCudaErrors(cudaMemcpy(dpi, hpi, sizeof(device_plane_info) * nplanes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dcp, _hcp, sizeof(device_contact_property) * nplanes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(dbi, 0, sizeof(device_body_info) * nContactObject));	
	/*checkCudaErrors(cudaMemset(db_force, 0, sizeof(double3) * nplanes));
	checkCudaErrors(cudaMemset(db_moment, 0, sizeof(double3) * nplanes));*/
// 	checkCudaErrors(cudaMemset(d_pair_count, 0, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMemset(d_old_pair_count, 0, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMemset(d_pair_start, 0, sizeof(unsigned int) * np));
// 	checkCudaErrors(cudaMemset(d_old_pair_start, 0, sizeof(unsigned int) * np));
	dbf = new device_body_force[nplanes];
	updataPlaneObjectData(true);
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

