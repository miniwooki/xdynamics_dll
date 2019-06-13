#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_object/xObject.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_algebra/xGridCell.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <list>
//#include <QtCore/QDebug>

xContactManager::xContactManager()
	: cpp(NULL)
	, ncontact(0)
	, cpmeshes(NULL)
	, cpplane(NULL)
	, d_pair_count_pp(NULL)
	, d_pair_count_ppl(NULL)
	, d_pair_count_ptri(NULL)
	, d_pair_id_pp(NULL)
	, d_pair_id_ppl(NULL)
	, d_pair_id_ptri(NULL)
	, d_tsd_pp(NULL)
	, d_tsd_ppl(NULL)
	, d_tsd_ptri(NULL)
	, d_Tmax(NULL)
	, d_RRes(NULL)
	, xcpl(NULL)
	, Tmax(NULL)
	, RRes(NULL)

{

}

xContactManager::~xContactManager()
{
	qDeleteAll(cots);
// 	if (cpp) delete cpp; cpp = NULL;
 	if (cpmeshes) delete cpmeshes; cpmeshes = NULL;
 	if (cpplane) delete cpplane; cpplane = NULL;
	if (Tmax) delete[] Tmax; Tmax = NULL;
	if (RRes) delete[] RRes; RRes = NULL;
	if (xcpl) delete[] xcpl; xcpl = NULL;

	if (d_pair_count_pp) checkCudaErrors(cudaFree(d_pair_count_pp)); d_pair_count_pp = NULL;
	if (d_pair_count_ppl) checkCudaErrors(cudaFree(d_pair_count_ppl)); d_pair_count_ppl = NULL;
	if (d_pair_count_ptri) checkCudaErrors(cudaFree(d_pair_count_ptri)); d_pair_count_ptri = NULL;
	if (d_pair_id_pp) checkCudaErrors(cudaFree(d_pair_id_pp)); d_pair_id_pp = NULL;
	if (d_pair_id_ppl) checkCudaErrors(cudaFree(d_pair_id_ppl)); d_pair_id_ppl = NULL;
	if (d_pair_id_ptri) checkCudaErrors(cudaFree(d_pair_id_ptri)); d_pair_id_ptri = NULL;
	if (d_tsd_pp) checkCudaErrors(cudaFree(d_tsd_pp)); d_tsd_pp = NULL;
	if (d_tsd_ppl) checkCudaErrors(cudaFree(d_tsd_ppl)); d_tsd_ppl = NULL;
	if (d_tsd_ptri) checkCudaErrors(cudaFree(d_tsd_ptri)); d_tsd_ptri = NULL;
	if (d_Tmax) checkCudaErrors(cudaFree(d_Tmax)); d_Tmax = NULL;
	if (d_RRes) checkCudaErrors(cudaFree(d_RRes)); d_RRes = NULL;
}



xContact* xContactManager::CreateContactPair(
	std::string n, xContactForceModelType method, xObject* o1, xObject* o2, xContactParameterData& d)
{
	xContactPairType pt = getContactPair(o1->Shape(), o2->Shape());
	xContact *c = NULL;
	switch (pt)
	{
	case PARTICLE_PARTICLE:	cpp = new xParticleParticleContact(n); c = cpp;	break;
	case PARTICLE_CUBE:	c = new xParticleCubeContact(n, o1, o2); break;
	case PARTICLE_PANE:	c = new xParticlePlaneContact(n); break;
	case PARTICLE_MESH_SHAPE: c = new xParticleMeshObjectContact(n, o1, o2); cpmesh[c->Name()] = dynamic_cast<xParticleMeshObjectContact*>(c); break;
	}
	xMaterialPair mpp =
	{
		o1->Youngs(), o2->Youngs(),
		o1->Poisson(), o2->Poisson(),
		o1->Shear(), o2->Shear()
	};
	c->setContactForceModel(method);
	c->setFirstObject(o1);
	c->setSecondObject(o2);
	c->setMaterialPair(mpp);
	c->setRestitution(d.rest);
	c->setStiffnessRatio(d.rto);
	c->setFriction(d.mu);
	c->setCohesion(d.coh);
	c->setRollingFactor(d.rf);
	QString name = QString::fromStdString(n);
	cots[name] = c;
	//c->setContactParameters(rest, ratio, fric, cohesion);
	return c;
}

unsigned int xContactManager::setupParticlesMeshObjectsContact()
{
	unsigned int n = 0;
	if (cpmesh.size() && !cpmeshes)
	{
		cpmeshes = new xParticleMeshObjectsContact;
		n = cpmeshes->define(cpmesh);
	}
	return n;
}

void xContactManager::setupParticlesPlanesContact()
{
	unsigned int n = 0;
	foreach(xContact* xc, cots)
	{
		switch (xc->PairType())
		{
		case xContactPairType::PARTICLE_PANE: n++; break;
		case xContactPairType::PARTICLE_CUBE: n += 6; break;
		}
	}
	if (!n)	return;
	if (n && !cpplane)
		cpplane = new xParticlePlanesContact;
	cpplane->allocHostMemory(n);
	n = 0;
	foreach(xContact* xc, cots)
	{
		switch (xc->PairType())
		{
		case xContactPairType::PARTICLE_PANE: cpplane->define(n, dynamic_cast<xParticlePlaneContact*>(xc)); n++; break;
		case xContactPairType::PARTICLE_CUBE: cpplane->define(n, dynamic_cast<xParticleCubeContact*>(xc)); n += 6; break;
		}
	}
}

void xContactManager::insertContact(xContact* c)
{
	QString qname = c->Name();
	if (c->PairType() == xContactPairType::PARTICLE_MESH_SHAPE)
	{
	}
	//	cppos[c->Name()] = dynamic_cast<contact_particles_polygonObject*>(c);
	else
		cots[qname] = c;
}

xContact* xContactManager::Contact(QString n)
{
	QStringList keys = cots.keys();
	QStringList::const_iterator it = qFind(keys, n);
	if (it == keys.end() || !keys.size())
		return NULL;
	return cots[n];
}

QMap<QString, xContact*>& xContactManager::Contacts()
{
	return cots;
}

xParticleParticleContact* xContactManager::ContactParticles()
{
	return cpp;
}

xParticleMeshObjectsContact* xContactManager::ContactParticlesMeshObjects()
{
	return cpmeshes;
}

xParticlePlanesContact* xContactManager::ContactParticlesPlanes()
{
	return cpplane;
}

bool xContactManager::runCollision(
	double *pos, double *vel, double *omega,
	double *mass, double* inertia, double *force, double *moment,
	unsigned int *sorted_id, unsigned int *cell_start,
	unsigned int *cell_end, unsigned int *cluster_index,
	unsigned int np, unsigned int ns, unsigned int nc)
{
	if (xSimulation::Cpu())
	{
		hostCollision
		(
			(vector4d*)pos,
			(vector3d*)vel,
			(vector3d*)omega,
			mass,
			inertia,
			(vector3d*)force,
			(vector3d*)moment,
			sorted_id, cell_start, cell_end,
			cluster_index, np, ns, nc
		);
	}
	else if (xSimulation::Gpu())
	{
		deviceCollision(
			pos, vel, omega,
			mass, inertia, force, moment,
			sorted_id, cell_start, cell_end, np
		);
	}
	return true;
}

void xContactManager::update()
{
	if (cpmeshes)
	{
		cpmeshes->updateMeshObjectData();
	}
	// 	if (cppoly)
	// 	{
	// 		model::isSinglePrecision ?
	// 			cppoly->updatePolygonObjectData_f() :
	// 			cppoly->updatePolygonObjectData();
	// 	}
}

void xContactManager::allocPairList(unsigned int np)
{
	if (xSimulation::Cpu())
	{
		if (!xcpl)
			xcpl = new xContactPairList[np];
		Tmax = new vector3d[np];
		RRes = new double[np];
	}
	else
	{
		checkCudaErrors(cudaMalloc((void**)&d_pair_count_pp, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMalloc((void**)&d_pair_count_ppl, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMalloc((void**)&d_pair_count_ptri, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMalloc((void**)&d_pair_id_pp, sizeof(unsigned int) * np * MAX_P2P_COUNT));
		checkCudaErrors(cudaMalloc((void**)&d_pair_id_ppl, sizeof(unsigned int) * np * MAX_P2PL_COUNT));
		checkCudaErrors(cudaMalloc((void**)&d_pair_id_ptri, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
		checkCudaErrors(cudaMalloc((void**)&d_tsd_pp, sizeof(double2) * np * MAX_P2P_COUNT));
		checkCudaErrors(cudaMalloc((void**)&d_tsd_ppl, sizeof(double2) * np * MAX_P2PL_COUNT));
		checkCudaErrors(cudaMalloc((void**)&d_tsd_ptri, sizeof(double2) * np * MAX_P2MS_COUNT));
		// 		//checkCudaErrors(cudaMalloc((void**)&d_old_pair_count, sizeof(unsigned int) * np));
		// 		//checkCudaErrors(cudaMalloc((void**)&d_pair_start, sizeof(unsigned int) * np));
		// 		//checkCudaErrors(cudaMalloc((void**)&d_old_pair_start, sizeof(unsigned int) * np));
		// 		//checkCudaErrors(cudaMalloc((void**)&d_type_count, sizeof(int) * np * 2));
		checkCudaErrors(cudaMalloc((void**)&d_Tmax, sizeof(double3) * np));
		checkCudaErrors(cudaMalloc((void**)&d_RRes, sizeof(double) * np));
		checkCudaErrors(cudaMemset(d_pair_count_pp, 0, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMemset(d_pair_count_ppl, 0, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMemset(d_pair_count_ptri, 0, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMemset(d_pair_id_pp, 0, sizeof(unsigned int) * np * MAX_P2P_COUNT));
		checkCudaErrors(cudaMemset(d_pair_id_ppl, 0, sizeof(unsigned int) * np * MAX_P2PL_COUNT));
		checkCudaErrors(cudaMemset(d_pair_id_ptri, 0, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
		checkCudaErrors(cudaMemset(d_tsd_pp, 0, sizeof(double2) * np * MAX_P2P_COUNT));
		checkCudaErrors(cudaMemset(d_tsd_ppl, 0, sizeof(double2) * np * MAX_P2PL_COUNT));
		checkCudaErrors(cudaMemset(d_tsd_ptri, 0, sizeof(double2) * np * MAX_P2MS_COUNT));
		// 		//checkCudaErrors(cudaMalloc((void**)&d_old_pair_count, sizeof(unsigned int) * np));
		// 		//checkCudaErrors(cudaMalloc((void**)&d_pair_start, sizeof(unsigned int) * np));
		// 		//checkCudaErrors(cudaMalloc((void**)&d_old_pair_start, sizeof(unsigned int) * np));
		// 		//checkCudaErrors(cudaMalloc((void**)&d_type_count, sizeof(int) * np * 2));
		checkCudaErrors(cudaMemset(d_Tmax, 0, sizeof(double3) * np));
		checkCudaErrors(cudaMemset(d_RRes, 0, sizeof(double) * np));
	}
}

void xContactManager::updateCollisionPair(
	vector4d* pos, 
	unsigned int* sorted_id,
	unsigned int* cell_start,
	unsigned int* cell_end,
	unsigned int* cluster_index,
	unsigned int np,
	unsigned int nc)
{
	for (unsigned int i = 0; i < np + nc; i++)
	{
		unsigned int count = 0;
		vector3d p = new_vector3d(pos[i].x, pos[i].y, pos[i].z);
		double r = pos[i].w;
		if(cpplane)
			cpplane->updateCollisionPair(xcpl[i], r, p);
		vector3i gp = xGridCell::getCellNumber(p.x, p.y, p.z);
		vector3d old_cpt = new_vector3d(0.0, 0.0, 0.0);
		vector3d old_unit = new_vector3d(0.0, 0.0, 0.0);
		vector3i ctype = new_vector3i(0, 0, 0);
		for (int z = -1; z <= 1; z++) {
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					vector3i neigh = new_vector3i(gp.x + x, gp.y + y, gp.z + z);
					unsigned int hash = xGridCell::getHash(neigh);
					unsigned int sid = cell_start[hash];
					if (sid != 0xffffffff) {
						unsigned int eid = cell_end[hash];
						for (unsigned int j = sid; j < eid; j++) {
							unsigned int k = sorted_id[j];
							if (i != k && k < np)
							{
								//unsigned int ci = cluster_index[(k - np) * 2 + 1];
// 								if (i == ci)
// 									continue;
								vector3d jp = new_vector3d(pos[k].x, pos[k].y, pos[k].z);
								double jr = pos[k].w;
								cpp->updateCollisionPair(k, xcpl[i], r, jr, p, jp);
							}
							else if (k >= np)
							{
								if (cpmeshes->updateCollisionPair(k - np, xcpl[i], r, p, old_cpt, old_unit, ctype))
									count++;
							}
						}
					}
				}
			}
		}
		if (count > 1)
			std::cout << "mesh contact overlab occured." << std::endl;
	}
}

void xContactManager::deviceCollision(
	double *pos, double *vel, double *omega,
	double *mass, double* inertia, double *force, double *moment,
	unsigned int *sorted_id, unsigned int *cell_start,
	unsigned int *cell_end, unsigned int np)
{
	cu_calculate_p2p(1, pos, vel, omega, force, moment, mass,
		d_Tmax, d_RRes, d_pair_count_pp, d_pair_id_pp, d_tsd_pp, sorted_id,
		cell_start, cell_end, cpp->DeviceContactProperty(), np);

	if (cpplane)
	{
		if (cpplane->NumContact())
		{
			cu_plane_contact_force(1, cpplane->devicePlaneInfo(), pos, vel,
				omega, force, moment, mass,
				d_Tmax, d_RRes, d_pair_count_ppl, d_pair_id_ppl, d_tsd_ppl,
				np, cpplane->DeviceContactProperty());
		}
	}
	if (cpmeshes)
	{
		if (cpmeshes->NumContactObjects())
		{
			cpmeshes->updateMeshMassData();
			cu_particle_polygonObject_collision(1, cpmeshes->deviceTrianglesInfo(), cpmeshes->devicePolygonObjectMassInfo(),
				pos, vel, omega, force, moment, mass,
				d_Tmax, d_RRes, d_pair_count_ptri, d_pair_id_ptri, d_tsd_ptri,
				sorted_id, cell_start, cell_end, cpmeshes->DeviceContactProperty(), np);
			cpmeshes->getMeshContactForce();
		}
	}
	cu_decide_rolling_friction_moment(d_Tmax, d_RRes, inertia, omega, moment, np);
}

void xContactManager::hostCollision(
	vector4d *pos, 
	vector3d *vel, 
	vector3d *omega, 
	double *mass, 
	double* inertia,
	vector3d *force, 
	vector3d *moment, 
	unsigned int *sorted_id,
	unsigned int *cell_start,
	unsigned int *cell_end,
	unsigned int *cluster_index,
	unsigned int np,
	unsigned int ns,
	unsigned int nc)
{
	updateCollisionPair(pos, sorted_id, cell_start, cell_end, cluster_index, np, nc);
	for (unsigned int i = 0; i < np + nc; i++)
	{
		vector3d F = new_vector3d(0.0, 0.0, 0.0);
		vector3d M = new_vector3d(0.0, 0.0, 0.0);
		double R = 0;
		vector3d T = new_vector3d(0.0, 0.0, 0.0);
		xContactPairList* pairs = &xcpl[i];
		vector3d p = new_vector3d(pos[i].x, pos[i].y, pos[i].z);
		vector3d v = vel[i];
		vector3d o = omega[i];
		double m = mass[i];
		double j = inertia[i];
		double r = pos[i].w;
		cpplane->cpplCollision(pairs, r, m, p, v, o, R, T, F, M);
		cpp->cppCollision(pairs, i, pos, vel, omega, mass, R, T, F, M);
		cpmeshes->cppolyCollision(pairs, r, m, p, v, o, R, T, F, M);
		force[i] += F;
		moment[i] += M;

		vector3d _Tmax = j * xSimulation::dt * o - T;
		if (length(_Tmax) && R)
		{
			vector3d _Tr = R * (_Tmax / length(_Tmax));
			if (length(_Tr) >= length(_Tmax))
				_Tr = _Tmax;
			moment[i] += _Tr;
		}
	}
	// 		foreach(xPairData* d, pairs->PlanePair())
	// 		{
	// 
	// 		}
	// 		
	// 		foreach(xContact* c, cots)
	// 		{
	// // 			if (c->IgnoreTime() && (simulation::ctime > c->IgnoreTime()))
	// // 				continue;
	// 			if (c->IsEnabled())
	// 				c->collision(r, m, p, v, o, F, M);
	// 		}
	// 		vector3i gp = xGridCell::getCellNumber(p.x, p.y, p.z);
	// 		for (int z = -1; z <= 1; z++){
	// 			for (int y = -1; y <= 1; y++){
	// 				for (int x = -1; x <= 1; x++){
	// 					vector3i neigh = new_vector3i(gp.x + x, gp.y + y, gp.z + z);
	// 					unsigned int hash = xGridCell::getHash(neigh);
	// 					unsigned int sid = cell_start[hash];
	// 					if (sid != 0xffffffff){
	// 						unsigned int eid = cell_end[hash];
	// 						for (unsigned int j = sid; j < eid; j++){
	// 							unsigned int k = sorted_id[j];
	// 							if (i != k && k < np)
	// 							{
	// 								vector3d jp = new_vector3d(pos[k].x, pos[k].y, pos[k].z);
	// 								vector3d jv = vel[k];
	// 								vector3d jo = omega[k];
	// 								double jr = pos[k].w;
	// 								double jm = mass[k];
	// 								cpp->cppCollision(
	// 									r, jr,
	// 									m, jm,
	// 									p, jp,
	// 									v, jv,
	// 									o, jo,
	// 									F, M);
	// 							}
	// 							else if (k >= np)
	// 							{
	// // 								if (!cppoly->cppolyCollision(k - np, r, m, p, v, o, F, M))
	// // 								{
	// // 
	// // 								}
	// 							}
	// 						}
	// 					}
	// 				}
	// 			}
	// 		}
	// 		force[i] += F;
	// 		moment[i] += M;
	// 	}
}

unsigned int xContactManager::deviceContactCount(
	double *pos, double *vel, double *omega,
	double *mass, double *force, double *moment,
	unsigned int *sorted_id, unsigned int *cell_start,
	unsigned int *cell_end, unsigned int np)
{
	// 	cu_calculate_particle_particle_contact_count(
	// 		pos,
	// 		d_pppd,
	// 		d_old_pair_count,
	// 		d_pair_count,
	// 		d_pair_start,
	// 		sorted_id,
	// 		cell_start,
	// 		cell_end,
	// 		np);
	// 	if (cpmeshes && cpmeshes->NumContactObjects())
	// 	{
	// 		cu_calculate_particle_triangle_contact_count(
	// 			cpmeshes->deviceTrianglesInfo(),
	// 			pos,
	// 			d_pppd,
	// 			d_old_pair_count,
	// 			d_pair_count,
	// 			d_pair_start,
	// 			sorted_id,
	// 			cell_start,
	// 			cell_end,
	// 			np);
	// 	}
	// 	
	// 	if (cpplane && cpplane->NumPlanes())
	// 	{
	// 		cu_calculate_particle_plane_contact_count(
	// 			cpplane->devicePlaneInfo(),
	// 			d_pppd,
	// 			d_old_pair_count,
	// 			d_pair_count,
	// 			d_pair_start,
	// 			pos,
	// 			cpplane->NumPlanes(),
	// 			np);
	// 	}
	// 	std::cout << "before cu_sumation_contact_count " << std::endl;
	return 0; //cu_sumation_contact_count(d_pair_count, np);
}
