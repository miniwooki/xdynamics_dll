#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_object/xObject.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_algebra/xGridCell.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <list>
#include <QtCore/QDebug>

xContactManager::xContactManager()
	: cpp(NULL)
	, ncontact(0)
	, cpmeshes(NULL)
	, cpplane(NULL)
	, d_old_pair_count(NULL)
	, d_pair_count(NULL)
	, d_old_pair_start(NULL)
	, d_pair_start(NULL)
	, d_type_count(NULL)
	, d_pppd(NULL)
{

}

xContactManager::~xContactManager()
{
	qDeleteAll(cots);
	//if (cpp) delete cpp; cpp = NULL;
	if (cpmeshes) delete cpmeshes; cpmeshes = NULL;
	if (cpplane) delete cpplane; cpplane = NULL;

	if (d_pair_count) checkCudaErrors(cudaFree(d_pair_count)); d_pair_count = NULL;
	if (d_old_pair_start) checkCudaErrors(cudaFree(d_old_pair_start)); d_old_pair_start = NULL;
	if (d_pair_start) checkCudaErrors(cudaFree(d_pair_start)); d_pair_start = NULL;
	if (d_type_count) checkCudaErrors(cudaFree(d_type_count)); d_type_count = NULL;
	if (d_pppd) checkCudaErrors(cudaFree(d_pppd)); d_pppd = NULL;
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

bool xContactManager::runCollision(double *pos, double *vel, double* ep, double *omega, double *mass, double *force, double *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{
	if (xSimulation::Cpu())
	{
		hostCollision
			(
			(vector4d*)pos,
			(vector3d*)vel,
			(vector3d*)omega,
			mass,
			(vector3d*)force,
			(vector3d*)moment,
			sorted_id, cell_start, cell_end, np
			);
	}
	else if (xSimulation::Gpu())
	{
		deviceCollision(
			pos, vel, ep, omega,
			mass, force, moment,
			sorted_id, cell_start, cell_end, np
			);
	}
	return true;
}

void xContactManager::update()
{
	if (cpmeshes && ncontact)
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
	}
	else
	{
		checkCudaErrors(cudaMalloc((void**)&d_pair_count, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMalloc((void**)&d_old_pair_count, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMalloc((void**)&d_pair_start, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMalloc((void**)&d_old_pair_start, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMalloc((void**)&d_type_count, sizeof(int) * np * 2));

		checkCudaErrors(cudaMemset(d_pair_count, 0, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMemset(d_old_pair_count, 0, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMemset(d_pair_start, 0, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMemset(d_old_pair_start, 0, sizeof(unsigned int) * np));
		checkCudaErrors(cudaMemset(d_type_count, 0, sizeof(int) * np * 2));
	}
}

void xContactManager::updateCollisionPair(vector4d* pos, unsigned int* sorted_id, unsigned int* cell_start, unsigned int* cell_end, unsigned int np)
{
	for (unsigned int i = 0; i < np; i++)
	{
		vector3d p = new_vector3d(pos[i].x, pos[i].y, pos[i].z);
		double r = pos[i].w;
		cpplane->updateCollisionPair(xcpl[i], r, p);
		vector3i gp = xGridCell::getCellNumber(p.x, p.y, p.z);
		for (int z = -1; z <= 1; z++){
			for (int y = -1; y <= 1; y++){
				for (int x = -1; x <= 1; x++){
					vector3i neigh = new_vector3i(gp.x + x, gp.y + y, gp.z + z);
					unsigned int hash = xGridCell::getHash(neigh);
					unsigned int sid = cell_start[hash];
					if (sid != 0xffffffff){
						unsigned int eid = cell_end[hash];
						for (unsigned int j = sid; j < eid; j++){
							unsigned int k = sorted_id[j];
							if (i != k && k < np)
							{
								vector3d jp = new_vector3d(pos[k].x, pos[k].y, pos[k].z);
								double jr = pos[k].w;
								cpp->updateCollisionPair(k, xcpl[i], r, jr, p, jp);
							}
							else if (k >= np)
							{
								// 								if (!cppoly->cppolyCollision(k - np, r, m, p, v, o, F, M))
								// 								{
								// 
								// 								}
							}
						}
					}
				}
			}
		}
	}
}

void xContactManager::deviceCollision(
	double *pos, double *vel, double* ep, double *omega, 
	double *mass, double *force, double *moment, 
	unsigned int *sorted_id, unsigned int *cell_start, 
	unsigned int *cell_end, unsigned int np)
{
	checkCudaErrors(cudaMemcpy(d_old_pair_count, d_pair_count, sizeof(unsigned int) * np, cudaMemcpyDeviceToDevice));
	unsigned int nc = deviceContactCount(pos, vel, omega, mass, force, moment, sorted_id, cell_start, cell_end, np);
//	std::cout << "ncontact : " << nc << std::endl;
	if (!nc) return;
	checkCudaErrors(cudaMemcpy(d_old_pair_start, d_pair_start, sizeof(unsigned int) * np, cudaMemcpyDeviceToDevice));
	pair_data* d_old_pppd;
	checkCudaErrors(cudaMalloc((void**)&d_old_pppd, sizeof(pair_data) * ncontact));
	checkCudaErrors(cudaMemcpy(d_old_pppd, d_pppd, sizeof(pair_data) * ncontact, cudaMemcpyDeviceToDevice));
	if (d_pppd)
		checkCudaErrors(cudaFree(d_pppd));
	checkCudaErrors(cudaMalloc((void**)&d_pppd, sizeof(pair_data) * nc));
	cu_copy_old_to_new_pair(d_old_pair_count, d_pair_count, d_old_pair_start, d_pair_start, d_old_pppd, d_pppd, nc, np);

	cu_new_particle_particle_contact(
		pos, ep, vel, omega, mass, force, moment, 
		d_old_pppd, d_pppd, 
		d_old_pair_count, d_pair_count,
		d_old_pair_start, d_pair_start, 
		d_type_count, cpp->DeviceContactProperty(), 
		sorted_id, cell_start, cell_end, np);

	if (cpplane && cpplane->NumPlanes())
	{
		cu_new_particle_plane_contact(
			cpplane->devicePlaneInfo(), pos, vel, ep, omega,
			mass, force, moment,
			d_old_pair_count, d_pair_count,
			d_old_pair_start, d_pair_start, d_type_count,
			d_old_pppd, d_pppd, cpplane->DeviceContactProperty(),
			cpplane->NumContact(), np);
	}

	if (cpmeshes && cpmeshes->NumContactObjects())
	{
		cpmeshes->updateMeshMassData();
	//	qDebug() << "new_polygon_contact";
		cu_new_particle_polygon_object_contact(
			cpmeshes->deviceTrianglesInfo(), cpmeshes->devicePolygonObjectMassInfo(),
			d_old_pppd, d_pppd, d_old_pair_count, d_pair_count, d_old_pair_start, d_pair_start,
			d_type_count, pos, vel, ep, omega, force, moment, mass,
			sorted_id, cell_start, cell_end, cpmeshes->DeviceContactProperty(), np);
		cpmeshes->getMeshContactForce();
	}
	ncontact = nc;
	checkCudaErrors(cudaFree(d_old_pppd));
}

void xContactManager::hostCollision(vector4d *pos, vector3d *vel, vector3d *omega, double *mass, vector3d *force, vector3d *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{
	updateCollisionPair(pos, sorted_id, cell_start, cell_end, np);
	for (unsigned int i = 0; i < np; i++)
	{
		vector3d F = new_vector3d(0.0, 0.0, 0.0);
		vector3d M = new_vector3d(0.0, 0.0, 0.0);
		xContactPairList* pairs = &xcpl[i];
		vector3d p = new_vector3d(pos[i].x, pos[i].y, pos[i].z);
		vector3d v = vel[i];
		vector3d o = omega[i];
		double m = mass[i];
		double r = pos[i].w;
		cpplane->cpplCollision(pairs, r, m, p, v, o, F, M);
		cpp->cppCollision(pairs, i, pos, vel, omega, mass, F, M);
		force[i] += F;
		moment[i] += M;
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
	cu_calculate_particle_particle_contact_count(
		pos,
		d_pppd,
		d_old_pair_count,
		d_pair_count,
		d_pair_start,
		sorted_id,
		cell_start,
		cell_end,
		np);
	if (cpmeshes && cpmeshes->NumContactObjects())
	{
		cu_calculate_particle_triangle_contact_count(
			cpmeshes->deviceTrianglesInfo(),
			pos,
			d_pppd,
			d_old_pair_count,
			d_pair_count,
			d_pair_start,
			sorted_id,
			cell_start,
			cell_end,
			np);
	}
	
	if (cpplane && cpplane->NumPlanes())
	{
		cu_calculate_particle_plane_contact_count(
			cpplane->devicePlaneInfo(),
			d_pppd,
			d_old_pair_count,
			d_pair_count,
			d_pair_start,
			pos,
			cpplane->NumPlanes(),
			np);
	}

	return cu_sumation_contact_count(d_pair_count, np);
}
