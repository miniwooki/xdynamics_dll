#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_object/xObject.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_algebra/xGridCell.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <list>

xContactManager::xContactManager()
	: cpp(NULL)
	, cpmeshes(NULL)
	, cpplane(NULL)

{

}

xContactManager::~xContactManager()
{
	qDeleteAll(cots);
	if (cpp) delete cpp; cpp = NULL;
	if (cpmeshes) delete cpmeshes; cpmeshes = NULL;
	if (cpplane) delete cpplane; cpplane = NULL;

	
}

xContact* xContactManager::CreateContactPair(
	std::string n, xContactForceModelType method, xObject* o1, xObject* o2, xContactParameterData& d)
{
	xContactPairType pt = getContactPair(o1->Shape(), o2->Shape());
	xContact *c = NULL;
	switch (pt)
	{
	case PARTICLE_PARTICLE:	cpp = new xParticleParticleContact(n); c = cpp;	break;
	case PARTICLE_CUBE:	c = new xParticleCubeContact(n); break;
	case PARTICLE_PANE:	c = new xParticlePlaneContact(n); break;
	case PARTICLE_MESH_SHAPE: c = new xParticleMeshObjectContact(n); cpmesh[c->Name()] = dynamic_cast<xParticleMeshObjectContact*>(c); break;
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

bool xContactManager::runCollision(double *pos, double *vel, double *omega, double *mass, double *force, double *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
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
			pos, vel, omega,
			mass, force, moment,
			sorted_id, cell_start, cell_end, np
			);
	}
	return true;
}

void xContactManager::update()
{
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

void xContactManager::deviceCollision(double *pos, double *vel, double *omega, double *mass, double *force, double *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{
	//cu_check_no_collision_pair(pos, d_pair_idx, d_pair_other, np);
// 	cu_calculate_contact_pair_count(pos, d_pair_idx, sorted_id, cell_start, cell_end, np);
// 	unsigned int npair = thrust::reduce(thrust::device, d_count, d_count + np);
// 	cu_update_contact_pair(pos, vel, omega, mass, d_start_each_pair)
	if (cpp)
	{
		cpp->cuda_collision(pos, vel, omega, mass, force, moment, sorted_id, cell_start, cell_end, np);
	}
	if (cpplane)
	{
		cpplane->cuda_collision(pos, vel, omega, mass, force, moment, sorted_id, cell_start, cell_end, np);
	}
	foreach(xContact* c, cots)
	{
// 		if (c->IgnoreTime() && (simulation::ctime > c->IgnoreTime()))
// 			continue;
		if (c->IsEnabled())
			c->cuda_collision(
			pos, vel, omega, mass, force, moment, sorted_id, cell_start, cell_end, np);
	}
// 	if (cppoly)
// 	{
// 		//qDebug() << "pass_cuda_collision_cppoly0";
// 		cppoly->cuda_collision(pos, vel, omega, mass, force, moment, sorted_id, cell_start, cell_end, np);
// 		//qDebug() << "pass_cuda_collision_cppoly1";
// 	}
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
