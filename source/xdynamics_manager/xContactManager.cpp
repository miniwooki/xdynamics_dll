#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_object/xObject.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_algebra/xGridCell.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <list>
#include <sstream>
#include "xdynamics_manager/xDynamicsManager.h"
//#include "xdynamics_parallel/xParallelDEM_decl.cuh"
//#include <QtCore/QDebug>

xContactManager::xContactManager()
	: cpp(NULL)
	, ncontact(0)
	, ncobject(0)
	, d_Tmax(NULL)
	, d_RRes(NULL)
	, xcpl(NULL)
	, Tmax(NULL)
	, RRes(NULL)

{

}

xContactManager::~xContactManager()
{
	if (cots.size()) cots.delete_all();

	if (Tmax) delete[] Tmax; Tmax = NULL;
	if (RRes) delete[] RRes; RRes = NULL;
	if (xcpl) delete[] xcpl; xcpl = NULL;

	if (d_Tmax) checkCudaErrors(cudaFree(d_Tmax)); d_Tmax = NULL;
	if (d_RRes) checkCudaErrors(cudaFree(d_RRes)); d_RRes = NULL;
}

void xContactManager::CreateContactPair(
	std::string n, xContactForceModelType method, xObject* o1, xObject* o2, xContactParameterData& d)
{
	xContact *ct = NULL;
	if (xContact::ContactForceModel() == NO_DEFINE_CONTACT_MODEL)
		xContact::setContactForceModel(method);
	else
	{
		if (xContact::ContactForceModel() != method)
		{
			throw runtime_error("Contact force model of " + n + " is not matched. " + ForceModelString(xContact::ContactForceModel()) + "!=" + ForceModelString(method));
		}		
	}
	xContactPairType pt = getContactPair(o1->Shape(), o2->Shape());
	xContact *_c = nullptr;
	switch (pt)
	{
		case PARTICLE_PARTICLE:	
			cpp = new xParticleParticleContact(n, o1, o2); 
			cpp->setContactParameters(d);
			cots.insert(n, cpp);
			break;
		case PARTICLE_CUBE:	
		{
			xCubeObject* cube = dynamic_cast<xCubeObject*>(o1->Shape() == CUBE_SHAPE ? o1 : o2);
			xParticleObject* particles = dynamic_cast<xParticleObject*>(o1->Shape() == PARTICLES ? o1 : o2);
			for (unsigned int i = 0; i < 6; i++)
			{
				std::stringstream ss;
				ss << n << i;
				xPlaneObject* plane = cube->Planes() + i;
				xParticlePlaneContact *c = new xParticlePlaneContact(ss.str(), particles, plane);
				c->setContactParameters(d);
				cpplanes.insert(cpplanes.size(), c);
				cots.insert(ss.str(), c);
			}
			break;
		}			
		case PARTICLE_PANE:	
		{
			xParticlePlaneContact *c = new xParticlePlaneContact(n, o1, o2); _c = c;
			c->setContactParameters(d);
			cpplanes.insert(cpplanes.size(), c);
			cots.insert(n, c);
			break;
		}		
		case PARTICLE_CYLINDER:
		{
			xParticleCylinderContact *c = new xParticleCylinderContact(n, o1, o2);
			c->setContactParameters(d);
			cpcylinders.insert(cpcylinders.size(), c);
			cots.insert(n, c);
			break;
		}
		case PARTICLE_MESH_SHAPE: 
		{
			xParticleMeshObjectContact* c = new xParticleMeshObjectContact(n, o1, o2); _c = c;
			c->setContactParameters(d);
			cpmeshes.insert(cpmeshes.size(), c);
			cots.insert(n, c);
			break;// cpmesh.insert(c->Name(), dynamic_cast<xParticleMeshObjectContact*>(c)); break;
		}		
	}
}

void xContactManager::defineContacts(unsigned int np)
{
	xParticlePlaneContact::local_initialize();
	xParticleMeshObjectContact::local_initialize();
	xParticleCylinderContact::local_initialize();
	for (xmap<xstring, xContact*>::iterator it = cots.begin(); it != cots.end(); it.next())
	{
		switch (it.value()->PairType())
		{
		case PARTICLE_MESH_SHAPE:
			dynamic_cast<xParticleMeshObjectContact*>(it.value())->define(cpmeshes.size(), np);
			break;
		case PARTICLE_PANE:
			dynamic_cast<xParticlePlaneContact*>(it.value())->define(cpplanes.size(), np);
			break;
		case PARTICLE_CYLINDER:
			dynamic_cast<xParticleCylinderContact*>(it.value())->define(cpcylinders.size(), np);
			break;
		case PARTICLE_PARTICLE:
			dynamic_cast<xParticleParticleContact*>(it.value())->define(0, np);
			break;
		}
	}

}

//unsigned int xContactManager::setupParticlesMeshObjectsContact()
//{
//	//unsigned int n = 0;
//	//xmap<int, xContact*>::iterator it = cpmeshes.begin();
//	//while (it.has_next())
//	//{
//	//	dynamic_cast<xParticleMeshObjectContact*>(it.value())->define();
//	//}
//	///*for(xmap<int, xContact*>::iterator it = cpmeshes.begin(); it !=cpmeshes.end())
//	//if (cpmesh.size() && !cpmeshes)
//	//{
//	//	cpmeshes = new xParticleMeshObjectsContact;
//	//	n = cpmeshes->define(cpmesh);
//	//}*/
//	//return xParticleMeshObjectContact::GetNumMeshSphere();
//}

//void xContactManager::setupParticlesCylindersContact()
//{
//	//unsigned int n = 0;
//	//for(xmap<xstring, xContact*>::iterator it = cots.begin(); it != cots.end(); it.next())
//	//	if (it.value()->PairType() == xContactPairType::PARTICLE_CYLINDER)
//	//		n++;
//	//if (!n) return;
//	//if (n && !cpcylinders)
//	//	cpcylinders = new xParticleCylindersContact;
//	//cpcylinders->allocHostMemory(n);
//	//unsigned int cnt = 0;
//	//for (xmap<xstring, xContact*>::iterator it = cots.begin(); it != cots.end(); it.next())
//	//{
//	//	if (it.value()->PairType() == xContactPairType::PARTICLE_CYLINDER)
//	//	{
//	//		cpcylinders->define(cnt, dynamic_cast<xParticleCylinderContact*>(it.value()));
//	//		cnt++;//	case xContactPairType::PARTICLE_CUBE: cpplane->define(n, dynamic_cast<xParticleCubeContact*>(xc)); n += 6; break;
//	//	}
//	//}
//
//}

void xContactManager::setNumClusterObject(unsigned int nc)
{
	ncobject = nc;
}

void xContactManager::setupParticlesPlanesContact()
{
	/*unsigned int n = 0;
	for (xmap<xstring, xContact*>::iterator it = cots.begin(); it != cots.end(); it.next())
	{
		switch (it.value()->PairType())
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
	for (xmap<xstring, xContact*>::iterator it = cots.begin(); it != cots.end(); it.next())
	{
		switch (it.value()->PairType())
		{
		case xContactPairType::PARTICLE_PANE: cpplane->define(n, dynamic_cast<xParticlePlaneContact*>(it.value())); n++; break;
		case xContactPairType::PARTICLE_CUBE: cpplane->define(n, dynamic_cast<xParticleCubeContact*>(it.value())); n += 6; break;
		}
	}*/
}

xmap<int, xParticleMeshObjectContact*>& xContactManager::PMContacts()
{
	return cpmeshes;
	// TODO: 여기에 반환 구문을 삽입합니다.
}

xmap<int, xParticlePlaneContact*>& xContactManager::PPLContacts()
{
	return cpplanes;
}

xmap<int, xParticleCylinderContact*>& xContactManager::PCYLContacts()
{
	return cpcylinders;// TODO: 여기에 반환 구문을 삽입합니다.
}

void xContactManager::insertContact(xContact* c)
{
	//QString qname = c->Name();
	if (c->PairType() == xContactPairType::PARTICLE_MESH_SHAPE)
	{
	}
	//	cppos[c->Name()] = dynamic_cast<contact_particles_polygonObject*>(c);
	else
		cots.insert(c->Name(), c);
}

xContact* xContactManager::Contact(std::string n)
{
	xstring xn = n;
	//QStringList keys = cots.keys();
//	QStringList::const_iterator it = qFind(keys, n);
	xmap<xstring, xContact*>::iterator it = cots.find(xn);
	if (it == cots.end() || !cots.size())
		return NULL;
	return it.value();// cots[n];
}

xmap<xstring, xContact*>& xContactManager::Contacts()
{
	return cots;
}

xParticleParticleContact* xContactManager::ContactParticles()
{
	return cpp;
}

//xParticleMeshObjectsContact* xContactManager::ContactParticlesMeshObjects()
//{
////	return cpmeshes;
//}
//
//xParticlePlanesContact* xContactManager::ContactParticlesPlanes()
//{
//	//return cpplane;
//}
//
//xParticleCylindersContact* xContactManager::ContactParticlesCylinders()
//{
//	//return cpcylinders;
//}

bool xContactManager::runCollision(
	double *pos, double* cpos, double* ep, double *vel, double *ev,
	double *mass, double* inertia, double *force, double *moment,
	unsigned int *sorted_id, unsigned int *cell_start,
	unsigned int *cell_end,
	xClusterInformation* xci,
	unsigned int np)
{
	if (xSimulation::Cpu())
	{
		hostCollision
		(
			(vector4d*)pos,
			(vector4d*)cpos,
			(vector3d*)vel,
			(euler_parameters*)ep,
			(euler_parameters*)ev,
			mass,
			inertia,
			(vector3d*)force,
			(vector3d*)moment,
			sorted_id, cell_start, cell_end,
			xci,
			np
		);
	}
	else if (xSimulation::Gpu())
	{
		deviceCollision(
			pos, cpos, ep, vel, ev,
			mass, inertia, force, moment,
			sorted_id, cell_start, cell_end, xci, np
		);
	}
	return true;
}

void xContactManager::update()
{
	for (xmap<xstring, xContact*>::iterator it = cots.begin(); it != cots.end(); it.next())
	{
		it.value()->update();
	}
		/*if (cpmeshes)
	{
		cpmeshes->updateMeshObjectData();
	}
	if (cpplane)
	{
		cpplane->updataPlaneObjectData();
	}
	if (cpcylinders)
	{
		cpcylinders->updateCylinderObjectData();
	}*/
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
		checkXerror(cudaMalloc((void**)&d_Tmax, sizeof(double3) * np));
		checkXerror(cudaMalloc((void**)&d_RRes, sizeof(double) * np));
		checkXerror(cudaMemset(d_Tmax, 0, sizeof(double3) * np));
		checkXerror(cudaMemset(d_RRes, 0, sizeof(double) * np));
	}
}

void xContactManager::SaveStepResult(unsigned int pt, unsigned int np)
{
	/*unsigned int *count = NULL, *id = NULL;
	double2 *tsd = NULL;*/
	if (xSimulation::Gpu())
	{
		if (cpp)
		{
			cpp->savePartData(np);
		}			
		if (cpplanes.size())
		{
			xParticlePlaneContact::savePartData(np);
		}			
		if (cpcylinders.size())
		{
			xParticleCylinderContact::savePartData(np);
		}			
		if (cpmeshes.size())
		{
			xParticleMeshObjectContact::savePartData(np);
		}			
	}
}

void xContactManager::set_from_part_result(std::fstream & fs)
{
	if (cpp)
	{

	}
}

void xContactManager::updateCollisionPair(
	vector4d* pos, 
	unsigned int* sorted_id,
	unsigned int* cell_start,
	unsigned int* cell_end,
	xClusterInformation* xci,
	unsigned int np)
{

	for (unsigned int i = 0; i < np; i++)
	{
		unsigned int count = 0;
		unsigned int neach = 1;
		unsigned int kcount = 0;
		if (ncobject)
			for (unsigned int j = 0; j < ncobject; j++)
				if (i >= xci[j].sid && i < xci[j].sid + xci[j].count * xci[j].neach)
					neach = xci[j].neach;
		vector3d p = new_vector3d(pos[i].x, pos[i].y, pos[i].z);
		double r = pos[i].w;
		for (xmap<int, xParticlePlaneContact*>::iterator it = cpplanes.begin(); it != cpplanes.end(); it.next())
			it.value()->updateCollisionPair(i, r, p);
		for(xmap<int, xParticleCylinderContact*>::iterator it = cpcylinders.begin(); it != cpcylinders.end(); it.next())
			it.value()->updateCollisionPair(i, r, p);
		vector3i gp = xGridCell::getCellNumber(p.x, p.y, p.z);
		unsigned int old_id = 0;
		vector3d old_cpt = new_vector3d(FLT_MAX, FLT_MAX, FLT_MAX);
		vector3d old_unit = new_vector3d(FLT_MAX, FLT_MAX, FLT_MAX);
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
							unsigned int di = neach == 1 ? 2 : (i >= k ? i - k : k - i);
							if (i != k && k < np && !(di <= neach))
							{
								if (cpp)
								{
									vector3d jp = new_vector3d(pos[k].x, pos[k].y, pos[k].z);
									double jr = pos[k].w;
									unsigned int ck = k / neach;
									cpp->updateCollisionPair(i, k, ncobject ? 1 : 0, r, jr, p, jp);
								}
							}
							else if (k >= np)
							{
								for (xmap<int, xParticleMeshObjectContact*>::iterator it = cpmeshes.begin(); it != cpmeshes.end(); it.next())
								{
									if(it.value()->check_this_mesh(k-np))
										it.value()->updateCollisionPair(k - np, r, p, old_id, old_cpt, old_unit, ctype);
								}								
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
	double *pos, double* cpos, double* ep, double *vel, double *ev,
	double *mass, double* inertia, double *force, double *moment,
	unsigned int *sorted_id, unsigned int *cell_start,
	unsigned int *cell_end, xClusterInformation* xci, unsigned int np)
{
	xmap<xstring, xContact*>::iterator it = cots.begin();
	while (it.has_next())
	{
		xContact* cot = it.value();
		cot->collision_gpu(pos, cpos, xci, ep, vel, ev, mass, inertia, force, moment, d_Tmax, d_RRes, sorted_id, cell_start, cell_end, np);
		it.next();
	}
	cu_decide_rolling_friction_moment(d_Tmax, d_RRes, inertia, ep, ev, moment, np);

	if (xci)
	{
		//if (cpp)
		//{
		//	cu_clusters_contact(pos, cpos, ep, vel, ev, force, moment, mass,
		//		d_Tmax, d_RRes, d_pair_count_pp, d_pair_id_pp, d_tsd_pp, sorted_id,
		//		cell_start, cell_end, cpp->DeviceContactProperty(), xci, np);
		//}
		//if (cpplane)
		//{
		//	if (cpplane->NumContact())
		//	{
		//		cu_cluster_plane_contact(
		//			cpplane->devicePlaneInfo(),
		//			cpplane->devicePlaneBodyInfo(),
		//			cpplane->deviceBodyForceAndMoment(),
		//			cpplane->DeviceContactProperty(),
		//			pos, cpos, ep, vel,
		//			ev, force, moment, mass,
		//			d_Tmax, d_RRes, d_pair_count_ppl, d_pair_id_ppl, d_tsd_ppl, xci,
		//			np, cpplane->NumPlanes());
		//		cpplane->getPlaneContactForce();
		//	}
		//}
		//if (cpmeshes)
		//{
		//	if (cpmeshes->NumContactObjects())
		//	{
		//		////cpmeshes->updateMeshMassData();
		//		//unsigned int bPolySphere = 0;
		//		//unsigned int ePolySphere = 0;
		//		//unsigned int cnt = 0;
		//		//device_body_force *dbfm = cpmeshes->deviceBodyForceAndMoment();
		//		//memset(dbfm, 0, sizeof(device_body_force) * cpmeshes->NumContactObjects());
		//		//for (xmap<xstring, xParticleMeshObjectContact*>::iterator it = cpmesh.begin(); it != cpmesh.end(); it.next())// (xParticleMeshObjectContact* cpm, cpmesh)
		//		//{
		//		//	xParticleMeshObjectContact* cpm = it.value();
		//		//	ePolySphere = cpm->MeshObject()->NumTriangle();
		//		//	cu_cluster_meshes_contact(
		//		//		cpmeshes->deviceTrianglesInfo(),
		//		//		cpmeshes->devicePolygonObjectMassInfo(),
		//		//		pos, cpos, ep, vel, ev, force, moment, cpmeshes->DeviceContactProperty(), mass,
		//		//		d_Tmax, d_RRes, d_pair_count_ptri, d_pair_id_ptri, d_tsd_ptri,
		//		//		sorted_id, cell_start, cell_end, xci, bPolySphere, ePolySphere, np);
		//		//	bPolySphere += cpm->MeshObject()->NumTriangle();
		//		//	dbfm[cnt].force.x += reduction(xContact::deviceBodyForceX(), np);
		//		//	dbfm[cnt].force.y += reduction(xContact::deviceBodyForceY(), np);
		//		//	dbfm[cnt].force.z += reduction(xContact::deviceBodyForceZ(), np);
		//		//	dbfm[cnt].moment.x += reduction(xContact::deviceBodyMomentX(), np);
		//		//	dbfm[cnt].moment.y += reduction(xContact::deviceBodyMomentY(), np);
		//		//	dbfm[cnt].moment.z += reduction(xContact::deviceBodyMomentZ(), np);
		//		//	//dbfm[cnt].force = reductionD3(xContact::deviceBodyForce(), np);
		//		//	//dbfm[cnt].moment = reductionD3(xContact::deviceBodyMoment(), np);
		//		//	cnt++;
		//		//}

		//		//cpmeshes->getMeshContactForce();
		//	}
		//}
		//if (cpcylinders)
		//{
		//	cu_cluster_cylinder_contact(
		//		cpcylinders->deviceCylinderInfo(),
		//		cpcylinders->deviceCylinderBodyInfo(),
		//		cpcylinders->deviceCylinderBodyForceAndMoment(), xci,
		//		cpcylinders->DeviceContactProperty(),
		//		pos, cpos, ep, vel, ev, force, moment, mass,
		//		d_Tmax, d_RRes, d_pair_count_pcyl, d_pair_id_pcyl, d_tsd_pcyl,
		//		np, cpcylinders->NumContact());
		//	cpcylinders->getCylinderContactForce();
		//}
		//cu_decide_cluster_rolling_friction_moment(d_Tmax, d_RRes, inertia, ep, ev, moment, xci, np);
	}
	else
	{
		//if (cpp)
		//{
		//	cu_calculate_p2p(pos, ep, vel, ev, force, moment, mass,
		//		d_Tmax, d_RRes, d_pair_count_pp, d_pair_id_pp, d_tsd_pp, sorted_id,
		//		cell_start, cell_end, cpp->DeviceContactProperty(), np);
		//}
		//if (cpplane)
		//{
		//	if (cpplane->NumContact())
		//	{
		//		cu_plane_contact_force(cpplane->devicePlaneInfo(),
		//			cpplane->devicePlaneBodyInfo(), cpplane->DeviceContactProperty(),
		//			cpplane->deviceBodyForceAndMoment(),
		//			pos, ep, vel,
		//			ev, force, moment, mass,
		//			d_Tmax, d_RRes, d_pair_count_ppl, d_pair_id_ppl, d_tsd_ppl, np, cpplane->NumPlanes());
		//		cpplane->getPlaneContactForce();
		//	}
		//}
		//if (cpcylinders)
		//{
		//	if (cpcylinders->NumContact())
		//	{

		//		device_body_force* dbfm = cpcylinders->deviceCylinderBodyForceAndMoment();
		//		memset(dbfm, 0, sizeof(device_body_force) * cpcylinders->NumContact());
		//		for (unsigned int i = 0 ; i < cpcylinders->NumContact(); i++)
		//		{
		//			cu_cylinder_contact_force(i, cpcylinders->deviceCylinderInfo(),
		//				cpcylinders->deviceCylinderBodyInfo(),
		//				cpcylinders->DeviceContactProperty(),
		//				pos, ep, vel, ev, force, moment, mass,
		//				d_Tmax, d_RRes, d_pair_count_pcyl, d_pair_id_pcyl, d_tsd_pcyl, np, cpcylinders->NumContact());
		//			dbfm[i].force.x += reduction(xContact::deviceBodyForceX(), np);
		//			dbfm[i].force.y += reduction(xContact::deviceBodyForceY(), np);
		//			dbfm[i].force.z += reduction(xContact::deviceBodyForceZ(), np);
		//			dbfm[i].moment.x += reduction(xContact::deviceBodyMomentX(), np);
		//			dbfm[i].moment.y += reduction(xContact::deviceBodyMomentY(), np);
		//			dbfm[i].moment.z += reduction(xContact::deviceBodyMomentZ(), np);
		//			/*dbfm[i].force = dbfm[i].force + reductionD3(xContact::deviceBodyForce(), np);
		//			dbfm[i].moment = dbfm[i].moment + reductionD3(xContact::deviceBodyMoment(), n*///p);
		//		}
		//		
		//		cpcylinders->getCylinderContactForce();
		//	}			
		//}
		//if (cpmeshes)
		//{


		//	if (cpmeshes->NumContactObjects())
		//	{
		//		////cpmeshes->updateMeshMassData();
		//		//unsigned int bPolySphere = 0;
		//		//unsigned int ePolySphere = 0;
		//		//unsigned int cnt = 0;
		//		//device_body_force *dbfm = cpmeshes->deviceBodyForceAndMoment();
		//		//memset(dbfm, 0, sizeof(device_body_force) * cpmeshes->NumContactObjects());
		//		//for (xmap<xstring, xParticleMeshObjectContact*>::iterator it = cpmesh.begin(); it != cpmesh.end(); it.next())
		//		//{
		//		//	xParticleMeshObjectContact* cpm = it.value();
		//		//	ePolySphere = cpm->MeshObject()->NumTriangle();
		//		//	cu_particle_polygonObject_collision(
		//		//		cpmeshes->deviceTrianglesInfo(),
		//		//		cpmeshes->devicePolygonObjectMassInfo(), 
		//		//		pos, ep, vel, ev, force, moment, mass,
		//		//		d_Tmax, d_RRes, d_pair_count_ptri, d_pair_id_ptri, d_tsd_ptri, 
		//		//		cpmeshes->SphereData(),
		//		//		sorted_id, cell_start, cell_end, 
		//		//		cpmeshes->DeviceContactProperty(),
		//		//		np,
		//		//		bPolySphere, ePolySphere);
		//		//	bPolySphere += cpm->MeshObject()->NumTriangle();
		//		//	dbfm[cnt].force.x += reduction(xContact::deviceBodyForceX(), np);
		//		//	dbfm[cnt].force.y += reduction(xContact::deviceBodyForceY(), np);
		//		//	dbfm[cnt].force.z += reduction(xContact::deviceBodyForceZ(), np);
		//		//	dbfm[cnt].moment.x += reduction(xContact::deviceBodyMomentX(), np);
		//		//	dbfm[cnt].moment.y += reduction(xContact::deviceBodyMomentY(), np);
		//		//	dbfm[cnt].moment.z += reduction(xContact::deviceBodyMomentZ(), np);
		//		//	/*dbfm[cnt].force = reductionD3(xContact::deviceBodyForce(), np);
		//		//	dbfm[cnt].moment = reductionD3(xContact::deviceBodyMoment(), np);*/
		//		//	cnt++;
		//	//	}
		//		//cpmeshes->getMeshContactForce();
		//	}
		//}
		
	}
}

void xContactManager::hostCollision(
	vector4d *pos,
	vector4d *cpos,
	vector3d *vel,
	euler_parameters *ep,
	euler_parameters *ev,
	double *mass,
	double* inertia,
	vector3d *force,
	vector3d *moment,
	unsigned int *sorted_id,
	unsigned int *cell_start,
	unsigned int *cell_end,
	xClusterInformation* xci,
	unsigned int np)
{

	updateCollisionPair(pos, sorted_id, cell_start, cell_end, xci, np);
	for (unsigned int i = 0; i < np; i++)
	{
		unsigned int neach = 0;
		unsigned int ci = i;
		if (ncobject)
		{
			for (unsigned int j = 0; j < ncobject; j++)
			{
				if (i >= xci[j].sid && i < xci[j].sid + xci[j].count * xci[j].neach)
				{
					neach = xci[j].neach;
					ci = i / neach;
				}
			}
		}
		vector3d F = new_vector3d(0.0, 0.0, 0.0);
		vector3d M = new_vector3d(0.0, 0.0, 0.0);
		double R = 0;
		vector3d T = new_vector3d(0.0, 0.0, 0.0);
		vector3d o = ToAngularVelocity(ep[ci], ev[ci]);
		//xContactPairList* pairs = &xcpl[i];
		
		for (xmap<xstring, xContact*>::iterator it = cots.begin(); it != cots.end(); it.next())
		{
			it.value()->collision_cpu(pos, ep, vel, ev, mass, R, T, F, M, ncobject, xci, cpos);
			//it.value()->collision();
		}
		/*if (cpplane)
			cpplane->cpplCollision(pairs, i, r, m, p, v, o, R, T, F, M, ncobject, xci, cpos);
		if (cpp)
			cpp->cppCollision(pairs, i, pos, cpos, vel, ep, ev, mass, R, T, F, M, xci, ncobject);
		if (cpmeshes)
			cpmeshes->cppolyCollision(pairs, i, r, m, p, v, o, R, T, F, M, ncobject, xci, cpos);
		if (cpcylinders)
			cpcylinders->pcylCollision(pairs, i, r, m, p, v, o, R, T, F, M, ncobject, xci, cpos);*/

		force[i] += F;
		moment[i] += M;
		double j = inertia[ci];
		vector3d _Tmax = j * xSimulation::dt * o - T;
		if (length(_Tmax) && R)
		{
			vector3d _Tr = R * (_Tmax / length(_Tmax));
			if (length(_Tr) >= length(_Tmax))
				_Tr = _Tmax;
			moment[i] += _Tr;
		}
	}
}