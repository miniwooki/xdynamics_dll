#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_object/xObject.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_algebra/xGridCell.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <list>
#include <sstream>
#include "xdynamics_manager/xDynamicsManager.h"
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
	if (cots.size()) cots.delete_all();//qDeleteAll(cots);
// 	if (cpp) delete cpp; cpp = NULL;
 	//if (cpmeshes) delete cpmeshes; cpmeshes = NULL;
 	//if (cpplane) delete cpplane; cpplane = NULL;
	if (Tmax) delete[] Tmax; Tmax = NULL;
	if (RRes) delete[] RRes; RRes = NULL;
	if (xcpl) delete[] xcpl; xcpl = NULL;
	//if (cpcylinders) delete cpcylinders; cpcylinders = NULL;
	//if (cpcylinders.size())
	//{
	//	qDeleteAll(cpcylinders);
	//}

//	if (d_pair_count_pp) checkCudaErrors(cudaFree(d_pair_count_pp)); d_pair_count_pp = NULL;
//	//if (d_pair_count_ppl) checkCudaErrors(cudaFree(d_pair_count_ppl)); d_pair_count_ppl = NULL;
//	//if (d_pair_count_ptri) checkCudaErrors(cudaFree(d_pair_count_ptri)); d_pair_count_ptri = NULL;
//	if (d_pair_count_pcyl) checkCudaErrors(cudaFree(d_pair_count_pcyl)); d_pair_count_pcyl = NULL;
//	if (d_pair_id_pp) checkCudaErrors(cudaFree(d_pair_id_pp)); d_pair_id_pp = NULL;
//	//if (d_pair_id_ppl) checkCudaErrors(cudaFree(d_pair_id_ppl)); d_pair_id_ppl = NULL;
//	//if (d_pair_id_ptri) checkCudaErrors(cudaFree(d_pair_id_ptri)); d_pair_id_ptri = NULL;
//	if (d_pair_id_pcyl) checkCudaErrors(cudaFree(d_pair_id_pcyl)); d_pair_id_pcyl = NULL;
//	if (d_tsd_pp) checkCudaErrors(cudaFree(d_tsd_pp)); d_tsd_pp = NULL;
//	//if (d_tsd_ppl) checkCudaErrors(cudaFree(d_tsd_ppl)); d_tsd_ppl = NULL;
//	//if (d_tsd_ptri) checkCudaErrors(cudaFree(d_tsd_ptri)); d_tsd_ptri = NULL;
//	if (d_tsd_pcyl) checkCudaErrors(cudaFree(d_tsd_pcyl)); d_tsd_pcyl = NULL;
//	if (d_Tmax) checkCudaErrors(cudaFree(d_Tmax)); d_Tmax = NULL;
//	if (d_RRes) checkCudaErrors(cudaFree(d_RRes)); d_RRes = NULL;
//
//	if (pair_count_pp) delete[] pair_count_pp; pair_count_pp = NULL;
//	//if (pair_count_ppl) delete[] pair_count_ppl; pair_count_ppl = NULL;
//	//if (pair_count_ptri) delete[] pair_count_ptri; pair_count_ptri = NULL;
//	if (pair_count_pcyl) delete[] pair_count_pcyl; pair_count_pcyl = NULL;
//	if (pair_id_pp) delete[] pair_id_pp; pair_id_pp = NULL;
////	if (pair_id_ppl) delete[] pair_id_ppl; pair_id_ppl = NULL;
//	//if (pair_id_ptri) delete[] pair_id_ptri; pair_id_ptri = NULL;
//	if (pair_id_pcyl) delete[] pair_id_pcyl; pair_id_pcyl = NULL;
//	if (tsd_pp) delete[] tsd_pp; tsd_pp = NULL;
////	if (tsd_ppl) delete[] tsd_ppl; tsd_ppl = NULL;
//	///if (tsd_ptri) delete[] tsd_ptri; tsd_ptri = NULL;
//	if (tsd_pcyl) delete[] tsd_pcyl; tsd_pcyl = NULL;
	if (Tmax) delete[] Tmax; Tmax = NULL;
	if (RRes) delete[] RRes; RRes = NULL;
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

	switch (pt)
	{
	case PARTICLE_PARTICLE:	cpp = new xParticleParticleContact(n); break;
	case PARTICLE_CUBE:	
	{
		xCubeObject* cube = dynamic_cast<xCubeObject*>(o1->Shape() == CUBE_SHAPE ? o1 : o2);
		xParticleObject* particles = dynamic_cast<xParticleObject*>(o1->Shape() == PARTICLES ? o1 : o2);
		for (unsigned int i = 0; i < 6; i++)
		{
			std::stringstream ss;
			ss << n << i;
			xPlaneObject* plane = cube->Planes() + i;
			xParticlePlaneContact *c = new xParticlePlaneContact(ss.str(), o1, o2); break;
			cpplanes.insert(cpplanes.size(), c);
			cots.insert(ss.str(), c);
		}
		break;
	}			
	case PARTICLE_PANE:	
	{
		xParticlePlaneContact *c = new xParticlePlaneContact(n, o1, o2);
		cpplanes.insert(cpplanes.size(), c);
		cots.insert(n, c);
		break;
	}		
	case PARTICLE_MESH_SHAPE: 
	{
		xParticleMeshObjectContact* c = new xParticleMeshObjectContact(n, o1, o2);
		cpmeshes.insert(cpmeshes.size(), c);
		cots.insert(n, c);
		break;// cpmesh.insert(c->Name(), dynamic_cast<xParticleMeshObjectContact*>(c)); break;
	}		
	//case PARTICLE_CYLINDER: c = new xParticleCylinderContact(n, o1, o2); break;
	}
	//xMaterialPair mpp =
	//{
	//	o1->Youngs(), o2->Youngs(),
	//	o1->Poisson(), o2->Poisson(),
	//	o1->Shear(), o2->Shear()
	//};
	
	//c->setContactForceModel(method);
	//c->setFirstObject(o1);
	//c->setSecondObject(o2);
	//c->setMaterialPair(mpp);
	//c->setRestitution(d.rest);
	//c->setStiffnessRatio(d.rto);
	//c->setStaticFriction(d.mu_s);
	//c->setFriction(d.mu);
	//c->setCohesion(d.coh);
	//c->setRollingFactor(d.rf);
	//xstring name = n;
	//cots.insert(name, c);
	//c->setContactParameters(rest, ratio, fric, cohesion);
	//return c;
}

void xContactManager::defineContacts(unsigned int np)
{
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
	}

}

unsigned int xContactManager::setupParticlesMeshObjectsContact()
{
	//unsigned int n = 0;
	//xmap<int, xContact*>::iterator it = cpmeshes.begin();
	//while (it.has_next())
	//{
	//	dynamic_cast<xParticleMeshObjectContact*>(it.value())->define();
	//}
	///*for(xmap<int, xContact*>::iterator it = cpmeshes.begin(); it !=cpmeshes.end())
	//if (cpmesh.size() && !cpmeshes)
	//{
	//	cpmeshes = new xParticleMeshObjectsContact;
	//	n = cpmeshes->define(cpmesh);
	//}*/
	//return xParticleMeshObjectContact::GetNumMeshSphere();
}

void xContactManager::setupParticlesCylindersContact()
{
	//unsigned int n = 0;
	//for(xmap<xstring, xContact*>::iterator it = cots.begin(); it != cots.end(); it.next())
	//	if (it.value()->PairType() == xContactPairType::PARTICLE_CYLINDER)
	//		n++;
	//if (!n) return;
	//if (n && !cpcylinders)
	//	cpcylinders = new xParticleCylindersContact;
	//cpcylinders->allocHostMemory(n);
	//unsigned int cnt = 0;
	//for (xmap<xstring, xContact*>::iterator it = cots.begin(); it != cots.end(); it.next())
	//{
	//	if (it.value()->PairType() == xContactPairType::PARTICLE_CYLINDER)
	//	{
	//		cpcylinders->define(cnt, dynamic_cast<xParticleCylinderContact*>(it.value()));
	//		cnt++;//	case xContactPairType::PARTICLE_CUBE: cpplane->define(n, dynamic_cast<xParticleCubeContact*>(xc)); n += 6; break;
	//	}
	//}

}

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
		//if (cpp)
		//{
		//	/*pair_count_pp = new unsigned int[np];
		//	pair_id_pp = new unsigned int[np * MAX_P2P_COUNT];
		//	tsd_pp = new double[2 * np * MAX_P2P_COUNT];
		//	checkXerror(cudaMalloc((void**)&d_pair_count_pp, sizeof(unsigned int) * np));
		//	checkXerror(cudaMalloc((void**)&d_pair_id_pp, sizeof(unsigned int) * np * MAX_P2P_COUNT));
		//	checkXerror(cudaMalloc((void**)&d_tsd_pp, sizeof(double2) * np * MAX_P2P_COUNT));
		//	checkXerror(cudaMemset(d_pair_count_pp, 0, sizeof(unsigned int) * np));
		//	checkXerror(cudaMemset(d_pair_id_pp, 0, sizeof(unsigned int) * np * MAX_P2P_COUNT));
		//	checkXerror(cudaMemset(d_tsd_pp, 0, sizeof(double2) * np * MAX_P2P_COUNT));
		//	xDynamicsManager::This()->XResult()->set_p2p_contact_data((int)MAX_P2P_COUNT);*/
		//}
		//if (cpplane)
		//{
		//	pair_count_ppl = new unsigned int[np];
		//	pair_id_ppl = new unsigned int[np * MAX_P2PL_COUNT];
		//	tsd_ppl = new double[2 * np * MAX_P2PL_COUNT];
		//	checkXerror(cudaMalloc((void**)&d_pair_count_ppl, sizeof(unsigned int) * np));
		//	checkXerror(cudaMalloc((void**)&d_pair_id_ppl, sizeof(unsigned int) * np * MAX_P2PL_COUNT));
		//	checkXerror(cudaMalloc((void**)&d_tsd_ppl, sizeof(double2) * np * MAX_P2PL_COUNT));
		//	checkXerror(cudaMemset(d_pair_count_ppl, 0, sizeof(unsigned int) * np));
		//	checkXerror(cudaMemset(d_pair_id_ppl, 0, sizeof(unsigned int) * np * MAX_P2PL_COUNT));
		//	checkXerror(cudaMemset(d_tsd_ppl, 0, sizeof(double2) * np * MAX_P2PL_COUNT));
		//	xDynamicsManager::This()->XResult()->set_p2pl_contact_data((int)MAX_P2PL_COUNT);
		//}
		//if (cpcylinders)
		//{
		//	pair_count_pcyl = new unsigned int[np];
		//	pair_id_pcyl = new unsigned int[np * MAX_P2CY_COUNT];
		//	tsd_pcyl = new double[2 * np * MAX_P2CY_COUNT];
		//	checkXerror(cudaMalloc((void**)&d_pair_count_pcyl, sizeof(unsigned int) * np));
		//	checkXerror(cudaMalloc((void**)&d_pair_id_pcyl, sizeof(unsigned int) * np * MAX_P2CY_COUNT));
		//	checkXerror(cudaMalloc((void**)&d_tsd_pcyl, sizeof(double2) * np * MAX_P2CY_COUNT));
		//	checkXerror(cudaMemset(d_pair_count_pcyl, 0, sizeof(unsigned int) * np));
		//	checkXerror(cudaMemset(d_pair_id_pcyl, 0, sizeof(unsigned int) * np * MAX_P2CY_COUNT));
		//	checkXerror(cudaMemset(d_tsd_pcyl, 0, sizeof(double2) * np * MAX_P2CY_COUNT));
		//	xDynamicsManager::This()->XResult()->set_p2cyl_contact_data((int)MAX_P2CY_COUNT);
		//}
		//if (cpmeshes)
		//{
		////	pair_count_ptri = new unsigned int[np];
		//	//pair_id_ptri = new unsigned int[np * MAX_P2MS_COUNT];
		//	//tsd_ptri = new double[2 * np * MAX_P2MS_COUNT];
		//	//checkXerror(cudaMalloc((void**)&d_pair_count_ptri, sizeof(unsigned int) * np));
		//	//checkXerror(cudaMalloc((void**)&d_pair_id_ptri, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
		//	//checkXerror(cudaMalloc((void**)&d_tsd_ptri, sizeof(double2) * np * MAX_P2MS_COUNT));
		//	//checkXerror(cudaMemset(d_pair_count_ptri, 0, sizeof(unsigned int) * np));
		//	//checkXerror(cudaMemset(d_pair_id_ptri, 0, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
		//	//checkXerror(cudaMemset(d_tsd_ptri, 0, sizeof(double2) * np * MAX_P2MS_COUNT));
		////	xDynamicsManager::This()->XResult()->set_p2tri_contact_data((int)MAX_P2MS_COUNT);
		//}
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
			//checkXerror(cudaMemcpy(pair_count_pp, d_pair_count_pp, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
			//checkXerror(cudaMemcpy(pair_id_pp, d_pair_id_pp, sizeof(unsigned int) * np * MAX_P2P_COUNT, cudaMemcpyDeviceToHost));
			//checkXerror(cudaMemcpy(tsd_pp, d_tsd_pp, sizeof(double2) * np * MAX_P2P_COUNT, cudaMemcpyDeviceToHost));
			//xDynamicsManager::This()->XResult()->save_p2p_contact_data(pair_count_pp, pair_id_pp, tsd_pp);
		}			
		if (cpplanes.size())
		{
			xParticlePlaneContact::savePartData(np);
			
		}			
		if (cpcylinders.size())
		{
			xParticleCylinderContact::savePartData(np);
			/*checkXerror(cudaMemcpy(pair_count_pcyl, d_pair_count_pcyl, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
			checkXerror(cudaMemcpy(pair_id_pcyl, d_pair_id_pcyl, sizeof(unsigned int) * np * MAX_P2CY_COUNT, cudaMemcpyDeviceToHost));
			checkXerror(cudaMemcpy(tsd_pcyl, d_tsd_pcyl, sizeof(double2) * np * MAX_P2CY_COUNT, cudaMemcpyDeviceToHost));
			xDynamicsManager::This()->XResult()->save_p2cyl_contact_data(pair_count_pcyl, pair_id_pcyl, tsd_pcyl);*/
		}			
		if (cpmeshes.size())
		{
			xParticleMeshObjectContact::savePartData(np);
			/*checkXerror(cudaMemcpy(pair_count_ptri, d_pair_count_ptri, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
			checkXerror(cudaMemcpy(pair_id_ptri, d_pair_id_ptri, sizeof(unsigned int) * np * MAX_P2MS_COUNT, cudaMemcpyDeviceToHost));
			checkXerror(cudaMemcpy(tsd_ptri, d_tsd_ptri, sizeof(double2) * np * MAX_P2MS_COUNT, cudaMemcpyDeviceToHost));*/
			//xDynamicsManager::This()->XResult()->save_p2tri_contact_data(pair_count_ptri, pair_id_ptri, tsd_ptri);
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
	/*for (unsigned int i = 0; i < np; i++)
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
		if (cpplane)
			cpplane->updateCollisionPair(xcpl[i], r, p);
		if (cpcylinders)
			cpcylinders->updateCollisionPair(xcpl[i], r, p);
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
									cpp->updateCollisionPair(k, ncobject ? 1 : 0, xcpl[i], r, jr, p, jp);
								}
							}
							else if (k >= np)
							{
								if (cpmeshes->updateCollisionPair(k - np, xcpl[i], r, p, old_id, old_cpt, old_unit, ctype))
									count++;
							}
						}
					}
				}
			}
		}
		if (count > 1)
			std::cout << "mesh contact overlab occured." << std::endl;
	}*/
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
		cot->collision(
			pos, ep, vel, ev, mass, inertia, force, moment, 
			d_Tmax, d_RRes, sorted_id, cell_start, cell_end, np);
		it.next();
	}
	cu_decide_rolling_friction_moment(d_Tmax, d_RRes, inertia, ep, ev, moment, np);
	/*for (xmap<int, xContact*>::iterator it = cpplanes.begin(); it != cpplanes.end(); it.next())
	{
		it.value()->collision(it.key(), pos, ep, vel, ev, mass, inertia, force, moment, sorted_id, cell_start, cell_end, np);
	}*/


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
		xContactPairList* pairs = &xcpl[i];
		vector3d p = new_vector3d(pos[i].x, pos[i].y, pos[i].z);
		vector3d v = vel[ci];
		vector3d o = ToAngularVelocity(ep[ci], ev[ci]);

		double m = mass[ci];
		double j = inertia[ci];
		double r = pos[i].w;
		for (xmap<xstring, xContact*>::iterator it = cots.begin(); it != cots.end(); it.next());
		{
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