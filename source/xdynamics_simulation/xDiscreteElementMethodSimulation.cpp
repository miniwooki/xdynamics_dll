#include "xdynamics_simulation/xDiscreteElementMethodSimulation.h"
#include "xdynamics_algebra/xGridCell.h"
#include "xdynamics_object/xSpringDamperForce.h"
//#include <QtCore/QFile>
#include <fstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>

xDiscreteElementMethodSimulation::xDiscreteElementMethodSimulation()
	: xSimulation()
	, xdem(NULL)
	, dtor(NULL)
	, xcm(NULL)
	, xci(NULL)
	, dxsdci(NULL)
	, dxsdc_data(NULL)
	, nco(0)
	, nTsdaConnection(0)
	, nTsdaConnectionList(0)
	, nTsdaConnectionValue(0)
	, isSaveMemory(false)
{

}

xDiscreteElementMethodSimulation::~xDiscreteElementMethodSimulation()
{
	clearMemory();
}

int xDiscreteElementMethodSimulation::Initialize(xDiscreteElementMethodModel* _xdem, xContactManager* _xcm)
{
	xdem = _xdem;
	double maxRadius = 0;
	xParticleManager* xpm = xdem->XParticleManager();
	//np = xpm->NumParticleWithCluster();
	np = xpm->NumParticle();
	ns = xpm->NumCluster();
	nco = xpm->nClusterObject();
	if (ns == 0) ns = np;
	//nSingleSphere = xpm->nSingleSphere();
	//nClusterSphere = ns - xpm->nSingleSphere();
	xcm = _xcm;
// 	if (xpm->RealTimeCreating()/* && !xpm->ChangedParticleModel()*/)
// 	{
// 		if (xpm->OneByOneCreating())
// 			per_np = static_cast<unsigned int>((1.0 / xpm->NumCreatingPerSecond()) / simulation::dt);
// 		else
// 		{
// 			per_np = static_cast<unsigned int>((pm->TimeCreatingPerGroup() + 1e-9) / simulation::dt);
// 			//pm->setCreatingPerGroupIterator();
// 		}
// 
// 	}
// 	else
// 		per_np = 0;
	if (xcm)
	{
		nPolySphere = xcm->setupParticlesMeshObjectsContact();
		xcm->setupParticlesPlanesContact();
		xcm->allocPairList(np);
		if (nPolySphere)
			maxRadius = xcm->ContactParticlesMeshObjects()->MaxRadiusOfPolySphere();
		xcm->setNumClusterObject(nco);
	/*	if (xdem->XParticleManager()->NumClusterSet())
			xdem->XParticleManager()->SetClusterInformation();*/
	}
	

	allocationMemory(ns, np);

	memset(pos, 0, sizeof(double) * np * 4);
	if (cpos)
	{
		memset(cpos, 0, sizeof(double) * ns * 4);
		//memset(cindex, 0, sizeof(unsigned int) * n);
	}
		
	memset(ep, 0, sizeof(double) * ns * 4);
	memset(vel, 0, sizeof(double) * ns * 3);
	memset(acc, 0, sizeof(double) * ns * 3);
	memset(avel, 0, sizeof(double) * ns * 4);
	memset(aacc, 0, sizeof(double) * ns * 4);
	memset(force, 0, sizeof(double) * np * 3);
	memset(moment, 0, sizeof(double) * np * 3);
	for (unsigned int i = 0; i < ns; i++)
	{
		//vel[0] = -0.5;
		acc[i * 3 + 0] = 0.0;// mass[i] * xModel::gravity.x;
		acc[i * 3 + 1] = xModel::gravity.y;
		acc[i * 3 + 2] = 0.0;// mass[i] * xModel::gravity.z;
		ep[i * 4 + 0] = 1.0;
	}
	xpm->CopyPosition(pos, cpos, ep, np);
	if(xci) xpm->CopyClusterInformation(xci, rcloc);
	xpm->SetMassAndInertia(mass, inertia);
	for (unsigned int i = 0; i < np; i++)
	{
		double r = pos[i * 4 + 3];
		//vector3d cm = 
		if (r > maxRadius)
			maxRadius = r;
	}
	

	dtor = new xNeiborhoodCell;
	// 	switch (md->SortType())
	// 	{
	// 	case grid_base::NEIGHBORHOOD: dtor = new neighborhood_cell; break;
	// 	}
	if (dtor)
	{
		dtor->setWorldOrigin(new_vector3d(-1.0, -1.0, -1.0));
		dtor->setGridSize(new_vector3ui(128, 128, 128));
		dtor->setCellSize(maxRadius * 2.0);
		dtor->initialize(np + nPolySphere);
	}
	// 	switch (md->IntegrationType())
	// 	{
	// 	case dem_integrator::VELOCITY_VERLET: itor = new velocity_verlet; break;
	// 	}
	if (xdem->XSpringDamperForce())
		xdem->XSpringDamperForce()->initializeFreeLength(pos);
	if (xSimulation::Gpu())
	{
		checkCudaErrors(cudaMemcpy(dpos, pos, sizeof(double) * np * 4, cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpy(dep, ep, sizeof(double) * ns * 4, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dvel, vel, sizeof(double) * ns * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dacc, acc, sizeof(double) * ns * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(davel, avel, sizeof(double) * ns * 4, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(daacc, aacc, sizeof(double) * ns * 4, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dforce, force, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dmoment, moment, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dmass, mass, sizeof(double) * ns, cudaMemcpyHostToDevice));
		if(ns < np)
			checkCudaErrors(cudaMemcpy(diner, inertia, sizeof(double) * ns * 3, cudaMemcpyHostToDevice));
		else
			checkCudaErrors(cudaMemcpy(diner, inertia, sizeof(double) * np, cudaMemcpyHostToDevice));
		if (xci)
		{
			checkCudaErrors(cudaMemcpy(dcpos, cpos, sizeof(double) * ns * 4, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(dxci, xci, sizeof(xClusterInformation) * xpm->nClusterObject(), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(drcloc, rcloc, sizeof(double) * xpm->nClusterEach() * 3, cudaMemcpyHostToDevice));
		}
		if (nTsdaConnection && xdem->XSpringDamperForce()->xSpringDamperConnection())
		{
			checkCudaErrors(cudaMemcpy(dxsdci, xdem->XSpringDamperForce()->xSpringDamperConnection(), sizeof(xSpringDamperConnectionInformation) * nTsdaConnection, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(dxsdc_data, xdem->XSpringDamperForce()->xSpringDamperConnectionList(), sizeof(xSpringDamperConnectionData) * nTsdaConnectionList, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(dxsdc_kc, xdem->XSpringDamperForce()->xSpringDamperCoefficientValue(), sizeof(xSpringDamperCoefficient) * nTsdaConnectionValue, cudaMemcpyHostToDevice));
			//checkCudaErrors(cudaMemcpy(dxsd_free_length, xdem->XSpringDamperForce()->FreeLength(), sizeof(double) * nTsdaConnectionList, cudaMemcpyHostToDevice));
		}
			
		if (xcm)
		{
			if (xcm->ContactParticles())
				xcm->ContactParticles()->cudaMemoryAlloc(np);
			if (xcm->ContactParticlesMeshObjects())
				xcm->ContactParticlesMeshObjects()->cudaMemoryAlloc(np);
			if (xcm->ContactParticlesPlanes())
				xcm->ContactParticlesPlanes()->cudaMemoryAlloc(np);
		}
		device_dem_parameters dp;
		dp.np = np;
		dp.nCluster = ns < np ? ns : 0;
		dp.nClusterObject = xpm->nClusterObject();
		dp.rollingCondition = false;// xmd->RollingCondition();
		dp.nsphere = 0;
		if (xcm->ContactParticlesPlanes())
			dp.nplane = xcm->ContactParticlesPlanes()->NumPlanes();
		else
			dp.nplane = 0;
		dp.ncell = dtor->nCell();
		dp.nTsdaConnection = nTsdaConnection;
		dp.nTsdaConnectionList = nTsdaConnectionList;
		dp.grid_size.x = xGridCell::gs.x;
		dp.grid_size.y = xGridCell::gs.y;
		dp.grid_size.z = xGridCell::gs.z;
		dp.dt = xSimulation::dt;
		dp.half2dt = 0.5 * dp.dt * dp.dt;
		dp.cell_size = xGridCell::cs;
		dp.cohesion = 0.0;
		dp.gravity.x = xModel::gravity.x;
		dp.gravity.y = xModel::gravity.y;
		dp.gravity.z = xModel::gravity.z;
		dp.world_origin.x = xGridCell::wo.x;
		dp.world_origin.y = xGridCell::wo.y;
		dp.world_origin.z = xGridCell::wo.z;
		setDEMSymbolicParameter(&dp);
	}
	else
	{
		dpos = pos;
		dcpos = cpos;
		dep = ep;
		//drot = rot;
		dvel = vel;
		dacc = acc;
		davel = avel;
		daacc = aacc;
		dforce = force;
		dmoment = moment;
		dmass = mass;
		diner = inertia;
		dxci = xci;
		drcloc = rcloc;
		if (xdem->XSpringDamperForce())
		{
			dxsdci = xdem->XSpringDamperForce()->xSpringDamperConnection();
			dxsdc_data = xdem->XSpringDamperForce()->xSpringDamperConnectionList();
			dxsdc_kc = xdem->XSpringDamperForce()->xSpringDamperCoefficientValue();
			//dxsd_free_length = xdem->XSpringDamperForce()->FreeLength();
		}
	}
	if (isSaveMemory)
		xdem->XParticleManager()->AllocParticleResultMemory(xSimulation::npart, np);
	//dynamic_cast<neighborhood_cell*>(dtor)->reorderElements(pos, (double*)cm->HostSphereData(), np, nPolySphere);
	return xDynamicsError::xdynamicsSuccess;
}

bool xDiscreteElementMethodSimulation::Initialized()
{
	return isInitilize;
}

QString xDiscreteElementMethodSimulation::SaveStepResult(unsigned int pt, double ct)
{
	double *rp = NULL;
	double *rv = NULL;
	if (isSaveMemory)
	{
		rp = xdem->XParticleManager()->GetPositionResultPointer(pt);
		rv = xdem->XParticleManager()->GetVelocityResultPointer(pt);
		if (xSimulation::Gpu())
		{
			checkCudaErrors(cudaMemcpy(rp, dpos, sizeof(double) * np * 4, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(rv, dvel, sizeof(double) * ns * 3, cudaMemcpyDeviceToHost));
		}
		else
		{
			memcpy(rp, dpos, sizeof(double) * np * 4);
			memcpy(rv, dvel, sizeof(double) * ns * 3);
		}
	}
	if (xSimulation::Gpu())
	{
		checkCudaErrors(cudaMemcpy(pos, dpos, sizeof(double) * np * 4, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vel, dvel, sizeof(double) * ns * 3, cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpy(ep, dep, sizeof(double) * np * 4, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(avel, davel, sizeof(double) * ns * 4, cudaMemcpyDeviceToHost));
	}
	//char pname[256] = { 0, };
 	QString fname = xModel::path + xModel::name;
	QString part_name;
	part_name.sprintf("part%04d", pt);
	fname.sprintf("%s/part%04d.bin", fname.toUtf8().data(), pt);
	std::fstream qf(fname.toStdString());
	qf.open(fname.toStdString(), std::ios::binary | std::ios::out);
	if (qf.is_open())
	{
		qf.write((char*)&ct, sizeof(double));
		qf.write((char*)&np, sizeof(unsigned int));
		qf.write((char*)pos, sizeof(double) * np * 4);
		qf.write((char*)vel, sizeof(double) * ns * 3);
		//qf.write((char*)ep, sizeof(double) * np * 4);
		qf.write((char*)avel, sizeof(double) * ns * 4);
	}
	qf.close();
	/*qf.open(QIODevice::WriteOnly);*/
	partList.push_back(fname);
	//vector4d* h_sphere = xcm->ContactParticlesMeshObjects()->GetCurrentSphereData();
	//foreach(xParticleMeshObjectContact* cpm, xcm->ContactParticlesMeshObjects())
	//{
	//	xMeshObject* pobj = cpm->MeshObject();
	//	QString m_path = fname + "/" + pobj->Name() + "/";
	//	xUtilityFunctions::CreateDirectory(m_path.toUtf8().data());
	//	m_path += pobj->Name() + Q
	//}
	return fname;
}

void xDiscreteElementMethodSimulation::ExportResults(std::fstream& of)
{
	foreach(QString s, partList)
	{
		of << s.toStdString() << endl;
	}
}

void xDiscreteElementMethodSimulation::EnableSaveResultToMemory(bool b)
{
	isSaveMemory = b;
}

void xDiscreteElementMethodSimulation::updateObjectFromMBD()
{
	if (xcm)
		xcm->update();
}

// vector4d xDiscreteElementMethodSimulation::GlobalSphereInertiaForce(const euler_parameters& ev, const double j, const euler_parameters& ep)
// {
// 	double GvP0 = -ev.e1*ep.e0 + ev.e0*ep.e1 + ev.e3*ep.e2 - ev.e2*ep.e3;
// 	double GvP1 = -ev.e2*ep.e0 - ev.e3*ep.e1 + ev.e0*ep.e2 + ev.e1*ep.e3;
// 	double GvP2 = -ev.e3*ep.e0 + ev.e2*ep.e1 - ev.e1*ep.e2 + ev.e0*ep.e3;
// 	return vector4d
// 	{
// 		8 * (-ev.e1*j*GvP0 - ev.e2*j*GvP1 - ev.e3*j*GvP2),
// 		8 * (ev.e0*j*GvP0 - ev.e3*j*GvP1 + ev.e2*j*GvP2),
// 		8 * (ev.e3*j*GvP0 + ev.e0*j*GvP1 - ev.e1*j*GvP2),
// 		8 * (-ev.e2*j*GvP0 + ev.e1*j*GvP1 + ev.e0*j*GvP2)
// 	};
// }
// 
// vector4d xDiscreteElementMethodSimulation::GlobalSphereInertiaForce(const vector4d& ev, const double j, const vector4d& ep)
// {
// 	double GvP0 = -ev.y*ep.x + ev.x*ep.y + ev.w*ep.z - ev.z*ep.w;
// 	double GvP1 = -ev.z*ep.x - ev.w*ep.y + ev.x*ep.z + ev.y*ep.w;
// 	double GvP2 = -ev.w*ep.x + ev.z*ep.y - ev.y*ep.z + ev.x*ep.w;
// 	return vector4d
// 	{
// 		8 * (-ev.y*j*GvP0 - ev.z*j*GvP1 - ev.w*j*GvP2),
// 		8 * (ev.x*j*GvP0 - ev.w*j*GvP1 + ev.z*j*GvP2),
// 		8 * (ev.w*j*GvP0 + ev.x*j*GvP1 - ev.y*j*GvP2),
// 		8 * (-ev.z*j*GvP0 + ev.y*j*GvP1 + ev.x*j*GvP2)
// 	};
// }


void xDiscreteElementMethodSimulation::clearMemory()
{
	if (dtor) delete dtor; dtor = NULL;
	if (mass) delete[] mass; mass = NULL;
	if (inertia) delete[] inertia; inertia = NULL;
	if (pos) delete[] pos; pos = NULL;
	if (cpos) delete[] cpos; cpos = NULL;
	//if (cindex) delete[] cindex; cindex = NULL;
	if (ep) delete[] ep; ep = NULL;
	//if (rot) delete[] rot; rot = NULL;
	if (vel) delete[] vel; vel = NULL;
	if (acc) delete[] acc; acc = NULL;
	if (avel) delete[] avel; avel = NULL;
	if (aacc) delete[] aacc; aacc = NULL;
	if (force) delete[] force; force = NULL;
	if (moment) delete[] moment; moment = NULL;
	if (xci) delete[] xci; xci = NULL;
	if (rcloc) delete[] rcloc; rcloc = NULL;
	//if (dxsdci) delete[] dxsdci; dxsdci = NULL;
	//if (dxsdc_list) delete[] dxsdc_list; dxsdc_list = NULL;

	if (xSimulation::Gpu())
	{
		if (dmass) checkCudaErrors(cudaFree(dmass)); dmass = NULL;
		if (diner) checkCudaErrors(cudaFree(diner)); diner = NULL;
		if (dpos) checkCudaErrors(cudaFree(dpos)); dpos = NULL;
		if (dcpos) checkCudaErrors(cudaFree(dcpos)); dcpos = NULL;
		if (dep) checkCudaErrors(cudaFree(dep)); dep = NULL;
		//if (drot) checkCudaErrors(cudaFree(drot)); drot = NULL;
		if (dvel) checkCudaErrors(cudaFree(dvel)); dvel = NULL;
		if (dacc) checkCudaErrors(cudaFree(dacc)); dacc = NULL;
		if (davel) checkCudaErrors(cudaFree(davel)); davel = NULL;
		if (daacc) checkCudaErrors(cudaFree(daacc)); daacc = NULL;
		if (dforce) checkCudaErrors(cudaFree(dforce)); dforce = NULL;
		if (dmoment) checkCudaErrors(cudaFree(dmoment)); dmoment = NULL;
		if (dxci) checkCudaErrors(cudaFree(dxci)); dxci = NULL;
		if (drcloc) checkCudaErrors(cudaFree(drcloc)); drcloc = NULL;
		if (dxsdci) checkCudaErrors(cudaFree(dxsdci)); dxsdci = NULL;
		if (dxsdc_data) checkCudaErrors(cudaFree(dxsdc_data)); dxsdc_data = NULL;
		if (dxsdc_kc) checkCudaErrors(cudaFree(dxsdc_kc)); dxsdc_kc = NULL;
		//if (dxsd_free_length) checkCudaErrors(cudaFree(dxsd_free_length)); dxsd_free_length = NULL;
	}
}

void xDiscreteElementMethodSimulation::allocationMemory(unsigned int np, unsigned int rnp)
{
	clearMemory();
	mass = new double[np];
	pos = new double[rnp * 4];
	if (np < rnp)
	{
		cpos = new double[np * 4];
		inertia = new double[np * 3];
		xci = new xClusterInformation[xdem->XParticleManager()->nClusterObject()];
		rcloc = new double[xdem->XParticleManager()->nClusterEach() * 3];
	//	cindex = new unsigned int[rnp];
	}
	else
	{
		inertia = new double[np];
	}
	ep = new double[np * 4];
	//rot = new double[np * 4];
	vel = new double[np * 3];
	acc = new double[np * 3];
	avel = new double[np * 4];
	aacc = new double[np * 4];
	force = new double[rnp * 3];
	moment = new double[rnp * 3];


	if (xSimulation::Gpu())
	{
		checkCudaErrors(cudaMalloc((void**)&dmass, sizeof(double) * np));
		if(np < rnp)
			checkCudaErrors(cudaMalloc((void**)&diner, sizeof(double) * np * 3));
		else
			checkCudaErrors(cudaMalloc((void**)&diner, sizeof(double) * np));
		checkCudaErrors(cudaMalloc((void**)&dpos, sizeof(double) * rnp * 4));
		checkCudaErrors(cudaMalloc((void**)&dep, sizeof(double) * np * 4));
		checkCudaErrors(cudaMalloc((void**)&dvel, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&dacc, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&davel, sizeof(double) * np * 4));
		checkCudaErrors(cudaMalloc((void**)&daacc, sizeof(double) * np * 4));
		checkCudaErrors(cudaMalloc((void**)&dforce, sizeof(double) * rnp * 3));
		checkCudaErrors(cudaMalloc((void**)&dmoment, sizeof(double) * rnp * 3));
		if (xdem->XParticleManager()->nClusterObject())
		{
			checkCudaErrors(cudaMalloc((void**)&dcpos, sizeof(double) * np * 4));
			checkCudaErrors(cudaMalloc((void**)&dxci, sizeof(xClusterInformation) * xdem->XParticleManager()->nClusterObject()));
			checkCudaErrors(cudaMalloc((void**)&drcloc, sizeof(double) * xdem->XParticleManager()->nClusterEach() * 3));
		}
		
		if (xdem->XSpringDamperForce())
		{
			nTsdaConnection = xdem->XSpringDamperForce()->NumSpringDamperConnection();
			nTsdaConnectionList = xdem->XSpringDamperForce()->NumSpringDamperConnectionList();
			nTsdaConnectionValue = xdem->XSpringDamperForce()->NumSpringDamperConnectionValue();
			checkCudaErrors(cudaMalloc((void**)&dxsdci, sizeof(xSpringDamperConnectionInformation)*nTsdaConnection));
			checkCudaErrors(cudaMalloc((void**)&dxsdc_data, sizeof(xSpringDamperConnectionData)* nTsdaConnectionList));
			checkCudaErrors(cudaMalloc((void**)&dxsdc_kc, sizeof(xSpringDamperCoefficient) * nTsdaConnectionValue));
			//checkCudaErrors(cudaMalloc((void**)&dxsd_free_length, sizeof(double) * nTsdaConnectionList));
		}		
	}
}
