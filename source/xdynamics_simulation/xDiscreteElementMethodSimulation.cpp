#include "xdynamics_simulation/xDiscreteElementMethodSimulation.h"
#include "xdynamics_algebra/xGridCell.h"
//#include <QtCore/QFile>
#include <fstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>

xDiscreteElementMethodSimulation::xDiscreteElementMethodSimulation()
	: xSimulation()
	, xdem(NULL)
	, dtor(NULL)
	, xcm(NULL)
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
	np = xpm->NumParticle();
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
		if (xdem->XParticleManager()->NumClusterSet())
			xdem->XParticleManager()->SetClusterInformation();
	}

	allocationMemory();

	memset(pos, 0, sizeof(double) * np * 4);
	memset(ep, 0, sizeof(double) * np * 4);
	memset(vel, 0, sizeof(double) * np * 3);
	memset(acc, 0, sizeof(double) * np * 3);
	memset(avel, 0, sizeof(double) * np * 3);
	memset(aacc, 0, sizeof(double) * np * 3);
	memset(force, 0, sizeof(double) * np * 3);
	memset(moment, 0, sizeof(double) * np * 3);

	xpm->CopyPosition(pos, np);
	xpm->SetMassAndInertia(mass, inertia);
	for (unsigned int i = 0; i < np; i++)
	{
		double r = pos[i * 4 + 3];
	//	vel[0] = -0.1;
		force[i * 3 + 0] = mass[i] * xModel::gravity.x;
		force[i * 3 + 1] = mass[i] * xModel::gravity.y;
		force[i * 3 + 2] = mass[i] * xModel::gravity.z;
		ep[i * 4 + 0] = 1.0;
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

	if (xSimulation::Gpu())
	{
		checkCudaErrors(cudaMemcpy(dpos, pos, sizeof(double) * np * 4, cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpy(dep, ep, sizeof(double) * np * 4, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dvel, vel, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dacc, acc, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(davel, avel, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(daacc, aacc, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dforce, force, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dmoment, moment, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dmass, mass, sizeof(double) * np, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(diner, inertia, sizeof(double) * np, cudaMemcpyHostToDevice));
		
		if (xcm)
		{
			
			if (xcm->ContactParticles())
				xcm->ContactParticles()->cudaMemoryAlloc(np);
			if (xcm->ContactParticlesMeshObjects())
				xcm->ContactParticlesMeshObjects()->cudaMemoryAlloc(np);
			if (xcm->ContactParticlesPlanes())
				xcm->ContactParticlesPlanes()->cudaMemoryAlloc(np);
			// 			if (xcm->ContactParticlesPolygonObjects())
			// 				xcm->ContactParticlesPolygonObjects()->cudaMemoryAlloc();
// 			foreach(xContact* c, xcm->Contacts())
// 				c->cudaMemoryAlloc();
		}
		device_dem_parameters dp;
		dp.np = np;
		dp.rollingCondition = false;// xmd->RollingCondition();
		dp.nsphere = 0;
		if (xcm->ContactParticlesPlanes())
			dp.nplane = xcm->ContactParticlesPlanes()->NumPlanes();
		else
			dp.nplane = 0;
		dp.ncell = dtor->nCell();
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
		//dep = ep;
		//drot = rot;
		dvel = vel;
		dacc = acc;
		davel = avel;
		daacc = aacc;
		dforce = force;
		dmoment = moment;
		dmass = mass;
		diner = inertia;
	}
	if (per_np)
		np = 0;
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
			checkCudaErrors(cudaMemcpy(rv, dvel, sizeof(double) * np * 3, cudaMemcpyDeviceToHost));
		}
		else
		{
			memcpy(rp, dpos, sizeof(double) * np * 4);
			memcpy(rv, dvel, sizeof(double) * np * 3);
		}
	}
	if (xSimulation::Gpu())
	{
		checkCudaErrors(cudaMemcpy(pos, dpos, sizeof(double) * np * 4, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vel, dvel, sizeof(double) * np * 3, cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpy(ep, dep, sizeof(double) * np * 4, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(avel, davel, sizeof(double) * np * 3, cudaMemcpyDeviceToHost));
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
		qf.write((char*)vel, sizeof(double) * np * 3);
		//qf.write((char*)ep, sizeof(double) * np * 4);
		qf.write((char*)avel, sizeof(double) * np * 3);
	}
	qf.close();
	/*qf.open(QIODevice::WriteOnly);*/
	partList.push_back(fname);
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
	if (ep) delete[] ep; ep = NULL;
	//if (rot) delete[] rot; rot = NULL;
	if (vel) delete[] vel; vel = NULL;
	if (acc) delete[] acc; acc = NULL;
	if (avel) delete[] avel; avel = NULL;
	if (aacc) delete[] aacc; aacc = NULL;
	if (force) delete[] force; force = NULL;
	if (moment) delete[] moment; moment = NULL;
	if (xSimulation::Gpu())
	{
		if (dmass) checkCudaErrors(cudaFree(dmass)); dmass = NULL;
		if (diner) checkCudaErrors(cudaFree(diner)); diner = NULL;
		if (dpos) checkCudaErrors(cudaFree(dpos)); dpos = NULL;
		if (dep) checkCudaErrors(cudaFree(dep)); dep = NULL;
		//if (drot) checkCudaErrors(cudaFree(drot)); drot = NULL;
		if (dvel) checkCudaErrors(cudaFree(dvel)); dvel = NULL;
		if (dacc) checkCudaErrors(cudaFree(dacc)); dacc = NULL;
		if (davel) checkCudaErrors(cudaFree(davel)); davel = NULL;
		if (daacc) checkCudaErrors(cudaFree(daacc)); daacc = NULL;
		if (dforce) checkCudaErrors(cudaFree(dforce)); dforce = NULL;
		if (dmoment) checkCudaErrors(cudaFree(dmoment)); dmoment = NULL;
	}
}

void xDiscreteElementMethodSimulation::allocationMemory()
{
	clearMemory();
	mass = new double[np];
	inertia = new double[np];
	pos = new double[np * 4];
	ep = new double[np * 4];
	//rot = new double[np * 4];
	vel = new double[np * 3];
	acc = new double[np * 3];
	avel = new double[np * 3];
	aacc = new double[np * 3];
	force = new double[np * 3];
	moment = new double[np * 3];

	if (xSimulation::Gpu())
	{
		checkCudaErrors(cudaMalloc((void**)&dmass, sizeof(double) * np));
		checkCudaErrors(cudaMalloc((void**)&diner, sizeof(double) * np));
		checkCudaErrors(cudaMalloc((void**)&dpos, sizeof(double) * np * 4));
		checkCudaErrors(cudaMalloc((void**)&dep, sizeof(double) * np * 4));
		checkCudaErrors(cudaMalloc((void**)&dvel, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&dacc, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&davel, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&daacc, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&dforce, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&dmoment, sizeof(double) * np * 3));
	}
}
