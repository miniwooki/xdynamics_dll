#include "xdynamics_object/xParticleCubeContact.h"
#include "xdynamics_object/xParticlePlaneContact.h"
#include "xdynamics_object/xCubeObject.h"
#include "xdynamics_object/xParticleObject.h"
#include <sstream>

xParticleCubeContact::xParticleCubeContact()
	: xContact()
	, cu(NULL)
	, p(NULL)
	, dpi(NULL)
{

}

xParticleCubeContact::xParticleCubeContact(std::string _name, xObject* o1, xObject *o2)
	: xContact(_name, PARTICLE_CUBE)
	, cu(NULL)
	, p(NULL)
	, dpi(NULL)
{
	for(int i = 0; i < 6; i++) cpplanes[i] = nullptr;
	if (o1->Shape() == CUBE_SHAPE)
	{
		cu = dynamic_cast<xCubeObject*>(o1);
		p = dynamic_cast<xParticleObject*>(o2);
	}
	else
	{
		cu = dynamic_cast<xCubeObject*>(o2);
		p = dynamic_cast<xParticleObject*>(o1);
	}
	mpp = { o1->Youngs(), o2->Youngs(), o1->Poisson(), o2->Poisson(), o1->Shear(), o2->Shear() };
}

xParticleCubeContact::~xParticleCubeContact()
{
	if (dpi) checkCudaErrors(cudaFree(dpi)); dpi = NULL;
	for (int i = 0; i < 6; i++) {
		if (cpplanes[i]) delete cpplanes[i]; cpplanes[i] = nullptr;
	}
}

xCubeObject* xParticleCubeContact::CubeObject()
{
	return cu;
}

void xParticleCubeContact::collision_gpu(
	double *pos, double* cpos, xClusterInformation* xci,
	double *ep, double *vel, double *ev,
	double *mass, double* inertia,
	double *force, double *moment,
	double *tmax, double* rres,
	unsigned int *sorted_id,
	unsigned int *cell_start,
	unsigned int *cell_end,
	unsigned int np)
{
	for (int i = 0; i < 6; i++) {
		if (cpplanes[i]) {
			cpplanes[i]->collision_gpu(
				pos, cpos, xci, ep, vel, ev,
				mass, inertia, force, moment, tmax, rres, sorted_id,
				cell_start, cell_end, np);
		}
	}
}

void xParticleCubeContact::collision_cpu(
	vector4d * pos, euler_parameters * ep, vector3d * vel,
	euler_parameters * ev, double* mass, double & rres, vector3d & tmax,
	vector3d & force, vector3d & moment, unsigned int nco,
	xClusterInformation * xci, vector4d * cpos)
{
}

void xParticleCubeContact::savePartData(unsigned int np)
{
	//for (int i = 0; i < 6; i++) {
	//	cpplanes[i]->savePartData(np);
	//}
}

void xParticleCubeContact::cuda_collision(
	double *pos, double *vel, double *omega,
	double *mass, double *force, double *moment,
	unsigned int *sorted_id, unsigned int *cell_start,
	unsigned int *cell_end, unsigned int np)
{
//	cu_cube_contact_force(1, dpi, pos, vel, omega, force, moment, mass, np, dcp);
}

void xParticleCubeContact::cudaMemoryAlloc(unsigned int np)
{
	/*xContact::cudaMemoryAlloc(np);
	device_plane_info *_dpi = new device_plane_info[6];
	xPlaneObject* pl = cu->Planes();
	for (unsigned i = 0; i < 6; i++)
	{
		xPlaneObject pe = pl[i];
		_dpi[i].l1 = pe.L1();
		_dpi[i].l2 = pe.L2();
		_dpi[i].xw = make_double3(pe.XW().x, pe.XW().y, pe.XW().z);
		_dpi[i].uw = make_double3(pe.UW().x, pe.UW().y, pe.UW().z);
		_dpi[i].u1 = make_double3(pe.U1().x, pe.U1().y, pe.U1().z);
		_dpi[i].u2 = make_double3(pe.U2().x, pe.U2().y, pe.U2().z);
		_dpi[i].pa = make_double3(pe.PA().x, pe.PA().y, pe.PA().z);
		_dpi[i].pb = make_double3(pe.PB().x, pe.PB().y, pe.PB().z);
		_dpi[i].w2 = make_double3(pe.W2().x, pe.W2().y, pe.W2().z);
		_dpi[i].w3 = make_double3(pe.W3().x, pe.W3().y, pe.W3().z);
		_dpi[i].w4 = make_double3(pe.W4().x, pe.W4().y, pe.W4().z);
	}

	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_plane_info) * 6));
	checkCudaErrors(cudaMemcpy(dpi, _dpi, sizeof(device_plane_info) * 6, cudaMemcpyHostToDevice));
	delete[] _dpi;*/
}

void xParticleCubeContact::update()
{
	cu->updateCube();
	for (int i = 0; i < 6; i++) {
		cpplanes[i]->update();
	}
	//if (xSimulation::Gpu())
	//{
	//	euler_parameters ep = cu->EulerParameters();
	//	euler_parameters ed = cu->DEulerParameters();
	//	double bi[15] =
	//	{
	//		cu->Mass(),
	//		cu->Position().x, cu->Position().y, cu->Position().z,
	//		cu->Velocity().x, cu->Velocity().y, cu->Velocity().z,
	//		ep.e0, ep.e1, ep.e2, ep.e3,
	//		ed.e0, ed.e1, ed.e2, ed.e3
	//	};
	//	checkXerror(cudaMemcpy(dbi, bi, DEVICE_BODY_MEM_SIZE, cudaMemcpyHostToDevice));
	//	//cu_update_meshObjectData(dvList, dsphere, dlocal, dti, dbi, po->NumTriangle());
	//}
}

void xParticleCubeContact::define(unsigned int idx, unsigned int np)
{
	xContactParameterData cpd = {
		xContact::restitution,
		xContact::stiffnessRatio,
		xContact::s_friction,
		xContact::friction,
		xContact::cohesion,
		xContact::rolling_factor
	};
	for (int i = 0; i < 6; i++) {
		std::stringstream ss;
		ss << name << i;
		//xstring _n = ss.str();
		xPlaneObject* plane = cu->Planes() + i;
		cpplanes[i] = new xParticlePlaneContact(ss.str(), p, plane);
		cpplanes[i]->setContactParameters(cpd);
		cpplanes[i]->define(idx, np);
	}	
}
