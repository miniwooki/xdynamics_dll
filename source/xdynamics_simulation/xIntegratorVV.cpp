#include "xdynamics_simulation/xIngegratorVV.h"
#include "xdynamics_parallel/xParallelDEM_decl.cuh"
#include "xdynamics_manager/XDiscreteElementMethodModel.h"

xIntegratorVV::xIntegratorVV()
{

}

xIntegratorVV::~xIntegratorVV()
{

}

int xIntegratorVV::Initialize(xDiscreteElementMethodModel* _xdem, xContactManager* _xcm)
{
	int ret = xDiscreteElementMethodSimulation::Initialize(_xdem, _xcm);	
	return ret;
}

int xIntegratorVV::OneStepSimulation(double ct, unsigned int cstep)
{
// 	if (per_np && !((cstep - 1) % per_np) && np < md->ParticleManager()->Np())
// 		md->ParticleManager()->OneByOneCreating() ? np += md->ParticleManager()->NextCreatingOne(np) : np += md->ParticleManager()->NextCreatingPerGroup();
	//std::cout << "cstep : " << cstep << " ";
	this->updatePosition(dpos, dvel, dacc, np);
	dtor->detection(dpos, (nPolySphere ? xcm->ContactParticlesMeshObjects()->SphereData() : NULL), np, nPolySphere);
	if (xcm)
	{
		xcm->runCollision(
			dpos, dvel, davel,
			dmass, dforce, dmoment,
			dtor->sortedID(), dtor->cellStart(), dtor->cellEnd(), np);
	}
	this->updateVelocity(dvel, dacc, davel, daacc, dforce, dmoment, dmass, diner, np);
	return 0;
}

void xIntegratorVV::updatePosition(double* dpos, double* dvel, double* dacc, unsigned int np)
{
	if (xSimulation::Gpu())
		vv_update_position(dpos, dvel, dacc, np);
	else
	{
		vector4d* p = (vector4d*)dpos;
	//	euler_parameters* r = (euler_parameters*)drot;
		vector3d* v = (vector3d*)dvel;
		vector3d* a = (vector3d*)dacc;
	//	euler_parameters* rv = (euler_parameters*)davel;
	//////	euler_parameters* ra = (euler_parameters*)daacc;
		double sqt_dt = 0.5 * xSimulation::dt * xSimulation::dt;
		for (unsigned int i = 0; i < np; i++){
			vector3d old_p = new_vector3d(p[i].x, p[i].y, p[i].z);
			vector3d new_p = old_p + xSimulation::dt * v[i] + sqt_dt * a[i];
			p[i] = new_vector4d(new_p.x, new_p.y, new_p.z, p[i].w);
			//r[i] = r[i] + xSimulation::dt * rv[i] + sqt_dt * ra[i];
		}
	}
}

void xIntegratorVV::updateVelocity(
	double *dvel, double* dacc, double *domega, 
	double* dalpha, double *dforce, double* dmoment, 
	double *dmass, double* dinertia, unsigned int np)
{
	if (xSimulation::Gpu())
		vv_update_velocity(dvel, dacc, domega, dalpha, dforce, dmoment, dmass, dinertia, np);
	else
	{
		double inv_m = 0;
		double inv_i = 0;
	//	vector3d* r = (vector3d *)drot;
		vector3d* v = (vector3d*)dvel;
		vector3d* o = (vector3d*)domega;
		vector3d* a = (vector3d*)dacc;
		vector3d* aa = (vector3d*)dalpha;
		vector3d* f = (vector3d*)dforce;
		vector3d* m = (vector3d*)dmoment;
		for (unsigned int i = 0; i < np; i++){
			inv_m = 1.0 / dmass[i];
			inv_i = 1.0 / dinertia[i];
			v[i] += 0.5 * xSimulation::dt * a[i];
			o[i] += 0.5 * xSimulation::dt * aa[i];
			a[i] = inv_m * f[i];
			//vector4d rF = 2.0 * GMatrix(r[i]) * m[i] + GlobalSphereInertiaForce(o[i], dinertia[i], r[i]);
			aa[i] = inv_i * m[i];
			v[i] += 0.5 * xSimulation::dt * a[i];
			o[i] += 0.5 * xSimulation::dt * aa[i];
			f[i] = dmass[i] * xModel::gravity;
			m[i] = new_vector3d(0.0, 0.0, 0.0);
		}
	}
}