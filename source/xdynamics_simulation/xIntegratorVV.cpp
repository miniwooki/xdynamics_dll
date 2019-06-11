#include "xdynamics_simulation/xIngegratorVV.h"
#include "xdynamics_parallel/xParallelDEM_decl.cuh"
#include "xdynamics_manager/XDiscreteElementMethodModel.h"

xIntegratorVV::xIntegratorVV()
	: m_np(0)
{

}

xIntegratorVV::~xIntegratorVV()
{

}

int xIntegratorVV::Initialize(xDiscreteElementMethodModel* _xdem, xContactManager* _xcm)
{
	int ret = xDiscreteElementMethodSimulation::Initialize(_xdem, _xcm);	
	m_np = xDiscreteElementMethodSimulation::ns;
	return ret;
}

int xIntegratorVV::OneStepSimulation(double ct, unsigned int cstep)
{
	m_np = xdem->XParticleManager()->ExcuteCreatingCondition(ct, cstep, m_np);
// 	if (per_np && !((cstep - 1) % per_np) && np < md->ParticleManager()->Np())
// 		md->ParticleManager()->OneByOneCreating() ? np += md->ParticleManager()->NextCreatingOne(np) : np += md->ParticleManager()->NextCreatingPerGroup();
	//std::cout << "cstep : " << cstep << " ";
	this->updatePosition(dpos, dvel, dacc, dep, davel, daacc, m_np);
	dtor->detection(dpos, (nPolySphere ? xcm->ContactParticlesMeshObjects()->SphereData() : NULL), m_np, nPolySphere);
//	std::cout << "after detection " << std::endl;
	if (xcm)
	{
		xcm->runCollision(
			dpos, dvel, davel,
			dmass, diner, dforce, dmoment,
			dtor->sortedID(), dtor->cellStart(), dtor->cellEnd(),
			xdem->XParticleManager()->ClusterIndex(), m_np, nSingleSphere, nClusterSphere);
	}
	this->updateVelocity(dvel, dacc, dep, davel, daacc, dforce, dmoment, dmass, diner, m_np);
	return 0;
}

void xIntegratorVV::updatePosition(
	double* dpos, double* dvel, double* dacc,
	double* ep, double* ev, double* ea, double* o, unsigned int np)
{
	if (xSimulation::Gpu())
		vv_update_position(dpos, dvel, dacc,/* ep, ev, ea,*/ np);
	else if (np != ns)
	{
		updateClusterPosition(dpos, NULL, dvel, dacc, ep, o, ea,
			xdem->XParticleManager()->ClusterIndex(), xdem->XParticleManager()->ClusterCount(),
			xdem->XParticleManager()->ClusterBegin(), xdem->XParticleManager()->ClusterRelativeLocation(), np);
	}
	else
	{
		vector4d* p = (vector4d*)dpos;
		vector3d* v = (vector3d*)dvel;
		vector3d* a = (vector3d*)dacc;
		double sqt_dt = 0.5 * xSimulation::dt * xSimulation::dt;
		for (unsigned int i = 0; i < np; i++) {
			vector3d old_p = new_vector3d(p[i].x, p[i].y, p[i].z);
			vector3d new_p = old_p + xSimulation::dt * v[i] + sqt_dt * a[i];
			p[i] = new_vector4d(new_p.x, new_p.y, new_p.z, p[i].w);
		}
	}
}

void xIntegratorVV::updateVelocity(
	double *dvel, double* dacc, double* ep, double *domega, 
	double* dalpha, double *dforce, double* dmoment, 
	double *dmass, double* dinertia, unsigned int np)
{
	if (xSimulation::Gpu())
		vv_update_velocity(dvel, dacc,/* ep,*/ domega, dalpha, dforce, dmoment, dmass, dinertia, np);
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

void xIntegratorVV::updateClusterPosition(
	double *dpos, double* dcpos, double * dvel, double * dacc, 
	double *ep, double * domega, double * dalpha,
	unsigned int* cidx, unsigned int* ccnt, unsigned int* cbegin, double* cdata, unsigned int np)
{
	vector3d* p = (vector3d*)dcpos;
	vector3d* v = (vector3d*)dvel;
	vector3d* a = (vector3d*)dacc;
	vector3d* w = (vector3d*)domega;
	vector3d* wd = (vector3d*)dalpha;
	vector4d* q = (vector4d*)ep;
	vector3d* rloc = (vector3d*)cdata;
	vector4d* rp = (vector4d*)dpos;
	double sqt_dt = 0.5 * xSimulation::dt * xSimulation::dt;
	for (unsigned int i = 0; i < np; i++) {
		vector3d old_p = new_vector3d(p[i].x, p[i].y, p[i].z);
		vector3d new_p = old_p + xSimulation::dt * v[i] + sqt_dt * a[i];
		p[i] = new_vector3d(new_p.x, new_p.y, new_p.z);
		matrix34d L = LMatrix(q[i]);
		vector4d ev = L * w[i];
		vector4d ea = 0.5 * L * a[i] - 0.25*dot(w[i], w[i]) * q[i];
		q[i] = q[i] + xSimulation::dt * ev + sqt_dt * ea;
		q[i] = normalize(q[i]);
		unsigned int id = cidx[i];
		unsigned int begin = cbegin[id * 2 + 0];
		unsigned int loc = cbegin[id * 2 + 1];
		unsigned int sid = begin + (i - begin) * ccnt[id];
		for (unsigned int j = 0; j < ccnt[cidx[i]]; j++)
		{
			vector3d m_pos = p[i] + ToGlobal(new_euler_parameters(q[i]), rloc[loc + j]);
			rp[sid + j] = new_vector4d(m_pos.x, m_pos.y, m_pos.z, rp[sid + j].w);
		}
	}
}

void xIntegratorVV::updateClusterVelocity(
	double *dpos, 
	double* dcpos, 
	double * dvel, 
	double * dacc, 
	double * ep,
	double * domega, 
	double * dalpha, 
	double * dforce, 
	double * dmoment, 
	double * dmass, 
	double * dinertia,
	unsigned int* cidx, 
	unsigned int* ccnt, 
	unsigned int* cbegin,
	double* cdata,
	unsigned int np,
	unsigned int nc)
{
	double inv_m = 0;
	double inv_i = 0;
	//	vector3d* r = (vector3d *)drot;
	vector3d* cp = (vector3d*)dcpos;
	vector3d* v = (vector3d*)dvel;
	vector3d* o = (vector3d*)domega;
	vector3d* a = (vector3d*)dacc;
	vector3d* aa = (vector3d*)dalpha;
	vector3d* f = (vector3d*)dforce;
	vector3d* m = (vector3d*)dmoment;
	vector4d* q = (vector4d*)ep;
	vector3d* J = (vector3d*)dinertia;
	vector3d* rloc = (vector3d*)cdata;
	vector2ui* cid = (vector2ui*)cidx;
	for (unsigned int i = 0; i < np + nc; i++) {
		unsigned int id = cid[i].x;// cidx[i];
		unsigned int begin = cbegin[id * 2 + 0];
		unsigned int loc = cbegin[id * 2 + 1];
		unsigned int sid = begin + (i - begin) * ccnt[id];
		vector3d F = new_vector3d(0.0, 0.0, 0.0);
		vector3d T = new_vector3d(0.0, 0.0, 0.0);
		vector3d LT = new_vector3d(0.0, 0.0, 0.0);
		euler_parameters e = new_euler_parameters(q[i]);
		for (unsigned int j = 0; j < ccnt[id]; j++)
		{
			vector3d gp = cp[i] + ToGlobal(e, rloc[loc + j]);
			vector3d dr = gp - cp[i];
			F += f[sid + j];
			T += m[sid + j];
			LT += cross(dr, f[sid + j]);
		}
		F += dmass[i] * xModel::gravity;
		vector3d n_prime = ToLocal(e, T + LT);
		vector3d w_prime = ToLocal(e, o[i]);
		vector3d rhs = n_prime - Tilde(o[i]) * new_vector3d(J[id].x * w_prime.x, J[id].y * w_prime.y, J[id].z * w_prime.z);
		vector3d wd_prime = new_vector3d(rhs.x / J[id].x, rhs.x / J[id].y, rhs.z / J[id].z);
		inv_m = 1.0 / dmass[i];
		v[i] += 0.5 * xSimulation::dt * a[i];
		o[i] += 0.5 * xSimulation::dt * aa[i];
		a[i] = inv_m * F;
		aa[i] = ToGlobal(e, wd_prime);
		v[i] += 0.5 * xSimulation::dt * a[i];
		o[i] += 0.5 * xSimulation::dt * aa[i];
		f[i] = new_vector3d(0.0, 0.0, 0.0);// dmass[i] * xModel::gravity;
		m[i] = new_vector3d(0.0, 0.0, 0.0);
	}
}

//void xIntegratorVV::integrationQuaternion(
//	double * quat, double * omega, double * moment, double* alpha, 
//	double * inertia, unsigned int np)
//{
//	vector3d* w = (vector3d*)omega;
//	vector3d* dw = (vector3d*)alpha;
//	vector3d* T = (vector3d*)moment;
//	vector4d* q = (vector4d*)quat;
//	for (unsigned int i = 0; i < np; i++)
//	{
//		double I = inertia[i];
//		vector3d _T = xUtilityFunctions::QuaternionRotation(q[i], T[i]);
//		vector3d _w = xUtilityFunctions::QuaternionRotation(q[i], w[i]);
//		vector3d _dw = (_T - cross(_w, I * _w)) / I;
//		vector3d _w_14 = _w + 0.25 * xSimulation::dt * _dw;
//		vector3d _w_12 = _w + 0.5 * xSimulation::dt * _dw;
//
//	}
//}
