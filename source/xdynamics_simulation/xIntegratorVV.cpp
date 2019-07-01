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
	if (xDiscreteElementMethodSimulation::ns > xDiscreteElementMethodSimulation::np)
		m_np = xDiscreteElementMethodSimulation::np;

	return ret;
}

int xIntegratorVV::OneStepSimulation(double ct, unsigned int cstep)
{
	m_np = xdem->XParticleManager()->ExcuteCreatingCondition(ct, cstep, m_np);
	this->updatePosition(dpos, dcpos, dvel, dacc, dep, davel, daacc, m_np);
	dtor->detection(dpos, (nPolySphere ? xcm->ContactParticlesMeshObjects()->SphereData() : NULL), nco ? np : m_np, nPolySphere);
//	std::cout << "after detection " << std::endl;
	if (xcm)
	{
		xcm->runCollision(
			dpos, dvel, davel,
			dmass, diner, dforce, dmoment,
			dtor->sortedID(), dtor->cellStart(), dtor->cellEnd(), dxci,
			nco ? np : m_np);
	}
	this->updateVelocity(dvel, dacc, dep, davel, daacc, dforce, dmoment, dmass, diner, m_np);
	return 0;
}

void xIntegratorVV::updatePosition(
	double* dpos, double* dcpos, double* dvel, double* dacc,
	double* ep, double* ev, double* ea, unsigned int np)
{
	if (xSimulation::Gpu())
	{
		if (xDiscreteElementMethodSimulation::np != ns)
			vv_update_cluster_position(dpos, dcpos, ep, drcloc, dvel, dacc, ev, ea, dxci, ns);
		else
			vv_update_position(dpos, dvel, dacc,/* ep, ev, ea,*/ np);
	}
	else if (xDiscreteElementMethodSimulation::np != ns)
	{
		updateClusterPosition(dpos, dcpos, dvel, dacc, ep, ev, ea, ns);
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
	{
		if(xDiscreteElementMethodSimulation::np != ns)
			vv_update_cluster_velocity(dcpos, ep, dvel, dacc,/* ep,*/ domega, dalpha, dforce, dmoment, drcloc, dmass, dinertia, dxci, np);
		else
			vv_update_velocity(dvel, dacc,/* ep,*/ domega, dalpha, dforce, dmoment, dmass, dinertia, np);
	}
		
	else if (xDiscreteElementMethodSimulation::np != ns)
	{
		updateClusterVelocity(dpos, dcpos, dvel, dacc, dep, domega, dalpha, dforce, dmoment, dmass, dinertia, np);
	}
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
			a[i] = inv_m * (f[i] + dmass[i] * xModel::gravity);
			//vector4d rF = 2.0 * GMatrix(r[i]) * m[i] + GlobalSphereInertiaForce(o[i], dinertia[i], r[i]);
			aa[i] = inv_i * m[i];
			v[i] += 0.5 * xSimulation::dt * a[i];
			o[i] += 0.5 * xSimulation::dt * aa[i];
			f[i] = new_vector3d(0.0, 0.0, 0.0);
			m[i] = new_vector3d(0.0, 0.0, 0.0);
		}
	}
}

void xIntegratorVV::updateClusterPosition(
	double *dpos, double* dcpos, double * dvel, double * dacc, 
	double *ep, double * domega, double * dalpha, unsigned int np)
{
	vector4d* p = (vector4d*)dcpos;
	vector3d* v = (vector3d*)dvel;
	vector3d* a = (vector3d*)dacc;
	vector3d* w = (vector3d*)domega;
	vector3d* wd = (vector3d*)dalpha;
	vector4d* q = (vector4d*)ep;
	vector3d* rloc = (vector3d*)drcloc;
	vector4d* rp = (vector4d*)dpos;
	double sqt_dt = 0.5 * xSimulation::dt * xSimulation::dt;
	for (unsigned int i = 0; i < np; i++) {
		unsigned int neach = 0;
		unsigned int seach = 0;
		unsigned int sbegin = 0;
		for (unsigned int j = 0; j < nco; j++)
		{
			xClusterInformation xc = xci[j];
			if (i >= xc.sid && i < xc.sid + xc.count * xc.neach)
			{
				neach = xc.neach;
				break;
			}
			seach += xc.neach;
			sbegin += xc.count * xc.neach;
		}

		vector3d old_p = new_vector3d(p[i].x, p[i].y, p[i].z);
		vector3d new_p = old_p + xSimulation::dt * v[i] + sqt_dt * a[i];
		p[i] = new_vector4d(new_p.x, new_p.y, new_p.z, p[i].w);
		matrix34d L = LMatrix(q[i]);
		vector4d ev = L * w[i];
		vector4d ea = 0.5 * L * wd[i] - 0.25*dot(w[i], w[i]) * q[i];
		q[i] = q[i] + xSimulation::dt * ev + sqt_dt * ea;
		q[i] = normalize(q[i]);
		double norm = length(q[i]);
		//unsigned int id = cidx[i];
		//unsigned int begin = cbegin[id * 2 + 0];
		//unsigned int loc = cbegin[id * 2 + 1];
		unsigned int sid = sbegin + i * neach;// begin + (i - begin) * ccnt[id];
		for (unsigned int j = 0; j < neach; j++)
		{
			vector3d cp = new_vector3d(p[i].x, p[i].y, p[i].z);
			vector3d m_pos = cp + ToGlobal(new_euler_parameters(q[i]), rloc[seach + j]);
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
	unsigned int dnp)
{
	double inv_m = 0;
	double inv_i = 0;
	//	vector3d* r = (vector3d *)drot;
	vector4d* cp = (vector4d*)dcpos;
	vector3d* v = (vector3d*)dvel;
	vector3d* o = (vector3d*)domega;
	vector3d* a = (vector3d*)dacc;
	vector3d* aa = (vector3d*)dalpha;
	vector3d* f = (vector3d*)dforce;
	vector3d* m = (vector3d*)dmoment;
	vector4d* q = (vector4d*)ep;
	vector3d* J = (vector3d*)dinertia;
	vector3d* rloc = (vector3d*)drcloc;
	//vector2ui* cid = (vector2ui*)cidx;
	for (unsigned int i = 0; i < dnp; i++) {
		unsigned int neach = 0;
		unsigned int seach = 0;
		unsigned int sbegin = 0;
		for (unsigned int j = 0; j < nco; j++)
		{
			xClusterInformation xc = xci[j];
			if (i >= xc.sid && i < xc.sid + xc.count * xc.neach)
			{
				neach = xc.neach;
				break;
			}
			seach += xc.neach;
			sbegin += xc.count * xc.neach;
		}
		unsigned int sid = sbegin + i * neach;
		//unsigned int id = cid[i].x;// cidx[i];
		//unsigned int begin = cbegin[id * 2 + 0];
		//unsigned int loc = cbegin[id * 2 + 1];
		//unsigned int sid = begin + (i - begin) * ccnt[id];
		vector3d F = new_vector3d(0.0, 0.0, 0.0);
		vector3d T = new_vector3d(0.0, 0.0, 0.0);
		vector3d LT = new_vector3d(0.0, 0.0, 0.0);
		euler_parameters e = new_euler_parameters(q[i]);
		for (unsigned int j = 0; j < neach; j++)
		{
			vector3d cpos = new_vector3d(cp[i].x, cp[i].y, cp[i].z);
			vector3d gp = cpos + ToGlobal(e, rloc[seach + j]);
			vector3d dr = gp - cpos;
			F += f[sid + j];
			//vector3d T_f = cross(dr, f[sid + j]);
			T += m[sid + j];
			LT += cross(dr, f[sid + j]);
			f[sid + j] = new_vector3d(0.0, 0.0, 0.0);// dmass[i] * xModel::gravity;
			m[sid + j] = new_vector3d(0.0, 0.0, 0.0);
		}
		//F = F / 2.0;
		F += dmass[i] * xModel::gravity;
		vector3d n_prime = ToLocal(e, T + LT);
		vector3d w_prime = ToLocal(e, o[i]);
		vector3d rhs = n_prime - Tilde(o[i]) * new_vector3d(J[i].x * w_prime.x, J[i].y * w_prime.y, J[i].z * w_prime.z);
		vector3d wd_prime = new_vector3d(rhs.x / J[i].x, rhs.y / J[i].y, rhs.z / J[i].z);
		inv_m = 1.0 / dmass[i];
		v[i] += 0.5 * xSimulation::dt * a[i];
		o[i] += 0.5 * xSimulation::dt * aa[i];
		a[i] = inv_m * F;
		aa[i] = ToGlobal(e, wd_prime);
		v[i] += 0.5 * xSimulation::dt * a[i];
		o[i] += 0.5 * xSimulation::dt * aa[i];
		
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
