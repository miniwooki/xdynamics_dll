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
	std::cout << m_np << std::endl;
	this->updatePosition(dpos, dcpos, dvel, dacc, dep, davel, daacc, m_np);
	dtor->detection(dpos, (nPolySphere ? xcm->ContactParticlesMeshObjects()->SphereData() : NULL), nco ? np : m_np, nPolySphere);
//	std::cout << "after detection " << std::endl;
	if (xcm)
	{
		xcm->runCollision(
			dpos, dcpos, dep, dvel, davel,
			dmass, diner, dforce, dmoment,
			dtor->sortedID(), dtor->cellStart(), dtor->cellEnd(), dxci,
			nco ? np : m_np);
	}	
	if (xdem->XSpringDamperForce())
			SpringDamperForce();
	this->updateVelocity(dvel, dacc, dep, davel, daacc, dforce, dmoment, dmass, diner, m_np);
	return 0;
}

void xIntegratorVV::updatePosition(
	double* dpos, double* dcpos, double* dvel, double* dacc,
	double* dep, double* dev, double* dea, unsigned int np)
{
	if (xSimulation::Gpu())
	{
		if (xDiscreteElementMethodSimulation::np != ns)
			vv_update_cluster_position(dpos, dcpos, dep, drcloc, dvel, dacc, dev, dea, dxci, ns);
		else
			vv_update_position(dpos, dep, dvel, dev, dacc, dea,/* ep, ev, ea,*/ np);
	}
	else if (xDiscreteElementMethodSimulation::np != ns)
	{
		updateClusterPosition(dpos, dcpos, dvel, dacc, dep, dev, dea, ns);
	}
	else
	{
		vector4d* p = (vector4d*)dpos;
		vector3d* v = (vector3d*)dvel;
		vector3d* a = (vector3d*)dacc;
		euler_parameters* ep = (euler_parameters*)dep;
		euler_parameters* ev = (euler_parameters*)dev;
		euler_parameters* ea = (euler_parameters*)dea;
		double sqt_dt = 0.5 * xSimulation::dt * xSimulation::dt;
		for (unsigned int i = 0; i < np; i++) {
			vector3d old_p = new_vector3d(p[i].x, p[i].y, p[i].z);
			vector3d new_p = old_p + xSimulation::dt * v[i] + sqt_dt * a[i];
			euler_parameters new_ep = ep[i] + xSimulation::dt * ev[i] + sqt_dt * ea[i];
			ep[i] = normalize(new_ep);
			p[i] = new_vector4d(new_p.x, new_p.y, new_p.z, p[i].w);
		}
	}
}

void xIntegratorVV::updateVelocity(
	double *dvel, double* dacc, double* dep, double *dev, 
	double* dea, double *dforce, double* dmoment, 
	double *dmass, double* dinertia, unsigned int np)
{
	if (xSimulation::Gpu())
	{
		if(xDiscreteElementMethodSimulation::np != ns)
			vv_update_cluster_velocity(dcpos, dep, dvel, dacc,/* ep,*/ dev, dea, dforce, dmoment, drcloc, dmass, dinertia, dxci, np);
		else
			vv_update_velocity(dvel, dacc, dep, dev, dea, dforce, dmoment, dmass, dinertia, np);
	}
		
	else if (xDiscreteElementMethodSimulation::np != ns)
	{
		updateClusterVelocity(dpos, dcpos, dvel, dacc, dep, dev, dea, dforce, dmoment, dmass, dinertia, np);
	}
	else
	{
		double inv_m = 0;
		double inv_i = 0;
		euler_parameters* ep = (euler_parameters*)dep;
		vector3d* v = (vector3d*)dvel;
		euler_parameters* ev = (euler_parameters*)dev;
		vector3d* a = (vector3d*)dacc;
		euler_parameters* ea = (euler_parameters*)dea;
		vector3d* f = (vector3d*)dforce;
		vector3d* m = (vector3d*)dmoment;
		vector3d* iner = (vector3d*)dinertia;
		for (unsigned int i = 0; i < np; i++){
			matrix33d J = { iner[i].x, 0, 0, 0, iner[i].y, 0, 0, 0, iner[i].z };
			//std::cout << "force : [" << f[i].x << " " << f[i].y << " " << f[i].z << "]" << std::endl;
		//	std::cout << "moment : [" << m[i].x << " " << m[i].y << " " << m[i].z << "]" << std::endl;
			vector3d n_prime = ToLocal(ep[i], m[i]);
			euler_parameters m_ea = CalculateUCEOM(J, ep[i], ev[i], n_prime);
			//std::cout << "euler_acc : [" << m_ea.e0 << ", " << m_ea.e1 << ", " << m_ea.e2 << ", " << m_ea.e3 << "]" << std::endl;
			inv_m = 1.0 / dmass[i];
		//	std::cout << "inverse mass : " << inv_m << std::endl;
			//inv_i = 1.0 / dinertia[i];
			v[i] += 0.5 * xSimulation::dt * a[i];
			ev[i] += 0.5 * xSimulation::dt * ea[i];
			a[i] = inv_m * (f[i] + dmass[i] * xModel::gravity);
			//vector4d rF = 2.0 * GMatrix(r[i]) * m[i] + GlobalSphereInertiaForce(o[i], dinertia[i], r[i]);
			ea[i] = m_ea;// inv_i * m[i];
			v[i] += 0.5 * xSimulation::dt * a[i];
			ev[i] += 0.5 * xSimulation::dt * ea[i];
			f[i] = new_vector3d(0.0, 0.0, 0.0);
			m[i] = new_vector3d(0.0, 0.0, 0.0);
		}
	}
}

void xIntegratorVV::updateClusterPosition(
	double *dpos, double* dcpos, double * dvel, double * dacc, 
	double *dep, double * dev, double * dea, unsigned int np)
{
	vector4d* p = (vector4d*)dcpos;
	vector3d* v = (vector3d*)dvel;
	vector3d* a = (vector3d*)dacc;
	euler_parameters* ev = (euler_parameters*)dev;
	euler_parameters* ea = (euler_parameters*)dea;
	euler_parameters* ep = (euler_parameters*)dep;
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
		/*matrix34d L = LMatrix(ep[i]);
		vector4d ev = L * w[i];
		vector4d ea = 0.5 * L * wd[i] - 0.25*dot(w[i], w[i]) * q[i];*/
		ep[i] = ep[i] + xSimulation::dt * ev[i] + sqt_dt * ea[i];
		ep[i] = normalize(ep[i]);
	//	double th = 2.0 * acos(ep[i].e0) * (180.0 / M_PI);
	//	std::cout << "theta : " << th << std::endl;
		double norm = length(ep[i]);
		//unsigned int id = cidx[i];
		//unsigned int begin = cbegin[id * 2 + 0];
		//unsigned int loc = cbegin[id * 2 + 1];
		unsigned int sid = sbegin + i * neach;// begin + (i - begin) * ccnt[id];
		for (unsigned int j = 0; j < neach; j++)
		{
			vector3d cp = new_vector3d(p[i].x, p[i].y, p[i].z);
			vector3d m_pos = cp + ToGlobal(ep[i], rloc[seach + j]);
			rp[sid + j] = new_vector4d(m_pos.x, m_pos.y, m_pos.z, rp[sid + j].w);
		}
	}
}

void xIntegratorVV::updateClusterVelocity(
	double *dpos, 
	double* dcpos, 
	double * dvel, 
	double * dacc, 
	double * dep,
	double * dev, 
	double * dea, 
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
	euler_parameters* ev = (euler_parameters*)dev;
	vector3d* a = (vector3d*)dacc;
	euler_parameters* ea = (euler_parameters*)dea;
	vector3d* f = (vector3d*)dforce;
	vector3d* m = (vector3d*)dmoment;
	vector4d* ep = (vector4d*)dep;
	vector3d* in = (vector3d*)dinertia;
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
		euler_parameters e = new_euler_parameters(ep[i]);
		for (unsigned int j = 0; j < neach; j++)
		{
			F += f[sid + j];
			T += m[sid + j];
			f[sid + j] = new_vector3d(0.0, 0.0, 0.0);// dmass[i] * xModel::gravity;
			m[sid + j] = new_vector3d(0.0, 0.0, 0.0);
			
		}
		//T = new_vector3d(0.0, 0.0,-1.0);
		//F = F / 2.0;
		F += dmass[i] * xModel::gravity;
		matrix33d J = { 0, };
		J.a00 = in[i].x; J.a11 = in[i].y; J.a22 = in[i].z;
		vector3d n_prime = ToLocal(e, T + LT);
		euler_parameters m_ea = CalculateUCEOM(J, e, ev[i], n_prime);
		/*matrix34d G = GMatrix(e);
		matrix34d Gd = GMatrix(ev[i]);
		matrix34d JPL = 2.0 * J * G;
		vector3d LH = G * (4.0 * (Gd * (J * (G * ev[i]))));
		matrix44d lhs = new_matrix44d(JPL, e);
		vector4d r0 = new_vector4d(LH.x, LH.y, LH.z, dot(ev[i], ev[i]));
		
		vector4d rhs = new_vector4d(n_prime.x, n_prime.y, n_prime.z, 0.0) - r0;*/
		//vector3d w_prime = ToLocal(e, o[i]);
		//vector3d rhs = n_prime - Tilde(o[i]) * new_vector3d(J[i].x * w_prime.x, J[i].y * w_prime.y, J[i].z * w_prime.z);
		//vector3d wd_prime = new_vector3d(rhs.x / J[i].x, rhs.y / J[i].y, rhs.z / J[i].z);
		inv_m = 1.0 / dmass[i];
		v[i] += 0.5 * xSimulation::dt * a[i];
		ev[i] += 0.5 * xSimulation::dt * ea[i];
		a[i] = inv_m * F;
		ea[i] = m_ea;// ToEulerParameters(Inverse4X4(lhs) * rhs);// ToGlobal(e, wd_prime);
		v[i] += 0.5 * xSimulation::dt * a[i];
		ev[i] += 0.5 * xSimulation::dt * ea[i];
		
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
