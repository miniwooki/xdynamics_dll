#include "xdynamics_object/xParticlePlaneContact.h"
#include "xdynamics_object/xPlaneObject.h"
#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_manager/xDynamicsManager.h"

unsigned int xParticlePlaneContact::defined_count = 0;
bool xParticlePlaneContact::allocated_static = false;

double* xParticlePlaneContact::d_tsd_ppl = nullptr;
unsigned int* xParticlePlaneContact::d_pair_count_ppl = nullptr;
unsigned int* xParticlePlaneContact::d_pair_id_ppl = nullptr;

double* xParticlePlaneContact::tsd_ppl = nullptr;
unsigned int* xParticlePlaneContact::pair_count_ppl = nullptr;
unsigned int* xParticlePlaneContact::pair_id_ppl = nullptr;

xParticlePlaneContact::xParticlePlaneContact()
	: xContact()
	, p(nullptr)
	, pe(nullptr)
	//, allocated_static(false)
{

}

xParticlePlaneContact::xParticlePlaneContact(std::string _name, xObject* o1, xObject* o2)
	: xContact(_name, PARTICLE_PANE)
	, p(nullptr)
	, pe(nullptr)
	//, allocated_static(false)
{
	if (o1 && o2)
	{
		if (o1->Shape() == PLANE_SHAPE)
		{
			pe = dynamic_cast<xPlaneObject*>(o1);
			p = dynamic_cast<xParticleObject*>(o2);
		}
		else
		{
			pe = dynamic_cast<xPlaneObject*>(o2);
			p = dynamic_cast<xParticleObject*>(o1);
		}
		mpp = { o1->Youngs(), o2->Youngs(), o1->Poisson(), o2->Poisson(), o1->Shear(), o2->Shear() };
	}	
}

xParticlePlaneContact::~xParticlePlaneContact()
{
	if (d_pair_count_ppl) checkCudaErrors(cudaFree(d_pair_count_ppl)); d_pair_count_ppl = NULL;
	if (d_pair_id_ppl) checkCudaErrors(cudaFree(d_pair_id_ppl)); d_pair_id_ppl = NULL;
	if (d_tsd_ppl) checkCudaErrors(cudaFree(d_tsd_ppl)); d_tsd_ppl = NULL;
	if (pair_count_ppl) delete[] pair_count_ppl; pair_count_ppl = NULL;
	if (pair_id_ppl) delete[] pair_id_ppl; pair_id_ppl = NULL;
	if (tsd_ppl) delete[] tsd_ppl; tsd_ppl = NULL;
	if (dpi) delete[] dpi; dpi = NULL;
	if (pair_count_ppl) delete[] pair_count_ppl; pair_count_ppl = NULL;
	if (tsd_ppl) delete[] tsd_ppl; tsd_ppl = NULL;
}

void xParticlePlaneContact::define(unsigned int idx, unsigned int np)
{
	id = defined_count;
	xContact::define(idx, np);
	//host_plane_info hpi =
	//{
	//	id,
	//	pe->L1(), pe->L2(),
	//	pe->U1(), pe->U2(),
	//	pe->UW(), pe->XW(),
	//	pe->PA(), pe->PB(),
	//	pe->W2(), pe->W3(), pe->W4()
	//};
	if (xSimulation::Gpu())
	{
		if (!allocated_static)
		{
			pair_count_ppl = new unsigned int[np];
			pair_id_ppl = new unsigned int[np * MAX_P2MS_COUNT];
			tsd_ppl = new double[2 * np * MAX_P2MS_COUNT];
			checkXerror(cudaMalloc((void**)&d_pair_count_ppl, sizeof(unsigned int) * np));
			checkXerror(cudaMalloc((void**)&d_pair_id_ppl, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
			checkXerror(cudaMalloc((void**)&d_tsd_ppl, sizeof(double2) * np * MAX_P2MS_COUNT));
			checkXerror(cudaMemset(d_pair_count_ppl, 0, sizeof(unsigned int) * np));
			checkXerror(cudaMemset(d_pair_id_ppl, 0, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
			checkXerror(cudaMemset(d_tsd_ppl, 0, sizeof(double2) * np * MAX_P2MS_COUNT));
			
			allocated_static = true;
		}
		checkXerror(cudaMalloc((void**)&dpi, sizeof(device_plane_info)));		
		checkXerror(cudaMalloc((void**)&dbi, sizeof(device_body_info)));
		//checkXerror(cudaMemcpy(dpi, &hpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
		
		update();
	}
	xDynamicsManager::This()->XResult()->set_p2pl_contact_data((int)MAX_P2PL_COUNT);
	defined_count++;
}

// void xParticlePlaneContact::initialize()
// {
// 	xParticlePlaneContact::local_initialize();
// }

void xParticlePlaneContact::local_initialize()
{
	defined_count = 0;
}

void xParticlePlaneContact::update()
{
	//device_body_info *bi = NULL;
	//if (nContactObject || is_first_set_up)
		//bi = new device_body_info[nContactObject];

	//QMapIterator<unsigned int, xPlaneObject*> it(pair_ip);
	//for (xmap<unsigned int, xPlaneObject*>::iterator it = pair_ip.begin(); it != pair_ip.end(); it.next())// (it.hasNext())
	//{
		//it.next();
		//unsigned int id = it.key();
		//xPlaneObject* p = it.value();
		/*if (p->MovingObject())
		{*/
			//host_plane_info new_hpi = { 0, };
			//new_hpi.ismoving = hpi.ismoving;
	hpi.id = id;
	hpi.xw = pe->Position() + pe->toGlobal(pe->LocalPoint(0));
	hpi.w2 = pe->Position() + pe->toGlobal(pe->LocalPoint(1));
	hpi.w3 = pe->Position() + pe->toGlobal(pe->LocalPoint(2));
	hpi.w4 = pe->Position() + pe->toGlobal(pe->LocalPoint(3));
	hpi.pa = hpi.w2 - hpi.xw;
	hpi.pb = hpi.w4 - hpi.xw;
	hpi.l1 = length(hpi.pa);
	hpi.l2 = length(hpi.pb);
	hpi.u1 = hpi.pa / hpi.l1;
	hpi.u2 = hpi.pb / hpi.l2;
	hpi.uw = cross(hpi.u1, hpi.u2);
	//hpi[id] = new_hpi;

	//if (xSimulation::Gpu())
	checkXerror(cudaMemcpy(dpi, &hpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
		//}
	//}
	if (xSimulation::Gpu())
	{
		unsigned int mcnt = 0;
	//	for (xmap<unsigned int, xContact*>::iterator it = pair_contact.begin(); it != pair_contact.end(); it.next())
			//foreach(xContact* xc, pair_contact)
		//{
			//xPointMass* pm = NULL;
			//xContact* xc = it.value();
			//if (xc->PairType() == PARTICLE_CUBE)
				//pm = dynamic_cast<xParticleCubeContact*>(xc)->CubeObject();
			//else if (xc->PairType() == PARTICLE_PANE)
				//pm = dynamic_cast<xParticlePlaneContact*>(xc)->PlaneObject();
		euler_parameters ep = pe->EulerParameters();
		euler_parameters ed = pe->DEulerParameters();
		host_body_info hbi = {
			pe->Mass(),
			pe->Position().x, pe->Position().y, pe->Position().z,
			pe->Velocity().x, pe->Velocity().y, pe->Velocity().z,
			ep.e0, ep.e1, ep.e2, ep.e3,
			ed.e0, ed.e1, ed.e2, ed.e3
		};
		//}
		checkXerror(cudaMemcpy(dbi, &hbi, sizeof(device_body_info), cudaMemcpyHostToDevice));
	}

}

xPlaneObject* xParticlePlaneContact::PlaneObject()
{
	return iobj->Shape() == PLANE_SHAPE ? dynamic_cast<xPlaneObject*>(iobj) : dynamic_cast<xPlaneObject*>(jobj);
}

void xParticlePlaneContact::setPlane(xPlaneObject* _pe)
{
	//pe = _pe;
}

//void xParticlePlaneContact::save_contact_result(unsigned int pt, unsigned int np)
//{
//	if (xSimulation::Gpu())
//	{
//		checkXerror(cudaMemcpy(pair_count_ppl, d_pair_count_ppl, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
//		checkXerror(cudaMemcpy(pair_id_ppl, d_pair_id_ppl, sizeof(unsigned int) * np * MAX_P2PL_COUNT, cudaMemcpyDeviceToHost));
//		checkXerror(cudaMemcpy(tsd_ppl, d_tsd_ppl, sizeof(double2) * np * MAX_P2PL_COUNT, cudaMemcpyDeviceToHost));
//		xDynamicsManager::This()->XResult()->save_p2pl_contact_data(pair_count_ppl, pair_id_ppl, tsd_ppl);
//	}	
//}

void xParticlePlaneContact::collision(
	double *pos, double *ep, double *vel, double *ev,
	double *mass, double* inertia,
	double *force, double *moment,
	double *tmax, double* rres,
	unsigned int *sorted_id,
	unsigned int *cell_start,
	unsigned int *cell_end,
	unsigned int np)
{
	if (xSimulation::Gpu())
	{
		double fm[6] = { 0, };
		cu_plane_contact_force(dpi, dbi, dcp, pos, ep, vel, ev, force, moment, mass,
			tmax, rres, d_pair_count_ppl, d_pair_id_ppl, d_tsd_ppl, np);
		if (pe->isDynamicsBody())
		{
			fm[0] = reduction(xContact::deviceBodyForceX(), np);
			fm[1] = reduction(xContact::deviceBodyForceY(), np);
			fm[2] = reduction(xContact::deviceBodyForceZ(), np);
			fm[3] = reduction(xContact::deviceBodyMomentX(), np);
			fm[4] = reduction(xContact::deviceBodyMomentY(), np);
			fm[5] = reduction(xContact::deviceBodyMomentZ(), np);
			pe->addAxialForce(fm[0], fm[1], fm[2]);
			pe->addAxialMoment(fm[3], fm[4], fm[5]);
		}
	}	
}

//void xParticlePlaneContact::alloc_memories(unsigned int np)
//{
//	/*xContact::alloc_memories(np);
//	if (xSimulation::Gpu())
//	{
//		checkXerror(cudaMalloc((void**)&dpi, sizeof(device_plane_info)));
//		checkXerror(cudaMalloc((void**)&d_pair_count_ppl, sizeof(unsigned int) * np));
//		checkXerror(cudaMalloc((void**)&d_pair_id_ppl, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
//		checkXerror(cudaMalloc((void**)&d_tsd_ppl, sizeof(double2) * np * MAX_P2MS_COUNT));
//		checkXerror(cudaMemset(d_pair_count_ppl, 0, sizeof(unsigned int) * np));
//		checkXerror(cudaMemset(d_pair_id_ppl, 0, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
//		checkXerror(cudaMemset(d_tsd_ppl, 0, sizeof(double2) * np * MAX_P2MS_COUNT));
//		pair_count_ppl = new unsigned int[np];
//		pair_id_ppl = new unsigned int[np * MAX_P2MS_COUNT];
//		tsd_ppl = new double[2 * np * MAX_P2MS_COUNT];
//	}*/
//}

void xParticlePlaneContact::savePartData(unsigned int np)
{
	checkXerror(cudaMemcpy(pair_count_ppl, d_pair_count_ppl, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
	checkXerror(cudaMemcpy(pair_id_ppl, d_pair_id_ppl, sizeof(unsigned int) * np * MAX_P2PL_COUNT, cudaMemcpyDeviceToHost));
	checkXerror(cudaMemcpy(tsd_ppl, d_tsd_ppl, sizeof(double2) * np * MAX_P2PL_COUNT, cudaMemcpyDeviceToHost));
	xDynamicsManager::This()->XResult()->save_p2pl_contact_data(pair_count_ppl, pair_id_ppl, tsd_ppl);
}

bool xParticlePlaneContact::detect_contact(vector4f& p, xPlaneObject& pe, vector3f& cpoint)
{
	vector3d pos =
	{
		static_cast<double>(p.x),
		static_cast<double>(p.y),
		static_cast<double>(p.z)
	};
	double r = static_cast<double>(p.w);
	vector3d dp = pos - pe.XW();// xw;
	vector3d wp = new_vector3d(dot(dp, pe.U1()), dot(dp, pe.U2()), dot(dp, pe.UW()));
	vector3d u;
	double a_l1 = pow(wp.x - pe.L1(), 2.0);
	double b_l2 = pow(wp.y - pe.L2(), 2.0);
	double sqa = wp.x * wp.x;
	double sqb = wp.y * wp.y;
	double sqc = wp.z * wp.z;
	double sqr = r * r;

//	vector3d dp = pos - pe.XW();
	vector3d uu = pe.UW() / length(pe.UW());
	int pp = -xsign(dot(dp, pe.UW()));
	u = -uu;
	double collid_dist = r - abs(dot(dp, u));
	bool is_contact = collid_dist > 0;
	vector3d _cp = pos + r * u;
	cpoint = 
	{
	static_cast<float>(_cp.x),
	static_cast<float>(_cp.y),
	static_cast<float>(_cp.z)
	};
	return is_contact;
}

double xParticlePlaneContact::particle_plane_contact_detection(	xPlaneObject* _pe, vector3d& u, vector3d& xp, vector3d& wp, double r)
{
// 	double a_l1 = pow(wp.x - _pe->L1(), 2.0);
// 	double b_l2 = pow(wp.y - _pe->L2(), 2.0);
// 	double sqa = wp.x * wp.x;
// 	double sqb = wp.y * wp.y;
// 	double sqc = wp.z * wp.z;
// 	double sqr = r*r;
// 
// 	// The sphere contacts with the wall face
// 	if (abs(wp.z) < r && (wp.x > 0 && wp.x < _pe->L1()) && (wp.y > 0 && wp.y < _pe->L2())){
// 		vector3d dp = xp - _pe->XW();
// 		vector3d uu = _pe->UW() / length(_pe->UW());
// 		int pp = 0;// -sign(dot(dp, _pe->UW()));
// 		u = pp * uu;
// 		double collid_dist = r - abs(dot(dp, u));
// 		return collid_dist;
// 	}
// 
// 	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->XW();
// 		double h = length(Xsw);
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	else if (wp.x > _pe->L1() && wp.y < 0 && (a_l1 + sqb + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->W2();
// 		double h = length(Xsw);
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	else if (wp.x > _pe->L1() && wp.y > _pe->L2() && (a_l1 + b_l2 + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->W3();
// 		double h = length(Xsw);
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	else if (wp.x < 0 && wp.y > _pe->L2() && (sqa + b_l2 + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->W4();
// 		double h = length(Xsw);
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	if ((wp.x > 0 && wp.x < _pe->L1()) && wp.y < 0 && (sqb + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->XW();
// 		vector3d wj_wi = _pe->W2() - _pe->XW();
// 		vector3d us = wj_wi / length(wj_wi);
// 		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
// 		double h = length(h_star);
// 		u = -h_star / h;
// 		return r - h;
// 	}
// 	else if ((wp.x > 0 && wp.x < _pe->L1()) && wp.y > _pe->L2() && (b_l2 + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->W4();
// 		vector3d wj_wi = _pe->W3() - _pe->W4();
// 		vector3d us = wj_wi / length(wj_wi);
// 		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
// 		double h = length(h_star);
// 		u = -h_star / h;
// 		return r - h;
// 
// 	}
// 	else if ((wp.x > 0 && wp.y < _pe->L2()) && wp.x < 0 && (sqr + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->XW();
// 		vector3d wj_wi = _pe->W4() - _pe->XW();
// 		vector3d us = wj_wi / length(wj_wi);
// 		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
// 		double h = length(h_star);
// 		u = -h_star / h;
// 		return r - h;
// 	}
// 	else if ((wp.x > 0 && wp.y < _pe->L2()) && wp.x > _pe->L1() && (a_l1 + sqc) < sqr){
// 		vector3d Xsw = xp - _pe->W2();
// 		vector3d wj_wi = _pe->W3() - _pe->W2();
// 		vector3d us = wj_wi / length(wj_wi);
// 		vector3d h_star = Xsw - (dot(Xsw, us)) * us;
// 		double h = length(h_star);
// 		u = -h_star / h;
// 		return r - h;
// 	}
// 
// 
 	return -1.0f;
}
