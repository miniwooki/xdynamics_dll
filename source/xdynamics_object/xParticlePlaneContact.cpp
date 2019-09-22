#include "xdynamics_object/xParticlePlaneContact.h"
#include "xdynamics_object/xPlaneObject.h"

xParticlePlaneContact::xParticlePlaneContact()
	: xContact()
{

}

xParticlePlaneContact::xParticlePlaneContact(std::string _name)
	: xContact(_name, PARTICLE_PANE)
{

}

xParticlePlaneContact::xParticlePlaneContact(const xContact& xc)
	: xContact(xc)
{

}

xParticlePlaneContact::~xParticlePlaneContact()
{

}

xPlaneObject* xParticlePlaneContact::PlaneObject()
{
	return iobj->Shape() == PLANE_SHAPE ? dynamic_cast<xPlaneObject*>(iobj) : dynamic_cast<xPlaneObject*>(jobj);
}

void xParticlePlaneContact::setPlane(xPlaneObject* _pe)
{
	//pe = _pe;
}

void xParticlePlaneContact::cuda_collision(double *pos, double *vel, double *omega, double *mass, double *force, double *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{
	//geometry_motion_condition gmc = pe->MotionCondition();
// 	if (gmc.enable && simulation::ctime >= gmc.st)
// 	{
// 		hpi.xw = pe->XW();
// 		hpi.w2 = pe->W2();
// 		hpi.w3 = pe->W3();
// 		hpi.w4 = pe->W4();
// 		hpi.pa = pe->PA();
// 		hpi.pb = pe->PB();
// 		hpi.u1 = pe->U1();
// 		hpi.u2 = pe->U2();
// 		hpi.uw = pe->UW();
// 
// 		checkCudaErrors(cudaMemcpy(dpi, &hpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
// 	}
	//cu_plane_contact_force(1, dpi, pos, vel, omega, force, moment, mass, np, dcp);
}

// void xParticlePlaneContact::updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos, double rj = 0, vector3d posj = new_vector3d(0.0, 0.0, 0.0))
// {
// 	vector3d dp = pos - pe->XW();
// 	vector3d wp = new_vector3d(dot(dp, pe->U1()), dot(dp, pe->U2()), dot(dp, pe->UW()));
// 	vector3d u;
// 
// 	double cdist = particle_plane_contact_detection(pe, u, pos, wp, r);
// 	if (cdist > 0){
// 		xPairData pd = { PLANE_SHAPE, id, cdist, u.x, u.y, u.z };
// 		xcpl.insertPlaneContactPair(pd);
// 	}
// }

void xParticlePlaneContact::collision(
	double r, double m, vector3d& pos, vector3d& vel, vector3d& omega, vector3d& F, vector3d& M)
{
	// 	simulation::isGpu()
	// 		? cu_plane_contact_force
	// 		(
	// 		1, dpi, pos, vel, omega, force, moment, mass, np, dcp
	// 		)
	// 		: hostCollision
	// 		(
	// 		pos, vel, omega, mass, force, moment, np
	// 		);
	// 
	// 	return true;
	//singleCollision(pe, m, r, pos, vel, omega, F, M);
	// 	;// force[i] += F;
	// 	;// moment[i] += M;
}

void xParticlePlaneContact::cudaMemoryAlloc_planeObject()
{
	//device_plane_info *_dpi = new device_plane_info;
// 	hpi.l1 = pe->L1();
// 	hpi.l2 = pe->L2();
// 	hpi.xw = pe->XW();
// 	hpi.uw = pe->UW();
// 	hpi.u1 = pe->U1();
// 	hpi.u2 = pe->U2();
// 	hpi.pa = pe->PA();
// 	hpi.pb = pe->PB();
// 	hpi.w2 = pe->W2();
// 	hpi.w3 = pe->W3();
// 	hpi.w4 = pe->W4();
// 	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_plane_info)));
// 	checkCudaErrors(cudaMemcpy(dpi, &hpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
	//delete _dpi;
}

void xParticlePlaneContact::cudaMemoryAlloc(unsigned int np)
{
	xContact::cudaMemoryAlloc(np);
	cudaMemoryAlloc_planeObject();
}

bool xParticlePlaneContact::detect_contact(vector4f& p, xPlaneObject& pe)
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

void xParticlePlaneContact::singleCollision(
	xPlaneObject* _pe, double mass, double rad, vector3d& pos, vector3d& vel,
	vector3d& omega, vector3d& force, vector3d& moment)
{
// 	vector3d dp = pos - _pe->XW();
// 	vector3d wp = new_vector3d(dot(dp, _pe->U1()), dot(dp, _pe->U2()), dot(dp, _pe->UW()));
// 	vector3d u;
// 
// 	double cdist = particle_plane_contact_detection(_pe, u, pos, wp, rad);
// 	if (cdist > 0){
// 		double rcon = rad - 0.5 * cdist;
// 		vector3d cp = rcon * u;
// 		//unsigned int ci = (unsigned int)(i / particle_cluster::perCluster());
// 		//vector3d c2p = cp - ps->getParticleClusterFromParticleID(ci)->center();
// 		vector3d dv = -(vel);// +omega.cross(rad * u));
// 
// 		xContactParameters c = getContactParameters(
// 			rad, 0.0,
// 			mass, 0.0,
// 			mpp.Ei, mpp.Ej,
// 			mpp.Pri, mpp.Prj,
// 			mpp.Gi, mpp.Gj);
// 		switch (force_model)
// 		{
// 		case DHS: DHSModel(c, cdist, cp, dv, u, force, moment); break;
// 		}
// 	}
}