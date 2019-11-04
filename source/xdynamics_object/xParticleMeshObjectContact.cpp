#include "xdynamics_object/xParticleMeshObjectContact.h"
#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_object/xMeshObject.h"

// int xParticleMeshObjectContact::nmoving = 0;
double* xParticlePlaneContact::d_tsd_ppl = nullptr;
unsigned int* xParticlePlaneContact::d_pair_count_ppl = nullptr;
unsigned int* xParticlePlaneContact::d_pair_id_ppl = nullptr;

double* xParticlePlaneContact::tsd_ppl = nullptr;
unsigned int* xParticlePlaneContact::pair_count_ppl = nullptr;
unsigned int* xParticlePlaneContact::pair_id_ppl = nullptr;

unsigned int xParticleMeshObjectContact::n_mesh_sphere = 0;
//double xParticleMeshObjectContact::max_sphere_radius = 0;

xParticleMeshObjectContact::xParticleMeshObjectContact()
	: xContact()
	, id(0)
	, p(nullptr)
	, po(nullptr)
	, hsphere(nullptr)
	, hlocal(nullptr)
	, allocated_static(false)
{

}

xParticleMeshObjectContact::xParticleMeshObjectContact(std::string _name, xObject* o1, xObject* o2)
	: xContact(_name, PARTICLE_MESH_SHAPE)
	, id(0)
	, p(nullptr)
	, po(nullptr)
	, hsphere(nullptr)
	, hlocal(nullptr)
	, allocated_static(false)
{
	if (o1->Shape() == MESH_SHAPE)
	{
		po = dynamic_cast<xMeshObject*>(o1);
		p = dynamic_cast<xParticleObject*>(o2);
	}
	else
	{
		po = dynamic_cast<xMeshObject*>(o2);
		p = dynamic_cast<xParticleObject*>(o1);
	}
	po->splitTriangles(po->RefinementSize());
}

xParticleMeshObjectContact::~xParticleMeshObjectContact()
{
	if (d_pair_count_ptri) checkCudaErrors(cudaFree(d_pair_count_ptri)); d_pair_count_ptri = NULL;
	if (d_pair_id_ptri) checkCudaErrors(cudaFree(d_pair_id_ptri)); d_pair_id_ptri = NULL;
	if (d_tsd_ptri) checkCudaErrors(cudaFree(d_tsd_ptri)); d_tsd_ptri = NULL;
	if (dsphere) checkXerror(cudaFree(dsphere)); dsphere = NULL;
	if (dlocal) checkXerror(cudaFree(dlocal)); dlocal = NULL;
	if (dvList) checkXerror(cudaFree(dvList)); dvList = NULL;
	if (dti) checkXerror(cudaFree(dti)); dti = NULL;
	if (dbi) checkXerror(cudaFree(dbi)); dbi = NULL;

	if (pair_count_ptri) delete[] pair_count_ptri; pair_count_ptri = NULL;
	if (pair_id_ptri) delete[] pair_id_ptri; pair_id_ptri = NULL;
	if (tsd_ptri) delete[] tsd_ptri; tsd_ptri = NULL;

	if (hsphere) delete[] hsphere; hsphere = nullptr;
	if (hlocal) delete[] hlocal; hlocal = nullptr;
}

void xParticleMeshObjectContact::define(unsigned int idx, unsigned int np)
{
	xContact::define(idx, np);
	
	n_mesh_sphere += po->NumTriangle();
	hsphere = new vector4d[po->NumTriangle()];
	hlocal = new vector3d[po->NumTriangle()];
	host_triangle_info* hti = new host_triangle_info[po->NumTriangle()];
	//xContactMaterialParameters hcp = { 0, };
	//xMaterialPair hmp = { 0, };

	double maxRadii = 0.0;
	unsigned int idx = 0;

	xContactMaterialParameters cp = { 0, };
	xMaterialPair xmp = { 0, };
	vector3d *vList = (vector3d *)po->VertexList();

	//hcp.restitution = this->Restitution();
	//hcp.stiffness_ratio = this->StiffnessRatio();
	//hcp.friction = this->Friction();
	//hcp.s_friction = this->StaticFriction();
	//hcp.rolling_friction = this->RollingFactor();
	//hcp.stiffness_multiplyer = this->StiffMultiplyer();
	//hmp = this->MaterialPropertyPair();

	unsigned int vi = 0;

	double* t_radius = new double[po->NumTriangle()];
	for (unsigned int i = 0; i < po->NumTriangle(); i++)
	{
		hti[i].id = i;
		hti[i].sid = 0;
		vector3d pos = po->Position();
		euler_parameters ep = po->EulerParameters();
		//unsigned int s = vi * 9;
		vector3d p = pos + ToGlobal(ep, vList[vi++]);
		vector3d q = pos + ToGlobal(ep, vList[vi++]);
		vector3d r = pos + ToGlobal(ep, vList[vi++]);
		hti[i].P = new_vector3d(p.x, p.y, p.z);
		hti[i].Q = new_vector3d(q.x, q.y, q.z);
		hti[i].R = new_vector3d(r.x, r.y, r.z);
		//hpi[i].indice.z = vi++;
		vector4d csph = xUtilityFunctions::FitSphereToTriangle(p, q, r, 0.8);
		//double rad = length(ctri - hpi[i].P);
		if (csph.w > maxRadii)
			maxRadii = csph.w;
		hsphere[i] = csph;// new_vector4d(ctri.x, ctri.y, ctri.z, rad);
		vector3d r_pos = ToLocal(ep, (new_vector3d(csph.x - pos.x, csph.y - pos.y, csph.z - pos.z)));
		hlocal[i] = r_pos;
		t_radius[i] = csph.w;
	}
	delete[] t_radius;

	if (xSimulation::Cpu())
		dsphere = (double *)hsphere;
	else
	{
		checkXerror(cudaMalloc((void**)&dsphere, sizeof(double) * po->NumTriangle() * 4));
		checkXerror(cudaMalloc((void**)&dlocal, sizeof(double) * po->NumTriangle() * 3));
		checkXerror(cudaMalloc((void**)&dvList, sizeof(double) * po->NumTriangle() * 9));
		checkXerror(cudaMalloc((void**)&dti, sizeof(device_triangle_info) * po->NumTriangle()));
		checkXerror(cudaMalloc((void**)&dbi, sizeof(device_body_info)));
		checkXerror(cudaMemcpy(dsphere, hsphere, sizeof(double) * po->NumTriangle() * 4, cudaMemcpyHostToDevice));
		checkXerror(cudaMemcpy(dti, hti, sizeof(device_triangle_info) * po->NumTriangle(), cudaMemcpyHostToDevice));
		checkXerror(cudaMemcpy(dlocal, hlocal, sizeof(vector3d) * po->NumTriangle(), cudaMemcpyHostToDevice));
		checkXerror(cudaMemcpy(dvList, vList, sizeof(double) * po->NumTriangle() * 9, cudaMemcpyHostToDevice));
		if (!allocated_static)
		{
			checkXerror(cudaMalloc((void**)&d_pair_count_ptri, sizeof(unsigned int) * np));
			checkXerror(cudaMalloc((void**)&d_pair_id_ptri, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
			checkXerror(cudaMalloc((void**)&d_tsd_ptri, sizeof(double2) * np * MAX_P2MS_COUNT));
			checkXerror(cudaMemset(d_pair_count_ptri, 0, sizeof(unsigned int) * np));
			checkXerror(cudaMemset(d_pair_id_ptri, 0, sizeof(unsigned int) * np * MAX_P2MS_COUNT));
			checkXerror(cudaMemset(d_tsd_ptri, 0, sizeof(double2) * np * MAX_P2MS_COUNT));
			pair_count_ptri = new unsigned int[np];
			pair_id_ptri = new unsigned int[np * MAX_P2MS_COUNT];
			tsd_ptri = new double[2 * np * MAX_P2MS_COUNT];
			allocated_static;
		}		
	}
	update();
	gps.max_radius = maxRadii;
}

void xParticleMeshObjectContact::update()
{
	if (xSimulation::Gpu())
	{
		euler_parameters ep = po->EulerParameters();
		euler_parameters ed = po->DEulerParameters();
		device_body_info bi = 
		{
			po->Mass(),
			po->Position().x, po->Position().y, po->Position().z,
			po->Velocity().x, po->Velocity().y, po->Velocity().z,
			ep.e0, ep.e1, ep.e2, ep.e3,
			ed.e0, ed.e1, ed.e2, ed.e3
		};
		checkXerror(cudaMemcpy(dbi, &bi, sizeof(device_body_info), cudaMemcpyHostToDevice));
		cu_update_meshObjectData(dvList, dsphere, dlocal, dti, dbi, n_mesh_sphere);
	}
}

double * xParticleMeshObjectContact::MeshSphere()
{
	return xSimulation::Gpu() ? dsphere : (double*)hsphere;
}

void xParticleMeshObjectContact::collision(
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
		cu_particle_polygonObject_collision(
			dti, dbi, pos, ep, vel, ev, force, moment, mass,
			tmax, rres, d_pair_count_ptri, d_pair_id_ptri, d_tsd_ptri,
			dsphere, sorted_id, cell_start, cell_end, dcp, np);
		if (po->isDynamicsBody())
		{
			fm[0] = reduction(xContact::deviceBodyForceX(), np);
			fm[1] = reduction(xContact::deviceBodyForceY(), np);
			fm[2] = reduction(xContact::deviceBodyForceZ(), np);
			fm[3] = reduction(xContact::deviceBodyMomentX(), np);
			fm[4] = reduction(xContact::deviceBodyMomentY(), np);
			fm[5] = reduction(xContact::deviceBodyMomentZ(), np);
			po->addAxialForce(fm[0], fm[1], fm[2]);
			po->addAxialMoment(fm[3], fm[4], fm[5]);
		}		
	}
}

void xParticleMeshObjectContact::savePartData(unsigned int np)
{
	if (xSimulation::Gpu())
	{
		checkXerror(cudaMemcpy(pair_count_ptri, d_pair_count_ptri, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
		checkXerror(cudaMemcpy(pair_id_ptri, d_pair_id_ptri, sizeof(unsigned int) * np * MAX_P2MS_COUNT, cudaMemcpyDeviceToHost));
		checkXerror(cudaMemcpy(tsd_ptri, d_tsd_ptri, sizeof(double2) * np * MAX_P2MS_COUNT, cudaMemcpyDeviceToHost));
		xDynamicsManager::This()->XResult()->save_p2tri_contact_data(pair_count_ptri, pair_id_ptri, tsd_ptri);
	}
}

unsigned int xParticleMeshObjectContact::GetNumMeshSphere()
{
	return n_mesh_sphere;
}

// void contact_particles_polygonObject::allocPolygonInformation(unsigned int _nPolySphere)
// {
// //	nPolySphere = _nPolySphere;
// 	//hsphere = new VEC4D[nPolySphere];
// //	hpi = new host_polygon_info[nPolySphere];
// }
// 
// void contact_particles_polygonObject::definePolygonInformation(
// 	unsigned int id, unsigned int bPolySphere, 
// 	unsigned int ePolySphere, double *vList, unsigned int *iList)
// {
// // 	unsigned int a, b, c;
// // 	maxRadii = 0;
// // 	for (unsigned int i = bPolySphere; i < bPolySphere + ePolySphere; i++)
// // 	{
// // 		a = iList[i * 3 + 0];
// // 		b = iList[i * 3 + 1];
// // 		c = iList[i * 3 + 2];
// // 		host_polygon_info po;
// // 		po.id = id;
// // 		po.P = VEC3D(vList[a * 3 + 0], vList[a * 3 + 1], vList[a * 3 + 2]);
// // 		po.Q = VEC3D(vList[b * 3 + 0], vList[b * 3 + 1], vList[b * 3 + 2]);
// // 		po.R = VEC3D(vList[c * 3 + 0], vList[c * 3 + 1], vList[c * 3 + 2]);
// // 		po.V = po.Q - po.P;
// // 		po.W = po.R - po.P;
// // 		po.N = po.V.cross(po.W);
// // 		po.N = po.N / po.N.length();
// // 		hpi[i] = po;
// // 		VEC3D M1 = (po.Q + po.P) / 2;
// // 		VEC3D M2 = (po.R + po.P) / 2;
// // 		VEC3D D1 = po.N.cross(po.V);
// // 		VEC3D D2 = po.N.cross(po.W);
// // 		double t = 0;
// // 		if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
// // 		{
// // 			t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
// // 		}
// // 		else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
// // 		{
// // 			t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
// // 		}
// // 		else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
// // 		{
// // 			t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
// // 		}
// // 		VEC3D Ctri = M1 + t * D1;
// // 		VEC4D sph;
// // 		sph.w = (Ctri - po.P).length();
// // 		sph.x = Ctri.x; sph.y = Ctri.y; sph.z = Ctri.z;
// // 		//com += Ctri;
// // 		// 		while (abs(fc - ft) > 0.00001)
// // 		// 		{
// // 		// 			d = ft * sph.w;
// // 		// 			double p = d / po.N.length();
// // 		// 			VEC3D _c = Ctri - p * po.N;
// // 		// 			sph.x = _c.x; sph.y = _c.y; sph.z = _c.z;
// // 		// 			sph.w = (_c - po.P).length();
// // 		// 			fc = d / sph.w;
// // 		// 		}
// // 		if (sph.w > maxRadii)
// // 			maxRadii = sph.w;
// // 		hsphere[i] = sph;
// // 	}
// //	com = com / ntriangle;
// }

// bool contact_particles_polygonObject::hostCollision(
// 	double *m_pos, double *m_vel, 
// 	double *m_omega, double *m_mass, 
// 	double *m_force, double *m_moment, 
// 	unsigned int *sorted_id, unsigned int *cell_start, 
// 	unsigned int *cell_end, unsigned int np)
// {
// // 	unsigned int _np = 0;
// // 	VEC3I neigh, gp;
// // 	double dist, cdist, mag_e, ds;
// // 	unsigned int hash, sid, eid;
// // 	contactParameters c;
// // 	VEC3D ipos, jpos, rp, u, rv, Fn, Ft, e, sh, M;
// // 	VEC4D *pos = (VEC4D*)m_pos;
// // 	VEC3D *vel = (VEC3D*)m_vel;
// // 	VEC3D *omega = (VEC3D*)m_omega;
// // 	VEC3D *fr = (VEC3D*)m_force;
// // 	VEC3D *mm = (VEC3D*)m_moment;
// // 	double* ms = m_mass;
// // 	double dt = simulation::ctime;
// // 	for (unsigned int i = 0; i < np; i++){
// // 		ipos = VEC3D(pos[i].x, pos[i].y, pos[i].z);
// // 		gp = grid_base::getCellNumber(pos[i].x, pos[i].y, pos[i].z);
// // 		for (int z = -1; z <= 1; z++){
// // 			for (int y = -1; y <= 1; y++){
// // 				for (int x = -1; x <= 1; x++){
// // 					neigh = VEC3I(gp.x + x, gp.y + y, gp.z + z);
// // 					hash = grid_base::getHash(neigh);
// // 					sid = cell_start[hash];
// // 					if (sid != 0xffffffff){
// // 						eid = cell_end[hash];
// // 						for (unsigned int j = sid; j < eid; j++){
// // 							unsigned int k = sorted_id[j];
// // 							if (i == k || k >= np)
// // 								continue;
// // 							jpos = VEC3D(pos[k].x, pos[k].y, pos[k].z);// toVector3();
// // 							rp = jpos - ipos;
// // 							dist = rp.length();
// // 							cdist = (pos[i].w + pos[k].w) - dist;
// // 							//double rcon = pos[i].w - cdist;
// // 							unsigned int rid = 0;
// // 							if (cdist > 0){
// // 								u = rp / dist;
// // 								VEC3D cp = ipos + pos[i].w * u;
// // 								//unsigned int ci = (unsigned int)(i / particle_cluster::perCluster());
// // 								//VEC3D c2p = cp - ps->getParticleClusterFromParticleID(ci)->center();
// // 								//double rcon = pos[i].w - 0.5 * cdist;
// // 								rv = vel[k] + omega[k].cross(-pos[k].w * u) - (vel[i] + omega[i].cross(pos[i].w * u));
// // 								c = getContactParameters(
// // 									pos[i].w, pos[k].w,
// // 									ms[i], ms[k],
// // 									mpp.Ei, mpp.Ej,
// // 									mpp.pri, mpp.prj,
// // 									mpp.Gi, mpp.Gj);
// // 								switch (f_type)
// // 								{
// // 								case DHS: DHSModel(c, cdist, cp, rv, u, Fn, Ft); break;
// // 								}
// // 
// // 								fr[i] += Fn/* + Ft*/;
// // 								mm[i] += M;
// // 							}
// // 						}
// // 					}
// // 				}
// // 			}
// // 		}
// // 	}
// 	return true;
// }
// 
// collision_particles_polygonObject::collision_particles_polygonObject()
// {
// 
// }
// 
// collision_particles_polygonObject::collision_particles_polygonObject(
// 	QString& _name, 
// 	modeler* _md, 
// 	particle_system *_ps, 
// 	polygonObject * _poly, 
// 	tContactModel _tcm)
// 	: collision(_name, _md, _ps->name(), _poly->objectName(), PARTICLES_POLYGONOBJECT, _tcm)
// 	, ps(_ps)
// 	, po(_poly)
// {
// 
// }
// 
// collision_particles_polygonObject::~collision_particles_polygonObject()
// {
// 
// }
// 
// bool collision_particles_polygonObject::collid(double dt)
// {
// 	return true;
// }
// 
// bool collision_particles_polygonObject::cuCollid(
// 	double *dpos, double *dvel,
// 	double *domega, double *dmass,
// 	double *dforce, double *dmoment, unsigned int np)
// {
// 	double3 *mforce;
// 	double3 *mmoment;
// 	double3 *mpos;
// 	VEC3D _mp;
// 	double3 _mf = make_double3(0.0, 0.0, 0.0);
// 	double3 _mm = make_double3(0.0, 0.0, 0.0);
// 	if (po->pointMass())
// 		_mp = po->pointMass()->Position();
// 	checkCudaErrors(cudaMalloc((void**)&mpos, sizeof(double3)));
// 	checkCudaErrors(cudaMemcpy(mpos, &_mp, sizeof(double3), cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&mforce, sizeof(double3)*ps->numParticle()));
// 	checkCudaErrors(cudaMalloc((void**)&mmoment, sizeof(double3)*ps->numParticle()));
// 	checkCudaErrors(cudaMemset(mforce, 0, sizeof(double3)*ps->numParticle()));
// 	checkCudaErrors(cudaMemset(mmoment, 0, sizeof(double3)*ps->numParticle()));
// 
// 	switch (tcm)
// 	{
// 	case HMCM: 
// 		cu_particle_polygonObject_collision(
// 			0, po->devicePolygonInfo(), po->deviceSphereSet(), po->deviceMassInfo(), 
// 			dpos, dvel, domega,
// 			dforce, dmoment, dmass,
// 			gb->cuSortedID(), gb->cuCellStart(), gb->cuCellEnd(), dcp, 
// 			ps->numParticle(), mpos, mforce, mmoment, _mf, _mm); 
// 		break;
// 	}
// 	
// 	_mf = reductionD3(mforce, ps->numParticle());
// 	if (po->pointMass()){
// 		po->pointMass()->addCollisionForce(VEC3D(_mf.x, _mf.y, _mf.z));
// 	}
// 	_mm = reductionD3(mmoment, ps->numParticle());
// 	if (po->pointMass()){
// 		po->pointMass()->addCollisionMoment(VEC3D(_mm.x, _mm.y, _mm.z));
// 	}
// 	checkCudaErrors(cudaFree(mforce)); mforce = NULL;
// 	checkCudaErrors(cudaFree(mmoment)); mmoment = NULL;
// 	checkCudaErrors(cudaFree(mpos)); mpos = NULL;
// 	return true;
// }
// 
// VEC3D collision_particles_polygonObject::particle_polygon_contact_detection(host_polygon_info& hpi, VEC3D& p, double pr)
// {
// 	VEC3D a = hpi.P.To<double>();
// 	VEC3D b = hpi.Q.To<double>();
// 	VEC3D c = hpi.R.To<double>();
// 	VEC3D ab = b - a;
// 	VEC3D ac = c - a;
// 	VEC3D ap = p - a;
// 
// 	double d1 = ab.dot(ap);// dot(ab, ap);
// 	double d2 = ac.dot(ap);// dot(ac, ap);
// 	if (d1 <= 0.0 && d2 <= 0.0){
// 		//	*wc = 0;
// 		return a;
// 	}
// 
// 	VEC3D bp = p - b;
// 	double d3 = ab.dot(bp);
// 	double d4 = ac.dot(bp);
// 	if (d3 >= 0.0 && d4 <= d3){
// 		//	*wc = 0;
// 		return b;
// 	}
// 	double vc = d1 * d4 - d3 * d2;
// 	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){
// 		//	*wc = 1;
// 		double v = d1 / (d1 - d3);
// 		return a + v * ab;
// 	}
// 
// 	VEC3D cp = p - c;
// 	double d5 = ab.dot(cp);
// 	double d6 = ac.dot(cp);
// 	if (d6 >= 0.0 && d5 <= d6){
// 		//	*wc = 0;
// 		return c;
// 	}
// 
// 	double vb = d5 * d2 - d1 * d6;
// 	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){
// 		//	*wc = 1;
// 		double w = d2 / (d2 - d6);
// 		return a + w * ac; // barycentric coordinates (1-w, 0, w)
// 	}
// 
// 	// Check if P in edge region of BC, if so return projection of P onto BC
// 	double va = d3 * d6 - d5 * d4;
// 	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){
// 		//	*wc = 1;
// 		double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
// 		return b + w * (c - b); // barycentric coordinates (0, 1-w, w)
// 	}
// 	//*wc = 2;
// 	// P inside face region. Compute Q through its barycentric coordinates (u, v, w)
// 	double denom = 1.0 / (va + vb + vc);
// 	double v = vb * denom;
// 	double w = vc * denom;
// 
// 	return a + v * ab + w * ac; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
// 	//return 0.f;
// }
// 
// bool collision_particles_polygonObject::collid_with_particle(unsigned int i, double dt)
// {
// 	double overlap = 0.f;
// 	VEC4D ipos = ps->position()[i];
// 	VEC3D ivel = ps->velocity()[i];
// 	VEC3D iomega = ps->angVelocity()[i];
// 	double ir = ipos.w;
// 	VEC3D m_moment = 0.f;
// 	VEC3I neighbour_pos = 0;
// 	unsigned int grid_hash = 0;
// 	VEC3D single_force = 0.f;
// 	VEC3D shear_force = 0.f;
// 	VEC3I gridPos = gb->getCellNumber(ipos.x, ipos.y, ipos.z);
// 	unsigned int sindex = 0;
// 	unsigned int eindex = 0;
// 	VEC3D ip = VEC3D(ipos.x, ipos.y, ipos.z);
// 	double ms = ps->mass()[i];
// 	unsigned int np = md->numParticle();
// 	for (int z = -1; z <= 1; z++){
// 		for (int y = -1; y <= 1; y++){
// 			for (int x = -1; x <= 1; x++){
// 				neighbour_pos = VEC3I(gridPos.x + x, gridPos.y + y, gridPos.z + z);
// 				grid_hash = gb->getHash(neighbour_pos);
// 				sindex = gb->cellStart(grid_hash);
// 				if (sindex != 0xffffffff){
// 					eindex = gb->cellEnd(grid_hash);
// 					for (unsigned int j = sindex; j < eindex; j++){
// 						unsigned int k = gb->sortedID(j);
// 						if (k >= np)
// 						{
// 							k -= np;
// 							VEC3D cp = particle_polygon_contact_detection(po->hostPolygonInfo()[k], ip, ir);
// 							VEC3D distVec = ip - cp;
// 							double dist = distVec.length();
// 							overlap = ir - dist;
// 							if (overlap > 0)
// 							{
// 								VEC3D unit = -po->hostPolygonInfo()[k].N;
// 								VEC3D dv = -(ivel + iomega.cross(ir * unit));
// 								constant c = getConstant(ir, 0, ms, 0, ps->youngs(), po->youngs(), ps->poisson(), po->poisson(), ps->shear(), po->shear());
// 								double fsn = -c.kn * pow(overlap, 1.5);
// 								single_force = (fsn + c.vn * dv.dot(unit)) * unit;
// 								//std::cout << k << ", " << single_force.x << ", " << single_force.y << ", " << single_force.z << std::endl;
// 								VEC3D e = dv - dv.dot(unit) * unit;
// 								double mag_e = e.length();
// 								if (mag_e){
// 									VEC3D s_hat = e / mag_e;
// 									double ds = mag_e * dt;
// 									shear_force = min(c.ks * ds + c.vs * dv.dot(s_hat), c.mu * single_force.length()) * s_hat;
// 									m_moment = (ir*unit).cross(shear_force);
// 								}
// 								ps->force()[i] += single_force;
// 								ps->moment()[i] += m_moment;
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	return true;
// }