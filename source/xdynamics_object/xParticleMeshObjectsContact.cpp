#include "xdynamics_object/xParticleMeshObjectsContact.h"
#include "xdynamics_manager/xModel.h"
#include "xdynamics_simulation/xSimulation.h"

xParticleMeshObjectsContact::xParticleMeshObjectsContact()
	: xContact()
	, hsphere(NULL)
	, dsphere(NULL)
	, hpi(NULL)
	, dpi(NULL)
	, hcp(NULL)
	, nPobjs(NULL)
	, dvList(NULL)
	, diList(NULL)
	, hpmi(NULL)
	, dpmi(NULL)
	, maxRadius(0)
	, npolySphere(0)
	, ncontact(0)
{

}

xParticleMeshObjectsContact::~xParticleMeshObjectsContact()
{
	if (hsphere) delete[] hsphere; hsphere = NULL;
	if (hpi) delete[] hpi; hpi = NULL;
	if (hcp) delete[] hcp; hcp = NULL;
	if (xmps) delete[] xmps; xmps = NULL;
	if (hpmi) delete[] hpmi; hpmi = NULL;
	//qDeleteAll(pair_ip);
	if (dsphere) checkCudaErrors(cudaFree(dsphere)); dsphere = NULL;
	if (dpi) checkCudaErrors(cudaFree(dpi)); dpi = NULL;
	if (dvList) checkCudaErrors(cudaFree(dvList)); dvList = NULL;
	if (diList) checkCudaErrors(cudaFree(diList)); diList = NULL;
	if (dpmi) checkCudaErrors(cudaFree(dpmi)); dpmi = NULL;
}

unsigned int xParticleMeshObjectsContact::define(QMap<QString, xParticleMeshObjectContact*>& cpmesh)
{
	foreach(xParticleMeshObjectContact* cpm, cpmesh)
	{
		xMeshObject* pobj = cpm->MeshObject();
		npolySphere += pobj->NumTriangle();
	}
	nPobjs = cpmesh.size();
	hsphere = new vector4d[npolySphere];
	hpi = new host_mesh_info[npolySphere];
	hcp = new xContactMaterialParameters[nPobjs];
	xmps = new xMaterialPair[nPobjs];
	hpmi = new host_mesh_mass_info[nPobjs];
	// 	if (simulation::isCpu())
	// 		dsphere = (double*)hsphere;
// 	if (!pct)
// 	{
// 		pct = new polygonContactType[nPobjs];
// 		memset(pct, 0, sizeof(polygonContactType) * nPobjs);
// 	}
	unsigned int bPolySphere = 0;
	unsigned int ePolySphere = 0;
	double maxRadii = 0.0;
	unsigned int idx = 0;
	foreach(xParticleMeshObjectContact* cpm, cpmesh)
	{
		xContactMaterialParameters cp = { 0, };
		xMaterialPair xmp = { 0, };
		xMeshObject* pobj = cpm->MeshObject();
		vector3d *vList = (vector3d *)pobj->VertexList();
//		unsigned int a, b, c;
		cp.restitution = cpm->Restitution();
		cp.stiffness_ratio = cpm->StiffnessRatio();
		cp.friction = cpm->Friction();
		xmp = cpm->MaterialPropertyPair();
		hcp[idx] = cp;
		xmps[idx] = xmp;
		pair_ip[idx] = pobj;
		ePolySphere += pobj->NumTriangle();
		unsigned int vi = 0;
		for (unsigned int i = bPolySphere; i < ePolySphere; i++)
		{
//			host_mesh_info po;
			hpi[i].id = idx;
			vector3d pos = pobj->Position();
			euler_parameters ep = pobj->EulerParameters();
			//unsigned int s = vi * 9;
			hpi[i].P = pos + ToGlobal(ep, vList[vi++]);
			hpi[i].Q = pos + ToGlobal(ep, vList[vi++]);
			hpi[i].R = pos + ToGlobal(ep, vList[vi++]);
			
			vector3d ctri = xUtilityFunctions::CenterOfTriangle(hpi[i].P, hpi[i].Q, hpi[i].R);
			double rad = length(ctri - hpi[i].P);
			if (rad > maxRadii)
				maxRadii = rad;
			hsphere[i] = new_vector4d(ctri.x, ctri.y, ctri.z, rad);
			//hpi[i] = po;
			//vi++;
		}
		bPolySphere += pobj->NumTriangle();
		idx++;
	}
	maxRadius = maxRadii;
	return npolySphere;
}

bool xParticleMeshObjectsContact::cppolyCollision(
	unsigned int idx, double r, double m,
	vector3d& p, vector3d& v, vector3d& o, vector3d& F, vector3d& M)
{
	unsigned int ct = 0;
	unsigned int nc = 0;
	unsigned int i = hpi[idx].id;
	// 	if (pct[i])
	// 		return false;
	xPointMass* pm = pair_ip[i];
	xMaterialPair mat = xmps[i];
	xContactMaterialParameters cpa = hcp[i];
	//polygonContactType _pct;
	vector3d u;
	restitution = cpa.restitution;
	friction = cpa.friction;
	stiffnessRatio = cpa.stiffness_ratio;
	vector3d mp = pm->Position();
	vector3d mv = pm->Velocity();
	vector3d mo = pm->AngularVelocity();
	vector3d cpt = particle_polygon_contact_detection(hpi[idx], p, r);
	vector3d po2cp = cpt - mp;
	vector3d distVec = p - cpt;
	double dist = length(distVec);
	double cdist = r - dist;
	vector3d m_f, m_m;
	if (cdist > 0)
	{
		ncontact++;
		vector3d qp = hpi[idx].Q - hpi[idx].P;
		vector3d rp = hpi[idx].R - hpi[idx].P;
		u = -cross(qp, rp);
		u = u / length(u);// .length();
		double rcon = r - 0.5 * cdist;
		vector3d cp = rcon * u;
		vector3d dv = mv + cross(mo, po2cp) - (v + cross(o, r * u));
		xContactParameters c = getContactParameters(
			r, 0.0,
			m, 0.0,
			mat.Ei, mat.Ej,
			mat.Pri, mat.Prj,
			mat.Gi, mat.Gj);
// 		switch (force_model)
// 		{
// 		//case DHS: DHSModel(c, cdist, cp, dv, u, m_f, m_m); break;
// 		}

		F += m_f;
		M += m_m;
		pm->addContactForce(-m_f.x, -m_f.y, -m_f.z);
		vector3d m_mm = -cross(po2cp, m_f);
		pm->addContactMoment(m_mm.x, m_mm.y, m_mm.z);
		return true;
	}
	return false;
}

void xParticleMeshObjectsContact::updateMeshObjectData(xVectorD& q, xVectorD& qd)
{
	unsigned int bPolySphere = 0;
	unsigned int ePolySphere = 0;
	//hpmi = new device_po_mass_info[nPobjs];
	//device_polygon_mass_info* dpmi = NULL;
	//checkCudaErrors(cudaMalloc((void**)&dpmi, sizeof(device_polygon_mass_info) * nPobjs));
	QMapIterator<unsigned int, xMeshObject*> po(pair_ip);
	while (po.hasNext())
	{
		po.next();
		unsigned int id = po.key();
		xMeshObject* mesh = po.value();
		unsigned int xid = mesh->xpmIndex() * xModel::OneDOF();
		//xMeshObject* p = po.value();
		vector3d pos = new_vector3d(q(xid + 0), q(xid + 1), q(xid + 2));
		euler_parameters ep = new_euler_parameters(q(xid + 3), q(xid + 4), q(xid + 5), q(xid + 6));
		vector3d vel = new_vector3d(qd(xid + 0), qd(xid + 1), qd(xid + 2));// p->Velocity();
		euler_parameters ev = new_euler_parameters(qd(xid + 3), qd(xid + 4), qd(xid + 5), qd(xid + 6));
		vector3d omega = 2.0 * GMatrix(ep) * ev;
		hpmi[id] =
		{
			pos.x, pos.y, pos.z,
			ep.e0, ep.e1, ep.e2, ep.e3,
			vel.x, vel.y, vel.z,
			omega.x, omega.y, omega.z,
			0.0, 0.0, 0.0,
			0.0, 0.0, 0.0
		};
	}
	if (xSimulation::Gpu())
	{
		checkCudaErrors(cudaMemcpy(dpmi, hpmi, sizeof(device_mesh_mass_info) * nPobjs, cudaMemcpyHostToDevice));
		foreach(xMeshObject* pobj, pair_ip)
		{
			ePolySphere += pobj->NumTriangle();
			cu_update_meshObjectData(dpmi, dvList, dsphere, dpi, ePolySphere - bPolySphere);
			bPolySphere += ePolySphere;
		}
	}
	else
	{
		foreach(xMeshObject* mesh, pair_ip)
		{
			//xMeshObject* pobj = cpm->MeshObject();
			vector3d *vList = (vector3d *)mesh->VertexList();
//			unsigned int a, b, c;
			ePolySphere += mesh->NumTriangle();
			unsigned int vi = 0;
			for (unsigned int i = bPolySphere; i < ePolySphere; i++)
			{
				//host_mesh_info po;
				//hpi[i].id = idx;
				//unsigned int s = vi * 9;
				vector3d pos = hpmi[hpi[i].id].pos;
				euler_parameters ep = hpmi[hpi[i].id].ep;
				hpi[i].P = pos + ToGlobal(ep, vList[vi++]);
				hpi[i].Q = pos + ToGlobal(ep, vList[vi++]);
				hpi[i].R = pos + ToGlobal(ep, vList[vi++]);

				vector3d ctri = xUtilityFunctions::CenterOfTriangle(hpi[i].P, hpi[i].Q, hpi[i].R);
				double rad = length(ctri - hpi[i].P);
// 				if (rad > maxRadii)
// 					maxRadii = rad;
				hsphere[i] = new_vector4d(ctri.x, ctri.y, ctri.z, rad);
				//hpi[i] = po;
				//vi++;
			}
			bPolySphere += mesh->NumTriangle();
		}
		
	}
}

void xParticleMeshObjectsContact::updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos)
{

}

void xParticleMeshObjectsContact::cudaMemoryAlloc(unsigned int np)
{
	device_contact_property *_hcp = new device_contact_property[nPobjs];
	for (unsigned int i = 0; i < nPobjs; i++)
	{
		_hcp[i] = { xmps[i].Ei, xmps[i].Ej, xmps[i].Pri, xmps[i].Prj, xmps[i].Gi, xmps[i].Gj,
			hcp[i].restitution, hcp[i].friction, hcp[i].rolling_friction, hcp[i].cohesion, hcp[i].stiffness_ratio };
	}
	checkCudaErrors(cudaMalloc((void**)&dsphere, sizeof(double) * npolySphere * 4));
	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_mesh_info) * npolySphere));
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property) * nPobjs));
	checkCudaErrors(cudaMalloc((void**)&dvList, sizeof(double) * npolySphere * 9));
	checkCudaErrors(cudaMalloc((void**)&dpmi, sizeof(device_mesh_mass_info) * nPobjs));
	//checkCudaErrors(cduaMalloc((void**)&diList, sizeof(unsigned int) * ))
	checkCudaErrors(cudaMemcpy(dsphere, hsphere, sizeof(double) * npolySphere * 4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dpi, hpi, sizeof(device_mesh_info) * npolySphere, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dcp, _hcp, sizeof(device_contact_property) * nPobjs, cudaMemcpyHostToDevice));
	unsigned int bPolySphere = 0;
	unsigned int ePolySphere = 0;
	foreach(xMeshObject* pobj, pair_ip)
	{
		//polygonObject* pobj = cppo->PolygonObject();
		double *vList = pobj->VertexList();
		ePolySphere += pobj->NumTriangle();
		//unsigned int *iList = pobj->IndexList();
		checkCudaErrors(cudaMemcpy(dvList + bPolySphere * 9, vList, sizeof(double) * pobj->NumTriangle() * 9, cudaMemcpyHostToDevice));
		bPolySphere += ePolySphere;
	}
	delete[] _hcp;
}

void xParticleMeshObjectsContact::cuda_collision(
	double *pos, double *vel, double *omega,
	double *mass, double *force, double *moment,
	unsigned int *sorted_id, unsigned int *cell_start,
	unsigned int *cell_end, unsigned int np)
{
	QMapIterator<unsigned int, xMeshObject*> po(pair_ip);
	cu_particle_meshObject_collision(1, dpi, dsphere, dpmi, pos, vel, omega, force, moment, mass, sorted_id, cell_start, cell_end, dcp, np);
	checkCudaErrors(cudaMemcpy(hpmi, dpmi, sizeof(device_mesh_mass_info) * nPobjs, cudaMemcpyDeviceToHost));
	//	po.toFront();
	while (po.hasNext())
	{
		po.next();
		unsigned int id = po.key();
		xPointMass* p = po.value();
		p->addContactForce(hpmi[id].force.x, hpmi[id].force.y, hpmi[id].force.z);
		//p->setCollisionMoment(VEC3D(hpmi[id].moment.x, hpmi[id].moment.y, hpmi[id].moment.z));
	}
	//	checkCudaErrors(cudaFree(dpmi));
//	delete[] hpmi;
}

void xParticleMeshObjectsContact::setZeroCollisionForce()
{
	foreach(xPointMass* pobj, pair_ip)
	{
		pobj->setContactForce(0.0, 0.0, 0.0);
		pobj->setContactMoment(0.0, 0.0, 0.0);
	}
}

vector3d xParticleMeshObjectsContact::particle_polygon_contact_detection(host_mesh_info& hpi, vector3d& p, double r)
{
	vector3d a = hpi.P;
	vector3d b = hpi.Q;
	vector3d c = hpi.R;
	vector3d ab = b - a;
	vector3d ac = c - a;
	vector3d ap = p - a;

	double d1 = dot(ab, ap);
	double d2 = dot(ac, ap);
	if (d1 <= 0.0 && d2 <= 0.0){
		//	*wc = 0;
		//_pct = VERTEX;
		return a;
	}

	vector3d bp = p - b;
	double d3 = dot(ab, bp);
	double d4 = dot(ac, bp);
	if (d3 >= 0.0 && d4 <= d3){
		//	*wc = 0;
		//_pct = VERTEX;
		return b;
	}
	double vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){
		//	*wc = 1;
		double v = d1 / (d1 - d3);
	//	_pct = EDGE;
		return a + v * ab;
	}

	vector3d cp = p - c;
	double d5 = dot(ab, cp);
	double d6 = dot(ac, cp);
	if (d6 >= 0.0 && d5 <= d6){
		//	*wc = 0;
		//_pct = VERTEX;
		return c;
	}

	double vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){
		//	*wc = 1;
		double w = d2 / (d2 - d6);
		//_pct = EDGE;
		return a + w * ac; // barycentric coordinates (1-w, 0, w)
	}

	// Check if P in edge region of BC, if so return projection of P onto BC
	double va = d3 * d6 - d5 * d4;
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){
		//	*wc = 1;
		double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		//_pct = EDGE;
		return b + w * (c - b); // barycentric coordinates (0, 1-w, w)
	}
	//*wc = 2;
	// P inside face region. Compute Q through its barycentric coordinates (u, v, w)
	double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;
	//_pct = FACE;
	return a + v * ab + w * ac; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
}