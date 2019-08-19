#include "xdynamics_object/xParticleMeshObjectsContact.h"
#include "xdynamics_manager/xModel.h"
#include "xdynamics_simulation/xSimulation.h"
//#include "xdynamics_parallel/xParallelCommon_decl.cuh"

xParticleMeshObjectsContact::xParticleMeshObjectsContact()
	: xContact()
	, hsphere(NULL)
	, dsphere(NULL)
	, dlocal(NULL)
	, hlocal(NULL)
	, hpi(NULL)
	, dpi(NULL)
	, hcp(NULL)
	, nPobjs(NULL)
	, dvList(NULL)
	, diList(NULL)
	//, hpmi(NULL)
	, dbi(NULL)
	, dbf(NULL)
	, maxRadius(0)
	, npolySphere(0)
	, ncontact(0)
	, nmoving(0)
{

}

xParticleMeshObjectsContact::~xParticleMeshObjectsContact()
{
	if (hsphere) delete[] hsphere; hsphere = NULL;
	if (hlocal) delete[] hlocal; hlocal = NULL;
	if (hpi) delete[] hpi; hpi = NULL;
	if (hcp) delete[] hcp; hcp = NULL;
	if (xmps) delete[] xmps; xmps = NULL;
	//if (hpmi) delete[] hpmi; hpmi = NULL;
	//qDeleteAll(pair_ip);
	if (xSimulation::Gpu())
	{
		if (dsphere) checkCudaErrors(cudaFree(dsphere)); dsphere = NULL;
		if (dlocal) checkCudaErrors(cudaFree(dlocal)); dlocal = NULL;
		if (dpi) checkCudaErrors(cudaFree(dpi)); dpi = NULL;
		if (dvList) checkCudaErrors(cudaFree(dvList)); dvList = NULL;
		if (diList) checkCudaErrors(cudaFree(diList)); diList = NULL;
		if (dbi) checkCudaErrors(cudaFree(dbi)); dbi = NULL;
		if (dbf) checkCudaErrors(cudaFree(dbf)); dbf = NULL;
	}
	
}

vector4d * xParticleMeshObjectsContact::GetCurrentSphereData()
{
	if (xSimulation::Gpu())
		checkCudaErrors(cudaMemcpy(hsphere, dsphere, sizeof(vector4d) * npolySphere, cudaMemcpyDeviceToHost));
	return hsphere;
}

unsigned int xParticleMeshObjectsContact::NumSphereData()
{
	return npolySphere;
}

unsigned int xParticleMeshObjectsContact::define(QMap<QString, xParticleMeshObjectContact*>& cpmesh)
{
	foreach(xParticleMeshObjectContact* cpm, cpmesh)
	{
		xMeshObject* pobj = cpm->MeshObject();
		if (pobj->MovingObject() || pobj->CompulsionMovingObject())
			nmoving++;
		npolySphere += pobj->NumTriangle();
	}
	nPobjs = cpmesh.size();
	hsphere = new vector4d[npolySphere];
	hlocal = new vector3d[npolySphere];
	hpi = new host_mesh_info[npolySphere];
	hcp = new xContactMaterialParameters[nPobjs];
	xmps = new xMaterialPair[nPobjs];
	//hpmi = new host_mesh_mass_info[nPobjs];
	//memset(hpmi, 0, sizeof(host_mesh_mass_info) * nPobjs);
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
		cp.rolling_friction = cpm->RollingFactor();
		cp.stiffness_multiplyer = cpm->StiffMultiplyer();
		xmp = cpm->MaterialPropertyPair();
		hcp[idx] = cp;
		xmps[idx] = xmp;
		pair_ip[idx] = pobj;
		ePolySphere += pobj->NumTriangle();
		unsigned int vi = 0;
		//vector3d* hlocal = new vector3d[npolySphere];
		double* t_radius = new double[npolySphere];
		for (unsigned int i = bPolySphere; i < ePolySphere; i++)
		{
			//			host_mesh_info po;
			hpi[i].id = idx;
			hpi[i].sid = bPolySphere;
			vector3d pos = pobj->Position();
			euler_parameters ep = pobj->EulerParameters();
			//unsigned int s = vi * 9;
			hpi[i].P = pos + ToGlobal(ep, vList[vi++]);
			//hpi[i].indice.x = vi++;
			hpi[i].Q = pos + ToGlobal(ep, vList[vi++]);
			//hpi[i].indice.y = vi++;
			hpi[i].R = pos + ToGlobal(ep, vList[vi++]);
			//hpi[i].indice.z = vi++;

			//vector3d ctri = xUtilityFunctions::CenterOfTriangle(hpi[i].P, hpi[i].Q, hpi[i].R);
			vector4d csph = xUtilityFunctions::FitSphereToTriangle(hpi[i].P, hpi[i].Q, hpi[i].R, 0.8);
			//double rad = length(ctri - hpi[i].P);
			if (csph.w > maxRadii)
				maxRadii = csph.w;
			hsphere[i] = csph;// new_vector4d(ctri.x, ctri.y, ctri.z, rad);
			vector3d r_pos = ToLocal(ep, (new_vector3d(csph.x - pos.x, csph.y - pos.y, csph.z - pos.z)));
			hlocal[i] = r_pos;
			t_radius[i] = csph.w;
		}
	//	ExportTriangleSphereLocalPosition(pobj->Name().toStdString(), bPolySphere, ePolySphere, hlocal, t_radius);
		bPolySphere += pobj->NumTriangle();
		delete[] t_radius;
		//delete[] hlocal; hlocal = NULL;
		/*std::fstream fs;
		fs.open("C:/xdynamics/tri_sphere.txt", std::ios::out);
		for (unsigned int i = 0; i < pobj->NumTriangle(); i++)
		{
			fs << hsphere[i].x << " " << hsphere[i].y << " " << hsphere[i].z << " " << hsphere[i].w << std::endl;
		}
		fs.close();*/
		idx++;
	}
	
	maxRadius = maxRadii;
	if (xSimulation::Cpu())
		dsphere = (double *)hsphere;
	return npolySphere;
}

void xParticleMeshObjectsContact::particle_triangle_contact_force(
	xTrianglePairData* d, double r, double m,
	vector3d& p, vector3d& v, vector3d& o,
	double &res, vector3d &tmax, vector3d& F, vector3d& M)
{
	vector3d m_fn = new_vector3d(0.0, 0.0, 0.0);
	vector3d m_m = new_vector3d(0, 0, 0);
	vector3d m_ft = new_vector3d(0, 0, 0);
	unsigned int j = hpi[d->id].id;
	xPointMass* hmmi = pair_ip[j];
	vector3d mp = hmmi->Position();
	vector3d mo = 2.0 * GMatrix(hmmi->EulerParameters()) * hmmi->DEulerParameters();
	vector3d mv = hmmi->Velocity();
	//host_mesh_mass_info hmmi = hpmi[j];
	double rcon = r - 0.5 * d->gab;
	vector3d u = new_vector3d(d->nx, d->ny, d->nz);
	//vector3d rc = r * u;
	vector3d dcpr = new_vector3d(d->cpx, d->cpy, d->cpz) - p;
	vector3d po2cp = new_vector3d(d->cpx - mp.x, d->cpy - mp.y, d->cpz - mp.z);
///vector3d mvel = new_vector3d(mp.x, mp.y, mp.z);
	//vector3d momega = new_vector3d(hmmi.ox, hmmi.oy, hmmi.oz);
	vector3d rv = mv + cross(mo, po2cp) - (v + cross(o, dcpr));
	xContactMaterialParameters cmp = hcp[j];
	xContactParameters c = getContactParameters(
		r, 0.0,
		m, 0.0,
		xmps[j].Ei, xmps[j].Ej,
		xmps[j].Pri, xmps[j].Prj,
		xmps[j].Gi, xmps[j].Gj,
		cmp.restitution, cmp.stiffness_ratio,
		cmp.friction, cmp.rolling_friction, cmp.cohesion);
	switch (force_model)
	{
	case DHS: DHSModel(c, d->gab, d->delta_s, d->dot_s, cmp.cohesion, rv, u, m_fn, m_ft); break;
	}
	RollingResistanceForce(c.rfric, r, 0.0, dcpr, m_fn, m_ft, res, tmax);
	vector3d nforce = m_fn + m_ft;
	F += nforce;
	M += cross(dcpr, nforce);
	hmmi->addContactForce(-nforce.x, -nforce.y, -nforce.z);
	/*hpmi[j].fx += -nforce.x;
	hpmi[j].fy += -nforce.y;
	hpmi[j].fz += -nforce.z;*/
	vector3d tmoment = -cross(po2cp, nforce);
	hmmi->addContactMoment(tmoment.x, tmoment.y, tmoment.z);
	/*hpmi[j].mx = tmoment.x;
	hpmi[j].my = tmoment.y;
	hpmi[j].mz = tmoment.z;*/
}

bool xParticleMeshObjectsContact::cppolyCollision(
	xContactPairList* pairs, unsigned int i, double r, double m,
	vector3d& p, vector3d& v, vector3d& o,
	double &res, vector3d &tmax, vector3d& F, vector3d& M,
	unsigned int nco, xClusterInformation* xci, vector4d* cpos)
{
	unsigned int ci = 0;
	unsigned int neach = 1;
	vector3d cp = p;
	if (nco && cpos)
	{
		for (unsigned int j = 0; j < nco; j++)
			if (i >= xci[j].sid && i < xci[j].sid + xci[j].count * xci[j].neach)
				neach = xci[j].neach;
		//ck = j / neach;
		ci = i / neach;
		cp = new_vector3d(cpos[ci].x, cpos[ci].y, cpos[ci].z);
	}
	
	foreach(xTrianglePairData* d, pairs->TrianglePair())
	{
		particle_triangle_contact_force(d, r, m, cp, v, o, res, tmax, F, M);
	}
	if (!pairs->TrianglePair().size())
	{
		foreach(xTrianglePairData* d, pairs->TriangleLinePair())
		{
			particle_triangle_contact_force(d, r, m, cp, v, o, res, tmax, F, M);
		}
			
	}
	if (!pairs->TrianglePair().size() && !pairs->TriangleLinePair().size())
	{
		foreach(xTrianglePairData* d, pairs->TrianglePointPair())
		{
			particle_triangle_contact_force(d, r, m, cp, v, o, res, tmax, F, M);
		}
	}
		
		
	/*foreach(xTrianglePairData* d, pairs->TrianglePair())
	{
		if(!d->isc)
			
	}*/
	return true;
//	unsigned int ct = 0;
//	unsigned int nc = 0;
//	unsigned int i = hpi[idx].id;
//	// 	if (pct[i])
//	// 		return false;
//	xPointMass* pm = pair_ip[i];
//	xMaterialPair mat = xmps[i];
//	xContactMaterialParameters cpa = hcp[i];
//	//polygonContactType _pct;
//	vector3d u;
//	restitution = cpa.restitution;
//	friction = cpa.friction;
//	stiffnessRatio = cpa.stiffness_ratio;
//	vector3d mp = pm->Position();
//	vector3d mv = pm->Velocity();
//	vector3d mo = pm->AngularVelocity();
//	vector3d cpt = particle_polygon_contact_detection(hpi[idx], p, r);
//	vector3d po2cp = cpt - mp;
//	vector3d distVec = p - cpt;
//	double dist = length(distVec);
//	double cdist = r - dist;
//	vector3d m_f, m_m;
//	if (cdist > 0)
//	{
//		ncontact++;
//		vector3d qp = hpi[idx].Q - hpi[idx].P;
//		vector3d rp = hpi[idx].R - hpi[idx].P;
//		u = -cross(qp, rp);
//		u = u / length(u);// .length();
//		double rcon = r - 0.5 * cdist;
//		vector3d cp = rcon * u;
//		vector3d dv = mv + cross(mo, po2cp) - (v + cross(o, r * u));
//		xContactParameters c = getContactParameters(
//			r, 0.0,
//			m, 0.0,
//			mat.Ei, mat.Ej,
//			mat.Pri, mat.Prj,
//			mat.Gi, mat.Gj);
//// 		switch (force_model)
//// 		{
//// 		//case DHS: DHSModel(c, cdist, cp, dv, u, m_f, m_m); break;
//// 		}
//
//		F += m_f;
//		M += m_m;
//		pm->addContactForce(-m_f.x, -m_f.y, -m_f.z);
//		vector3d m_mm = -cross(po2cp, m_f);
//		pm->addContactMoment(m_mm.x, m_mm.y, m_mm.z);
//		return true;
//	}
	return false;
}

void xParticleMeshObjectsContact::updateMeshObjectData(bool is_first_set_up)
{
	unsigned int bPolySphere = 0;
	unsigned int ePolySphere = 0;
	//QMapIterator<unsigned int, xMeshObject*> po(pair_ip);
	//double *hep = new double[nPobjs * 4];
	//while (po.hasNext())
	//{
	//	po.next();
	//	unsigned int id = po.key();
	//	xMeshObject* mesh = po.value();
	//	unsigned int xid = mesh->xpmIndex() * xModel::OneDOF();
	//	//xMeshObject* p = po.value();
	//	vector3d pos = mesh->Position();//new_vector3d(q(xid + 0), q(xid + 1), q(xid + 2));
	//	euler_parameters ep = mesh->EulerParameters();//new_euler_parameters(q(xid + 3), q(xid + 4), q(xid + 5), q(xid + 6));
	//	vector3d vel = mesh->Velocity();//new_vector3d(qd(xid + 0), qd(xid + 1), qd(xid + 2));// p->Velocity();
	//	euler_parameters ev = mesh->DEulerParameters();//new_euler_parameters(qd(xid + 3), qd(xid + 4), qd(xid + 5), qd(xid + 6));
	//	vector3d omega = 2.0 * GMatrix(ep) * ev;
	//	hpmi[id] =
	//	{
	//		mesh->Mass(),
	//		pos.x, pos.y, pos.z,
	//		vel.x, vel.y, vel.z,
	//		omega.x, omega.y, omega.z,
	//		0.0, 0.0, 0.0,
	//		0.0, 0.0, 0.0
	//	};
	//	hep[id * 4 + 0] = ep.e0;
	//	hep[id * 4 + 1] = ep.e1;
	//	hep[id * 4 + 2] = ep.e2;
	//	hep[id * 4 + 3] = ep.e3;
	//}

	if ((xSimulation::Gpu() && nmoving) || is_first_set_up)
	{
		unsigned int mcnt = 0;
		device_body_info *bi = NULL;
		if (nPobjs)
			bi = new device_body_info[nPobjs];
		//checkCudaErrors(cudaMemcpy(dpmi, hpmi, sizeof(device_mesh_mass_info) * nPobjs, cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpy(dep, hep, sizeof(double4) * nPobjs, cudaMemcpyHostToDevice));
		foreach(xMeshObject* pobj, pair_ip)
		{
			euler_parameters ep = pobj->EulerParameters(), ed = pobj->DEulerParameters();
			bi[mcnt] = {
				pobj->Mass(),
				pobj->Position().x, pobj->Position().y, pobj->Position().z,
				pobj->Velocity().x, pobj->Velocity().y, pobj->Velocity().z,
				ep.e0, ep.e1, ep.e2, ep.e3,
				ed.e0, ed.e1, ed.e2, ed.e3
			};
			checkCudaErrors(cudaMemcpy(dbi, bi, sizeof(device_body_info) * nPobjs, cudaMemcpyHostToDevice));
			mcnt++;
		}
		foreach(xMeshObject* pobj, pair_ip)
		{
			ePolySphere += pobj->NumTriangle();
			cu_update_meshObjectData(dvList, dsphere, dlocal, dpi, dbi, ePolySphere - bPolySphere);
			bPolySphere += pobj->NumTriangle();// ePolySphere;
		}
	}
	else
	{
		foreach(xMeshObject* mesh, pair_ip)
		{
			//xMeshObject* pobj = cpm->MeshObject();
			vector3d *vList = (vector3d *)mesh->VertexList();
			ePolySphere += mesh->NumTriangle();
			unsigned int vi = 0;
			for (unsigned int i = bPolySphere; i < ePolySphere; i++)
			{
				//host_mesh_info po;
				//hpi[i].id = idx;
				//unsigned int s = vi * 9;
				unsigned int id = hpi[i].id;
				vector3d pos = mesh->Position();
				euler_parameters ep = mesh->EulerParameters();
				//host_mesh_mass_info m = hpmi[id];
				//vector3d pos = new_vector3d(m.px, m.py, m.pz);// [hpi[i].id].pos;
				//euler_parameters ep = new_euler_parameters(
				//	hep[id * 4 + 0],
				//	hep[id * 4 + 1],
				//	hep[id * 4 + 2],
				//	hep[id * 4 + 3]);
				/*hpi[i].P = pos + ToGlobal(ep, vList[vi++]);
				hpi[i].Q = pos + ToGlobal(ep, vList[vi++]);
				hpi[i].R = pos + ToGlobal(ep, vList[vi++]);
				vector3d ctri = xUtilityFunctions::CenterOfTriangle(hpi[i].P, hpi[i].Q, hpi[i].R);
				double rad = length(ctri - hpi[i].P);*/
				vector3d gsph = pos + ToGlobal(ep, hlocal[i]);
				hsphere[i] = new_vector4d(gsph.x, gsph.y, gsph.z, hsphere[i].w);
				///hsphere[i] = new_vector4d(ctri.x, ctri.y, ctri.z, rad);
			}
			bPolySphere += mesh->NumTriangle();
		}
	}
	//delete[] hep;
}

void xParticleMeshObjectsContact::updateMeshMassData()
{
	//QMapIterator<unsigned int, xMeshObject*> xmo(pair_ip);
	//while (xmo.hasNext())
	//{
	//	xmo.next();
	//	unsigned int id = xmo.key();
	//	xMeshObject* o = xmo.value();
	//	vector3d pos = o->Position();
	//	vector3d vel = o->Velocity();
	//	euler_parameters ep = o->EulerParameters();
	//	euler_parameters ev = o->DEulerParameters();
	//	vector3d omega = 2.0 * GMatrix(ep) * ev;		
	//	hpmi[id] =
	//	{
	//		o->Mass(),
	//		pos.x, pos.y, pos.z,
	//		vel.x, vel.y, vel.z,
	//		omega.x, omega.y, omega.z,
	//		0.0, 0.0, 0.0,
	//		0.0, 0.0, 0.0
	//	};
	//}
	//if (dpmi)
	//{
	//	checkCudaErrors(cudaMemcpy(dpmi, hpmi, sizeof(device_mesh_mass_info) * nPobjs, cudaMemcpyHostToDevice));
	//}
}

void xParticleMeshObjectsContact::getMeshContactForce()
{
	if (nmoving)
	{
		QMapIterator<unsigned int, xMeshObject*> xmo(pair_ip);
		while (xmo.hasNext())
		{
			xmo.next();
			unsigned int id = xmo.key();
			xMeshObject* o = xmo.value();
			if (o->MovingObject())
			{
				//std::cout << "mesh contact force : [" << dbf[id].force.x << ", " << dbf[id].force.y << ", " << dbf[id].force.z << "]" << std::endl;
				o->addContactForce(dbf[id].force.x, dbf[id].force.y, dbf[id].force.z);
				o->addContactMoment(dbf[id].moment.x, dbf[id].moment.y, dbf[id].moment.z);
			}
		}
	}
//	checkCudaErrors(cudaMemcpy(hpmi, dpmi, sizeof(device_mesh_mass_info) * nPobjs, cudaMemcpyDeviceToHost));
	
}

bool xParticleMeshObjectsContact::updateCollisionPair(
	unsigned int id, xContactPairList& xcpl, double r, 
	vector3d pos, unsigned int &oid, vector3d& ocpt, vector3d& ounit, vector3i& ctype)
{
	int t = -1;
	unsigned int k = 0;
	host_mesh_info hmi = hpi[id];
//	host_mesh_mass_info hmmi = hpmi[hmi.id];
	vector3d jpos = new_vector3d(hsphere[id].x, hsphere[id].y, hsphere[id].z);
	double jr = hsphere[id].w;
	vector3d cpt = particle_polygon_contact_detection(hmi, pos, r, t);
	
	vector3d rp = cpt - pos;
	double dist = length(rp);
	vector3d unit = rp / dist;
	//if (t != 2)
	//{
	//	oid = id;
	//	ocpt = cpt;
	//	ounit = unit;
	//	return false;
	//}
	//vector3d po2cp = cpt - new_vector3d(hmmi.px, hmmi.py, hmmi.pz);
	//double cdist = r - length(pos - cpt);
	double cdist = r - dist;
	if (cdist > 0)
	{
		/*vector3d qp = hmi.Q - hmi.P;
		vector3d rp = hmi.R - hmi.P;
		vector3d unit = -cross(qp, rp);*/
		
		//cpt = 0.5 * (jpos + pos);
		//unit = -unit / length(unit);
		if (t != 0)
		{
			bool overlab = checkOverlab(ctype, ocpt, cpt, ounit, unit);

			if (overlab)
				return false;

			ocpt = cpt;
			ounit = unit;
		}
		
		//*(&(ctype.x) + t) += 1;
		bool is_new = t == 0 ? xcpl.IsNewTriangleContactPair(id) : (t == 1 ? xcpl.IsNewTriangleLineContactPair(id) : xcpl.IsNewTrianglePointContactPair(id));
		if (is_new)
		{
			xTrianglePairData* pd = new xTrianglePairData;
			*pd = { MESH_SHAPE, true, 0, id, 0, 0, cdist, unit.x, unit.y, unit.z, cpt.x, cpt.y, cpt.z };
			t == 0 ? xcpl.insertTriangleContactPair(pd) : (t == 1 ? xcpl.insertTriangleLineContactPair(pd) : xcpl.insertTrianglePointContactPair(pd));
			//xcpl.insertTriangleContactPair(pd);
		}	
		else
		{
			xTrianglePairData *pd = t == 0 ? xcpl.TrianglePair(id) : (t== 1 ? xcpl.TriangleLinePair(id) : xcpl.TrianglePointPair(id));
			pd->gab = cdist;
			pd->cpx = cpt.x;
			pd->cpy = cpt.y;
			pd->cpz = cpt.z;
			pd->nx = unit.x;
			pd->ny = unit.y;
			pd->nz = unit.z;
		}
		return true;
	}
	else
	{
		xcpl.deleteTrianglePairData(id);
		xcpl.deleteTriangleLinePairData(id);
		xcpl.deleteTrianglePointPairData(id);
		//t == 0 ? xcpl.deleteTrianglePairData(id) : (t==1 ? xcpl.deleteTriangleLinePairData(id) : xcpl.deleteTrianglePointPairData(id));
	}
	return false;
}

void xParticleMeshObjectsContact::updateCollisionPairLineOrVertex(
	double r, vector3d & pos, unsigned int& oid, vector3d & ocpt, vector3d & ounit, xContactPairList & xcpl)
{
	vector3d rp = ocpt - pos;
	double dist = length(rp);
	vector3d unit = rp / dist;
	host_mesh_info hmi = hpi[oid];
	//host_mesh_mass_info hmmi = hpmi[hmi.id];
	xMeshObject* mesh = pair_ip[hmi.id];
	vector3d po2cp = ocpt - mesh->Position();// new_vector3d(hmmi.px, hmmi.py, hmmi.pz);
	//double cdist = r - length(pos - cpt);
	double cdist = r - dist;
	if (cdist > 0)
	{
		/*vector3d qp = hmi.Q - hmi.P;
		vector3d rp = hmi.R - hmi.P;
		vector3d unit = -cross(qp, rp);*/

		//cpt = 0.5 * (jpos + pos);
		//unit = -unit / length(unit);
		//bool overlab = checkOverlab(ctype, ocpt, cpt, ounit, unit);

		/*if (overlab)
			return false;

		ocpt = cpt;
		ounit = unit;
		*(&(ctype.x) + t) += 1;*/
		if (xcpl.IsNewTriangleContactPair(oid))
		{
			xTrianglePairData* pd = new xTrianglePairData;
			*pd = { MESH_SHAPE, true, 0, oid, 0, 0, cdist, unit.x, unit.y, unit.z, ocpt.x, ocpt.y, ocpt.z };
			xcpl.insertTriangleContactPair(pd);
		}
		else
		{
			xTrianglePairData *pd = xcpl.TrianglePair(oid);
			pd->gab = cdist;
			pd->cpx = ocpt.x;
			pd->cpy = ocpt.y;
			pd->cpz = ocpt.z;
			pd->nx = unit.x;
			pd->ny = unit.y;
			pd->nz = unit.z;
		}
	}
	else
	{
		xcpl.deleteTrianglePairData(oid);
	}
}

void xParticleMeshObjectsContact::cudaMemoryAlloc(unsigned int np)
{
	device_contact_property *_hcp = new device_contact_property[nPobjs];
	for (unsigned int i = 0; i < nPobjs; i++)
	{
		_hcp[i] = { xmps[i].Ei, xmps[i].Ej, xmps[i].Pri, xmps[i].Prj, xmps[i].Gi, xmps[i].Gj,
			hcp[i].restitution, hcp[i].friction, hcp[i].rolling_friction, hcp[i].cohesion, hcp[i].stiffness_ratio, hcp[i].stiffness_multiplyer };
	}
	checkCudaErrors(cudaMalloc((void**)&dsphere, sizeof(double) * npolySphere * 4));
	checkCudaErrors(cudaMalloc((void**)&dlocal, sizeof(double) * npolySphere * 3));
	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_triangle_info) * npolySphere));
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property) * nPobjs));
	checkCudaErrors(cudaMalloc((void**)&dvList, sizeof(double) * npolySphere * 9));
	checkCudaErrors(cudaMalloc((void**)&dbi, sizeof(device_body_info) * nPobjs));
	
	//checkCudaErrors(cduaMalloc((void**)&diList, sizeof(unsigned int) * ))
	checkCudaErrors(cudaMemcpy(dsphere, hsphere, sizeof(double) * npolySphere * 4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dpi, hpi, sizeof(device_triangle_info) * npolySphere, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dcp, _hcp, sizeof(device_contact_property) * nPobjs, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dlocal, hlocal, sizeof(vector3d) * npolySphere, cudaMemcpyHostToDevice));
	unsigned int bPolySphere = 0;
	unsigned int ePolySphere = 0;
	foreach(xMeshObject* pobj, pair_ip)
	{
		//polygonObject* pobj = cppo->PolygonObject();
		double *vList = pobj->VertexList();
		ePolySphere += pobj->NumTriangle();
		//unsigned int *iList = pobj->IndexList();
		checkCudaErrors(cudaMemcpy(dvList + bPolySphere * 9, vList, sizeof(double) * pobj->NumTriangle() * 9, cudaMemcpyHostToDevice));
		bPolySphere += pobj->NumTriangle();// ePolySphere;
	}
	dbf = new device_body_force[nPobjs];
	updateMeshObjectData(true);
	delete[] _hcp;
}

void xParticleMeshObjectsContact::cuda_collision(
	double *pos, double *vel, double *omega,
	double *mass, double *force, double *moment,
	unsigned int *sorted_id, unsigned int *cell_start,
	unsigned int *cell_end, unsigned int np)
{
// 	QMapIterator<unsigned int, xMeshObject*> po(pair_ip);
// 	cu_particle_meshObject_collision(1, dpi, dsphere, dpmi, pos, vel, omega, force, moment, mass, sorted_id, cell_start, cell_end, dcp, np);
// 	checkCudaErrors(cudaMemcpy(hpmi, dpmi, sizeof(device_mesh_mass_info) * nPobjs, cudaMemcpyDeviceToHost));
// 	//	po.toFront();
// 	while (po.hasNext())
// 	{
// 		po.next();
// 		unsigned int id = po.key();
// 		xPointMass* p = po.value();
// 		p->addContactForce(hpmi[id].force.x, hpmi[id].force.y, hpmi[id].force.z);
// 		//p->setCollisionMoment(VEC3D(hpmi[id].moment.x, hpmi[id].moment.y, hpmi[id].moment.z));
// 	}
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

device_body_force * xParticleMeshObjectsContact::deviceBodyForceAndMoment()
{
	return dbf;
}

device_triangle_info* xParticleMeshObjectsContact::deviceTrianglesInfo()
{
	return dpi;
}

device_body_info* xParticleMeshObjectsContact::devicePolygonObjectMassInfo()
{
	return dbi;
}

void xParticleMeshObjectsContact::ExportTriangleSphereLocalPosition(std::string & name, unsigned int b, unsigned int e, vector3d* hlocal, double *rad)
{
	std::string path = xModel::makeFilePath(name) + ".tsd";
	std::fstream fs;
	fs.open(path, std::ios::out | std::ios::binary);
	vector3d* data = hlocal + b;
	unsigned int sz = e - b;
	fs.write((char*)&sz, sizeof(unsigned int));
	fs.write((char*)hlocal, sizeof(vector3d) * sz);
	fs.write((char*)rad, sizeof(double) * sz);
	fs.close();

}

vector3d xParticleMeshObjectsContact::particle_polygon_contact_detection(
	host_mesh_info& hpi, vector3d& p, double r,int& ct)
{
	vector3d a = hpi.P;
	vector3d b = hpi.Q;
	vector3d c = hpi.R;
	vector3d ab = b - a;
	vector3d ac = c - a;
	vector3d ap = p - a;
	vector3d bp = p - b;
	vector3d cp = p - c;
	double d1 = dot(ab, ap);
	double d2 = dot(ac, ap);
	double d3 = dot(ab, bp);
	double d4 = dot(ac, bp);
	double d5 = dot(ab, cp);
	double d6 = dot(ac, cp);
	double va = d3 * d6 - d5 * d4;
	double vb = d5 * d2 - d1 * d6;
	double vc = d1 * d4 - d3 * d2;

	//if (va > 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
	
	vector3d cpt = new_vector3d(0, 0, 0);
	
	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { ct = 1; cpt = a + (d1 / (d1 - d3)) * ab; }
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { ct = 1; cpt = a + (d2 / (d2 - d6)) * ac; }
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) { ct = 1; cpt = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b); }


	if (d1 <= 0.0 && d2 <= 0.0) { ct = 2; cpt = a; }
	if (d3 >= 0.0 && d4 <= d3) { ct = 2; cpt = b; }
	if (d6 >= 0.0 && d5 <= d6) { ct = 2; cpt = c; }

	if (va >= 0 && vb >= 0 && vc >= 0)
	{
		double denom = 1.0 / (va + vb + vc);
		vector3d v = vb * denom * ab;
		vector3d w = vc * denom * ac;
		cpt = a + v + w;
		ct = 0;
		//ct = 2;
		//if (_dist > 0) return _cpt;
		//return _cpt; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
	}

	return cpt;
	//ct = 2;
	// P inside face region. Comu0te Q through its barycentric coordinates (u, v, w)
	/*double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;*/
	
}

bool xParticleMeshObjectsContact::checkOverlab(vector3i ctype, vector3d p, vector3d c, vector3d u0, vector3d u1)
{
	bool b_over = false;
	if (p.x >= c.x - 1e-9 && p.x <= c.x + 1e-9)
		b_over = true;
	if (p.y >= c.y - 1e-9 && p.y <= c.y + 1e-9)
		b_over = true;
	if (p.z >= c.z - 1e-9 && p.z <= c.z + 1e-9)
		b_over = true;

	if (/*(ctype.y || ctype.z) &&*/ !b_over)
	{
		if (u0.x >= u1.x - 1e-9 && u0.x <= u1.x + 1e-9)
			if (u0.y >= u1.y - 1e-9 && u0.y <= u1.y + 1e-9)
				if (u0.z >= u1.z - 1e-9 && u0.z <= u1.z + 1e-9)
					b_over = true;
	}
	if (!b_over)
	{
		
	}
	return b_over;
}