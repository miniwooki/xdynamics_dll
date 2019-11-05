#include "xdynamics_object/xParticleMeshObjectContact.h"
#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_object/xMeshObject.h"
#include "xdynamics_manager/xDynamicsManager.h"

unsigned int xParticleMeshObjectContact::defined_count = 0;
bool xParticleMeshObjectContact::allocated_static = false;
// int xParticleMeshObjectContact::nmoving = 0;
double* xParticleMeshObjectContact::d_tsd_ptri = nullptr;
unsigned int* xParticleMeshObjectContact::d_pair_count_ptri = nullptr;
unsigned int* xParticleMeshObjectContact::d_pair_id_ptri = nullptr;

double* xParticleMeshObjectContact::tsd_ptri = nullptr;
unsigned int* xParticleMeshObjectContact::pair_count_ptri = nullptr;
unsigned int* xParticleMeshObjectContact::pair_id_ptri = nullptr;

unsigned int xParticleMeshObjectContact::n_mesh_sphere = 0;
//double xParticleMeshObjectContact::max_sphere_radius = 0;

xParticleMeshObjectContact::xParticleMeshObjectContact()
	: xContact()
	, id(0)
	, p(nullptr)
	, po(nullptr)
	, hsphere(nullptr)
	, hlocal(nullptr)
{

}

xParticleMeshObjectContact::xParticleMeshObjectContact(std::string _name, xObject* o1, xObject* o2)
	: xContact(_name, PARTICLE_MESH_SHAPE)
	, id(0)
	, p(nullptr)
	, po(nullptr)
	, hsphere(nullptr)
	, hlocal(nullptr)
{
	if (o1 && o2)
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
		mpp = { o1->Youngs(), o2->Youngs(), o1->Poisson(), o2->Poisson(), o1->Shear(), o2->Shear() };
		po->splitTriangles(po->RefinementSize());
	}
	
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
	if (hti) delete[] hti; hti = nullptr;
}

void xParticleMeshObjectContact::define(unsigned int idx, unsigned int np)
{
	id = defined_count;
	xContact::define(idx, np);

	hsphere = new vector4d[po->NumTriangle()];
	hlocal = new vector3d[po->NumTriangle()];
	hti = new host_triangle_info[po->NumTriangle()];
	//xContactMaterialParameters hcp = { 0, };
	//xMaterialPair hmp = { 0, };

	double maxRadii = 0.0;
	//unsigned int idx = 0;

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
		hti[i].tid = n_mesh_sphere + i;
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
	n_mesh_sphere += po->NumTriangle();
	defined_count++;
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

// void xParticleMeshObjectContact::initialize()
// {
// 	xParticleMeshObjectContact::local_initialize();
// }

double * xParticleMeshObjectContact::MeshSphere()
{
	return xSimulation::Gpu() ? dsphere : (double*)hsphere;
}

void xParticleMeshObjectContact::collision_gpu(
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
	if (xSimulation::Gpu())
	{
		double fm[6] = { 0, };
		if(!cpos) cu_particle_polygonObject_collision(dti, dbi, pos, ep, vel, ev, force, moment, mass, tmax, rres, d_pair_count_ptri, d_pair_id_ptri, d_tsd_ptri, dsphere, sorted_id, cell_start, cell_end, dcp, np);
		else if (cpos) cu_cluster_meshes_contact(dti, dbi, pos, cpos, ep, vel, ev, force, moment, dcp, mass, tmax, rres, d_pair_count_ptri, d_pair_id_ptri, d_tsd_ptri, sorted_id, cell_start, cell_end, xci, np);
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

void xParticleMeshObjectContact::collision_cpu(
	vector4d * pos, euler_parameters * ep, vector3d * vel,
	euler_parameters * ev, double* mass, double & rres, vector3d & tmax,
	vector3d & force, vector3d & moment, unsigned int nco,
	xClusterInformation * xci, vector4d * cpos)
{
	
	
	for (xmap<unsigned int, xTrianglePairData*>::iterator it = triangle_pair.begin(); it != triangle_pair.end(); it.next())
	{
		unsigned int id = it.value()->id;
		unsigned int ci = id;
		unsigned int neach = 1;
		vector3d cp = new_vector3d(pos[id].x, pos[id].y, pos[id].z);
		double r = pos[id].w;
		double m = mass[id];
		if (nco)
		{
			for (unsigned int j = 0; j < nco; j++)
				if (id >= xci[j].sid && id < xci[j].sid + xci[j].count * xci[j].neach)
				{
					neach = xci[j].neach; ci = id / neach;
				}
			cp = new_vector3d(cpos[ci].x, cpos[ci].y, cpos[ci].z);
			r = cpos[ci].w;
		}
//		double m = mass[ci];
		vector3d v = vel[ci];
		vector3d o = ToAngularVelocity(ep[ci], ev[ci]);
		particle_triangle_contact_force(it.value(), r, m, cp, v, o, rres, tmax, force, moment);
	}
	if (!triangle_pair.size())
	{
		for (xmap<unsigned int, xTrianglePairData*>::iterator it = triangle_line_pair.begin(); it != triangle_line_pair.end(); it.next())
		{
			unsigned int id = it.value()->id;
			unsigned int ci = id;
			unsigned int neach = 1;
			vector3d cp = new_vector3d(pos[id].x, pos[id].y, pos[id].z);
			double r = pos[id].w;
			double m = mass[id];
			if (nco)
			{
				for (unsigned int j = 0; j < nco; j++)
					if (id >= xci[j].sid && id < xci[j].sid + xci[j].count * xci[j].neach)
					{
						neach = xci[j].neach; ci = id / neach;
					}
				cp = new_vector3d(cpos[ci].x, cpos[ci].y, cpos[ci].z);
				r = cpos[ci].w;
			}
//			double m = mass[ci];
			vector3d v = vel[ci];
			vector3d o = ToAngularVelocity(ep[ci], ev[ci]);
			particle_triangle_contact_force(it.value(), r, m, cp, v, o, rres, tmax, force, moment);
		}

	}
	if (!triangle_pair.size() && !triangle_line_pair.size())
	{
		for (xmap<unsigned int, xTrianglePairData*>::iterator it = triangle_point_pair.begin(); it != triangle_point_pair.end(); it.next())
		{
			unsigned int id = it.value()->id;
			unsigned int ci = id;
			unsigned int neach = 1;
			vector3d cp = new_vector3d(pos[id].x, pos[id].y, pos[id].z);
			double r = pos[id].w;
			double m = mass[id];
			if (nco)
			{
				for (unsigned int j = 0; j < nco; j++)
					if (id >= xci[j].sid && id < xci[j].sid + xci[j].count * xci[j].neach)
					{
						neach = xci[j].neach; ci = id / neach;
					}
				cp = new_vector3d(cpos[ci].x, cpos[ci].y, cpos[ci].z);
				r = cpos[ci].w;
			}
		//	double m = mass[ci];
			vector3d v = vel[ci];
			vector3d o = ToAngularVelocity(ep[ci], ev[ci]);
			particle_triangle_contact_force(it.value(), r, m, cp, v, o, rres, tmax, force, moment);
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

void xParticleMeshObjectContact::local_initialize()
{
	defined_count = 0;
}

bool xParticleMeshObjectContact::check_this_mesh(unsigned int idx)
{
	if (idx >= hti[0].id && idx < hti[0].id)
		return true;
	return false;
}

vector3d xParticleMeshObjectContact::particle_polygon_contact_detection(host_triangle_info& hpi, vector3d& p, double r, int& ct)
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
	}
	return cpt;
}

bool xParticleMeshObjectContact::checkOverlab(vector3i ctype, vector3d p, vector3d c, vector3d u0, vector3d u1)
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

bool xParticleMeshObjectContact::updateCollisionPair(
	unsigned int id, double r, vector3d pos, unsigned int &oid, vector3d& ocpt, vector3d& ounit, vector3i& ctype)
{
	int t = -1;
	unsigned int k = 0;
	host_triangle_info hmi = hti[id];
	//	host_mesh_mass_info hmmi = hpmi[hmi.id];
	vector3d jpos = new_vector3d(hsphere[id].x, hsphere[id].y, hsphere[id].z);
	double jr = hsphere[id].w;
	vector3d cpt = particle_polygon_contact_detection(hmi, pos, r, t);

	vector3d rp = cpt - pos;
	double dist = length(rp);
	vector3d unit = rp / dist;
	double cdist = r - dist;
	xmap<unsigned int, xTrianglePairData*>* pair = t == 0 ? &triangle_pair : (t == 1 ? &triangle_line_pair : &triangle_point_pair);
	if (cdist > 0)
	{
		if (t != 0)
		{
			bool overlab = checkOverlab(ctype, ocpt, cpt, ounit, unit);

			if (overlab)
				return false;

			ocpt = cpt;
			ounit = unit;
		}

		//*(&(ctype.x) + t) += 1
		bool is_new = pair->find(id) == pair->end();
		if (is_new)
		{
			xTrianglePairData* pd = new xTrianglePairData;
			*pd = { MESH_SHAPE, true, 0, id, 0, 0, cdist, unit.x, unit.y, unit.z, cpt.x, cpt.y, cpt.z };
			pair->insert(id, pd);// t == 0 ? triangle_pair.insert(id, pd) : (t == 1 ? triangle_line_pair.insert(id, pd) : triangle_point_pair.insert(id, pd));
			//xcpl.insertTriangleContactPair(pd);
		}
		else
		{
			xTrianglePairData *pd = pair->find(id).value();// t == 0 ? xcpl.TrianglePair(id) : (t == 1 ? xcpl.TriangleLinePair(id) : xcpl.TrianglePointPair(id));
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
		delete pair->take(id);
	}
	return false;
}

void xParticleMeshObjectContact::particle_triangle_contact_force(
	xTrianglePairData* d, double r, double m,
	vector3d& p, vector3d& v, vector3d& o,
	double &res, vector3d &tmax, vector3d& F, vector3d& M)
{
	vector3d m_fn = new_vector3d(0.0, 0.0, 0.0);
	vector3d m_m = new_vector3d(0, 0, 0);
	vector3d m_ft = new_vector3d(0, 0, 0);
	//unsigned int j = hpi[d->id].id;
	xPointMass* hmmi = po;// pair_ip[j];
	vector3d mp = hmmi->Position();
	vector3d mo = 2.0 * GMatrix(hmmi->EulerParameters()) * hmmi->DEulerParameters();
	vector3d mv = hmmi->Velocity();
	//host_mesh_mass_info hmmi = hpmi[j];
	double rcon = r - 0.5 * d->gab;
	vector3d u = new_vector3d(d->nx, d->ny, d->nz);
	//vector3d rc = r * u;
	vector3d dcpr = new_vector3d(d->cpx, d->cpy, d->cpz) - p;
	vector3d po2cp = new_vector3d(d->cpx - mp.x, d->cpy - mp.y, d->cpz - mp.z);
	vector3d rv = mv + cross(mo, po2cp) - (v + cross(o, dcpr));
	//xContactMaterialParameters cmp = hcp[j];
	xContactParameters c = getContactParameters(
		r, 0.0,
		m, po->Mass(),
		mpp.Ei, mpp.Ej,
		mpp.Pri, mpp.Prj,
		mpp.Gi, mpp.Gj,
		restitution, stiffnessRatio, s_friction,
		friction, rolling_factor, cohesion);
	switch (xContact::ContactForceModel())
	{
	case DHS: DHSModel(c, d->gab, d->delta_s, d->dot_s, cohesion, rv, u, m_fn, m_ft); break;
	case HERTZ_MINDLIN_NO_SLIP: Hertz_Mindlin(c, d->gab, d->delta_s, d->dot_s, cohesion, rv, u, m_fn, m_ft); break;
	}
	RollingResistanceForce(c.rfric, r, 0.0, dcpr, m_fn, m_ft, res, tmax);
	vector3d nforce = m_fn + m_ft;
	F += nforce;
	M += cross(dcpr, nforce);
	hmmi->addContactForce(-nforce.x, -nforce.y, -nforce.z);
	vector3d tmoment = -cross(po2cp, nforce);
	hmmi->addContactMoment(tmoment.x, tmoment.y, tmoment.z);
}