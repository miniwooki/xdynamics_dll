#include "xdynamics_manager/xParticleMananger.h"
//#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_manager/xObjectManager.h"
#include <algorithm>
#include <random>
//#include <QtCore/QRandomGenerator>

xParticleManager::xParticleManager()
	: np(0)
	, ncluster(0)
	, r_pos(NULL)
	, r_vel(NULL)
	, n_cluster_object(0)
	, n_cluster_each(0)
	, n_mass_particle(0)
	, minimum_particle_density(FLT_MAX)
	, maximum_youngs_modulus(-FLT_MAX)
	, minimum_poisson_ratio(FLT_MAX)
	, minimum_radius(FLT_MAX)
	/*, isCluster(NULL)
	, cluster_index(NULL)
	, cluster_count(NULL)
	, cluster_begin(NULL)
	, cluster_set_location(NULL)
	
	
	, n_single_sphere(0)
	, n_cluster_sphere(0)*/
// 	, is_realtime_creating(false)
// 	, one_by_one(false)
{

}

xParticleManager::~xParticleManager()
{
	if (r_pos) delete[] r_pos; r_pos = NULL;
	if (r_vel) delete[] r_vel; r_vel = NULL;
	//xpcos.delete_all();
	/*if (isCluster) delete[] isCluster; isCluster = NULL;
	if (cluster_index) delete[] cluster_index; cluster_index = NULL;
	if (cluster_count) delete[] cluster_count; cluster_count = NULL;
	if (cluster_begin) delete[] cluster_begin; cluster_begin = NULL;
	if (cluster_set_location) delete[] cluster_set_location; cluster_set_location = NULL;*/
}

unsigned int xParticleManager::NumParticle()
{
	return np;
}

unsigned int xParticleManager::NumMassParticle()
{
	return n_mass_particle;
}

double xParticleManager::CriticalDensity()
{
	return minimum_particle_density;
}

double xParticleManager::CriticalPoisson()
{
	return minimum_poisson_ratio;
}

double xParticleManager::CriticalYoungs()
{
	return maximum_youngs_modulus;
}

double xParticleManager::CriticalRadius()
{
	return minimum_radius;
}

//unsigned int xParticleManager::NumClusterSet()
//{
//	return n_cluster_object;
//}

double* xParticleManager::GetPositionResultPointer(unsigned int pt)
{
	return r_pos + (np * 4 * pt);
}

double* xParticleManager::GetVelocityResultPointer(unsigned int pt)
{
	return r_vel + (np * 3 * pt);
}

// void xParticleManager::setRealTimeCreating(bool b)
// {
// 	is_realtime_creating = b;
// }
// 
// unsigned int xParticleManager::RealTimeCreating()
// {
// 	return is_realtime_creating;
// }
// 
// bool xParticleManager::OneByOneCreating()
// {
// 	return one_by_one;
// }

xParticleObject* xParticleManager::XParticleObject(std::string ws)
{
	//QStringList keys = xpcos.keys();
	if (!xpcos.size())
		return NULL;
	xmap<xstring, xParticleObject*>::iterator it = xpcos.find(ws);//QStringList::const_iterator it = qFind(keys, ws);
	if (it == xpcos.end())
		return NULL;
	return it.value();// xpcos[ws];
}

unsigned int xParticleManager::nClusterObject()
{
	return n_cluster_object;
}

unsigned int xParticleManager::NumCluster()
{
	return ncluster;
}

unsigned int xParticleManager::nClusterEach()
{
	return n_cluster_each;
}
//
//unsigned int xParticleManager::nSingleSphere()
//{
//	return n_single_sphere;
//}
//
//unsigned int xParticleManager::nClusterSphere()
//{
//	return n_cluster_sphere;
//}
//
//unsigned int xParticleManager::NumParticleWithCluster()
//{
//	return n_single_sphere + n_cluster_sphere;
//}
//
//unsigned int* xParticleManager::ClusterIndex()
//{
//	return cluster_index;
//}
//
//unsigned int * xParticleManager::ClusterCount()
//{
//	return cluster_count;
//}
//
//unsigned int * xParticleManager::ClusterBegin()
//{
//	return cluster_begin;
//}

unsigned int xParticleManager::GetNumCubeParticles(
	double dx, double dy, double dz, double min_radius, double max_radius)
{
	vector3i ndim;

	double r = min_radius;
	double diameter = 2.0 * r;
	vector3d dimension = new_vector3d(dx, dy, dz);
	ndim = ToVector3I(dimension / diameter) - new_vector3i(1, 1, 1);
	
	return ndim.x * ndim.y * ndim.z;
}

unsigned int xParticleManager::GetNumLineParticles(double sx, double sy, double sz, double ex, double ey, double ez, double min_radius, double max_radius)
{
	unsigned int n = 0;
	vector3d dp = new_vector3d(ex - sx, ey - sy, ez - sz);
	double len = length(dp);
	n = static_cast<unsigned int>(floor(len / (2.0 * max_radius)));
	return n;
}

unsigned int xParticleManager::GetNumPlaneParticles(double dx, unsigned int ny, double dz, double min_radius, double max_radius)
{
	vector3ui ndim;
	double r = max_radius;
	double diameter = 2.0 * r;
	vector3d dimension = new_vector3d(dx, 0, dz);
	ndim = ToVector3UI(dimension / diameter) - new_vector3ui(1, 0, 1);
	return ndim.x * ny * ndim.z;
}

unsigned int xParticleManager::GetNumCircleParticles(
	double d, double min_radius, double max_radius)
{
	double cr = 0.5 * d;
	unsigned int nr = static_cast<unsigned int>(cr / (2.0 * max_radius)) - 1;
	double rr = 2.0 * max_radius * nr + max_radius;
	unsigned int cnt = 0;
	double space = (cr - rr) / (nr + 1);
	for (unsigned int i = 1; i <= nr; i++)
	{
		double _r = i * (2.0 * max_radius + space);
		double dth = (2.0 * max_radius + space) / _r;
		unsigned int npr = static_cast<unsigned int>((2.0 * M_PI) / dth);
		dth = ((2.0 * M_PI) / npr);
		for (unsigned int j = 0; j < npr; j++)
		{
			cnt++;
		}
	}
	return cnt;
}

void xParticleManager::setCriticalMaterial(double d, double y, double p)
{
	if (minimum_particle_density > d) minimum_particle_density = d;
	if (maximum_youngs_modulus < y) maximum_youngs_modulus = y;
	if (minimum_poisson_ratio > p) minimum_poisson_ratio = p;
}

xParticleObject * xParticleManager::CreateLineParticle(std::string n, xMaterialType mt, unsigned int _np, xLineParticleData & d)
{
	xstring name = n;
	xParticleObject* xpo = new xParticleObject(n);
	vector4d* pos = xpo->AllocMemory(_np);
	//double* mass = xpo->Mass();
	xpo->setStartIndex(np);
	xpo->setMaterialType(mt);
	//n_single_sphere += _np;
	np += _np;
	xMaterial xm = GetMaterialConstant(mt);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	xpo->setMinRadius(d.minr);
	xpo->setMaxRadius(d.maxr);
	setCriticalMaterial(xm.density, xm.youngs, xm.poisson);

	if (d.minr == d.maxr)
	{
		double r = d.minr;
		double diameter = 2.0 * r;
		vector3d sp = new_vector3d(d.sx, d.sy, d.sz);
		vector3d ep = new_vector3d(d.ex, d.ey, d.ez);
		vector3d dp = ep - sp;
		vector3d dir = dp / length(dp);
		double gab = length(dp) - (diameter * (_np - 1));
		double space = gab / _np;
		for (unsigned int i = 0; i < _np; i++)
		{
			vector3d p = sp + i * (diameter + space) * dir;
			pos[i] = new_vector4d(p.x, p.y, p.z, r);
		}
		if (minimum_radius > d.minr) minimum_radius = d.minr;
	}
	//xpo->set
	SetMassAndInertia(xpo);
	xpcos.insert(name, xpo);
	xObjectManager::XOM()->addObject(xpo);
	return xpo;
}

xParticleObject* xParticleManager::CreateParticleFromList(
	std::string n, xMaterialType mt, unsigned int _np, vector4d* d, double* m)
{
	xstring name = n;
	xParticleObject* xpo = new xParticleObject(n);
	vector4d* pos = xpo->AllocMemory(_np);
	double* ms = xpo->Mass();
	xpo->setStartIndex(np);
	//n_single_sphere += _np;
	np += _np;
	xMaterial xm = GetMaterialConstant(mt);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	
	setCriticalMaterial(xm.density, xm.youngs, xm.poisson);
	double min_r = d[0].w;
	double max_r = d[0].w;
	for (unsigned int i = 0; i < _np; i++)
	{
		pos[xpo->StartIndex() + i] = new_vector4d(d[i].x, d[i].y, d[i].z, d[i].w);
		min_r = min(min_r, d[i].w);
		max_r = max(max_r, d[i].w);
		ms[i] = m[i];
	}
	if (minimum_radius > min_r) minimum_radius = min_r;
	xpo->setMinRadius(min_r);
	xpo->setMaxRadius(max_r);
	SetMassAndInertia(xpo);
	xpcos.insert(name, xpo);
	xObjectManager::XOM()->addObject(xpo);
	return xpo;
}

// xParticleObject* xParticleManager::CreateSPHParticles(xObject* xobj, double ps, unsigned int nlayer)
// {
// 	xParticleObject* xpo = new xParticleObject(xobj->Name().toStdString());
// 	//vector4d* pos = xpo->AllocMemory(_np);
// 	xpo->setStartIndex(np);
// 	xpo->setMaterialType(xobj->Material());
// 	//np += _np;
// 	xMaterial xm = GetMaterialConstant(xobj->Material());
// 	xpo->setDensity(xm.density);
// 	xpo->setYoungs(xm.youngs);
// 	xpo->setPoisson(xm.poisson);
// 	xpo->setMinRadius(ps * 0.5);
// 	xpo->setMaxRadius(ps * 0.5);
// 	unsigned int _np = xobj->create_sph_particles(ps, nlayer);
// 	np += _np;
// 	xobj->create_sph_particles(ps, nlayer, xpo->AllocMemory(_np));
// 	xpcos[xobj->Name()] = xpo;
// 	xObjectManager::XOM()->addObject(xpo);
// 	return xpo;
// }

// void xParticleManager::create_sph_particles_with_plane_shape(
// 	double dx, double dy, double lx, double ly, double lz, double ps)
// {
// 
// }

// unsigned int xParticleManager::GetNumSPHPlaneParticles(double dx, double dy, double ps)
// {
// 	unsigned int nx = static_cast<unsigned int>((dx / ps) + 1e-9) - 1;
// 	unsigned int ny = static_cast<unsigned int>((dy / ps) + 1e-9) - 1;
// 	return nx * ny;
// }

// xParticleObject* xParticleManager::CreateSPHPlaneParticleObject(
// 	std::string n, xMaterialType mt, xSPHPlaneObjectData& d)
// {
// 	QString name = QString::fromStdString(n);
// 	xParticleObject* xpo = new xParticleObject(n);
// 	//vector4d* pos = xpo->AllocMemory(_np);
// 	//xpo->setStartIndex(np);
// 	xpo->setMaterialType(mt);
// 	//np += _np;
// 	xMaterial xm = GetMaterialConstant(mt);
// 	xpo->setDensity(xm.density);
// 	xpo->setYoungs(xm.youngs);
// 	xpo->setPoisson(xm.poisson);
// 	xpo->setMinRadius(d.ps * 0.5);
// 	xpo->setMaxRadius(d.ps * 0.5);
// // 	unsigned int nx = static_cast<unsigned int>((d.dx / d.ps) + 1e-9) - 1;
// // 	unsigned int ny = static_cast<unsigned int>((d.dy / d.ps) + 1e-9) - 1;
// // 	unsigned int count = 0;
// // 	for (unsigned int x = 0; x < nx; x++)
// // 	{
// // 		double px = x * d.lx + d.ps;
// // 		for (unsigned int y = 0; y < ny; y++)
// // 		{
// // 			pos[count].x = px;
// // 			pos[count].y = y * d.ly + d.ps;
// // 			pos[count].z = 0.0;
// // 			count++;
// // 		}
// // 	}
// 	xpcos[name] = xpo;
// 	xObjectManager::XOM()->addObject(xpo);
// 	return xpo;
// }

void xParticleManager::SetCurrentParticlesFromPartResult(std::string path)
{
	if (c_filepath.size() == 0 && path.empty())
		return;
	std::string c_path = path.empty() ? c_filepath.toStdString() : path;
	std::fstream qf;
	std::cout << "SetCurrentParticleFromPartResult : " << path << std::endl;
	qf.open(c_path, std::ios::binary | std::ios::in);
	double m_ct = 0;
	unsigned int m_np = 0;
	unsigned int m_ns = 0;
	if (qf.is_open())
	{
		qf.read((char*)&m_ct, sizeof(double));
		qf.read((char*)&m_np, sizeof(unsigned int));
		qf.read((char*)&m_ns, sizeof(unsigned int));
		if (np != m_np)
		{
			std::cout << "Failed setCurrentParticlesFromPartResult" << std::endl;
			qf.close();
			return;
		}
		vector4d* m_pos = new vector4d[m_np];
		vector3d* m_vel = new vector3d[m_ns];
		vector3d* m_acc = new vector3d[m_ns];
		vector4d* m_ep = new vector4d[m_ns];
		vector4d* m_avel = new vector4d[m_ns];
		vector4d* m_aavel = new vector4d[m_ns];
		vector4d* m_cpos = NULL;
		if (m_np != m_ns)
			m_cpos = new vector4d[m_ns];
		qf.read((char*)m_pos, sizeof(vector4d) * m_np);
		qf.read((char*)m_vel, sizeof(vector3d) * m_ns);
		qf.read((char*)m_acc, sizeof(vector3d) * m_ns);
		qf.read((char*)m_ep, sizeof(vector4d) * m_ns);
		qf.read((char*)m_avel, sizeof(vector4d) * m_ns);
		qf.read((char*)m_aavel, sizeof(vector4d) * m_ns);
		if(m_cpos)
			qf.read((char*)m_cpos, sizeof(vector4d) * m_ns);
		for(xmap<xstring, xParticleObject*>::iterator it = xpcos.begin(); it != xpcos.end(); it.next())//foreach(xParticleObject* xpo, xpcos)
		{
			xParticleObject* xpo = it.value();
 			unsigned int sid = xpo->StartIndex();
			unsigned int xnp = xpo->NumParticle();
			unsigned int xns = xpo->NumCluster();
			vector4d* xpo_pos = xpo->Position();
			vector4d* xpo_ep = xpo->EulerParameters();
			memcpy(xpo_pos, m_pos + sid, sizeof(vector4d) * xnp);
			if (xns && m_cpos)
			{
				vector4d* xpo_cpos = xpo->ClusterPosition();
				unsigned int c_sid = xpo->StartClusterIndex();
				memcpy(xpo_cpos, m_cpos + c_sid, sizeof(vector4d) * xns);
				memcpy(xpo_ep, m_ep + c_sid, sizeof(vector4d) * xns);
			}
		/*	else
			{
				memcpy(xpo_ep, m_ep + sid, sizeof(vector4d) * xnp);
			}*/
		//	qf.read((char*)m_pos, sizeof()
		//	xpccs.erase(xpo->Name());
		}
		delete[] m_pos;
		delete[] m_vel;
		delete[] m_acc;
		delete[] m_ep;
		delete[] m_avel;
		delete[] m_aavel;
		if (m_cpos) delete[] m_cpos;
		
	}
	qf.close();
}

xParticleObject* xParticleManager::CreateCubeParticle(
	std::string n, xMaterialType mt, unsigned int _np, xCubeParticleData& d)
{
	xstring name = n;
	xParticleObject* xpo = new xParticleObject(n);
	vector4d* pos = xpo->AllocMemory(_np);
	//double* mass = xpo->Mass();
	xpo->setStartIndex(np);
	xpo->setMaterialType(mt);
	//n_single_sphere += _np;
	np += _np;
	xMaterial xm = GetMaterialConstant(mt);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	xpo->setMinRadius(d.minr);
	xpo->setMaxRadius(d.maxr);
	setCriticalMaterial(xm.density, xm.youngs, xm.poisson);

	double r = d.maxr;
	double diameter = 2.0 * r;
	vector3d dimension = new_vector3d(d.dx, d.dy, d.dz);
	vector3ui ndim = new_vector3ui
	(
		static_cast<unsigned int>(dimension.x / diameter) - 1,
		static_cast<unsigned int>(dimension.y / diameter) - 1,
		static_cast<unsigned int>(dimension.z / diameter) - 1
	);
	vector3d plen = diameter * ToVector3D(ndim);// .To<double>();
	vector3d space = dimension - plen;
	space.x /= ndim.x + 1;
	space.y /= ndim.y + 1;
	space.z /= ndim.z + 1;
	vector3d gab = new_vector3d(diameter + space.x, diameter + space.y, diameter + space.z);
	vector3d ran = r * space;
	unsigned int cnt = 0;
	for (unsigned int z = 0; z < ndim.z; z++) {
		double _z = d.lz + r + space.z;
		for (unsigned int y = 0; y < ndim.y; y++) {
			double _y = d.ly + r + space.y;
			for (unsigned int x = 0; x < ndim.x; x++) {
				double _x = d.lx + r + space.x;
				vector4d p = new_vector4d
				(
					_x + x * gab.x + ran.x * frand(),
					_y + y * gab.y + ran.y * frand(),
					_z + z * gab.z + ran.z * frand(), r
				);
				pos[cnt] = p;
				//mass[cnt] = 0.0;
				cnt++;
			}
		}
	}
	if (minimum_radius > d.minr) minimum_radius = d.minr;

	if (d.minr != d.maxr) {
		for (unsigned int i = 0; i < _np; i++) {
			pos[i].w = d.minr + (d.maxr - d.minr) * frand();
		}
	}
	//xpo->set
	SetMassAndInertia(xpo);
	xpcos.insert(name, xpo);
	xObjectManager::XOM()->addObject(xpo);
	return xpo;
}

xParticleObject* xParticleManager::CreatePlaneParticle(
	std::string n, xMaterialType mt, xPlaneParticleData& d)
{
	xstring name = n;
	xParticleObject* xpo = new xParticleObject(n);
//	vector4d* pos = xpo->AllocMemory(_np);

	xpo->setStartIndex(np);
	xpo->setMaterialType(mt);
	//np += _np;
	xMaterial xm = GetMaterialConstant(mt);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	xpo->setMinRadius(d.minr);
	xpo->setMaxRadius(d.maxr);
	setCriticalMaterial(xm.density, xm.youngs, xm.poisson);
	
	//double r = d.maxr;
	//unsigned int nr = static_cast<unsigned int>(cr / (2.0 * r)) - 1;
//	double rr = 2.0 * r * nr + r;
	//double space = (cr - rr) / (nr + 1);
//	unsigned int cnt = 0;
//	unsigned int k = 0;
	bool isStopCreating = false;
	vector3d u = new_vector3d(d.drx, d.dry, d.drz);
	vector3d pu = new_vector3d(-u.y, u.x, u.z);
	vector3d qu = cross(u, pu);
	vector3d sp = new_vector3d(d.lx, d.ly, d.lz);
	matrix33d A = { u.x, pu.x, qu.x, u.y, pu.y, qu.y, u.z, pu.z, qu.z };
	double cr = 0.0;
	
	double space = 1e-6;
	unsigned int nx = static_cast<unsigned int>(d.dx / (2.0 * d.maxr)) - 1;
	double space_x = (d.dx - 2.0 * nx * d.maxr) / (nx + 1);
	std::list<vector4d> plist;
	unsigned int nz = static_cast<unsigned int>(d.dz / (2.0 * d.maxr)) - 1;
	double space_z = (d.dz - 2.0 * nz * d.maxr) / (nz + 1);
	//space_z /= nz + 1;
	for (unsigned int i = 0; i < d.dy; i++)
	{
		bool is_end_dx = false;
		for(unsigned int z = 0; z < nz; z++)
		{
			double pr = 0.0;
			double x_pos = 0;
			double gab_z = (2.0 * d.maxr + space_z) * z + space_z;
			for(unsigned int x = 0; x < nx; x++)
			{
				//cr = d.minr + (d.minr != d.maxr ? (d.maxr - d.minr) * frand() : 0.0);
				x_pos = (2.0 * d.maxr + space_x) * x + space_x;//x_pos + (pr + cr + space);
				vector3d pp = new_vector3d(0.0, x_pos + d.maxr + space_x, gab_z + d.maxr + space_z);
				//if (pp.y + cr + space >= d.dx) break;
				pp = sp + A * pp;
				plist.push_back(new_vector4d(pp.x, pp.y, pp.z, d.maxr));
				//pr = cr;
				//cnt++;
			}
		}
	}
	if (plist.size())
	{
		unsigned int _np = static_cast<unsigned int>(plist.size());
		vector4d* pos = xpo->AllocMemory(_np);
		np += _np;
		unsigned int cnt = 0;
		for (std::list<vector4d>::iterator it = plist.begin(); it != plist.end(); it++) {
			pos[cnt] = *it;
			if(d.minr != d.maxr) 
				pos[cnt++].w = d.minr + (d.maxr - d.minr) * frand();
		}			
	}
	if (minimum_radius > d.minr) minimum_radius = d.minr;
	//SetMassAndInertia(xpo);
	SetMassAndInertia(xpo);
	xpcos.insert(name, xpo);
	xObjectManager::XOM()->addObject(xpo);
	return xpo;
}

double xParticleManager::GetMaxParticleRadius()
{
	return maximum_radius;
}

xParticleObject* xParticleManager::CreateCircleParticle(
	std::string n, xMaterialType mt, unsigned int _np, xCircleParticleData& d)
{
	xstring name = n;
	xParticleObject* xpo = new xParticleObject(n);
	vector4d* pos = xpo->AllocMemory(_np);
	//double* mass = xpo->Mass();
	
	xpo->setStartIndex(np);
	xpo->setMaterialType(mt);
	//n_single_sphere += _np;
	np += _np;
	xMaterial xm = GetMaterialConstant(mt);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	xpo->setMinRadius(d.minr);
	xpo->setMaxRadius(d.maxr);
	setCriticalMaterial(xm.density, xm.youngs, xm.poisson);
	SetMassAndInertia(xpo);
	double r = d.maxr;
	double cr = 0.5 * d.diameter;
	unsigned int nr = static_cast<unsigned int>(cr / (2.0 * r)) - 1;
	double rr = 2.0 * r * nr + r;
	double space = (cr - rr) / (nr + 1);
	unsigned int cnt = 0;
	unsigned int k = 0;
	bool isStopCreating = false;
	vector3d u = new_vector3d(d.dx, d.dy, d.dz);
	vector3d pu = new_vector3d(-u.y, u.x, u.z);
	vector3d qu = cross(u, pu);
	vector3d sp = new_vector3d(d.sx, d.sy, d.sz);
	matrix33d A = { u.x, pu.x, qu.x, u.y, pu.y, qu.y, u.z, pu.z, qu.z };
	/*while (1)
	{*/
		for (unsigned int i = 1; i <= nr; i++)
		{
			double _r = i * (2.0 * r + space);
			double dth = (2.0 * r + space) / _r;
			double th = 0.5 * k * r / _r;
			unsigned int npr = static_cast<unsigned int>((2.0 * M_PI) / dth);
			dth = ((2.0 * M_PI) / npr);
			//double rval = r * space * frand();
			for (unsigned int j = 0; j < npr; j++)
			{
				vector3d pp = new_vector3d(0, _r * sin(dth * j), _r * cos(dth * j));
				vector4d new_pp = new_vector4d(pp.x, pp.y, pp.z, r);				
				new_pp.x = pp.x * cos(th) + pp.z * sin(th);
				new_pp.z = -pp.x * sin(th) + pp.z * cos(th);
				vector3d gpp = sp + A * pp;
				pos[cnt] = new_vector4d(gpp.x, gpp.y, gpp.z, r);
			//	mass[cnt] = 0.0;
				cnt++;
				/*if (cnt == _np)
				{
					isStopCreating = true;
					break;
				}*/
			}
			//if (isStopCreating) break;
		}
	//	if (isStopCreating) break;
	//	k++;
	//}
	
	
	if (d.minr != d.maxr)
	{
		double dr = d.maxr - d.minr;
		srand(GetTickCount());
		for (unsigned int i = 0; i < _np; i++)
		{
			pos[xpo->StartIndex() + i].w = d.minr + dr * frand();
		}
	}
	if (minimum_radius > d.minr) minimum_radius = d.minr;
	SetMassAndInertia(xpo);
	xpcos.insert(name, xpo);
	xObjectManager::XOM()->addObject(xpo);
	return xpo;
}


xParticleObject* xParticleManager::CreateMassParticle(
	std::string n, xMaterialType mt, double rad, xPointMassData& d)
{
	xstring name = n;
	xParticleObject* xpo = new xParticleObject(n);
	vector4d* pos = xpo->AllocMemory(1);
	vector3d* iner = xpo->Inertia();
	double* mass = xpo->Mass();
	xpo->setStartIndex(np);
	xpo->setMaterialType(mt);
	xpo->setShapeForm(NO_SHAPE_AND_MASS);
	//n_single_sphere += _np;
	xpo->setMassIndex(np);
	np += 1;
	n_mass_particle += 1;
	xMaterial xm = GetMaterialConstant(mt);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	xpo->setMinRadius(rad);
	xpo->setMaxRadius(rad);
	setCriticalMaterial(xm.density, xm.youngs, xm.poisson);
	pos[0] = new_vector4d(d.px, d.py, d.pz, rad);
	iner[0] = new_vector3d(d.ixx, d.iyy, d.izz);
	mass[0] = d.mass;
	xpcos.insert(name, xpo);
	xObjectManager::XOM()->addObject(xpo);
	return xpo;
}

xParticleObject * xParticleManager::CreateClusterParticle(
	std::string n, xMaterialType mt, vector3d& loc, vector3i& grid, xClusterObject * xo)
{
	unsigned int _np = grid.x * grid.y * grid.z;
	unsigned int neach = xo->NumElement();
	unsigned int rnp = _np * neach;
	double min_rad = xo->ElementMinimumRadius();
	double max_rad = xo->ElementMaximumRadius();
	xstring name = n;
	xMaterial xm = GetMaterialConstant(mt);
	xParticleObject* xpo = new xParticleObject(n);
	xpo->setStartIndex(np);
	xpo->setClusterStartIndex(ncluster);
	xpo->setMaterialType(mt);
	xpo->setShapeForm(CLUSTER_SHAPE);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	xpo->setMinRadius(min_rad);
	xpo->setMaxRadius(max_rad);
	xpo->setEachCount(neach);
	xpo->setParticleShapeName(xo->Name());
	setCriticalMaterial(xm.density, xm.youngs, xm.poisson);
	vector4d* pos = xpo->AllocMemory(rnp);
	vector4d* cpos = xpo->AllocClusterMemory(_np);
	double* mass = xpo->Mass();
	vector3d* inertia = xpo->Inertia();
	//vector3d* inertia = xpo->AllocInertiaMemory(_np);
	
	euler_parameters* ep = (euler_parameters*)xpo->EulerParameters();
	//n_cluster_sphere += _np;
	ncluster += _np;
	np += _np * neach;
	double norm = 0;
	vector4d* rloc = xo->RelativeLocation();
	xpo->setRelativeLocation(xo->RelativeLocation());
	double c_rad = 0.0;
	unsigned int cnt = 0;
	double rad = max_rad;
	double dr = max_rad - min_rad;
	for (int x = 0; x < grid.x; x++)
	{
		for (int y = 0; y < grid.y; y++)
		{
			for (int z = 0; z < grid.z; z++)
			{
				vector3d cp = new_vector3d(
					loc.x + 2.0 * c_rad * x + x * 1e-6,
					loc.y + 2.0 * c_rad * y + y * 1e-6,
					loc.z + 2.0 * c_rad * z + z * 1e-6);
				vector3d rot = new_vector3d(180 * frand(), 180 * frand(), 180 * frand());
				euler_parameters m_ep = EulerAngleToEulerParameters(rot);
				matrix33d A = GlobalTransformationMatrix(m_ep);
				for (unsigned int j = 0; j < neach; j++)
				{
					vector3d m_pos = cp + A * new_vector3d(rloc[j].x, rloc[j].y, rloc[j].z);
					pos[cnt * neach + j] = new_vector4d(m_pos.x, m_pos.y, m_pos.z, rloc[j].w);
				}
				cpos[cnt] = new_vector4d(cp.x, cp.y, cp.z, rad);
				ep[cnt] = m_ep;
				norm = length(new_vector4d(m_ep.e0, m_ep.e1, m_ep.e2, m_ep.e3));
				if (!c_rad)	{
					c_rad = xUtilityFunctions::FitClusterRadius(pos, neach, 0.5 * (max_rad + min_rad));
				}
				cnt++;
			}
		}
	}
	SetClusterMassAndInertia(xpo);
	//if (minimum_radius > rad) minimum_radius = rad;
	xpcos.insert(name, xpo);
	xObjectManager::XOM()->addObject(xpo);
	n_cluster_each += neach;
	n_cluster_object++;
	return xpo;
}

void xParticleManager::CopyClusterInformation(xClusterInformation * xci, double* rcloc)
{
	unsigned int cnt = 0;
	vector3d* rloc = (vector3d*)rcloc;
	for (xmap<xstring, xParticleObject*>::iterator it = xpcos.begin(); it != xpcos.end(); it.next())// foreach(xParticleObject* xpo, xpcos)
	{
		xParticleObject* xpo = it.value();
		if (xpo->ShapeForm() == CLUSTER_SHAPE)
		{
			xci[cnt].sid = xpo->StartClusterIndex();
			xci[cnt].neach = xpo->EachCount();
			xci[cnt].count = xpo->NumCluster();
			memcpy(rloc + xci[cnt].sid, xpo->RelativeLocation(), sizeof(vector4d) * xci[cnt].neach);
			cnt++;
		}
	}
}

xmap<xstring, xParticleObject*>& xParticleManager::XParticleObjects()
{
	return xpcos;
}

bool xParticleManager::CopyPosition(
	double *pos, double* cpos, double* ep, unsigned int inp)
{
	for (xmap<xstring, xParticleObject*>::iterator it = xpcos.begin(); it != xpcos.end(); it.next())// foreach(xParticleObject* xpo, xpcos)
	{
		xParticleObject* xpo = it.value();
		xpo->CopyPosition(pos);
		if (cpos && xpo->ShapeForm() == CLUSTER_SHAPE)
			xpo->CopyClusterPosition(cpos, ep);
	}
	return true;
}


bool xParticleManager::CopyMassAndInertia(double * mass, double * inertia)
{
	for (xmap<xstring, xParticleObject*>::iterator it = xpcos.begin(); it != xpcos.end(); it.next())// foreach(xParticleObject* xpo, xpcos)
	{
		it.value()->CopyMassAndInertia(mass, (vector3d* )inertia);
	}
	return false;
}

bool xParticleManager::SetClusterMassAndInertia(xParticleObject* xpo)
{
	unsigned int c = 0;
	/*foreach(xParticleObject* xpo, xpcos)
	{*/
	double* mass = xpo->Mass();
	vector3d* inertia = xpo->Inertia();
	double d = xpo->Density();
	//vector4d* v = xpo->Position();
	unsigned int sid = xpo->StartIndex();
	vector4d* cpos = xpo->ClusterPosition();
	vector4d* rloc = xpo->RelativeLocation();

	for (unsigned int i = 0; i < xpo->NumCluster(); i++)
	{
		double m = d * (4.0 / 3.0) * M_PI * pow(cpos[i].w, 3.0);
		mass[i] = m * xpo->EachCount();
		double J = (2.0 / 5.0) * m * pow(cpos[i].w, 2.0);
		vector3d J3 = new_vector3d(0, 0, 0);
		vector3d m_pos = new_vector3d(cpos[i].x, cpos[i].y, cpos[i].z);
		for (unsigned int j = 0; j < xpo->EachCount(); j++)
		{
			vector4d dr = rloc[j];
			J3.x += m * (dr.y * dr.y + dr.z * dr.z);
			J3.y += m * (dr.x * dr.x + dr.z * dr.z);
			J3.z += m * (dr.x * dr.x + dr.y * dr.y);
		}
		inertia[i] = J3;
		c++;
	}
	xpo->setMassIndex(c);
	//}
	return false;
}

bool xParticleManager::SetMassAndInertia(xParticleObject* xpo)
{
	unsigned int c = 0;

	double d = xpo->Density();
	vector4d* pos = xpo->Position();
	double* mass = xpo->Mass();
	vector3d* inertia = xpo->Inertia();
	
	for (unsigned int i = 0; i < xpo->NumParticle(); i++)
	{
		double m = d * (4.0 / 3.0) * M_PI * pow(pos[i].w, 3.0);
		double J = (2.0 / 5.0) * m * pow(pos[i].w, 2.0);
		mass[i] = mass[i] ? mass[i] : m;
		inertia[i] = new_vector3d(J, J, J);
		//	}	
		c++;
	}
	xpo->setMassIndex(c);
		//}	
	//}
	/*foreach(xParticleObject* xpo, xpcos)
	{

		if (xpo->ShapeForm() == NO_SHAPE_AND_MASS)
		{
			vector4d* pos = xpo->Position();
			double* ms = xpo->Mass();
			vector3d* J = xpo->Inertia();
			unsigned int sid = xpo->StartIndex();
			unsigned int jid = np - n_mass_particle;
			xpo->setMassIndex(c);
			for (unsigned int i = 0; i < xpo->NumParticle(); i++)
			{
				if (xpo->ShapeForm() == NO_SHAPE_AND_MASS)
				{
					mass[i + sid] = ms[i];
					inertia[jid + i * 3 + 0] = J[i].x;
					inertia[jid + i * 3 + 1] = J[i].y;
					inertia[jid + i * 3 + 2] = J[i].z;
				}
			}
		}
	}*/

	return true;
}

void xParticleManager::ExportParticleDataForView(std::string path)
{
	std::fstream of;
	of.open(path, std::ios::out | std::ios::binary);
	of.write((char*)&np, sizeof(unsigned int));
	//double* m_mass = new double[np];
	for (xmap<xstring, xParticleObject*>::iterator it = xpcos.begin(); it != xpcos.end(); it.next())
	{
		xParticleObject* po = it.value();
		unsigned int _sid = po->StartIndex();
		unsigned int _np = po->NumParticle();
		unsigned int _cnp = po->NumCluster();
		double d[2] = { po->MinRadius(), po->MaxRadius() };
		double* _pos = (double *)po->Position();
		unsigned int mat = po->Material();
		unsigned int ns = static_cast<unsigned int>(po->Name().size());
		of.write((char*)&ns, sizeof(int));
		of.write((char*)po->Name().c_str(), sizeof(char) * ns);
		of.write((char*)&mat, sizeof(int));
		of.write((char*)&_sid, sizeof(unsigned int));
		of.write((char*)&_np, sizeof(unsigned int));
		of.write((char*)&_cnp, sizeof(unsigned int));
		of.write((char*)&d, sizeof(double) * 2);
		of.write((char*)_pos, sizeof(double) * _np * 4);
		of.write((char*)po->Mass(), sizeof(double) * (_cnp ? _cnp : _np));
		//memcpy(m_mass + _sid, po->Mass(), sizeof(double) * (_cnp ? _cnp : _np));
	}
	//of.write((char*)m_mass, sizeof(double) * np);
	//delete[] m_mass;
	of.close();
}

void xParticleManager::AllocParticleResultMemory(unsigned int npart, unsigned int np)
{
	if (r_pos) delete[] r_pos; r_pos = NULL;
	if (r_vel) delete[] r_vel; r_vel = NULL;
	unsigned int n = npart * np;
	r_pos = new double[n * 4];
	r_vel = new double[n * 3];
	memset(r_pos, 0, sizeof(double) * n * 4);
	memset(r_vel, 0, sizeof(double) * n * 3);
}

void xParticleManager::SetChangeParticlesFilePath(std::string path)
{
	c_filepath = path;
}

void xParticleManager::AddParticleCreatingCondition(xParticleObject* xpo, xParticleCreateCondition& xpcc)
{
	xpccs.insert(xpo->Name(), xpcc);
	random_device ran;
	mt19937_64 rnd(ran());
	uniform_int_distribution<int> range(0, xpo->EachCount() - 1);
	vector<unsigned int> iList;
	while (iList.size() != xpo->EachCount())
	{
		unsigned int ni = range(rnd);// ran.bounded(xpo->EachCount());
		if (std::find(iList.begin(), iList.end(), ni) != iList.end())
			continue;
		iList.push_back(ni);
	}

	unsigned int it = 0;
	unsigned int k = 0;
	vector<unsigned int>::iterator iter = iList.begin();
	vector<vector4d> pList;
	vector<double> mList;
	vector<vector3d> jList;
	vector4d* mpos = xpo->Position();
	double* mmass = xpo->Mass();
	vector3d* miner = xpo->Inertia();
	for (unsigned int i = 0; i < xpo->NumParticle(); i++)
	{
		pList.push_back(mpos[i]);
		mList.push_back(mmass[i]);
		jList.push_back(miner[i]);
	}
		
	np = np - xpo->NumParticle();
	xpo->resizeParticles(xpcc.count);
	np = np + xpo->NumParticle();
	while (1)
	{
		int sign = k % 2 ? 1 : -1;
		if ((k + 1) * xpo->EachCount() > xpo->NumParticle())
			it = xpo->NumParticle() - k * xpo->EachCount();
		else
			it = xpo->EachCount();
		vector4d* pos = xpo->Position() + k * xpo->EachCount();
		double* mass = xpo->Mass() + k * xpo->EachCount();
		vector3d* inertia = xpo->Inertia() + k * xpo->EachCount();

		unsigned int over = 0;
		for(unsigned int i = 0; i < iList.size(); i++)
		//foreach(vector4d v, pList)
		{
			if (over >= it) 
				break;
			while (*iter >= it) { iter++; }
			pos[*iter] = pList.at(i) + sign * xpo->MaxRadius() * frand() * new_vector4d(0.1, 0.1, 0.1, 0.0);
			mass[*iter] = mList.at(i);
			inertia[*iter] = jList.at(i);
			iter++;
			over++;
		}
		iter = iList.begin();
		//pList.clear();
		k++;
		if (k * xpo->EachCount() > xpo->NumParticle())
			break;

	}
}

unsigned int xParticleManager::ExcuteCreatingCondition(
	double ct, unsigned int cstep, unsigned int cnp)
{
	xmap<xstring, xParticleCreateCondition>::iterator it = xpccs.begin();
	for(; it != xpccs.end(); it.next())
	{
		if (cstep == 1)
			return it.value().neach;
		if (cnp > it.value().sid && cnp < it.value().sid + it.value().count)
		{
			if (!(cstep % it.value().nstep))
			{
				if (cnp + it.value().neach > np)
					return np;
				else
					return cnp + it.value().neach;
			}
				
		}
	}
	return cnp;
}
