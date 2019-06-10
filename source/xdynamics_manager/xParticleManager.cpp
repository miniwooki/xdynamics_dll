#include "xdynamics_manager/xParticleMananger.h"
#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_manager/xObjectManager.h"

xParticleManager::xParticleManager()
	: np(0)
	, r_pos(NULL)
	, r_vel(NULL)
	, cluster_index(NULL)
	, cluster_count(NULL)
	, cluster_begin(NULL)
	, cluster_set_location(NULL)
	, n_cluster_object(0)
	, n_cluster_each(0)
// 	, is_realtime_creating(false)
// 	, one_by_one(false)
{

}

xParticleManager::~xParticleManager()
{
	if (r_pos) delete[] r_pos; r_pos = NULL;
	if (r_vel) delete[] r_vel; r_vel = NULL;
	if (cluster_index) delete[] cluster_index; cluster_index = NULL;
	if (cluster_count) delete[] cluster_count; cluster_count = NULL;
	if (cluster_begin) delete[] cluster_begin; cluster_begin = NULL;
	if (cluster_set_location) delete[] cluster_set_location; cluster_set_location = NULL;
}

unsigned int xParticleManager::NumParticle()
{
	return np;
}

unsigned int xParticleManager::NumClusterSet()
{
	return n_cluster_object;
}

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

xParticleObject* xParticleManager::XParticleObject(QString& ws)
{
	QStringList keys = xpcos.keys();
	QStringList::const_iterator it = qFind(keys, ws);
	if (it == keys.end() || !keys.size())
		return NULL;
	return xpcos[ws];
}

unsigned int xParticleManager::GetNumCubeParticles(
	double dx, double dy, double dz, double min_radius, double max_radius)
{
	vector3i ndim;
	if (min_radius == max_radius)
	{
		double r = min_radius;
		double diameter = 2.0 * r;
		vector3d dimension = new_vector3d(dx, dy, dz);
		ndim = ToVector3I(dimension / diameter) - new_vector3i(1, 1, 1);
	}
	return ndim.x * ndim.y * ndim.z;
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

unsigned int xParticleManager::GetNumCircleParticles(double d, unsigned int ny, double min_radius, double max_radius)
{
	double r = max_radius;
	double cr = 0.5 * d;
	unsigned int nr = static_cast<unsigned int>(cr / (2.0 * r)) - 1;
	double rr = 2.0 * r * nr + r;
	double space = (cr - rr) / (nr + 1);
	unsigned int cnt = 1;
	for (unsigned int i = 1; i <= nr; i++)
	{
		double dth = (2.0 * r + space) / (i * (2.0 * r + space));
		unsigned int _np = static_cast<unsigned int>((2.0 * M_PI) / dth);
		cnt += _np;
	}
	return cnt * ny;
}

xParticleObject* xParticleManager::CreateParticleFromList(
	std::string n, xMaterialType mt, unsigned int _np, vector4d* d)
{
	QString name = QString::fromStdString(n);
	xParticleObject* xpo = new xParticleObject(n);
	vector4d* pos = xpo->AllocMemory(_np);
	xpo->setStartIndex(np);
	np += _np;
	xMaterial xm = GetMaterialConstant(mt);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	double min_r = d[0].w;
	double max_r = d[0].w;
	for (unsigned int i = 0; i < _np; i++)
	{
		pos[xpo->StartIndex() + i] = new_vector4d(d[i].x, d[i].y, d[i].z, d[i].w);
		min_r = min(min_r, d[i].w);
		max_r = max(max_r, d[i].w);
	}
	xpo->setMinRadius(min_r);
	xpo->setMaxRadius(max_r);
	xpcos[name] = xpo;
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
	std::fstream qf;
	std::cout << "SetCurrentParticleFromPartResult : " << path << std::endl;
	qf.open(path, std::ios::binary | std::ios::in);
	double m_ct = 0;
	unsigned int m_np = 0;
	if (qf.is_open())
	{
		qf.read((char*)&m_ct, sizeof(double));
		qf.read((char*)&m_np, sizeof(unsigned int));
		if (np != m_np)
		{
			std::cout << "Failed setCurrentParticlesFromPartResult" << std::endl;
			qf.close();
			return;
		}
		vector4d* m_pos = new vector4d[m_np];
		qf.read((char*)m_pos, sizeof(vector4d) * m_np);
		foreach(xParticleObject* xpo, xpcos)
		{
			unsigned int sid = xpo->StartIndex();
			unsigned int xnp = xpo->NumParticle();
			vector4d* xpo_pos = xpo->Position();
			memcpy(xpo_pos, m_pos + sid, sizeof(vector4d) * xnp);
		//	qf.read((char*)m_pos, sizeof()
		}
		delete[] m_pos;
	}
	qf.close();

}

xParticleObject* xParticleManager::CreateCubeParticle(
	std::string n, xMaterialType mt, unsigned int _np, xCubeParticleData& d)
{
	QString name = QString::fromStdString(n);
	xParticleObject* xpo = new xParticleObject(n);
	vector4d* pos = xpo->AllocMemory(_np);
	xpo->setStartIndex(np);
	xpo->setMaterialType(mt);
	np += _np;
	xMaterial xm = GetMaterialConstant(mt);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	xpo->setMinRadius(d.minr);
	xpo->setMaxRadius(d.maxr);

	if (d.minr == d.maxr)
	{
		double r = d.minr;
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
		for (unsigned int z = 0; z < ndim.z; z++){
			double _z = d.lz + r + space.z;
			for (unsigned int y = 0; y < ndim.y; y++){
				double _y = d.ly + r + space.y;
				for (unsigned int x = 0; x < ndim.x; x++){
					double _x = d.lx + r + space.x;
					vector4d p = new_vector4d
						(
						_x + x * gab.x + ran.x * frand(), 
						_y + y * gab.y + ran.y * frand(), 
						_z + z * gab.z + ran.z * frand(), r
						);
						pos[cnt] = p;
					cnt++;
				}
			}
		}
	}
	//xpo->set
	xpcos[name] = xpo;
	xObjectManager::XOM()->addObject(xpo);
	return xpo;
}

xParticleObject * xParticleManager::CreateClusterParticle(std::string n, xMaterialType mt, unsigned int _np, xClusterObject * xo)
{
	unsigned int neach = xo->NumElement();
	unsigned int rnp = _np * neach;
	double rad = xo->ElementRadius();
	QString name = QString::fromStdString(n);
	xParticleObject* xpo = new xParticleObject(n);
	vector4d* pos = xpo->AllocMemory(rnp);
	xpo->setStartIndex(np);
	xpo->setMaterialType(mt);
	np += rnp;
	xMaterial xm = GetMaterialConstant(mt);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	xpo->setMinRadius(rad);
	xpo->setMaxRadius(rad);
	xpo->setEachCount(neach);
	vector3d* rloc = xo->RelativeLocation();
	xpo->setRelativeLocation(xo->RelativeLocation());
	for (unsigned int i = 0; i < _np; i++)
	{
		vector3d cm = new_vector3d(0.0, 0.01, 0.0);
		for (unsigned int j = 0; j < neach; j++)
		{
			vector3d m_pos = cm + rloc[j];
			pos[i * neach + j] = new_vector4d(m_pos.x, m_pos.y, m_pos.z, rad);
		}
	}
	xpcos[name] = xpo;
	xObjectManager::XOM()->addObject(xpo);
	n_cluster_each += neach;
	n_cluster_object++;
	return xpo;
}

QMap<QString, xParticleObject*>& xParticleManager::XParticleObjects()
{
	return xpcos;
}

bool xParticleManager::CopyPosition(double *pos, unsigned int inp)
{
	foreach(xParticleObject* xpo, xpcos)
	{
		xpo->CopyPosition(pos);
	}
	return true;
}

bool xParticleManager::SetMassAndInertia(double *mass, double *inertia)
{
	foreach(xParticleObject* xpo, xpcos)
	{
		double d = xpo->Density();
		vector4d* v = xpo->Position();
		unsigned int sid = xpo->StartIndex();
		for (unsigned int i = 0; i < xpo->NumParticle(); i++)
		{
			mass[i + sid] = d * (4.0 / 3.0) * M_PI * pow(v[i].w, 3.0);
			inertia[i + sid] = (2.0 / 5.0) * mass[i + sid] * pow(v[i].w, 2.0);
		}		
	}
	return true;
}

void xParticleManager::ExportParticleDataForView(std::string path)
{
	std::fstream of;
	of.open(path, std::ios::out | std::ios::binary);
	of.write((char*)&np, sizeof(unsigned int));
	foreach(xParticleObject* po, xpcos)
	{
		unsigned int _sid = po->StartIndex();
		unsigned int _np = po->NumParticle();
		double d[2] = { po->MinRadius(), po->MaxRadius() };
		double* _pos = (double *)po->Position();
		int mat = po->Material();
		int ns = po->Name().size();
		of.write((char*)&ns, sizeof(int));
		of.write((char*)po->Name().toStdString().c_str(), sizeof(char) * ns);
		of.write((char*)&mat, sizeof(int));
		of.write((char*)&_sid, sizeof(unsigned int));
		of.write((char*)&_np, sizeof(unsigned int));
		of.write((char*)&d, sizeof(double) * 2);
		of.write((char*)_pos, sizeof(double) * _np * 4);
	}
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

void xParticleManager::SetClusterInformation()
{
	if (n_cluster_object)
	{
		cluster_index = new unsigned int[np];
		cluster_count = new unsigned int[n_cluster_object];
		cluster_begin = new unsigned int[n_cluster_object * 2];
		cluster_set_location = new double[n_cluster_each];
	}
	unsigned int idx = 0;
	unsigned int sum_each = 0;
	foreach(xParticleObject* po, xpcos)
	{
		if (po->Shape() == CLUSTER_SHAPE)
		{
			cluster_count[idx] = po->EachCount();
			cluster_begin[idx + 0] = po->StartIndex();
			cluster_begin[idx + 1] = sum_each;
			memcpy(cluster_set_location + sum_each, po->RelativeLocation(), sizeof(vector3d) * po->EachCount());
			for (unsigned int i = po->StartIndex(); i < po->NumParticle(); i++)
			{
				cluster_index[i] = idx;
			}
			idx++;
			sum_each += po->EachCount();
		}
	}
}

