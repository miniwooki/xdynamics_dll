#include "xdynamics_manager/xParticleMananger.h"
//#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_manager/xObjectManager.h"
#include <QtCore/QRandomGenerator>

xParticleManager::xParticleManager()
	: np(0)
	, r_pos(NULL)
	, r_vel(NULL)
	, n_cluster_object(0)
	, n_cluster_each(0)
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

xParticleObject* xParticleManager::XParticleObject(QString& ws)
{
	QStringList keys = xpcos.keys();
	QStringList::const_iterator it = qFind(keys, ws);
	if (it == keys.end() || !keys.size())
		return NULL;
	return xpcos[ws];
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

xParticleObject* xParticleManager::CreateParticleFromList(
	std::string n, xMaterialType mt, unsigned int _np, vector4d* d)
{
	QString name = QString::fromStdString(n);
	xParticleObject* xpo = new xParticleObject(n);
	vector4d* pos = xpo->AllocMemory(_np);
	xpo->setStartIndex(np);
	//n_single_sphere += _np;
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
	//n_single_sphere += _np;
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

xParticleObject* xParticleManager::CreateCircleParticle(
	std::string n, xMaterialType mt, unsigned int _np, xCircleParticleData& d)
{
	QString name = QString::fromStdString(n);
	xParticleObject* xpo = new xParticleObject(n);
	vector4d* pos = xpo->AllocMemory(_np);
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
	double r = d.maxr;
	double cr = 0.5 * d.diameter;
	unsigned int nr = static_cast<unsigned int>(cr / (2.0 * r)) - 1;
	double rr = 2.0 * r * nr + r;
	double space = (cr - rr) / (nr + 1);
	unsigned int cnt = 0;
	unsigned int k = 0;
	bool isStopCreating = false;
	
	while (1)
	{
		
		for (unsigned int i = 1; i <= nr; i++)
		{
			double _r = i * (2.0 * r + space);
			double dth = (2.0 * r + space) / _r;
			double th = 0.5 * k * r / _r;
			unsigned int npr = static_cast<unsigned int>((2.0 * M_PI) / dth);
			dth = ((2.0 * M_PI) / npr);
			for (unsigned int j = 0; j < npr; j++)
			{
				vector4d pp = new_vector4d(d.sx + _r * cos(dth * j), d.sy, d.sz + _r * sin(dth * j), r);
				vector4d new_pp = pp;				
				new_pp.x = pp.x * cos(th) + pp.z * sin(th);
				new_pp.z = -pp.x * sin(th) + pp.z * cos(th);
				pos[xpo->StartIndex() + cnt] = new_pp;
				cnt++;
				if (cnt == _np)
				{
					isStopCreating = true;
					break;
				}
			}
			if (isStopCreating) break;
		}
		if (isStopCreating) break;
		k++;
	}
	
	
	if (d.minr != d.maxr)
	{
		double dr = d.maxr - d.minr;
		srand(GetTickCount());
		for (unsigned int i = 0; i < _np; i++)
		{
			pos[xpo->StartIndex() + i].w = d.minr + dr * frand();
		}
	}
	xpcos[name] = xpo;
	xObjectManager::XOM()->addObject(xpo);
	return xpo;
}

xParticleObject * xParticleManager::CreateClusterParticle(
	std::string n, xMaterialType mt, unsigned int _np, xClusterObject * xo)
{
	unsigned int neach = xo->NumElement();
	unsigned int rnp = _np * neach;
	double rad = xo->ElementRadius();
	QString name = QString::fromStdString(n);
	xMaterial xm = GetMaterialConstant(mt);
	xParticleObject* xpo = new xParticleObject(n);
	xpo->setStartIndex(np);
	xpo->setClusterStartIndex(ncluster);
	xpo->setMaterialType(mt);
	xpo->setShapeForm(CLUSTER_SHAPE);
	xpo->setDensity(xm.density);
	xpo->setYoungs(xm.youngs);
	xpo->setPoisson(xm.poisson);
	xpo->setMinRadius(rad);
	xpo->setMaxRadius(rad);
	xpo->setEachCount(neach);
	vector4d* pos = xpo->AllocMemory(rnp);
	vector4d* cpos = xpo->AllocClusterMemory(_np);
	euler_parameters* ep = (euler_parameters*)xpo->EulerParameters();
	//n_cluster_sphere += _np;
	ncluster += _np;
	np += _np * neach;
	double norm = 0;
	vector3d* rloc = xo->RelativeLocation();
	xpo->setRelativeLocation(xo->RelativeLocation());
	for (unsigned int i = 0; i < _np; i++)
	{
		vector3d cp = new_vector3d(0.0, 0.005 * (i + 1), 0.0);
		vector3d rot = new_vector3d(180 * frand(), 180 * frand(), 180 * frand());
		//vector3d rot = new_vector3d(30,0,0);
		euler_parameters m_ep = EulerAngleToEulerParameters(rot);
		matrix33d A = GlobalTransformationMatrix(m_ep);
		for (unsigned int j = 0; j < neach; j++)
		{
			vector3d m_pos = cp + A * rloc[j];
			pos[i * neach + j] = new_vector4d(m_pos.x, m_pos.y, m_pos.z, rad);
		}
		cpos[i] = new_vector4d(cp.x, cp.y, cp.z, rad);
		ep[i] = m_ep;
		norm = length(new_vector4d(m_ep.e0, m_ep.e1, m_ep.e2, m_ep.e3));
	}
	xpcos[name] = xpo;
	xObjectManager::XOM()->addObject(xpo);
	n_cluster_each += neach;
	n_cluster_object++;
	return xpo;
}

void xParticleManager::CopyClusterInformation(xClusterInformation * xci, double* rcloc)
{
	unsigned int cnt = 0;
	vector3d* rloc = (vector3d*)rcloc;
	foreach(xParticleObject* xpo, xpcos)
	{
		if (xpo->ShapeForm() == CLUSTER_SHAPE)
		{
			xci[cnt].sid = xpo->StartClusterIndex();
			xci[cnt].neach = xpo->EachCount();
			xci[cnt].count = xpo->NumCluster();
			memcpy(rloc + xci[cnt].sid, xpo->RelativeLocation(), sizeof(vector3d) * xci[cnt].neach);
			cnt++;
		}
	}
}

QMap<QString, xParticleObject*>& xParticleManager::XParticleObjects()
{
	return xpcos;
}

bool xParticleManager::CopyPosition(
	double *pos, double* cpos, double* ep, unsigned int inp)
{
	foreach(xParticleObject* xpo, xpcos)
	{
		xpo->CopyPosition(pos);
		if (cpos && xpo->ShapeForm() == CLUSTER_SHAPE)
			xpo->CopyClusterPosition(cpos, ep);
	}
	return true;
}

bool xParticleManager::SetMassAndInertia(double *mass, double *inertia)
{
	unsigned int c = 0;
	foreach(xParticleObject* xpo, xpcos)
	{
		double d = xpo->Density();
		//vector4d* v = xpo->Position();
		unsigned int sid = xpo->StartIndex();
		if (xpo->ShapeForm() == CLUSTER_SHAPE)
		{
			vector4d* cpos = xpo->ClusterPosition();
			vector3d* rloc = xpo->RelativeLocation();
			for (unsigned int i = 0; i < xpo->NumCluster(); i++)
			{
				double m = d * (4.0 / 3.0) * M_PI * pow(cpos[i].w, 3.0);
				mass[i + sid] = m * xpo->EachCount();
				double J = (2.0 / 5.0) * m * pow(cpos[i].w, 2.0);
				vector3d J3 = new_vector3d(0, 0, 0);
				vector3d m_pos = new_vector3d(cpos[i].x, cpos[i].y, cpos[i].z);
				for (unsigned int j = 0; j < xpo->EachCount(); j++)
				{
					vector3d dr = rloc[j];
					J3.x += dr.y * dr.y + dr.z * dr.z;
					J3.y += dr.x * dr.x + dr.z * dr.z;
					J3.z += dr.x * dr.x + dr.y * dr.y;
				}
				inertia[(i + sid) * 3 + 0] = J + J3.x;
				inertia[(i + sid) * 3 + 1] = J + J3.y;
				inertia[(i + sid) * 3 + 2] = J + J3.z;
				//i += xpo->EachCount() - 1;
			}
		}
		else
		{
			vector4d* pos = xpo->Position();
			for (unsigned int i = 0; i < xpo->NumParticle(); i++)
			{
				double m = d * (4.0 / 3.0) * M_PI * pow(pos[i].w, 3.0);
				double J = (2.0 / 5.0) * m * pow(pos[i].w, 2.0);
				mass[i + sid] = m;
				inertia[i + sid] = J;
			}			
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

//void xParticleManager::SetClusterInformation()
//{
////	isCluster = new bool[]
//	//unsigned int ncs = np - n_single_sphere;
//	//if (n_cluster_object)
//	//{
//	//	cluster_index = new unsigned int[ncs * 2];
//	//	cluster_count = new unsigned int[n_cluster_object];
//	//	cluster_begin = new unsigned int[n_cluster_object * 2];
//	//	cluster_set_location = new vector3d[n_cluster_each];
//	//}
//	//unsigned int idx = 0;
//	//unsigned int cnt = 0;
//	//unsigned int sum_each = 0;
//	//foreach(xParticleObject* po, xpcos)
//	//{
//	//	if (po->ShapeForm() == CLUSTER_SHAPE)
//	//	{
//	//		/*cluster_count[idx] = po->EachCount();
//	//		cluster_begin[idx*2 + 0] = po->StartIndex();
//	//		cluster_begin[idx*2 + 1] = sum_each;*/
//	//		memcpy(cluster_set_location + sum_each, po->RelativeLocation(), sizeof(vector3d) * po->EachCount());
//	//		unsigned int t = po->NumParticle() / po->EachCount();
//	//		for (unsigned int i = po->StartIndex(); i < t; i++)
//	//		{
//	//			for (unsigned int j = 0; j < po->EachCount(); j++)
//	//			{
//	//				cluster_index[cnt * 2 + 0] = idx;
//	//				cluster_index[cnt * 2 + 1] = i;
//	//				cnt++;
//	//			}
//	//		}
//	//		sum_each += po->EachCount();
//	//	}
//	//	idx++;
//	//}
//}

void xParticleManager::AddParticleCreatingCondition(xParticleObject* xpo, xParticleCreateCondition& xpcc)
{
	xpccs.push_back(xpcc);
	QRandomGenerator qran;
	QVector<unsigned int> iList;
	while (iList.size() != xpo->EachCount())
	{
		unsigned int ni = qran.bounded(xpo->EachCount());
		if (qFind(iList.begin(), iList.end(), ni) != iList.end())
			continue;
		iList.push_back(ni);
	}

	unsigned int it = 0;
	unsigned int k = 0;
	QVector<unsigned int>::iterator iter = iList.begin();
	QList<vector4d> pList;
	
	while (1)
	{
		if ((k + 1) * xpo->EachCount() > xpo->NumParticle())
			it = xpo->NumParticle() - k * xpo->EachCount();
		else
			it = xpo->EachCount();
		vector4d* pos = pos = xpo->Position() + k * xpo->EachCount();
					
		for (unsigned int i = 0; i < it; i++)
		{
			pList.push_back(pos[i]);
		}
		unsigned int over = 0;
		foreach(vector4d v, pList)
		{
			if (over > it) break;
			while (*iter >= it) { iter++; }
			pos[*iter] = v;
			iter++;
			over++;
		}
		iter = iList.begin();
		pList.clear();
		k++;
		if (k * xpo->EachCount() > xpo->NumParticle())
			break;

	}
}

unsigned int xParticleManager::ExcuteCreatingCondition(
	double ct, unsigned int cstep, unsigned int cnp)
{
	foreach(xParticleCreateCondition xpcc, xpccs)
	{
		if (cstep == 1)
			return xpcc.neach;
		if (cnp > xpcc.sid && cnp < xpcc.sid + xpcc.count)
		{
			if (!(cstep % xpcc.nstep))
			{
				if (cnp + xpcc.neach > np)
					return np;
				else
					return cnp + xpcc.neach;
			}
				
		}
	}
	return cnp;
}
