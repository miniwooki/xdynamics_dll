#include "xdynamics_manager/xParticleMananger.h"
#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_manager/xObjectManager.h"

xParticleManager::xParticleManager()
	: np(0)
// 	, is_realtime_creating(false)
// 	, one_by_one(false)
{

}

xParticleManager::~xParticleManager()
{
	//qDeleteAll(xpcos);
}


unsigned int xParticleManager::NumParticle()
{
	return np;
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

xParticleObject* xParticleManager::CreateCubeParticle(
	std::string n, xMaterialType mt, unsigned int _np, xCubeParticleData& d)
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