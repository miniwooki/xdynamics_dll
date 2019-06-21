#include "xdynamics_object/xLineObject.h"
#include "xdynamics_manager/xSmoothedParticleHydrodynamicsModel.h"

xLineObject::xLineObject()
	: xPointMass()
{
	memset(&len, 0, sizeof(double) * 10);
}

xLineObject::xLineObject(std::wstring _name)
	: xPointMass(_name)
{
	memset(&len, 0, sizeof(double) * 10);
	xObject::shape = LINE_SHAPE;
}

xLineObject::~xLineObject()
{

}

bool xLineObject::define(vector3d p0, vector3d p1, vector3d n)
{
	len = length(p1 - p0);
	spoint = p0;
	epoint = p1;
	normal = n;
	tangential = (p1 - p0) / len;
	return true;
}

vector3d xLineObject::Normal() const
{
	return normal;
}

vector3d xLineObject::StartPoint() const
{
	return spoint;
}

vector3d xLineObject::EndPoint() const
{
	return epoint;
}

void xLineObject::SetupDataFromStructure(xLineObjectData& d)
{
	this->define(
		new_vector3d(d.p0x, d.p0y, d.p0z),
		new_vector3d(d.p1x, d.p1y, d.p1z),
		new_vector3d(d.nx, d.ny, d.nz));
	data = d;
}

unsigned int xLineObject::create_sph_particles(double ps, unsigned int nlayers, vector3d* p, xMaterialType* t)
{
	unsigned int nx = static_cast<unsigned int>((len / ps) + 1e-9);
	unsigned int count = 0;
	if (material == FLUID)
	{

	}
	else if (material == BOUNDARY){
		for (unsigned int i = 0; i <= nx; i++){
			vector3d _p = spoint + (i * ps) * tangential;
			if (xSmoothedParticleHydrodynamicsModel::XSPH()->CheckCorner(_p))
				continue;
			if (p){
				p[count] = new_vector3d(_p.x, _p.y, _p.z);
				t[count] = BOUNDARY;
				for (unsigned int j = 1; j <= nlayers; j++){
					vector3d _d = _p + (j * ps) * normal;
					p[count + j] = new_vector3d(_d.x, _d.y, _d.z);
					t[count + j] = DUMMY;
				}
			}
			count += 1 + nlayers;
		}
	}
	
	return count;
}

QVector<xCorner> xLineObject::get_sph_boundary_corners()
{
	vector3d t0 = normalize(spoint - epoint);
	vector3d t1 = normalize(epoint - spoint);
	xCorner c1 = { spoint.x, spoint.y, spoint.z, normal.x, normal.y, normal.z, t0.x, t0.y, t0.z };
	xCorner c2 = { epoint.x, epoint.y, epoint.z, normal.x, normal.y, normal.z, t1.x, t1.y, t1.z };
	QVector<xCorner> list(2);
	list[0] = c1;
	list[1] = c2;
	return list;
}

