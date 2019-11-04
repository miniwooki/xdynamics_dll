#include "xdynamics_object/xCubeObject.h"
#include "xdynamics_object/xPlaneObject.h"

xCubeObject::xCubeObject()
	: xPointMass(CUBE_SHAPE)
	, planes(NULL)
{
	memset(&ori.x, 0, sizeof(double) * 12);
}

xCubeObject::xCubeObject(std::string _name)
	: xPointMass(_name, CUBE_SHAPE)
	, planes(NULL)
{
	memset(&ori.x, 0, sizeof(double) * 12);
}

xCubeObject::xCubeObject(const xCubeObject& _cube)
	: xPointMass(*this)
	, ori(_cube.origin())
	, min_p(_cube.min_point())
	, max_p(_cube.max_point())
	, size(_cube.cube_size())
{
	planes = new xPlaneObject[6];
	memcpy(planes, _cube.Planes(), sizeof(xPlaneObject) * 6);
}

xCubeObject::~xCubeObject()
{
	if (planes) delete[] planes; planes = NULL;
}

bool xCubeObject::define(vector3d& min, vector3d& max)
{
	if (!planes)
		planes = new xPlaneObject[6];
	min_p = min;
	max_p = max;
	xPointMass::pos = 0.5 * (min + max);
	size.x = length((max_p - new_vector3d(min_p.x, max_p.y, max_p.z)));
	size.y = length((max_p - new_vector3d(max_p.x, min_p.y, max_p.z)));
	size.z = length((max_p - new_vector3d(max_p.x, max_p.y, min_p.z)));

	planes[0].define(this->isDynamicsBody(), min_p, min_p + new_vector3d(0, 0, size.z), min_p + new_vector3d(size.x, 0, 0));
	planes[1].define(this->isDynamicsBody(), min_p, min_p + new_vector3d(0, size.y, 0), min_p + new_vector3d(0, 0, size.z));
	planes[2].define(this->isDynamicsBody(), min_p + new_vector3d(size.x, 0, 0), min_p + new_vector3d(size.x, 0, size.z), min_p + new_vector3d(size.x, size.y, 0));
	planes[3].define(this->isDynamicsBody(), min_p, min_p + new_vector3d(size.x, 0, 0), min_p + new_vector3d(0, size.y, 0));
	planes[4].define(this->isDynamicsBody(), min_p + new_vector3d(0, 0, size.z), min_p + new_vector3d(0, size.y, size.z), min_p + new_vector3d(size.x, 0, size.z));
	planes[5].define(this->isDynamicsBody(), min_p + new_vector3d(0, size.y, 0), min_p + new_vector3d(size.x, size.y, 0), min_p + new_vector3d(0, size.y, size.z));
	return true;
}

vector3d xCubeObject::origin()
{
	return ori;
}

vector3d xCubeObject::origin() const
{
	return ori;
}

vector3d xCubeObject::min_point()
{
	return min_p;
}

vector3d xCubeObject::min_point() const
{
	return min_p;
}

vector3d xCubeObject::max_point()
{
	return max_p;
}

vector3d xCubeObject::max_point() const
{
	return max_p;
}

vector3d xCubeObject::cube_size()
{
	return size;
}

vector3d xCubeObject::cube_size() const
{
	return size;
}

xPlaneObject* xCubeObject::planes_data(int i) const
{
	return &(planes[i]);
}

void xCubeObject::SetupDataFromStructure(xCubeObjectData& d)
{
	this->define(
		new_vector3d(d.p0x, d.p0y, d.p0z),
		new_vector3d(d.p1x, d.p1y, d.p1z));
}

unsigned int xCubeObject::create_sph_particles(double ps, unsigned int nlayers, vector3d* p, xMaterialType* t)
{
	return 0;
}

//QVector<xCorner> xCubeObject::get_sph_boundary_corners()
//{
//	return QVector<xCorner>();
//}
