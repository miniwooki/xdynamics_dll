#include "xdynamics_object/xCylinderObject.h"

xCylinderObject::xCylinderObject()
{

}

xCylinderObject::xCylinderObject(std::string _name)
{

}

xCylinderObject::xCylinderObject(const xCylinderObject& _cube)
{

}

xCylinderObject::~xCylinderObject()
{

}

bool xCylinderObject::define(vector3d& min, vector3d& max)
{
	return true;
}

vector3d xCylinderObject::top_position() { return ptop; }

vector3d xCylinderObject::top_position() const { return ptop; }

vector3d xCylinderObject::bottom_position() { return pbase; }

vector3d xCylinderObject::bottom_position() const { return pbase; }

double xCylinderObject::cylinder_length() { return len_rr.x; }

double xCylinderObject::cylinder_length() const { return len_rr.x; }

double xCylinderObject::cylinder_top_radius() { return len_rr.y; }

double xCylinderObject::cylinder_top_radius() const { return len_rr.y; }

double xCylinderObject::cylinder_bottom_radius() { return len_rr.z; }

double xCylinderObject::cylinder_bottom_radius() const { return len_rr.z; }

void xCylinderObject::SetupDataFromStructure(xCylinderObjectData& d)
{
	len_rr = new_vector3d(d.length, d.r_top, d.r_bottom);
	pbase = new_vector3d(d.p0x, d.p0y, d.p0z);
	ptop = new_vector3d(d.p1x, d.p1y, d.p1z);
}

unsigned int xCylinderObject::create_sph_particles(double ps, unsigned int nlayers, vector3d* p /*= NULL*/, xMaterialType* t /*= NULL*/)
{
	return 0;
}

QVector<xCorner> xCylinderObject::get_sph_boundary_corners()
{
	return QVector<xCorner>();
}

