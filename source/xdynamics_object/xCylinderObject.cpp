#include "xdynamics_object/xCylinderObject.h"

xCylinderObject::xCylinderObject()
	: xPointMass(CYLINDER_SHAPE)
	, empty(NO_EMPTY_PART)
	, thickness(0)
{
	memset(&len_rr.x, 0, sizeof(double) * 9);
}

xCylinderObject::xCylinderObject(std::string _name)
	: xPointMass(_name, CYLINDER_SHAPE)
	, empty(NO_EMPTY_PART)
	, thickness(0)
{
	memset(&len_rr.x, 0, sizeof(double) * 9);
}

xCylinderObject::xCylinderObject(const xCylinderObject& _cyl)
	: xPointMass(*this)
	, len_rr(new_vector3d(_cyl.cylinder_length(), _cyl.cylinder_bottom_radius(), _cyl.cylinder_top_radius()))
	, pbase(_cyl.bottom_position())
	, ptop(_cyl.top_position())
{

}

xCylinderObject::~xCylinderObject()
{

}

bool xCylinderObject::define(vector3d& min, vector3d& max)
{
	return true;
}

xCylinderObject::empty_part xCylinderObject::empty_part_type() { return empty; }
xCylinderObject::empty_part xCylinderObject::empty_part_type() const { return empty; }

double xCylinderObject::cylinder_thickness()
{
	return thickness;
}

double xCylinderObject::cylinder_thickness() const
{
	return thickness;
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

void xCylinderObject::updateData()
{
	//matrix33d A = GlobalTransformationMatrix(e);
	pbase = toLocal(pbase);
	ptop = toLocal(ptop);
}

void xCylinderObject::SetupDataFromStructure(xCylinderObjectData& d)
{
	len_rr = new_vector3d(d.length, d.r_top, d.r_bottom);
	pbase = new_vector3d(d.p0x, d.p0y, d.p0z);
	ptop = new_vector3d(d.p1x, d.p1y, d.p1z);

	//vector3d to = ptop - pbase;
	//double _len = length(to);
	//double h_len = _len * 0.5;
	//vector3d u = to / length(to);
	///*vector3d _e = sin(M_PI * 0.5) * u;
	//double e0 = 1 - dot(_e, _e);
	//euler_parameters e = { e0, _e.x, _e.y, _e.z };
	//matrix33d A = GlobalTransformationMatrix(e);*/
	//vector3d pu = new_vector3d(-u.y, u.z, u.x);
	//vector3d qu = cross(u, pu);

	//matrix33d A = { u.x, pu.x, qu.x, u.y, pu.y, qu.y, u.z, pu.z, qu.z };

	//double trA = A.a00 + A.a11 + A.a22;
	//double e0 = sqrt((trA + 1) / 4.0);
	//double e1 = sqrt((2.0 * A.a00 - trA + 1) / 4.0);
	//double e2 = sqrt((2.0 * A.a11 - trA + 1) / 4.0);
	//double e3 = sqrt((2.0 * A.a22 - trA + 1) / 4.0);

	xPointMass::pos = 0.5 * (pbase + ptop);
	//xPointMass::ep = new_euler_parameters(e0, e1, e2, e3);
	//setupTransformationMatrix();
	pbase = xPointMass::toLocal(pbase - xPointMass::pos);
	ptop = xPointMass::toLocal(ptop - xPointMass::pos);
	empty = (empty_part)d.empty;
	thickness = d.thickness;
}

unsigned int xCylinderObject::create_sph_particles(double ps, unsigned int nlayers, vector3d* p /*= NULL*/, xMaterialType* t /*= NULL*/)
{
	return 0;
}

//QVector<xCorner> xCylinderObject::get_sph_boundary_corners()
//{
//	return QVector<xCorner>();
//}

