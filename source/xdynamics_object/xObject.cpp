#include "xdynamics_object/xObject.h"

int xObject::count = 0;

xObject::xObject(xShapeType _s)
	: id(-1)
	, vol(0)
	, d(0)
	, y(0)
	, p(0)
	, sm(0)
	, shape(_s)
	, is_dynamics_body(false)
	, is_compulsion_moving_object(false)
{
	id = count++;
}

xObject::xObject(std::string _name, xShapeType _s)
	: name(_name)
	, id(-1)
	, vol(0)
	, d(0)
	, y(0)
	, p(0)
	, sm(0)
	, shape(_s)
	, is_dynamics_body(false)
	, is_compulsion_moving_object(false)
{
	id = count++;
}

xObject::xObject(const xObject& o)
	: name(o.Name())
	, d(o.Density())
	, id(ObjectID())
	, y(o.Youngs())
	, p(o.Poisson())
	, sm(o.Shear())
	, vol(o.Volume())
	, is_dynamics_body(o.isDynamicsBody())
	, is_compulsion_moving_object(o.CompulsionMovingObject())
{

}

xObject::~xObject()
{

}

std::string xObject::info()
{
	return m_info.toStdString();
}

unsigned int xObject::create_sph_particles(double ps, unsigned int nlayers, vector3d* p /*= NULL*/, xMaterialType* t)
{
	return 0;
}

//QVector<xCorner> xObject::get_sph_boundary_corners()
//{
//	return QVector<xCorner>();
//}

void xObject::initialize()
{
	count = 0;
}

void xObject::setObjectID(int _id)
{
	id = _id;
}

std::string xObject::Name() const
{
	return name.toStdString();
}

std::string xObject::ConnectedGeometryName() const
{
	return connected_geometry_name.toStdString();
}

int xObject::ObjectID() const
{
	return id;
}

bool xObject::isDynamicsBody() const
{
	return is_dynamics_body;
}

double xObject::Density() const
{ 
	return d; 
}
double xObject::Youngs() const
{ 
	return y; 
}

double xObject::Shear() const
{
	return sm;
}

double xObject::Poisson() const
{ 
	return p;
}

double xObject::Volume() const
{
	return vol;
}
//
//bool xObject::MovingObject() const
//{
//	return is_mass_moving_object;
//}

bool xObject::CompulsionMovingObject() const
{
	return is_compulsion_moving_object;
}

void xObject::setDensity(double _d)
{
	d = _d;
}

void xObject::setYoungs(double _y)
{
	y = _y;
}

void xObject::setPoisson(double _p)
{
	p = _p;
}

void xObject::setDynamicsBody(bool ismo)
{
	is_dynamics_body = ismo;
}

xShapeType xObject::Shape() const
{
	return shape;
}

void xObject::setShapeType(xShapeType xst)
{
	shape = xst;
}

void xObject::setMaterialType(xMaterialType xmt)
{
	material = xmt;
	xMaterial xm = GetMaterialConstant(xmt);
	d = xm.density;
	y = xm.youngs;
	p = xm.poisson;
}

void xObject::setConnectedGeometryName(std::string n)
{
	connected_geometry_name = n;
}

void xObject::setCompulsionMovingObject(bool b)
{
	is_compulsion_moving_object = b;
}

void xObject::setMovingConstantMovingVelocity(vector3d v)
{
	const_vel = v;
}

void xObject::setInfo(std::string _info)
{
	m_info = _info;
}

xMaterialType xObject::Material() const
{
	return material;
}