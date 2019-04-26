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
{
	id = count++;
}

xObject::xObject(std::string _name, xShapeType _s)
	: name(QString::fromStdString(_name))
	, id(-1)
	, vol(0)
	, d(0)
	, y(0)
	, p(0)
	, sm(0)
	, shape(_s)
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
{

}

xObject::~xObject()
{

}

unsigned int xObject::create_sph_particles(double ps, vector4d* p /*= NULL*/)
{
	return 0;
}

QVector<xCorner> xObject::get_sph_boundary_corners()
{
	return QVector<xCorner>();
}

void xObject::setObjectID(int _id)
{
	id = _id;
}

QString xObject::Name() const
{
	return name;
}

int xObject::ObjectID() const
{
	return id;
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
}

xMaterialType xObject::Material() const
{
	return material;
}