#ifndef XOBJECT_H
#define XOBJECT_H

#include "xdynamics_decl.h"
#include "xdynamics_algebra/xAlgebraMath.h"
#include <QtCore/QString>
#include <QtCore/QVector>

class XDYNAMICS_API xObject
{
public:
	xObject(xShapeType _s = NO_SHAPE);
	xObject(std::string _name, xShapeType _s = NO_SHAPE);
	xObject(const xObject& o);
	virtual ~xObject();

	virtual unsigned int create_sph_particles(double ps, vector4d* p = NULL);
	virtual QVector<xCorner> get_sph_boundary_corners();

	void setObjectID(int _id);
	void setDensity(double _d);
	void setYoungs(double _y);
	void setPoisson(double _p);
	void setShapeType(xShapeType xst);
	void setMaterialType(xMaterialType xmt);

	double Density() const;// { return d; }
	double Youngs() const;// { return y; }
	double Poisson() const;// { return p; }
	double Shear() const;
	double Volume() const;
	xShapeType Shape() const;
	xMaterialType Material() const;

	QString Name() const;
	int ObjectID() const;

protected:
	static int count;
	QString name;			// Object name
	xShapeType shape;
	xMaterialType material;
	int id;
// 	static unsigned int count;
// 	dimension_type dim;
// 	unsigned int id;
// 	// pointMass of object
// 	VEC3D dia_iner0;			// Ixx, Iyy, Izz
// 	VEC3D sym_iner0;		// Ixy, Ixz, Iyz
 	double vol;				// volume
// 	QString name;
// 	geometry_use roll_type;
// 	geometry_type obj_type;
// 	material_type mat_type;
 	double d;		// density
 	double y;		// young's modulus
 	double p;		// poisson ratio
 	double sm;		// shear modulus
// 
// 	vobject* vobj;
// 	vobject* marker;
};


#endif