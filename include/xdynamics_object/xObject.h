#ifndef XOBJECT_H
#define XOBJECT_H

#include "xdynamics_decl.h"
#include "xdynamics_algebra/xAlgebraMath.h"
#include "xstring.h"
//#include <QtCore/QString>
//#include <QtCore/QVector>

class XDYNAMICS_API xObject
{
public:
	xObject(xShapeType _s = NO_SHAPE);
	xObject(std::string _name, xShapeType _s = NO_SHAPE);
	xObject(const xObject& o);
	virtual ~xObject();

	virtual unsigned int create_sph_particles(double ps, unsigned int nlayers, vector3d* p = NULL, xMaterialType* t = NULL);
	//virtual QVector<xCorner> get_sph_boundary_corners();

	static void initialize();
	void setObjectID(int _id);
	void setDensity(double _d);
	void setYoungs(double _y);
	void setPoisson(double _p);
	void setDynamicsBody(bool ismo);
	void setShapeType(xShapeType xst);
	void setMaterialType(xMaterialType xmt);
	void setConnectedGeometryName(std::string n);
	void setCompulsionMovingObject(bool b);
	void setMovingConstantMovingVelocity(vector3d v);

	double Density() const;// { return d; }
	double Youngs() const;// { return y; }
	double Poisson() const;// { return p; }
	double Shear() const;
	double Volume() const;
	//bool setDynamicsBody() const;
	bool CompulsionMovingObject() const;
	xShapeType Shape() const;
	xMaterialType Material() const;

	std::string Name() const;
	std::string ConnectedGeometryName() const;
	int ObjectID() const;
	bool isDynamicsBody() const;

protected:
	static int count;
	xstring name;			// Object name
	xstring connected_geometry_name;
	xShapeType shape;
	xMaterialType material;
	bool is_compulsion_moving_object;
	//bool is_mass_moving_object;
	int id;
	bool is_dynamics_body;
// 	static unsigned int count;
// 	dimension_type dim;
// 	unsigned int id;
// 	// pointMass of object
// 	VEC3D dia_iner0;			// Ixx, Iyy, Izz
	vector3d const_vel;
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
 
// 	vobject* vobj;
// 	vobject* marker;
};


#endif