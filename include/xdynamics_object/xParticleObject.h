#ifndef XPARTICLEOBJECT_H
#define XPARTICLEOBJECT_H

#include "xdynamics_object/xObject.h"

class XDYNAMICS_API xParticleObject : public xObject
{
public:
	xParticleObject();
	xParticleObject(std::wstring _name);
	virtual ~xParticleObject();

	void setStartIndex(unsigned int sid);
	void setClusterStartIndex(unsigned int csid);
	void setEachCount(unsigned int ec);
	void setMinRadius(double _mr);
	void setMaxRadius(double _mr);
	void setShapeForm(xShapeType xst);
	void setRelativeLocation(vector3d* rl);

	vector4d* AllocMemory(unsigned int _np);
	vector4d* AllocClusterMemory(unsigned int _np);

	void CopyPosition(double* _pos);
	void CopyClusterPosition(double* _pos, unsigned int* cindex);
	unsigned int StartIndex() const;
	unsigned int NumParticle() const;
	unsigned int EachCount() const;
	double MinRadius() const;
	double MaxRadius() const;
	xShapeType ShapeForm() const;
	vector4d* Position() const;
	vector3d* RelativeLocation() const;

	virtual unsigned int create_sph_particles(double ps, unsigned int nlayers, vector3d* p = NULL, xMaterialType* t = NULL);
	virtual QVector<xCorner> get_sph_boundary_corners();
// 	static unsigned int calculateNumCubeParticles(double dx, double dy, double dz, double min_radius, double max_radius);
// 	static unsigned int calculateNumPlaneParticles(double dx, unsigned int ny, double dy, double min_radius, double max_radius);
// 	static unsigned int calculateNumCircleParticles(double d, unsigned int ny, double min_radius, double max_radius);
// 
// 	vector4d* CreateCubeParticle(
// 		QString& n, xMaterialType type, unsigned int _np, double dx, double dy, double dz,
// 		double lx, double ly, double lz,
// 		double spacing, double min_radius, double max_radius,
// 		double youngs, double density, double poisson, double shear);

private:
	static unsigned int xpo_count;
	xShapeType form;
	unsigned int sid;
	unsigned int csid;
	unsigned int np;
	unsigned int cnp;
	unsigned int each;
	double min_radius;
	double max_radius;

	vector4d* pos;
	vector4d* cpos;
	vector3d *relative_loc;
};

#endif