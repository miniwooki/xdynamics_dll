#ifndef XPARTICLEOBJECT_H
#define XPARTICLEOBJECT_H

#include "xdynamics_object/xObject.h"

class XDYNAMICS_API xParticleObject : public xObject
{
public:
	xParticleObject();
	xParticleObject(std::string _name);
	virtual ~xParticleObject();

	void setStartIndex(unsigned int sid);
	void setMinRadius(double _mr);
	void setMaxRadius(double _mr);

	vector4d* AllocMemory(unsigned int _np);

	void CopyPosition(double* _pos);
	unsigned int StartIndex() const;
	unsigned int NumParticle() const;
	double MinRadius() const;
	double MaxRadius() const;
	vector4d* Position() const;

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
	unsigned int sid;
	unsigned int np;
	double min_radius;
	double max_radius;

	vector4d* pos;
};

#endif