#ifndef XLINEOBJECT_H
#define XLINEOBJECT_H

#include "xdynamics_object/xPointMass.h"
#include <QtCore/QString>

class XDYNAMICS_API xLineObject : public xPointMass
{
public:
	xLineObject();
	xLineObject(std::wstring _name);
	virtual ~xLineObject();

	bool define(vector3d p0, vector3d p1, vector3d n);

	vector3d Normal() const;
	vector3d StartPoint() const;
	vector3d EndPoint() const;

	void SetupDataFromStructure(xLineObjectData& d);

	virtual unsigned int create_sph_particles(double ps, unsigned int nlayers, vector3d* p = NULL, xMaterialType* t = NULL);
	virtual QVector<xCorner> get_sph_boundary_corners();

private:
	double len;
	vector3d normal;
	vector3d tangential;
	vector3d spoint;
	vector3d epoint;
	xLineObjectData data;
};

#endif