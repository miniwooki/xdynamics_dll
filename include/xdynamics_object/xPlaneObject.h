#ifndef XPLANEOBJECT_H
#define XPLANEOBJECT_H

#include "xdynamics_object/xPointMass.h"
//#include <QtCore/QString>

class XDYNAMICS_API xPlaneObject : public xPointMass
{
public:
	xPlaneObject();
	xPlaneObject(std::string _name);
	virtual ~xPlaneObject();

	bool define(bool isMovingObject, vector3d& _xw, vector3d& _pa, vector3d& _pb);
	bool define(vector3d& p0, vector3d& p1, vector3d& p2, vector3d& p3);
	double L1() const;// { return l1; }
	double L2() const;// { return l2; }
	vector3d U1() const;// { return u1; }
	vector3d U2() const;// { return u2; }
	vector3d UW() const;// { return uw; }
	vector3d XW() const;// { return xw; }
	vector3d PA() const;// { return pa; }
	vector3d PB() const;// { return pb; }
	vector3d W2() const;// { return w2; }
	vector3d W3() const;// { return w3; }
	vector3d W4() const;// { return w4; }
	vector3d MinPoint() const;// { return minp; }
	vector3d MaxPoint() const;// { return maxp; }

	vector3d LocalPoint(unsigned int i);

	void SetupDataFromStructure(xPlaneObjectData& d);

	virtual unsigned int create_sph_particles(double ps, unsigned int nlayers, vector3d* p = NULL, xMaterialType* t = NULL);
//	virtual QVector<xCorner> get_sph_boundary_corners();

private:
	double l1, l2;
	vector3d minp;
	vector3d maxp;
	vector3d u1;
	vector3d u2;
	vector3d uw;
	vector3d xw;
	vector3d pa;
	vector3d pb;

	vector3d w2;
	vector3d w3;
	vector3d w4;

	vector3d local_point[4];

	xPlaneObjectData data;
};

#endif