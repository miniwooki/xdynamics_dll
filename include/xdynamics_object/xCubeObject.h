#ifndef XCUBEOBJECT_H
#define XCUBEOBJECT_H

#include "xdynamics_object/xPointMass.h"
#include "xdynamics_object/xPlaneObject.h"
//#include <QtCore/QString>

//class xPlaneObject;

class XDYNAMICS_API xCubeObject : public xPointMass
{
public:
	xCubeObject();
	xCubeObject(std::string _name);
	xCubeObject(const xCubeObject& _cube);
	virtual ~xCubeObject();

	xPlaneObject* Planes() const { return planes; }
	//device_plane_info* deviceCubeInfo() { return dpi; }
	void updateCube();
	bool define(vector3d& min, vector3d& max);
	vector3d origin();// { return ori; }
	vector3d origin() const;// { return ori; }
	vector3d min_point();// { return min_p; }
	vector3d min_point() const;// { retursdsn min_p; }
	vector3d max_point();// { return max_p; }
	vector3d max_point() const;// { return max_p; }
	vector3d cube_size();// { return size; }
	vector3d cube_size() const;// { return size; }
	xPlaneObject* planes_data(int i) const;// { return &(planes[i]); }

	void SetupDataFromStructure(xCubeObjectData& d);

	virtual unsigned int create_sph_particles(double ps, unsigned int nlayers, vector3d* p = NULL, xMaterialType* t = NULL);
	//virtual QVector<xCorner> get_sph_boundary_corners();

private:
	xPlaneObject *planes;
	vector3d ori;
	vector3d min_p;
	vector3d max_p;
	vector3d size;
	vector3d local_plane_position[6];
};

#endif