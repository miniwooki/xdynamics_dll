#ifndef XCYLINDEROBJECT_H
#define XCYLINDEROBJECT_H

#include "xdynamics_object/xPointMass.h"
//#include "xdynamics_object/xPlaneObject.h"
#include <QtCore/QString>

//class xPlaneObject;

class XDYNAMICS_API xCylinderObject : public xPointMass
{
public:
	xCylinderObject();
	xCylinderObject(std::string _name);
	xCylinderObject(const xCylinderObject& _cube);
	virtual ~xCylinderObject();

	//xCylinderObject* Planes() const { return planes; }
	//device_plane_info* deviceCubeInfo() { return dpi; }

	bool define(vector3d& min, vector3d& max);
	vector3d top_position();
	vector3d top_position() const;
	vector3d bottom_position();
	vector3d bottom_position() const;
	double cylinder_length();
	double cylinder_length() const;
	double cylinder_top_radius();
	double cylinder_top_radius() const;
	double cylinder_bottom_radius();
	double cylinder_bottom_radius() const;

	void SetupDataFromStructure(xCylinderObjectData& d);

	virtual unsigned int create_sph_particles(double ps, unsigned int nlayers, vector3d* p = NULL, xMaterialType* t = NULL);
	virtual QVector<xCorner> get_sph_boundary_corners();

private:
	//xPlaneObject *planes;
	vector3d len_rr;// len, rbase, rtop;
	vector3d pbase;
	vector3d ptop;
	//	vector3d origin;
};
#endif