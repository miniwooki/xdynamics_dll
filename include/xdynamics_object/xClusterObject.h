#ifndef XCLUSTEROBJECT_H
#define XCLUSTEROBJECT_H

#include "xdynamics_object/xObject.h"

//class XDYNAMICS_API xCluster
//{
//public: 
//	xCluster();
//	~xCluster();
//
//	void setCluster(unsigned int num, vector3d* d);
//
//private:
//	
//	unsigned int start_index;
//	vector3d cm;
//	vector3d quaternion;
//	vector3d angle;
//	vector3d omega;
//	
//};

class XDYNAMICS_API xClusterObject : public xObject
{
public:
	xClusterObject();
	xClusterObject(std::wstring name);
	virtual ~xClusterObject();

	void setClusterSet(unsigned int num, double rad, vector3d* d);
	unsigned int NumElement();
	double ElementRadius();
	vector3d* RelativeLocation();

private:
	unsigned int nelement;
	double radius;
	vector3d *relative_loc;
};

#endif
