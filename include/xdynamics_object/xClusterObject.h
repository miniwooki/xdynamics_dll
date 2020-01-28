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
	xClusterObject(std::string name);
	virtual ~xClusterObject();

	void setClusterSet(unsigned int num, double min_rad, double max_rad, vector3d* d, bool isEachCluster);
	unsigned int NumElement();
	double ElementMinimumRadius();
	double ElementMaximumRadius();
	vector3d* RelativeLocation();
	bool IsRandomRadiusEachCluster();
	bool IsRandomRadius();

private:
	bool is_random_radius;
	bool is_each_cluster;
	unsigned int nelement;
	double min_radius;
	double max_radius;
	//double radius;
	vector3d *relative_loc;
};

#endif
