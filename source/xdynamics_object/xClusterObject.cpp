#include "xdynamics_object\xClusterObject.h"

//xCluster::xCluster()
//	: nelement(0)
//	, start_index(0)
//	, cm(new_vector3d(0.0, 0.0, 0.0))
//	, quaternion(new_vector3d(0.0, 0.0, 0.0))
//	, angle(new_vector3d(0.0, 0.0, 0.0))
//	, omega(new_vector3d(0.0, 0.0, 0.0))
//	, relative_loc(NULL)
//{
//
//}
//
//xCluster::~xCluster()
//{
//	if (relative_loc) delete[] relative_loc; relative_loc = NULL;
//}
//
//void xCluster::setCluster(unsigned int num, vector3d *d)
//{
//	relative_loc = new vector3d[num];
//	memcpy(relative_loc, d, sizeof(vector3d) * num);
//}
//
xClusterObject::xClusterObject()
	: nelement(0)
	, min_radius(0)
	, max_radius(0)
	, relative_loc(NULL)
{
}

xClusterObject::xClusterObject(std::string name)
	: xObject(name, CLUSTER_SHAPE)
	, nelement(0)
	, min_radius(0)
	, max_radius(0)
	, relative_loc(NULL)
{
}

xClusterObject::~xClusterObject()
{
	if (relative_loc) delete[] relative_loc; relative_loc = NULL;
}

void xClusterObject::setClusterSet(unsigned int num, double min_rad, double max_rad, vector3d * d, bool isEachCluster)
{
	is_each_cluster = isEachCluster;
	nelement = num;
	min_radius = min_rad;
	max_radius = max_rad;
	relative_loc = new vector3d[num];
	memcpy(relative_loc, d, sizeof(vector3d) * num);
}

unsigned int xClusterObject::NumElement()
{
	return nelement;
}

double xClusterObject::ElementMinimumRadius()
{
	return min_radius;
}

double xClusterObject::ElementMaximumRadius()
{
	return max_radius;
}

vector3d * xClusterObject::RelativeLocation()
{
	return relative_loc;
}

bool xClusterObject::IsRandomRadiusEachCluster()
{
	return is_each_cluster;
}

bool xClusterObject::IsRandomRadius()
{
	return min_radius == max_radius;
}
