#include "xdynamics_object/xClusterObject.h"

xClusterObject::xClusterObject()
	: nelement(0)
	, min_radius(0)
	, max_radius(0)
	, total_clusters(0)
	, relative_loc(NULL)
{
}

xClusterObject::xClusterObject(std::string name)
	: xObject(name, CLUSTER_SHAPE)
	, nelement(0)
	, min_radius(0)
	, max_radius(0)
	, total_clusters(0)
	, relative_loc(NULL)
{
}

//xClusterObject::xClusterObject(const xClusterObject& cobj)
//	: xObject(cobj)
//	, is_random_radius(cobj.IsRandomRadius())
//	, is_each_cluster(cobj.IsRandomRadiusEachCluster())
//	, nelement(cobj.NumElement())
//	, total_clusters(cobj.TotalClusters())
//	, min_radius(cobj.MinimumRadius())
//	, max_radius(cobj.MaximumRadius())
//{
//	relative_loc = new vector4d[nelement];
//	vector4d* rloc = cobj.RelativeLocation();
//	for (unsigned int i = 0; i < nelement; i++) {
//		relative_loc[i] = rloc[i];
//	}
//}

xClusterObject::~xClusterObject()
{
	if (relative_loc) delete[] relative_loc; relative_loc = NULL;
}

void xClusterObject::clearData()
{
	is_each_cluster = false;
	nelement = 0;
	min_radius = 0;
	max_radius = 0;
	if (relative_loc) delete relative_loc; relative_loc = nullptr;
}

void xClusterObject::setClusterSet(unsigned int num, double min_rad, double max_rad, vector4d * d, bool isEachCluster)
{
	is_each_cluster = isEachCluster;
	nelement = num;
	min_radius = min_rad;
	max_radius = max_rad;
	relative_loc = new vector4d[num];
	memcpy(relative_loc, d, sizeof(vector4d) * num);
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

vector4d * xClusterObject::RelativeLocation()
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

void xClusterObject::SetTotalClusters(unsigned int n)
{
	total_clusters = n;
}

unsigned int xClusterObject::TotalClusters()
{
	return total_clusters;
}

bool xClusterObject::IsRandomRadius() const
{
	return is_random_radius;
}

bool xClusterObject::IsRandomRadiusEachCluster() const
{
	return is_each_cluster;
}

unsigned int xClusterObject::NumElement() const
{
	return nelement;
}

unsigned int xClusterObject::TotalClusters() const
{
	return total_clusters;
}

double xClusterObject::MinimumRadius() const
{
	return min_radius;
}

double xClusterObject::MaximumRadius() const
{
	return max_radius;
}

vector4d * xClusterObject::RelativeLocation() const
{
	return relative_loc;
}
