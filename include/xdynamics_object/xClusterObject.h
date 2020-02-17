#ifndef XCLUSTEROBJECT_H
#define XCLUSTEROBJECT_H

#include "xdynamics_object/xObject.h"

class XDYNAMICS_API xClusterObject : public xObject
{
public:
	xClusterObject();
	xClusterObject(std::string name);
	//xClusterObject(const xClusterObject& cobj);
	virtual ~xClusterObject();

	void clearData();
	void setClusterSet(unsigned int num, double min_rad, double max_rad, vector4d* d, bool isEachCluster);
	unsigned int NumElement();
	double ElementMinimumRadius();
	double ElementMaximumRadius();
	vector4d* RelativeLocation();
	bool IsRandomRadiusEachCluster();
	bool IsRandomRadius();
	void SetTotalClusters(unsigned int n);
	unsigned int TotalClusters();

	bool IsRandomRadius() const;
	bool IsRandomRadiusEachCluster() const;
	unsigned int NumElement() const;
	unsigned int TotalClusters() const;
	double MinimumRadius() const;
	double MaximumRadius() const;
	vector4d *RelativeLocation() const;

private:
	bool is_random_radius;
	bool is_each_cluster;
	unsigned int nelement;
	unsigned int total_clusters;
	double min_radius;
	double max_radius;
	//double radius;
	//vector3ui dimension;
	//vector3d location;
	vector4d *relative_loc;
};

#endif
