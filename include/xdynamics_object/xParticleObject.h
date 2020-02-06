#ifndef XPARTICLEOBJECT_H
#define XPARTICLEOBJECT_H

#include "xdynamics_object/xObject.h"
#include "xmap.hpp"

class xClusterObject;

class XDYNAMICS_API xParticleObject : public xObject
{
public:

	typedef struct
	{
		unsigned int sidx, num, each;
	}xEachObjectData;

	xParticleObject();
	xParticleObject(std::string _name);
	virtual ~xParticleObject();

	void setStartIndex(unsigned int sid);
	void setClusterStartIndex(unsigned int csid);
	void setEachCount(unsigned int ec);
	void setMinRadius(double _mr);
	void setMaxRadius(double _mr);
	void setShapeForm(xShapeType xst);
	void setMassIndex(unsigned _mid);
	void resizeParticles(unsigned int new_np);
	vector4d* AllocMemory(unsigned int _np);
	//vector3d* AllocInertiaMemory(unsigned int _np);
	vector4d* AllocClusterMemory(unsigned int _np);

	void CopyPosition(double* _pos);
	void CopyMassAndInertia(double* _mass, vector3d* _inertia);
	void CopyClusterPosition(double* _pos, double* _ep);
	void CopyClusterPosition(unsigned int _sid, double* _pos, double* _ep);
	unsigned int StartIndex() const;
	unsigned int StartClusterIndex() const;
	unsigned int MassIndex() const;
	unsigned int NumParticle() const;
	unsigned int NumCluster() const;
	unsigned int EachCount() const;
	double MinRadius() const;
	double MaxRadius() const;
	xShapeType ShapeForm() const;
	vector4d* Position() const;
	vector4d* ClusterPosition() const;
	vector4d* EulerParameters() const;
	double* Mass() const;
	vector3d* Inertia() const;
	void setParticleShapeName(xstring s);
	std::string ParticleShapeName();
	bool IsInThisFromIndex(unsigned int id);
	void MoveParticle(unsigned int id, double x, double y, double z);
	virtual unsigned int create_sph_particles(double ps, unsigned int nlayers, vector3d* p = NULL, xMaterialType* t = NULL);
	void appendEachObject(xstring n, xParticleObject::xEachObjectData& co);
	//virtual QVector<xCorner> get_sph_boundary_corners();
// 	static unsigned int calculateNumCubeParticles(double dx, double dy, double dz, double min_radius, double max_radius);
// 	static unsigned int calculateNumPlaneParticles(double dx, unsigned int ny, double dy, double min_radius, double max_radius);
// 	static unsigned int calculateNumCircleParticles(double d, unsigned int ny, double min_radius, double max_radius);
// 
// 	vector4d* CreateCubeParticle(
// 		QString& n, xMaterialType type, unsigned int _np, double dx, double dy, double dz,
// 		double lx, double ly, double lz,
// 		double spacing, double min_radius, double max_radius,
// 		double youngs, double density, double poisson, double shear);
	xmap<xstring, xParticleObject::xEachObjectData>& EachObjects();
	void SetStartingPosition(unsigned int id, vector4d* p);
	vector4d* StartingPosition(unsigned int id);

private:
	static unsigned int xpo_count;
	xShapeType form;
	xstring particleShapeName;
	unsigned int sid;
	unsigned int csid;
	unsigned int mid;
	unsigned int np;
	unsigned int cnp;
	unsigned int each;
	double min_radius;
	double max_radius;

	vector4d* pos;
	vector4d* cpos;
	vector4d* ep;
	double* mass;
	vector3d* inertia;
	//vector4d *relative_loc;
	vector4d* spos[100];
	xmap<xstring, xEachObjectData> cobjects;
};

#endif