#ifndef XPARTICLEMESHOBJECTSCONTACT_H
#define XPARTICLEMESHOBJECTSCONTACT_H

#include "xdynamics_object/xConatct.h"
#include "xdynamics_object/xParticleMeshObjectContact.h"
#include <QtCore/QMap>
#include <QtCore/QString>

class xMeshObject;

class XDYNAMICS_API xParticleMeshObjectsContact : public xContact
{
	//enum polygonContactType { FACE = 0, VERTEX, EDGE };
	struct host_mesh_info
	{
		int id;
		//vector3ui indice;
		vector3d P;
		vector3d Q;
		vector3d R;
		vector3d V;
		vector3d W;
		vector3d N;
	};

	struct host_mesh_mass_info
	{
		double px, py, pz;
		double vx, vy, vz;
		double ox, oy, oz;
		double fx, fy, fz;
		double mx, my, mz;
		double e0, e1, e2, e3;
	};

public:
	xParticleMeshObjectsContact();
	virtual ~xParticleMeshObjectsContact();

	double MaxRadiusOfPolySphere() { return maxRadius; }
	double* SphereData() { return dsphere; }
	vector4d* HostSphereData() { return hsphere; }
	unsigned int define(QMap<QString, xParticleMeshObjectContact*>& cppos);
	bool cppolyCollision(
		unsigned int idx, double r, double m,
		vector3d& p, vector3d& v, vector3d& o, vector3d& F, vector3d& M);
	unsigned int NumContact() { return ncontact; }
	unsigned int NumContactObjects() { return nPobjs; }
	void setNumContact(unsigned int c) { ncontact = c; }
	void updateMeshObjectData();
	void updateMeshMassData();
	void getMeshContactForce();
	void updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos);
	virtual void cudaMemoryAlloc(unsigned int np);
	virtual void cuda_collision(
		double *pos, double *vel, double *omega,
		double *mass, double *force, double *moment,
		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np);
	void setZeroCollisionForce();
	device_triangle_info* deviceTrianglesInfo();
	device_mesh_mass_info* devicePolygonObjectMassInfo();

private:
	vector3d particle_polygon_contact_detection(host_mesh_info& dpi, vector3d& p, double r/*, polygonContactType& _pct*/);

	unsigned int ncontact;
	//polygonContactType *pct;
	double maxRadius;
	unsigned int nPobjs;
	unsigned int npolySphere;
	xContactMaterialParameters* hcp;
	xMaterialPair* xmps;
	// 	contact_parameter* dcp;
	vector4d *hsphere;
	double* dsphere;
	double* dvList;
	unsigned int* diList;
	//double* dvertexList;
	host_mesh_info* hpi;
	device_triangle_info* dpi;
	QMap<unsigned int, xMeshObject*> pair_ip;
	host_mesh_mass_info *hpmi;
	device_mesh_mass_info *dpmi;
};

#endif