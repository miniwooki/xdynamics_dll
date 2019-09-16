#ifndef XPARTICLEMESHOBJECTSCONTACT_H
#define XPARTICLEMESHOBJECTSCONTACT_H

#include "xdynamics_object/xConatct.h"
#include "xdynamics_object/xParticleMeshObjectContact.h"
//#include <QtCore/QMap>
//#include <QtCore/QString>

class xMeshObject;

class XDYNAMICS_API xParticleMeshObjectsContact : public xContact
{
	//enum polygonContactType { FACE = 0, VERTEX, EDGE };
	struct host_mesh_info
	{
		int id;
		unsigned int sid;
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
		double mass;
		double px, py, pz;
		double vx, vy, vz;
		double ox, oy, oz;
		double fx, fy, fz;
		double mx, my, mz;
	};

public:
	xParticleMeshObjectsContact();
	virtual ~xParticleMeshObjectsContact();

	double MaxRadiusOfPolySphere() { return maxRadius; }
	double* SphereData() { return dsphere; }
	vector4d* HostSphereData() { return hsphere; }
	vector4d* GetCurrentSphereData();
	unsigned int NumSphereData();
	unsigned int define(xmap<xstring, xParticleMeshObjectContact*>& cppos);
	bool cppolyCollision(
		xContactPairList* pairs, unsigned int i, double r, double m,
		vector3d& p, vector3d& v, vector3d& o,
		double &res, vector3d &tmax, vector3d& F, vector3d& M,
		unsigned int nco, xClusterInformation* xci, vector4d* cpos);
	void particle_triangle_contact_force(
		xTrianglePairData* d, double r, double m,
		vector3d& p, vector3d& v, vector3d& o,
		double &res, vector3d &tmax, vector3d& F, vector3d& M);
	unsigned int NumContact() { return ncontact; }
	unsigned int NumContactObjects() { return nPobjs; }
	void setNumContact(unsigned int c) { ncontact = c; }
	void updateMeshObjectData(bool is_first_set_up = false);
	void updateMeshMassData();
	void getMeshContactForce();
	bool updateCollisionPair(
		unsigned int id, xContactPairList& xcpl, double r, 
		vector3d pos, unsigned int &oid, vector3d& ocpt, vector3d& ounit, vector3i& ctype);
	void updateCollisionPairLineOrVertex(double r, vector3d& pos, unsigned int& oid, vector3d& ocpt, vector3d& ounit, xContactPairList& xcpl);
	virtual void cudaMemoryAlloc(unsigned int np);
	virtual void cuda_collision(
		double *pos, double *vel, double *omega,
		double *mass, double *force, double *moment,
		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np);
	void setZeroCollisionForce();
	device_body_force* deviceBodyForceAndMoment();
	device_triangle_info* deviceTrianglesInfo();
	device_body_info* devicePolygonObjectMassInfo();
	void ExportTriangleSphereLocalPosition(std::string& name, unsigned int b, unsigned int e, vector3d* hlocal, double *rad);

private:
	vector3d particle_polygon_contact_detection(
		host_mesh_info& dpi, vector3d& p, double r, int& t);
	bool checkOverlab(vector3i ctype, vector3d p, vector3d c, vector3d u0, vector3d u1);
	unsigned int ncontact;
	unsigned int nmoving;
	//polygonContactType *pct;
	double maxRadius;
	unsigned int nPobjs;
	unsigned int npolySphere;
	xContactMaterialParameters* hcp;
	xMaterialPair* xmps;
	// 	contact_parameter* dcp;
	vector4d *hsphere;
	double* dsphere;
	double* dlocal;
	vector3d* hlocal;
	double* dvList;
	unsigned int* diList;
	//double* dvertexList;
	host_mesh_info* hpi;
	device_triangle_info* dpi;
	xmap<unsigned int, xMeshObject*> pair_ip;
	//host_mesh_mass_info *hpmi;
//	double* dep;
	device_body_info *dbi;
	device_body_force *dbf;
};

#endif