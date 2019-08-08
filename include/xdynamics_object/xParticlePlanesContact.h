#ifndef XPARTICLEPLANESCONTACT_H
#define XPARTICLEPLANESCONTACT_H

#include "xdynamics_object/xConatct.h"
#include "xdynamics_object/xParticlePlaneContact.h"
#include <QtCore/QMap>
#include <QtCore/QString>

class xPlaneObject;

class XDYNAMICS_API xParticlePlanesContact : public xContact
{
	//enum polygonContactType { FACE = 0, VERTEX, EDGE };
	struct host_plane_info
	{
		bool ismoving;
		unsigned int mid;
		double l1, l2;
		vector3d u1;
		vector3d u2;
		vector3d uw;
		vector3d xw;
		vector3d pa;
		vector3d pb;
		vector3d w2;
		vector3d w3;
		vector3d w4;
	};

public:
	xParticlePlanesContact();
	virtual ~xParticlePlanesContact();

// 	double MaxRadiusOfPolySphere() { return maxRadius; }
// 	double* SphereData() { return dsphere; }
// 	vector4d* HostSphereData() { return hsphere; }
	void define(unsigned int i, xParticlePlaneContact* d);
	void define(unsigned int i, xParticleCubeContact* d);
	void allocHostMemory(unsigned int n);
	void updataPlaneObjectData();
	bool cpplCollision(
		xContactPairList* pairs, unsigned int i, double r, double m,
		vector3d& p, vector3d& v, vector3d& o, 
		double &R, vector3d& T, vector3d& F, vector3d& M,
		unsigned int nco, xClusterInformation* xci, vector4d* cpos);
	unsigned int NumContact();// { return ncontact; }
//	void setNumContact(unsigned int c) { ncontact = c; }
	//void updateMeshObjectData(xVectorD& q, xVectorD& qd);
	xContactMaterialParameters& ContactMaterialParameters(unsigned int id);
	void updateCollisionPair(xContactPairList& xcpl, double r, vector3d pos);
	virtual void cudaMemoryAlloc(unsigned int np);
	virtual void cuda_collision(
		double *pos, double *vel, double *omega,
		double *mass, double *force, double *moment,
		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np);
	//void setZeroCollisionForce();
	device_plane_info* devicePlaneInfo();
	device_body_info* devicePlaneBodyInfo();
	unsigned int NumPlanes();
	void getPlaneContactForce();

private:
	
	//vector3d particle_polygon_contact_detection(host_mesh_info& dpi, vector3d& p, double r/*, polygonContactType& _pct*/);
	double particle_plane_contact_detection(host_plane_info* _pe, vector3d& u, vector3d& xp, vector3d& wp, double r);
	
	unsigned int nplanes;
	unsigned int nmoving;
// 	unsigned int *d_old_pair_count;
// 	unsigned int *d_pair_count;
// 	unsigned int *d_old_pair_start;
// 	unsigned int *d_pair_start;
	/*particle_plane_pair_data* d_pppd;*/
	xContactMaterialParameters* hcmp;
	xMaterialPair* xmps;
	host_plane_info* hpi;
	device_plane_info* dpi;
	device_body_info* dbi;
	QMap<unsigned int, xPlaneObject*> pair_ip;

	//device_plane_info *hpmi;
	//device_mesh_mass_info *dpmi;
};

#endif