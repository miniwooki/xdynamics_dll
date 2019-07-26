#ifndef XPARTICLECYLINDERSCONTACT_H
#define XPARTICLECYLINDERSCONTACT_H

#include "xdynamics_decl.h"
#include "xdynamics_object/xConatct.h"
#include "xdynamics_object/xParticleCylinderContact.h"

class XDYNAMICS_API xParticleCylindersContact : public xContact
{
	struct host_cylinder_info
	{
		vector3d len_rr;// len, rbase, rtop;
		vector3d pbase;
		vector3d ptop;
		//double3 origin;
		//double3 vel;
		//double3 omega;
		//double4 ep;
	};
public:
	xParticleCylindersContact();
	virtual ~xParticleCylindersContact();

	void define(unsigned int i, xParticleCylinderContact * d);
	//void define(unsigned int i, xParticleCubeContact* d);
	void allocHostMemory(unsigned int n);
//	void updataPlaneObjectData();
	bool pcylCollision(
		xContactPairList* pairs, unsigned int i, double r, double m,
		vector3d& p, vector3d& v, vector3d& o,
		double &R, vector3d& T, vector3d& F, vector3d& M,
		unsigned int nco, xClusterInformation* xci, vector4d* cpos);
	unsigned int NumContact();
	//xContactMaterialParameters& ContactMaterialParameters(unsigned int id);
	void updateCollisionPair(xContactPairList& xcpl, double r, vector3d pos);
	virtual void cudaMemoryAlloc(unsigned int np);
	virtual void cuda_collision(
		double *pos, double *vel, double *omega,
		double *mass, double *force, double *moment,
		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np);
	//void setZeroCollisionForce();
	//device_plane_info* devicePlaneInfo();
	//unsigned int NumPlanes();

private:
	double particle_cylinder_contact_detection(
		host_cylinder_info& cinfo, xCylinderObject* c_ptr, vector3d& pt, vector3d& u, vector3d& cp, double r, bool& isInnerContact);
	double particle_cylinder_inner_base_or_top_contact_detection(
		host_cylinder_info& cinfo, xCylinderObject* c_ptr, vector3d& pt, vector3d& u, vector3d& cp, double r);
	//vector3d particle_polygon_contact_detection(host_mesh_info& dpi, vector3d& p, double r/*, polygonContactType& _pct*/);
	//double particle_plane_contact_detection(host_plane_info* _pe, vector3d& u, vector3d& xp, vector3d& wp, double r);
	QMap<unsigned int, xCylinderObject*> pair_ip;
	unsigned int ncylinders;
	xContactMaterialParameters* hcmp;
	xMaterialPair* xmps;
	host_cylinder_info *hci;
	device_cylinder_info* dci;
	//host_plane_info* hpi;
	//device_plane_info* dpi;
	//QMap<unsigned int, xPlaneObject*> pair_ip;
};

#endif
