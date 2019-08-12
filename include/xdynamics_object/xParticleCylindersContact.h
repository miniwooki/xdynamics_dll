#ifndef XPARTICLECYLINDERSCONTACT_H
#define XPARTICLECYLINDERSCONTACT_H

#include "xdynamics_decl.h"
#include "xdynamics_object/xConatct.h"
#include "xdynamics_object/xParticleCylinderContact.h"

class XDYNAMICS_API xParticleCylindersContact : public xContact
{
	struct host_cylinder_info
	{
		bool ismoving;
		unsigned int id;
		unsigned int mid;
		unsigned int empty_part;
		double thickness;
		vector3d len_rr;
		vector3d pbase;
		vector3d ptop;
	};
public:
	xParticleCylindersContact();
	virtual ~xParticleCylindersContact();

	void define(unsigned int i, xParticleCylinderContact * d);
	void allocHostMemory(unsigned int n);
	bool pcylCollision(
		xContactPairList* pairs, unsigned int i, double r, double m,
		vector3d& p, vector3d& v, vector3d& o,
		double &R, vector3d& T, vector3d& F, vector3d& M,
		unsigned int nco, xClusterInformation* xci, vector4d* cpos);
	unsigned int NumContact();
	void updateCylinderObjectData();
	void getCylinderContactForce();
	device_cylinder_info* deviceCylinderInfo();
	device_body_info* deviceCylinderBodyInfo();
	device_body_force* deviceCylinderBodyForceAndMoment();
	void updateCollisionPair(xContactPairList& xcpl, double r, vector3d pos);
	virtual void cudaMemoryAlloc(unsigned int np);
	virtual void cuda_collision(
		double *pos, double *vel, double *omega,
		double *mass, double *force, double *moment,
		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np);

private:
	double particle_cylinder_contact_detection(
		host_cylinder_info& cinfo, xCylinderObject* c_ptr, vector3d& pt, vector3d& u, vector3d& cp, double r, bool& isInnerContact);
	double particle_cylinder_inner_base_or_top_contact_detection(
		host_cylinder_info& cinfo, xCylinderObject* c_ptr, vector3d& pt, vector3d& u, vector3d& cp, double r);
	QMap<unsigned int, xCylinderObject*> pair_ip;
	unsigned int ncylinders;
	unsigned int nmoving;
	xContactMaterialParameters* hcmp;
	xMaterialPair* xmps;
	host_cylinder_info *hci;
	device_cylinder_info* dci;
	device_body_info* dbi;
	device_body_force* dbf;
};

#endif
