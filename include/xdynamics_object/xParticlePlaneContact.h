#ifndef XPARTICLEPLANECONTACT_H
#define XPARTICLEPLANECONTACT_H

#include "xdynamics_object/xConatct.h"

class xParticleObject;
class xPlaneObject;

class XDYNAMICS_API xParticlePlaneContact : public xContact
{
	struct host_plane_info
	{
		unsigned int id;
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
	xParticlePlaneContact();
	xParticlePlaneContact(std::string _name, xObject* o1, xObject* o2);
	//xParticlePlaneContact(const xContact& xc);
	virtual ~xParticlePlaneContact();

	xPlaneObject* PlaneObject();
	//void define(unsigned int idx, unsigned int np);
	void setPlane(xPlaneObject* _pe);
	//irtual void updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos, double rj = 0, vector3d posj = new_vector3d(0.0, 0.0, 0.0));
	virtual void collision(
		double *pos, double *ep, double *vel, double *ev,
		double *mass, double* inertia,
		double *force, double *moment,
		double *tmax, double* rres,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);
	
	virtual void define(unsigned int idx, unsigned int np);
	virtual void update();

	static void savePartData(unsigned int np);
	static bool detect_contact(vector4f& p, xPlaneObject& pl, vector3f& cpoint);

private:
	double particle_plane_contact_detection(xPlaneObject* _pe, vector3d& u, vector3d& xp, vector3d& wp, double r);
	friend class xParticleCubeContact;
	//void cudaMemoryAlloc_planeObject();
	bool allocated_static;
	unsigned int id;
	
	static double *d_tsd_ppl;
	static unsigned int *d_pair_count_ppl;
	static unsigned int *d_pair_id_ppl;

	static double* tsd_ppl;
	static unsigned int* pair_count_ppl;
	static unsigned int* pair_id_ppl;

	host_plane_info hpi;
	//host_body_info hbi;

	device_plane_info* dpi;
	device_body_info* dbi;
	device_body_force* dbf;

 	xParticleObject* p;
 	xPlaneObject *pe;
	//device_plane_info *dpi;
	//host_plane_info hpi;
};

#endif