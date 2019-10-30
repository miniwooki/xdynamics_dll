#ifndef XPARTICLEPLANECONTACT_H
#define XPARTICLEPLANECONTACT_H

#include "xdynamics_object/xConatct.h"

class xParticleObject;
class xPlaneObject;

class XDYNAMICS_API xParticlePlaneContact : public xContact
{
public:
	xParticlePlaneContact();
	xParticlePlaneContact(std::string _name, xObject* o1, xObject* o2);
	xParticlePlaneContact(const xContact& xc);
	virtual ~xParticlePlaneContact();

	xPlaneObject* PlaneObject();
	void setPlane(xPlaneObject* _pe);
	void save_contact_result(unsigned int pt, unsigned int np);
	//irtual void updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos, double rj = 0, vector3d posj = new_vector3d(0.0, 0.0, 0.0));
	virtual void collision(
		unsigned int id, double *pos, double *ep, double *vel, double *ev,
		double *mass, double* inertia,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);
	virtual void alloc_memories(unsigned int np);
	static bool detect_contact(vector4f& p, xPlaneObject& pl, vector3f& cpoint);

private:
	double particle_plane_contact_detection(xPlaneObject* _pe, vector3d& u, vector3d& xp, vector3d& wp, double r);
	friend class xParticleCubeContact;
	//void cudaMemoryAlloc_planeObject();

	unsigned int id;

	static double *d_tsd_ppl;
	static unsigned int *d_pair_count_ppl;
	static unsigned int *d_pair_id_ppl;

	double* tsd_ppl;
	unsigned int* pair_count_ppl;
	unsigned int* pair_id_ppl;

	device_plane_info* dpi;
	device_body_info* dbi;
	device_body_force* dbf;

 	xParticleObject* p;
 	xPlaneObject *pe;
	//device_plane_info *dpi;
	//host_plane_info hpi;
};

#endif