#ifndef XPARTICLEPLANECONTACT_H
#define XPARTICLEPLANECONTACT_H

#include "xdynamics_object/xConatct.h"

// class xParticleObject;
class xPlaneObject;

class XDYNAMICS_API xParticlePlaneContact : public xContact
{
public:
	xParticlePlaneContact();
	xParticlePlaneContact(std::string _name);
	xParticlePlaneContact(const xContact& xc);
	virtual ~xParticlePlaneContact();

	xPlaneObject* PlaneObject();
	void setPlane(xPlaneObject* _pe);
	virtual void cuda_collision(
		double *pos, double *vel,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);
	//irtual void updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos, double rj = 0, vector3d posj = new_vector3d(0.0, 0.0, 0.0));
	virtual void collision(double r, double m, vector3d& pos, vector3d& vel, vector3d& omega, vector3d& F, vector3d& M);
	virtual void cudaMemoryAlloc(unsigned int np);
	static bool detect_contact(vector4f& p, xPlaneObject& pl, vector3f& cpoint);

private:
	double particle_plane_contact_detection(xPlaneObject* _pe, vector3d& u, vector3d& xp, vector3d& wp, double r);
	friend class xParticleCubeContact;
	void singleCollision(
		xPlaneObject* _pe, double mass, double rad, vector3d& pos, vector3d& vel,
		vector3d& omega, vector3d& force, vector3d& moment);
	void cudaMemoryAlloc_planeObject();

// 	xObject* p;
// 	xPlaneObject *pe;
	//device_plane_info *dpi;
	//host_plane_info hpi;
};

#endif