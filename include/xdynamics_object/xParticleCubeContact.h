#ifndef XPARTICLECUBECONTACT_H
#define XPARTICLECUBECONTACT_H

#include "xdynamics_object/xConatct.h"

class xParticleObject;
class xCubeObject;

class XDYNAMICS_API xParticleCubeContact : public xContact
{
public:
	xParticleCubeContact();
	xParticleCubeContact(std::string _name, xObject* o1, xObject *o2);
	virtual ~xParticleCubeContact();

	xCubeObject* CubeObject();
	virtual void cuda_collision(
		double *pos, double *vel,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

	//virtual void updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos, double rj = 0, vector3d posj = new_vector3d(0.0, 0.0, 0.0));
	virtual void collision_gpu(
		double *pos, double* cpos, xClusterInformation* xci,
		double *ep, double *vel, double *ev,
		double *mass, double* inertia,
		double *force, double *moment,
		double *tmax, double* rres,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

	virtual void collision_cpu(
		xContactPairList* pairs, unsigned int pid, unsigned int cid, double r,
		vector4d* pos, euler_parameters* ep, vector3d* vel,
		euler_parameters* ev, double* mass, double& rres, vector3d& tmax,
		vector3d& force, vector3d& moment, unsigned int nco,
		xClusterInformation* xci, vector4d *cpos);

	virtual void cudaMemoryAlloc(unsigned int np);

private:

	xParticleObject* p;
	xCubeObject* cu;
	device_plane_info *dpi;
};

#endif