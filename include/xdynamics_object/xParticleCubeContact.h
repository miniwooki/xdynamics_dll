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
	virtual void collision(double r, double m, vector3d& pos, vector3d& vel, vector3d& omega, vector3d& F, vector3d& M);
	virtual void cudaMemoryAlloc(unsigned int np);

private:
	xParticleObject* p;
	xCubeObject* cu;
	device_plane_info *dpi;
};

#endif