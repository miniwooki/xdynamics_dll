#ifndef XPARTICLEPARTICLECONTACT_H
#define XPARTICLEPARTICLECONTACT_H

#include "xdynamics_decl.h"
#include "xdynamics_object/xConatct.h"

class XDYNAMICS_API xParticleParticleContact : public xContact
{
public:
	xParticleParticleContact();
	xParticleParticleContact(std::wstring _name);
	virtual ~xParticleParticleContact();

	void cppCollision(
		xContactPairList* pairs, unsigned int i,
		vector4d *pos, vector3d *vel,
		vector3d *omega, double *mass, 
		double &res, vector3d &tmax,
		vector3d& F, vector3d& M);
	void updateCollisionPair(unsigned int id, xContactPairList& xcpl, double ri, double rj, vector3d& posi, vector3d& posj);
	virtual void cuda_collision(
		double *pos, double *vel,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);
	virtual void cudaMemoryAlloc(unsigned int np);

	void deviceContactCount(double* pos, unsigned int *sorted_id, unsigned int *cstart, unsigned int *cend, unsigned int np);
//private:
// 	unsigned int *d_pair_idx;
// 	unsigned int *d_pair_other;
// 	double* d_tan;
};

#endif