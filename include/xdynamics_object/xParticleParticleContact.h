#ifndef XPARTICLEPARTICLECONTACT_H
#define XPARTICLEPARTICLECONTACT_H

#include "xdynamics_decl.h"
#include "xdynamics_object/xConatct.h"

class XDYNAMICS_API xParticleParticleContact : public xContact
{
public:
	xParticleParticleContact();
	xParticleParticleContact(std::string _name, xObject* o1, xObject* o2);
	virtual ~xParticleParticleContact();

	void cppCollision(
		xContactPairList* pairs, unsigned int i,
		vector4d *pos, vector4d* cpos, vector3d *vel, euler_parameters* ep,
		euler_parameters *ev, double *mass, 
		double &res, vector3d &tmax,
		vector3d& F, vector3d& M, xClusterInformation* xci, unsigned int nco);

	void updateCollisionPair(
		unsigned int ip, unsigned int jp, bool isc,
		double ri, double rj, vector3d& posi, vector3d& posj);

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
		vector4d * pos, euler_parameters * ep, vector3d * vel,
		euler_parameters * ev, double* mass, double & rres, vector3d & tmax,
		vector3d & force, vector3d & moment, unsigned int nco,
		xClusterInformation * xci, vector4d * cpos);

	//virtual void cudaMemoryAlloc(unsigned int np);
	virtual void define(unsigned int idx, unsigned int np);
	virtual void update();

	void savePartData(unsigned int np);
	void deviceContactCount(double* pos, unsigned int *sorted_id, unsigned int *cstart, unsigned int *cend, unsigned int np);

private:
	unsigned int *d_pair_count_pp;
	unsigned int *d_pair_id_pp;
	double *d_tsd_pp;

	unsigned int* pair_count_pp;
	unsigned int* pair_id_pp;
	double* tsd_pp;

	xmap<vector2ui, xPairData*> c_pairs;
};

#endif