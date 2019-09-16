#ifndef XPARTICLEMESHOBJECTCONTACT_H
#define XPARTICLEMESHOBJECTCONTACT_H

#include "xdynamics_object/xConatct.h"
#include "xdynamics_object/xMeshObject.h"

class xParticleObject;
class xMeshObject;

class XDYNAMICS_API xParticleMeshObjectContact : public xContact
{
public:
	xParticleMeshObjectContact();
	xParticleMeshObjectContact(std::string _name, xObject* o1, xObject* o2);
	virtual ~xParticleMeshObjectContact();

	virtual void cudaMemoryAlloc(unsigned int np);
	//void insertContactParameters(unsigned int id, double r, double rt, double fr);
	xMeshObject* MeshObject() { return po; }
	//virtual void updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos, double rj = 0, vector3d posj = new_vector3d(0.0, 0.0, 0.0));
	virtual void cuda_collision(
		double *pos, double *vel,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

private:
	double maxRadii;
	xParticleObject* p;
	xMeshObject* po;
};

#endif