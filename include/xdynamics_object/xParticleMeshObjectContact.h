#ifndef XPARTICLEMESHOBJECTCONTACT_H
#define XPARTICLEMESHOBJECTCONTACT_H

#include "xdynamics_object/xConatct.h"
#include "xdynamics_object/xMeshObject.h"

class xParticleObject;
class xMeshObject;

class XDYNAMICS_API xParticleMeshObjectContact : public xContact
{
	struct host_triangle_info
	{
		int id;
		unsigned int tid;
		//vector3ui indice;
		vector3d P;
		vector3d Q;
		vector3d R;
	};
public:
	xParticleMeshObjectContact();
	xParticleMeshObjectContact(std::string _name, xObject* o1, xObject* o2);
	virtual ~xParticleMeshObjectContact();

	//virtual void alloc_memories(unsigned int np);
	//void insertContactParameters(unsigned int id, double r, double rt, double fr);
	xMeshObject* MeshObject() { return po; }
	double* MeshSphere();
	
	//virtual void updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos, double rj = 0, vector3d posj = new_vector3d(0.0, 0.0, 0.0));
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
	//virtual void initialize();
	static void savePartData(unsigned int np);
	static unsigned int GetNumMeshSphere();
	//static double GetMaxSphereRadius();
	static void local_initialize();
private:
	//static double max_sphere_radius;
	//static int nmoving;
	
	static unsigned int defined_count;
	static bool allocated_static;
	static unsigned int n_mesh_sphere;
	unsigned int id;
	vector4d *hsphere;
	vector3d *hlocal;
	//host_triangle_info *hti;
	//xContactMaterialParameters *hcp;
	//xMaterialPair* hmp;
	double* dsphere;
	double* dlocal;
	double* dvList;
	// device particle-triangle contact parameters

	static double *d_tsd_ptri;
	//static double *d_tri_sph;
	static unsigned int *d_pair_count_ptri;
	static unsigned int *d_pair_id_ptri;
	device_triangle_info* dti;
	device_body_info *dbi;

	// host particle-triangle contact parameters
	static double* tsd_ptri;
	static unsigned int* pair_count_ptri;
	static unsigned int* pair_id_ptri;
	//device_triangle_info* hti;

	xParticleObject* p;
	xMeshObject* po;
};

#endif