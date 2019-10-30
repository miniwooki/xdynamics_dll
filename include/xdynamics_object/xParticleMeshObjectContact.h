#ifndef XPARTICLEMESHOBJECTCONTACT_H
#define XPARTICLEMESHOBJECTCONTACT_H

#include "xdynamics_object/xConatct.h"
#include "xdynamics_object/xMeshObject.h"

class xParticleObject;
class xMeshObject;

class XDYNAMICS_API xParticleMeshObjectContact : public xContact
{
	struct host_mesh_info
	{
		int id;
		unsigned int sid;
		//vector3ui indice;
		vector3d P;
		vector3d Q;
		vector3d R;
		vector3d V;
		vector3d W;
		vector3d N;
	};
public:
	xParticleMeshObjectContact();
	xParticleMeshObjectContact(std::string _name, xObject* o1, xObject* o2);
	virtual ~xParticleMeshObjectContact();

	virtual void alloc_memories(unsigned int np);
	//void insertContactParameters(unsigned int id, double r, double rt, double fr);
	xMeshObject* MeshObject() { return po; }
	void save_contact_result(unsigned int pt, unsigned int np);
	bool define();
	//virtual void updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos, double rj = 0, vector3d posj = new_vector3d(0.0, 0.0, 0.0));
	virtual void collision(
		double *pos, double *ep, double *vel, double *ev,
		double *mass, double* inertia,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

	static unsigned int GetNumMeshSphere();

private:
	static double max_sphere_radius;
	static int nmoving;
	static unsigned int n_mesh_sphere;
	vector4d *hsphere;
	vector3d *hlocal;
	host_mesh_info *hmi;
	xContactMaterialParameters *hcp;
	xMaterialPair* hmp;
	double* dsphere;
	// device particle-triangle contact parameters
	double *d_tsd_ptri;
	double *d_tri_sph;
	unsigned int *d_pair_count_ptri;
	unsigned int *d_pair_id_ptri;
	device_triangle_info* dti;
	device_body_info *dbi;

	// host particle-triangle contact parameters
	double* tsd_ptri;
	unsigned int* pair_count_ptri;
	unsigned int* pair_id_ptri;	
	device_triangle_info* hti;

	double maxRadii;
	xParticleObject* p;
	xMeshObject* po;
};

#endif