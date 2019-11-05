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
	
	bool checkOverlab(vector3i ctype, vector3d p, vector3d c, vector3d u0, vector3d u1);
	vector3d particle_polygon_contact_detection(host_triangle_info& hpi, vector3d& p, double r, int& ct);
	bool updateCollisionPair(unsigned int id, double r, vector3d pos, unsigned int &oid, vector3d& ocpt, vector3d& ounit, vector3i& ctype);
	void particle_triangle_contact_force(xTrianglePairData* d, double r, double m, vector3d& p, vector3d& v, vector3d& o, double &res, vector3d &tmax, vector3d& F, vector3d& M);

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

	virtual void define(unsigned int idx, unsigned int np);
	virtual void update();
	//virtual void initialize();
	static void savePartData(unsigned int np);
	static unsigned int GetNumMeshSphere();
	//static double GetMaxSphereRadius();
	static void local_initialize();
	bool check_this_mesh(unsigned int idx);
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
	host_triangle_info* hti;
	device_triangle_info* dti;
	device_body_info *dbi;

	// host particle-triangle contact parameters
	static double* tsd_ptri;
	static unsigned int* pair_count_ptri;
	static unsigned int* pair_id_ptri;
	//device_triangle_info* hti;

	xParticleObject* p;
	xMeshObject* po;

	xmap<unsigned int, xTrianglePairData*> triangle_pair;
	xmap<unsigned int, xTrianglePairData*> triangle_line_pair;
	xmap<unsigned int, xTrianglePairData*> triangle_point_pair;
};

#endif