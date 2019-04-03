#ifndef XPARALLELDEM_DECL_CUH
#define XPARALLELDEM_DECL_CUH

#include "xdynamics_decl.h"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>

#define MAXIMUM_PAIR_NUMBER 5

struct device_parameters
{
	bool rollingCondition;
	unsigned int np;
	unsigned int nsphere;
	unsigned int ncell;
	uint3 grid_size;
	double dt;
	double half2dt;
	double cell_size;
	double cohesion;
	double3 gravity;
	double3 world_origin;
};

struct pair_data
{
	bool enable;
	unsigned int type;
	unsigned int i;
	unsigned int j;
	double ds;
	double dots;
};

struct device_mesh_info
{
	int id;
	double3 P;
	double3 Q;
	double3 R;
	double3 V;
	double3 W;
	double3 N;
};

struct device_plane_info
{
	double l1, l2;
	double3 u1;
	double3 u2;
	double3 uw;
	double3 xw;
	double3 pa;
	double3 pb;
	double3 w2;
	double3 w3;
	double3 w4;
};

struct device_mesh_mass_info
{
	double3 origin;
	double4 ep;
	double3 vel;
	double3 omega;
	double3 force;
	double3 moment;
};

struct device_cylinder_info
{
	double len, rbase, rtop;
	double3 pbase;
	double3 ptop;
	double3 origin;
	double3 vel;
	double3 omega;
	double4 ep;
};

struct device_contact_property
{
	double Ei, Ej;
	double pri, prj;
	double Gi, Gj;
	double rest;
	double fric;
	double rfric;
	double coh;
	double sratio;
};


struct device_force_constant
{
	double kn;
	double vn;
	double ks;
	double vs;
	double mu;
	double ms;
};

struct device_force_constant_d
{
	double kn;
	double vn;
	double ks;
	double vs;
	double mu;
};

void XDYNAMICS_API setSymbolicParameter(device_parameters *h_paras);

void XDYNAMICS_API vv_update_position(double *pos, double *vel, double *acc, unsigned int np);
void XDYNAMICS_API vv_update_velocity(double *vel, double *acc, double *omega, double *alpha, double *force, double *moment, double* mass, double* iner, unsigned int np);

void XDYNAMICS_API cu_calculateHashAndIndex(unsigned int* hash, unsigned int* index, double *pos, unsigned int np);
void XDYNAMICS_API cu_calculateHashAndIndexForPolygonSphere(
	unsigned int* hash, unsigned int* index,
	unsigned int sid, unsigned int nsphere, double *sphere);
void XDYNAMICS_API cu_reorderDataAndFindCellStart(unsigned int* hash, unsigned int* index, unsigned int* cstart, unsigned int* cend, unsigned int* sorted_index, unsigned int np, /*unsigned int nsphere,*/ unsigned int ncell);

void XDYNAMICS_API cu_calculate_p2p(
	const int tcm, double* pos, double* vel,
	double* omega, double* force,
	double* moment, double* mass,
	unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, device_contact_property* cp, unsigned int np);

// Function for contact between particle and plane
void XDYNAMICS_API cu_plane_contact_force(
	const int tcm, device_plane_info* plan,
	double* pos, double* vel, double* omega,
	double* force, double* moment, double* mass,
	unsigned int np, device_contact_property *cp);

void XDYNAMICS_API cu_cube_contact_force(
	const int tcm, device_plane_info* plan,
	double* pos, double* vel, double* omega,
	double* force, double* moment, double* mass,
	unsigned int np, device_contact_property *cp);

// Function for contact between particle and polygonObject
void XDYNAMICS_API cu_particle_meshObject_collision(
	const int tcm, device_mesh_info* dpi, double* dsph, device_mesh_mass_info* dpmi,
	double* pos, double* vel, double* omega,
	double* force, double* moment, double* mass,
	unsigned int* sidx, unsigned int* cstart, unsigned int* cend, device_contact_property *cp,
	unsigned int np/*, double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm*/);

// Function for contact between particle and cylinder
void XDYNAMICS_API cu_cylinder_hertzian_contact_force(
	const int tcm, device_cylinder_info* cyl,
	double* pos, double* vel, double* omega,
	double* force, double* moment, double* mass,
	unsigned int np, device_contact_property* cp,
	double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm);

double3 XDYNAMICS_API reductionD3(double3* in, unsigned int np);

void XDYNAMICS_API cu_update_meshObjectData(
	device_mesh_mass_info *dpmi, double* vList,
	double* sphere, device_mesh_info* dpi, unsigned int ntriangle);

//void XDYNAMICS_API cu_calculate_contact_pair_count(double* pos, unsigned int *count, unsigned int* sidx, unsigned int* cstart, unsigned int* cend, unsigned int np);
//void XDYNAMICS_API cu_check_no_collision_pair(double* pos, unsigned int* pinfo, unsigned int* pother, unsigned int np);
//void XDYNAMICS_API cu_check_new_collision_pair(double* pos, unsigned int* pinfo, unsigned int* pdata, unsigned int* sorted_id, unsigned int* cstart, unsigned int *cend, unsigned int np);
//void XDYNAMICS_API cu_calculate_particle_collision_with_pair(double* pos, double* vel, double* omega, double* mass, double* ds, double* force, double* moment, unsigned int* pinfo, unsigned int* pother, device_contact_property* cp, unsigned int np);

void XDYNAMICS_API cu_calculate_particle_particle_contact_count(
	double* pos, pair_data* pairs, 
	unsigned int* old_pair_count, 
	unsigned int* pair_count, 
	unsigned int* pair_start, 
	unsigned int* sorted_id, 
	unsigned int* cell_start,
	unsigned int* cell_end, unsigned int np);
void XDYNAMICS_API cu_new_particle_particle_contact
(double* pos, double* vel, double* omega, double* mass, double* force, double* moment, 
pair_data* old_pairs, pair_data* pairs, 
unsigned int* old_pair_count, unsigned int* pair_count, 
unsigned int *old_pair_start, unsigned int *pair_start, 
int *type_count, device_contact_property* cp, 
unsigned int *sorted_id, unsigned int* cell_start, unsigned int* cell_end, unsigned int np);

unsigned int XDYNAMICS_API cu_calculate_particle_plane_contact_count(
	device_plane_info *plane, pair_data* old_pppd, 
	unsigned int* old_pair_count, unsigned int *count, 
	unsigned int *sidx, double* pos, unsigned int _nplanes, 
	unsigned int np);
void XDYNAMICS_API cu_copy_old_to_new_pair(unsigned int *old_count, unsigned int *new_count, unsigned int* old_sidx, unsigned int* new_sidx, pair_data* old_pppd, pair_data* new_pppd, unsigned int nc, unsigned int np);
void XDYNAMICS_API cu_new_particle_plane_contact(device_plane_info *plane, double* pos, double* vel, double* omega, double* mass, double* force, double* moment, unsigned int* old_pair_count, unsigned int *count, unsigned int *old_sidx, unsigned int *sidx, int* type_count, pair_data *old_pairs, pair_data *pairs, device_contact_property *cp, unsigned int nplanes, unsigned int np);
#endif



