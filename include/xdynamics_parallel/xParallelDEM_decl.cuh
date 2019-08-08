#ifndef XPARALLELDEM_DECL_CUH
#define XPARALLELDEM_DECL_CUH

#include "xdynamics_decl.h"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>
//#include "vector_types.h"
//#include <helper_math.h>
typedef unsigned int uint;
//#include <helper_functions.h>
//#include <helper_cuda.h>
#define MAX_P2P_COUNT  12
#define MAX_P2PL_COUNT 3
#define MAX_P2MS_COUNT 5
#define MAX_P2CY_COUNT 3
#define CUDA_THREADS_PER_BLOCK 128
//__constant__ device_parameters cte;
//double3 toDouble3(VEC3D& v3) { return double3(v3.x, v3.y, v3.z); }
//inline double3 change_cuda_double3(VEC3D& v3) { return make_double3(v3.x, v3.y, v3.z); }
struct device_wave_damping
{

};

struct device_dem_parameters
{
	bool rollingCondition;
	unsigned int np;
	unsigned int nmp;
	unsigned int nCluster;
	unsigned int nClusterObject;
	unsigned int nTsdaConnection;
	unsigned int nTsdaConnectionList;
	unsigned int nTsdaConnectionBodyData;
	unsigned int nsphere;
	unsigned int nplane;
	unsigned int ncylinder;
	unsigned int ncell;
	uint3 grid_size;
	double dt;
	double half2dt;
	double cell_size;
	double cohesion;
	double3 gravity;
	double3 world_origin;
};

struct device_triangle_info
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
	bool ismoving;
	unsigned int mid;
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
	double mass;
	double3 origin;
	double3 vel;
	double3 omega;
	double3 force;
	double3 moment;
};

struct device_body_info
{
	double3 pos;
	double3 vel;
	double3 force;
	double3 moment;
	double4 ep;
	double4 ed;
};

struct device_tsda_connection_body_data
{
	unsigned int id;
	unsigned int kc_id;
	unsigned int body_id;
	double init_l;
	double3 rpos;
};

struct device_cylinder_info
{
	bool ismoving;
	unsigned int id;
	unsigned int mid;
	unsigned int empty_part;
	double thickness;
	double3 len_rr;// len, rbase, rtop;
	double3 pbase;
	double3 ptop;
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
	//double coh_s;
	double sratio;
	double amp;
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

struct device_cluster_information
{
	unsigned int neach;
	unsigned int ncluster;
};

struct device_triangle_contact_info
{
	unsigned int id;
	double3 cpt;
};

void XDYNAMICS_API setDEMSymbolicParameter(device_dem_parameters *h_paras);

void XDYNAMICS_API vv_update_position(
	double *pos, double* ep, double *vel, double* ev, double *acc, double* ea, unsigned int np);
void XDYNAMICS_API vv_update_velocity(
	double *vel, double *acc, double* ep, double *ev, double *ea, double *force, double *moment, double* mass, double* iner, unsigned int np);

void XDYNAMICS_API cu_calculateHashAndIndex(unsigned int* hash, unsigned int* index, double *pos, unsigned int np);
void XDYNAMICS_API cu_calculateHashAndIndexForPolygonSphere(
	unsigned int* hash, unsigned int* index,
	unsigned int sid, unsigned int nsphere, double *sphere);
void XDYNAMICS_API cu_reorderDataAndFindCellStart(unsigned int* hash, unsigned int* index, unsigned int* cstart, unsigned int* cend, unsigned int* sorted_index, unsigned int np, /*unsigned int nsphere,*/ unsigned int ncell);

void XDYNAMICS_API cu_calculate_p2p(
	const int tcm, double* pos, double* ep, double* vel,
	double* omega, double* force,
	double* moment, double* mass, double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd,
	unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, device_contact_property* cp, unsigned int np);

// Function for contact between particle and plane
void XDYNAMICS_API cu_plane_contact_force(
	const int tcm, device_plane_info* plan, device_body_info* dbi, device_contact_property *cp,
	double* pos, double* ep, double* vel, double* omega,
	double* force, double* moment, double* mass,
	double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd,
	unsigned int np);

void XDYNAMICS_API cu_cube_contact_force(
	const int tcm, device_plane_info* plan,
	double* pos, double* ep, double* vel, double* omega,
	double* force, double* moment, double* mass,
	unsigned int np, device_contact_property *cp);

// Function for contact between particle and polygonObject
void XDYNAMICS_API cu_particle_polygonObject_collision(
	const int tcm, device_triangle_info* dpi, device_mesh_mass_info* dpmi,
	double* pos, double* ep, double* vel, double* omega,
	double* force, double* moment, double* mass,
	double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd, double* dsph,
	unsigned int* sidx, unsigned int* cstart, unsigned int* cend, device_contact_property *cp,
	unsigned int np/*, double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm*/);

// Function for contact between particle and cylinder
void XDYNAMICS_API cu_cylinder_contact_force(
	const int tcm, device_cylinder_info* cyl, device_body_info* bi, device_contact_property *cp,
	double* pos, double* ep, double* vel, double* omega,
	double* force, double* moment, double* mass,
	double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd,
	unsigned int np);

void XDYNAMICS_API cu_decide_rolling_friction_moment(
	double* tmax,
	double* rres,
	double* inertia,
	double* ep,
	double* ev,
	double* moment,
	unsigned int np);

double3 XDYNAMICS_API reductionD3(double3* in, unsigned int np);
void XDYNAMICS_API cu_update_meshObjectData(
	double *vList, double* sph, double* dlocal, device_triangle_info* poly,
	device_mesh_mass_info* dpmi, double* ep, unsigned int np);

void XDYNAMICS_API cu_clusters_contact(
	double* pos, double* cpos, double* ep, double* vel,
	double* omega, double* force,
	double* moment, double* mass, double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd,
	unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, device_contact_property* cp, xClusterInformation* xci, unsigned int np);

void XDYNAMICS_API cu_cluster_plane_contact(
	device_plane_info* plan,
	double* pos, double* cpos, double* ep, double* vel, double* omega,
	double* force, double* moment, double* mass,
	double* tmax, double* rres,
	unsigned int* pair_count, unsigned int *pair_id, double* tsd, xClusterInformation* xci,
	unsigned int np, device_contact_property *cp);

void XDYNAMICS_API cu_cluster_meshes_contact(
	device_triangle_info *dpi, device_mesh_mass_info* dpmi,
	double* pos, double* cpos, double *ep, double* vel, double* ev,
	double* force, double* moment,
	device_contact_property *cp, double* mass,
	double* tmax, double* rres,
	unsigned int* pair_count, unsigned int* pair_id,
	double* tsd, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend,
	xClusterInformation* xci, unsigned int np);

void XDYNAMICS_API vv_update_cluster_position(
	double *pos, double *cpos, double* ep, 
	double *rloc, double *vel, double *acc, 
	double* omega, double* alpha, xClusterInformation *xci, unsigned int np);

void XDYNAMICS_API vv_update_cluster_velocity(
	double* cpos, double* ep, double *vel, double *acc, double *omega,
	double *alpha, double *force, double *moment, double* rloc,
	double* mass, double* iner, xClusterInformation *xci, unsigned int np);

void XDYNAMICS_API cu_calculate_spring_damper_force(
	double* pos,
	double* vel,
	double* force,
	xSpringDamperConnectionInformation* xsdci,
	xSpringDamperConnectionData* xsdcd,
	xSpringDamperCoefficient* xsdkc,
	unsigned int nc);

void XDYNAMICS_API cu_calculate_spring_damper_connecting_body_force(
	double* pos,
	double* vel,
	double* ep,
	double* ev,
	double* mass,
	double* force,
	double* moment,
	device_tsda_connection_body_data* xsdbcd,
	xSpringDamperCoefficient* xsdkc,
	unsigned int nc);

#endif



