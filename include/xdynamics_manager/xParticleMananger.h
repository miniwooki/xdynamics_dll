#ifndef XPARTICLEMANAGER_H
#define XPARTICLEMANAGER_H

#include "xdynamics_decl.h"
#include "xdynamics_object/xParticleObject.h"
#include "xModel.h"
#include "xmap.hpp"

//class xParticleObject;
class xClusterObject;
class xObject;

class XDYNAMICS_API xParticleManager
{
public:
	xParticleManager();
	~xParticleManager();

	unsigned int NumParticle();
	//unsigned int NumClusterSet();
	bool CopyPosition(
		double *pos, double* cpos, double* ep, unsigned int inp);
	bool CopyMassAndInertia(double* mass, double* inertia);
	void CopyClusterInformation(xClusterInformation* xci, double* rcloc);
	bool SetClusterMassAndInertia(xParticleObject* xpo);
	bool SetMassAndInertia(xParticleObject* xpo);
	xmap<xstring, xParticleObject*>& XParticleObjects();
	xParticleObject* XParticleObject(std::string ws);
	void ExportParticleDataForView(std::string path);
// 	void setRealTimeCreating(bool b);
// 	bool OneByOneCreating();
	unsigned int nClusterObject();
	unsigned int NumCluster();
	unsigned int nClusterEach();
	unsigned int NumMassParticle();
	double CriticalDensity();
	double CriticalPoisson();
	double CriticalYoungs();
	double CriticalRadius();
	//unsigned int nSingleSphere();
	//unsigned int nClusterSphere();
	//unsigned int NumParticleWithCluster();
	//unsigned int* ClusterIndex();
	//unsigned int* ClusterCount();
	//unsigned int* ClusterBegin();
	//vector3d* ClusterRelativeLocation();

	double* GetPositionResultPointer(unsigned int pt);
	double* GetVelocityResultPointer(unsigned int pt);
// 	unsigned int RealTimeCreating();
	void AllocParticleResultMemory(unsigned int npart, unsigned int np);
	void SetCurrentParticlesFromPartResult(std::string path);
	//void SetClusterInformation();
	void AddParticleCreatingCondition(xParticleObject* xpo, xParticleCreateCondition& xpcc);
	unsigned int ExcuteCreatingCondition(double ct, unsigned int cstep, unsigned int cnp);
	//static unsigned int GetNumLineParticles(double len, double r0, double r1 = 0);
	static unsigned int GetNumCubeParticles(double dx, double dy, double dz, double min_radius, double max_radius);
	static unsigned int GetNumLineParticles(double sx, double sy, double sz, double ex, double ey, double ez, double min_radius, double max_radius);
	static unsigned int GetNumPlaneParticles(double dx, unsigned int ny, double dy, double min_radius, double max_radius);
	static unsigned int GetNumCircleParticles(double d, double min_radius, double max_radius);
//	static unsigned int GetNumSPHPlaneParticles(double dx, double dy, double ps);

	xParticleObject* CreateLineParticle(std::string n, xMaterialType mt, unsigned int _np, xLineParticleData& d);
	xParticleObject* CreateParticleFromList(std::string n, xMaterialType mt, unsigned int _np, vector4d* d, double *m);
	xParticleObject* CreateCubeParticle(std::string n, xMaterialType mt, unsigned int _np, xCubeParticleData& d);
	xParticleObject* CreateCircleParticle(std::string n, xMaterialType mt, unsigned int _np, xCircleParticleData& d);
	xParticleObject* CreateClusterParticle(std::string n, xMaterialType mt, vector3d& loc, vector3i& grid, xClusterObject* xo);
	xParticleObject* CreateMassParticle(std::string n, xMaterialType mt, double rad, xPointMassData& d);
	xParticleObject* CreatePlaneParticle(std::string n, xMaterialType mt, xPlaneParticleData& d);
//	xParticleObject* CreateSPHParticles(xObject* xobj, double ps, unsigned int nlayer);
//	xParticleObject* CreateBoundaryParticles(xObject* xobj, double lx, double ly, double lz, double ps);
	//xParticleObject* CreateSPHLineParticle(std::string n, xMa)
	//xParticleObject* CreateSPHPlaneParticleObject(std::string n, xMaterialType mt, xSPHPlaneObjectData& d);

private:
	void setCriticalMaterial(double d, double y, double p);
	//void create_sph_particles_with_plane_shape(double dx, double dy, double lx, double ly, double lz, double ps);
// 	bool is_realtime_creating;
// 	bool one_by_one;
	unsigned int n_cluster_object;
	unsigned int n_cluster_each;
	//unsigned int n_single_sphere;
	//unsigned int n_cluster_sphere; // the number of cluster set element
	unsigned int np;
	unsigned int ncluster;
	unsigned int n_mass_particle;
	//unsigned int num_xpo;
	//unsigned int per_np;
	//double per_time;
	xmap<xstring, xParticleObject*> xpcos;
	xmap<xstring, xParticleCreateCondition> xpccs;

	double *r_pos;
	double *r_vel;
	double minimum_particle_density;
	double maximum_youngs_modulus;
	double minimum_poisson_ratio;
	double minimum_radius;
	//bool *isCluster;
	//unsigned int *cluster_index;
	//unsigned int *cluster_count;
//unsigned int *cluster_begin;
	//vector3d* rcluster;
	xMaterialType *r_type;
};


#endif