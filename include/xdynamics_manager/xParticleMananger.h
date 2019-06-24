#ifndef XPARTICLEMANAGER_H
#define XPARTICLEMANAGER_H

#include "xdynamics_decl.h"
#include "xdynamics_object/xParticleObject.h"
#include "xModel.h"

//class xParticleObject;
class xClusterObject;
class xObject;

class XDYNAMICS_API xParticleManager
{
public:
	xParticleManager();
	~xParticleManager();

	unsigned int NumParticle();
	unsigned int NumClusterSet();
	bool CopyPosition(double *pos, double *cpos, double* ep, unsigned int* cindex, unsigned int inp);
	bool SetMassAndInertia(double *mass, double *inertia);
	QMap<QString, xParticleObject*>& XParticleObjects();
	xParticleObject* XParticleObject(QString& ws);
	void ExportParticleDataForView(std::string path);
// 	void setRealTimeCreating(bool b);
// 	bool OneByOneCreating();
	unsigned int nClusterObject();
	unsigned int nClusterEach();
	unsigned int nSingleSphere();
	unsigned int nClusterSphere();
	unsigned int NumParticleWithCluster();
	unsigned int* ClusterIndex();
	unsigned int* ClusterCount();
	unsigned int* ClusterBegin();
	vector3d* ClusterRelativeLocation();

	double* GetPositionResultPointer(unsigned int pt);
	double* GetVelocityResultPointer(unsigned int pt);
// 	unsigned int RealTimeCreating();
	void AllocParticleResultMemory(unsigned int npart, unsigned int np);
	void SetCurrentParticlesFromPartResult(std::string path);
	void SetClusterInformation();
	void AddParticleCreatingCondition(xParticleObject* xpo, xParticleCreateCondition& xpcc);
	unsigned int ExcuteCreatingCondition(double ct, unsigned int cstep, unsigned int cnp);
	//static unsigned int GetNumLineParticles(double len, double r0, double r1 = 0);
	static unsigned int GetNumCubeParticles(double dx, double dy, double dz, double min_radius, double max_radius);
	static unsigned int GetNumPlaneParticles(double dx, unsigned int ny, double dy, double min_radius, double max_radius);
	static unsigned int GetNumCircleParticles(double d, double min_radius, double max_radius);
//	static unsigned int GetNumSPHPlaneParticles(double dx, double dy, double ps);

	xParticleObject* CreateParticleFromList(std::string n, xMaterialType mt, unsigned int _np, vector4d* d);
	xParticleObject* CreateCubeParticle(std::string n, xMaterialType mt, unsigned int _np, xCubeParticleData& d);
	xParticleObject* CreateCircleParticle(std::string n, xMaterialType mt, unsigned int _np, xCircleParticleData& d);
	xParticleObject* CreateClusterParticle(std::string n, xMaterialType mt, unsigned int _np, xClusterObject* xo);
//	xParticleObject* CreateSPHParticles(xObject* xobj, double ps, unsigned int nlayer);
//	xParticleObject* CreateBoundaryParticles(xObject* xobj, double lx, double ly, double lz, double ps);
	//xParticleObject* CreateSPHLineParticle(std::string n, xMa)
	//xParticleObject* CreateSPHPlaneParticleObject(std::string n, xMaterialType mt, xSPHPlaneObjectData& d);

private:
	//void create_sph_particles_with_plane_shape(double dx, double dy, double lx, double ly, double lz, double ps);
// 	bool is_realtime_creating;
// 	bool one_by_one;
	unsigned int n_cluster_object;
	unsigned int n_cluster_each;
	unsigned int n_single_sphere;
	unsigned int n_cluster_sphere; // the number of cluster set element
	unsigned int np;
	//unsigned int num_xpo;
	//unsigned int per_np;
	//double per_time;
	QMap<QString, xParticleObject*> xpcos;
	QList<xParticleCreateCondition> xpccs;

	double *r_pos;
	double *r_vel;
	bool *isCluster;
	unsigned int *cluster_index;
	unsigned int *cluster_count;
	unsigned int *cluster_begin;
	vector3d* cluster_set_location;
	xMaterialType *r_type;
};

#endif