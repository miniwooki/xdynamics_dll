#ifndef XPARTICLEMANAGER_H
#define XPARTICLEMANAGER_H

#include "xdynamics_decl.h"
#include "xModel.h"

class xParticleObject;
class xObject;

class XDYNAMICS_API xParticleManager
{
public:
	xParticleManager();
	~xParticleManager();

	unsigned int NumParticle();
	bool CopyPosition(double *pos, unsigned int inp);
	bool SetMassAndInertia(double *mass, double *inertia);
	QMap<QString, xParticleObject*>& XParticleObjects();
	xParticleObject* XParticleObject(QString& ws);
	void ExportParticleDataForView(std::string path);
	void ImportParticleDataFromPartResult(std::string path);
// 	void setRealTimeCreating(bool b);
// 	bool OneByOneCreating();
	double* GetPositionResultPointer(unsigned int pt);
	double* GetVelocityResultPointer(unsigned int pt);
// 	unsigned int RealTimeCreating();
	void AllocParticleResultMemory(unsigned int npart, unsigned int np);
	//static unsigned int GetNumLineParticles(double len, double r0, double r1 = 0);
	static unsigned int GetNumCubeParticles(double dx, double dy, double dz, double min_radius, double max_radius);
	static unsigned int GetNumPlaneParticles(double dx, unsigned int ny, double dy, double min_radius, double max_radius);
	static unsigned int GetNumCircleParticles(double d, unsigned int ny, double min_radius, double max_radius);
//	static unsigned int GetNumSPHPlaneParticles(double dx, double dy, double ps);

	xParticleObject* CreateParticleFromList(std::string n, xMaterialType mt, unsigned int _np, vector4d* d);
	xParticleObject* CreateCubeParticle(std::string n, xMaterialType mt, unsigned int _np, xCubeParticleData& d);
//	xParticleObject* CreateSPHParticles(xObject* xobj, double ps, unsigned int nlayer);
//	xParticleObject* CreateBoundaryParticles(xObject* xobj, double lx, double ly, double lz, double ps);
	//xParticleObject* CreateSPHLineParticle(std::string n, xMa)
	//xParticleObject* CreateSPHPlaneParticleObject(std::string n, xMaterialType mt, xSPHPlaneObjectData& d);

private:
	//void create_sph_particles_with_plane_shape(double dx, double dy, double lx, double ly, double lz, double ps);
// 	bool is_realtime_creating;
// 	bool one_by_one;
	unsigned int np;
	//unsigned int num_xpo;
	//unsigned int per_np;
	//double per_time;
	QMap<QString, xParticleObject*> xpcos;

	double *r_pos;
	double *r_vel;
	xMaterialType *r_type;
};


#endif