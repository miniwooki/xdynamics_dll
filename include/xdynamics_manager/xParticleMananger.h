#ifndef XPARTICLEMANAGER_H
#define XPARTICLEMANAGER_H

#include "xdynamics_decl.h"
#include "xModel.h"

class xParticleObject;

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
// 	void setRealTimeCreating(bool b);
// 	bool OneByOneCreating();
// 
// 	unsigned int RealTimeCreating();

	static unsigned int GetNumCubeParticles(double dx, double dy, double dz, double min_radius, double max_radius);
	static unsigned int GetNumPlaneParticles(double dx, unsigned int ny, double dy, double min_radius, double max_radius);
	static unsigned int GetNumCircleParticles(double d, unsigned int ny, double min_radius, double max_radius);

	xParticleObject* CreateParticleFromList(std::string n, xMaterialType mt, unsigned int _np, vector4d* d);
	xParticleObject* CreateCubeParticle(std::string n, xMaterialType mt, unsigned int _np, xCubeParticleData& d);

private:
// 	bool is_realtime_creating;
// 	bool one_by_one;
	unsigned int np;
	//unsigned int num_xpo;
	//unsigned int per_np;
	//double per_time;
	QMap<QString, xParticleObject*> xpcos;
};



#endif