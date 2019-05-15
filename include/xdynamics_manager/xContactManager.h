#ifndef XCONTACTMANAGER_H
#define XCONTACTMANAGER_H

#include "xdynamics_decl.h"
#include "xModel.h"
#include "xdynamics_object/xParticleParticleContact.h"
#include "xdynamics_object/xParticlePlaneContact.h"
#include "xdynamics_object/xParticleCubeContact.h"
#include "xdynamics_object/xParticleMeshObjectsContact.h"
#include "xdynamics_object/xParticlePlanesContact.h"

class xObject;
class xContact;
class xContactPairList;

class XDYNAMICS_API xContactManager
{
public:
	xContactManager();
	~xContactManager();

// 	void Save(QTextStream& qts);
// 	void Open(QTextStream& qts, particleManager* pm, geometryObjects* objs);
	xContact* CreateContactPair(
		std::string n, xContactForceModelType method, xObject* fo, xObject* so, xContactParameterData& d);

	unsigned int setupParticlesMeshObjectsContact();
	void setupParticlesPlanesContact();
	//double* SphereData();
	//double* HostSphereData();
	//float* SphereData_f();
	void insertContact(xContact* c);
	xContact* Contact(QString n);// { return cots[n]; }
	//QMap<QString, QString>& Logs() { return logs; }
	QMap<QString, xContact*>& Contacts();// { return cots; }
	xParticleParticleContact* ContactParticles();// { return cpp; }
	xParticleMeshObjectsContact* ContactParticlesMeshObjects();// { return cpmeshes; }
	xParticlePlanesContact* ContactParticlesPlanes();

	bool runCollision(
		double *pos, double *vel, double* ep,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

	void update();
	void allocPairList(unsigned int np);

private:
	void updateCollisionPair(vector4d* pos, unsigned int* sorted_id, unsigned int* cell_start, unsigned int* cell_end, unsigned int np);
	void deviceCollision(
		double *pos, double *vel, double* ep,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);
	void hostCollision(
		vector4d *pos, vector3d *vel,
		vector3d *omega, double *mass,
		vector3d *force, vector3d *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

	unsigned int deviceContactCount(
		double *pos, double *vel,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

	unsigned int ncontact;
	int *d_type_count;
	unsigned int *d_old_pair_count;
	unsigned int *d_pair_count;
	unsigned int *d_old_pair_start;
	unsigned int *d_pair_start;
	pair_data* d_pppd;

	QMap<QString, xContact*> cots;
	QMap<QString, xParticleMeshObjectContact*> cpmesh;
	xContactPairList* xcpl;
	//QMap<QString, contact_particles_polygonObject*> cppos;
	xParticlePlanesContact* cpplane;
	xParticleParticleContact* cpp;
	xParticleMeshObjectsContact* cpmeshes;
	//contact_particles_particles* cpp;
	//contact_particles_polygonObjects* cppoly;
};

#endif