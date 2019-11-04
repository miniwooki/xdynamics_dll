#ifndef XCONTACTMANAGER_H
#define XCONTACTMANAGER_H

#include "xdynamics_decl.h"
#include "xModel.h"
#include "xdynamics_object/xParticleParticleContact.h"
#include "xdynamics_object/xParticlePlaneContact.h"
#include "xdynamics_object/xParticleCubeContact.h"
#include "xdynamics_object/xParticleMeshObjectContact.h"
#include "xdynamics_object/xParticleCylinderContact.h"
//#include "xdynamics_object/xParticleCylindersContact.h"

//#include <QtCore/qlist.h>

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
	void CreateContactPair(
		std::string n, xContactForceModelType method, xObject* fo, xObject* so, xContactParameterData& d);

	unsigned int setupParticlesMeshObjectsContact();
	void setupParticlesPlanesContact();
	void setupParticlesCylindersContact();
	void setNumClusterObject(unsigned int nc);
	void defineContacts(unsigned int np);
	//double* SphereData();
	//double* HostSphereData();
	//float* SphereData_f();
	xmap<int, xParticleMeshObjectContact*>& PMContacts();
	void insertContact(xContact* c);
	xContact* Contact(std::string n);// { return cots[n]; }
	//QMap<QString, QString>& Logs() { return logs; }
	xmap<xstring, xContact*>& Contacts();// { return cots; }
	xParticleParticleContact* ContactParticles();// { return cpp; }
	//xParticleMeshObjectsContact* ContactParticlesMeshObjects();// { return cpmeshes; }
	//xParticlePlanesContact* ContactParticlesPlanes();
	//xParticleCylindersContact* ContactParticlesCylinders();

	bool runCollision(
		double *pos, double* cpos, double* ep, double *vel,
		double *ev, double *mass, double* inertia,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		//unsigned int *cluster_index,
		xClusterInformation* xci,
		unsigned int np);

	void update();
	void allocPairList(unsigned int np);
	void SaveStepResult(unsigned int pt, unsigned int np);
	void set_from_part_result(std::fstream& fs);

private:
	void updateCollisionPair(
		vector4d* pos, 
		unsigned int* sorted_id,
		unsigned int* cell_start,
		unsigned int* cell_end,
		xClusterInformation* xci,
		unsigned int np);

	void deviceCollision(
		double *pos, double* cpos, double *ep, double *vel,
		double *ev, double *mass, double* inertia,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		xClusterInformation* xci,
		unsigned int np);

	void hostCollision(
		vector4d *pos, vector4d* cpos, vector3d *vel, euler_parameters* ep,
		euler_parameters *ev, double *mass, double *inertia,
		vector3d *force, vector3d *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		xClusterInformation* xci,
		unsigned int np);

	//unsigned int deviceContactCount(
	//	double *pos, double *ep, double *vel,
	//	double *omega, double *mass,
	//	double *force, double *moment,
	//	unsigned int *sorted_id,
	//	unsigned int *cell_start,
	//	unsigned int *cell_end,
	//	unsigned int np);

	unsigned int ncontact;
	unsigned int ncobject;
	//int *d_type_count;
	//unsigned int *d_old_pair_count;
	//unsigned int *d_pair_count_pp;
	////unsigned int *d_pair_count_ppl;
	////unsigned int *d_pair_count_ptri;
	////unsigned int *d_pair_count_pcyl;
	//unsigned int *d_pair_id_pp;
	////unsigned int *d_pair_id_ppl;
	////unsigned int *d_pair_id_ptri;
	////unsigned int *d_pair_id_pcyl;

	//double *d_tsd_pp;
	////double *d_tsd_ppl;
	////double *d_tsd_ptri;
	////double *d_tsd_pcyl;

	//unsigned int* pair_count_pp;
	////unsigned int* pair_count_ppl;
	////unsigned int* pair_count_ptri;
	////unsigned int* pair_count_pcyl;
	//unsigned int* pair_id_pp;
	////unsigned int* pair_id_ppl;
	////unsigned int* pair_id_ptri;
	////unsigned int* pair_id_pcyl;

	//double* tsd_pp;
	//double* tsd_ppl;
	//double* tsd_ptri;
	//double* tsd_pcyl;

	double* d_Tmax;
	double* d_RRes;
	vector3d* Tmax;
	double* RRes;
	unsigned int n_total_mesh_sphere;
	xmap<xstring, xContact*> cots;
	xParticleParticleContact* cpp;
	xmap<int, xParticlePlaneContact*> cpplanes;
	xmap<int, xParticleMeshObjectContact*> cpmeshes;
	xmap<int, xParticleCylinderContact*> cpcylinders;
	//QMap<QString, xParticleCylinderContact*> cpcylinder;
	xContactPairList* xcpl;
	//QMap<QString, contact_particles_polygonObject*> cppos;
	//xParticlePlanesContact* cpplane;
	
	//xParticleMeshObjectsContact* cpmeshes;
	//xParticleCylindersContact* cpcylinders;
};

#endif