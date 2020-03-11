#ifndef XCONTACTMANAGER_H
#define XCONTACTMANAGER_H

#include "xdynamics_decl.h"
#include "xModel.h"
#include "xdynamics_object/xParticleParticleContact.h"
#include "xdynamics_object/xParticlePlaneContact.h"
#include "xdynamics_object/xParticleCubeContact.h"
#include "xdynamics_object/xParticleMeshObjectContact.h"
#include "xdynamics_object/xParticleCylinderContact.h"
#include <map>
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

	//unsigned int setupParticlesMeshObjectsContact();
	void setupParticlesPlanesContact();
	//void setupParticlesCylindersContact();
	void setNumClusterObject(unsigned int nc);
	void defineContacts(unsigned int np);
	//double* SphereData();
	//double* HostSphereData();
	//float* SphereData_f();
	xmap<xstring, xParticleMeshObjectContact*>& PMContacts();
	xmap<xstring, xParticlePlaneContact*>& PPLContacts();
	xmap<xstring, xParticleCylinderContact*>& PCYLContacts();
	void insertContact(xContact* c);
	xContact* Contact(std::string n);// { return cots[n]; }
	//QMap<QString, QString>& Logs() { return logs; }
	xmap<xstring, xContact*>& Contacts();// { return cots; }
	xParticleParticleContact* ContactParticles();// { return cpp; }

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
	std::map<pair<unsigned int, unsigned int>, xPairData> CalculateCollisionPair(
		vector4d* pos,
		unsigned int* sorted_id,
		unsigned int* cell_start,
		unsigned int* cell_end,
		xClusterInformation* xci,
		unsigned int ncobject,
		unsigned int np);

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

	unsigned int ncontact;
	unsigned int ncobject;

	double max_sphere_radius;
	double* d_Tmax;
	double* d_RRes;
	vector3d* Tmax;
	double* RRes;
	unsigned int n_total_mesh_sphere;
	xmap<xstring, xContact*> cots;
	xParticleParticleContact* cpp;
	xmap<xstring, xParticlePlaneContact*> cpplanes;
	xmap<xstring, xParticleMeshObjectContact*> cpmeshes;
	xmap<xstring, xParticleCylinderContact*> cpcylinders;
	xmap<xstring, xParticleCubeContact*> cpcubes;
	xContactPairList* xcpl;
};

#endif