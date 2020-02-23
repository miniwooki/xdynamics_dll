#ifndef XDISCRETEELEMENTMETHODSIMULATION_H
#define XDISCRETEELEMENTMETHODSIMULATION_H

#include "xdynamics_decl.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_manager/xObjectManager.h"
#include "xdynamics_algebra/xNeiborhoodCell.h"
#include "xdynamics_manager/XDiscreteElementMethodModel.h"
#include "xlist.hpp"
//#include <QtCore/QString>

class XDYNAMICS_API xDiscreteElementMethodSimulation : public xSimulation
{
public:
	xDiscreteElementMethodSimulation();
	//~xDiscreteElementMethoSimulation();
	virtual ~xDiscreteElementMethodSimulation();

	virtual int Initialize(xObjectManager* _xobj, xDiscreteElementMethodModel* _xdem, xContactManager* _cm, bool is_set_result_memory = false);
	bool Initialized();
	/*bool initialize_f(contactManager* _cm);*/
	virtual int OneStepSimulation(double ct, unsigned int cstep) = 0;
	/*bool oneStepAnalysis_f(double ct, unsigned int cstep);*/
	bool SaveStepResult(unsigned int pt);
	void ExportResults(std::fstream& of);
	void EnableSaveResultToMemory(bool b);
	double CriticalTimeStep(double min_rad);
	void updateObjectFromMBD();
	double* Position();
	double* Velocity();
	void TranslationSpringDamperForce();
	void RotationSpringDamperForce();
	unsigned int num_particles();
	unsigned int num_clusters();
	unsigned int setupByLastSimulationFile(std::string ldr);
	unsigned int set_dem_data(double* cpos, double* pos, double* vel, double* acc, double* ep, double* ev, double* ea);

protected:
	void clearMemory();
	void allocationMemory(unsigned int np, unsigned int rnp);
	bool isInitilize;		// initialize 
	bool isSaveMemory;
	unsigned int np;
	unsigned int ns;
	unsigned int nco;
	unsigned int nMassParticle;
	unsigned int nPolySphere;
	unsigned int nTsdaConnection;
	unsigned int nTsdaConnectionList;
	unsigned int nTsdaConnectionValue;
	unsigned int nTsdaConnectionBody;
	unsigned int nTsdaConnectionBodyData;

	xDiscreteElementMethodModel* xdem;
	xGridCell* dtor;
	xParticleManager* xpm;
	xContactManager* xcm;
	xObjectManager* xom;
	//unsigned int *cindex;
	double *mass;
	double *inertia;
	double *pos;
	double *cpos;
	double *ep;
	double *vel;
	double *acc;
	double *avel;
	double *aacc;
	double *force;
	double *moment;
	double *rcloc;

	double *dmass;
	double *diner;
	double *dpos;
	double *dcpos;
	double *dep;
	double *dvel;
	double *dacc;
	double *davel;
	double *daacc;
	double *dforce;
	double *dmoment;
	double *drcloc;

	xClusterInformation *xci;
	xClusterInformation *dxci;

	xSpringDamperConnectionInformation *dxsdci;
	xSpringDamperConnectionData *dxsdc_data;
	xSpringDamperCoefficient *dxsdc_kc;
	xSpringDamperBodyConnectionInfo *dxsdc_body;
	xSpringDamperBodyConnectionData *dxsdc_body_data;
	device_tsda_connection_body_data *dxsd_cbd;

	xSpringDamperCoefficient *drsda_coefficient;
	xRSDAConnectionData *drsda_connection_data;
	xRSDAConnectionInformation *drsda_connection_information;
	xSpringDamperBodyConnectionInfo *drsda_attached_body_info;
	xRSDABodyAttachedData *drsda_body_attatched_data;

	xlist<xstring> partList;
};

#endif