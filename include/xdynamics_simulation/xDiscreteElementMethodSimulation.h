#ifndef XDISCRETEELEMENTMETHODSIMULATION_H
#define XDISCRETEELEMENTMETHODSIMULATION_H

#include "xdynamics_decl.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_algebra/xNeiborhoodCell.h"
#include "xdynamics_manager/XDiscreteElementMethodModel.h"
#include <QtCore/QString>

class XDYNAMICS_API xDiscreteElementMethodSimulation : public xSimulation
{
public:
	xDiscreteElementMethodSimulation();
	//~xDiscreteElementMethoSimulation();
	virtual ~xDiscreteElementMethodSimulation();

	virtual int Initialize(xDiscreteElementMethodModel* _xdem, xContactManager* _cm);
	bool Initialized();
	/*bool initialize_f(contactManager* _cm);*/
	virtual int OneStepSimulation(double ct, unsigned int cstep) = 0;
	/*bool oneStepAnalysis_f(double ct, unsigned int cstep);*/
	QString SaveStepResult(unsigned int pt, double ct);
	void ExportResults(std::fstream& of);
	void EnableSaveResultToMemory(bool b);
	void updateObjectFromMBD();
	double* HostPosition();
	double* HostVelocity();

protected:
	void clearMemory();
	void allocationMemory(unsigned int np, unsigned int rnp);
	bool isInitilize;		// initialize 
	bool isSaveMemory;
	unsigned int np;
	unsigned int ns;
	unsigned int nco;
	unsigned int nPolySphere;
	unsigned int nTsdaConnection;
	unsigned int nTsdaConnectionList;
	unsigned int nTsdaConnectionValue;
	unsigned int nTsdaConnectionBody;
	unsigned int nTsdaConnectionBodyData;

	xDiscreteElementMethodModel* xdem;
	xGridCell* dtor;
	xContactManager* xcm;
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
	/*xSpringDamperBodyConnectionInfo *dxsdc_body;
	xSpringDamperBodyConnectionData *dxsdc_body_data;*/
	device_tsda_connection_body_data *dxsd_cbd;

	QList<QString> partList;
};

#endif