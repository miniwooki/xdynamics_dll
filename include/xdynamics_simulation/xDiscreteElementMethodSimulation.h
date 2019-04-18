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
	//QString saveResult_f(float *vp, float* vv, double ct, unsigned int pt);
	//void setIntegratorType(dem_integrator_type itype) { itor_type = itype; }
	//void saveFinalResult(QFile& qf);
	//void saveFinalResult_f(QFile& qf);
	//void setStartingData(startingModel* stm);


protected:
	//void applyMassForce();
// 	vector4d GlobalSphereInertiaForce(const euler_parameters& ev, const double j, const euler_parameters& ep);
// 	vector4d GlobalSphereInertiaForce(const vector4d& ev, const double j, const vector4d& ep);
	void clearMemory();
	void allocationMemory();
	//dem_integrator_type itor_type;
	bool isInitilize;		// initialize 
	bool isSaveMemory;
	unsigned int np;
	unsigned int per_np;
	unsigned int nPolySphere;
	xDiscreteElementMethodModel* xdem;
	xGridCell* dtor;
	//dem_integrator* itor;
	xContactManager* xcm;
	double *mass;
	double *inertia;
	double *pos;
	double *vel;
	double *acc;
	double *avel;
	double *aacc;
	double *force;
	double *moment;

	double *dmass;
	double *diner;
	double *dpos;
	double *dvel;
	double *dacc;
	double *davel;
	double *daacc;
	double *dforce;
	double *dmoment;

	QList<QString> partList;
};

#endif