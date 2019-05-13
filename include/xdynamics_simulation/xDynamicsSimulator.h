#ifndef XDYNAMICSSIMULATOR_H
#define XDYNAMICSSIMULATOR_H

#include "xdynamics_decl.h"
#include "xdynamics_simulation/xMultiBodySimulation.h"
#include "xdynamics_simulation/xDiscreteElementMethodSimulation.h"
#include "xdynamics_simulation/xSmoothedParticleHydrodynamicsSimulation.h"
#include "xdynamics_manager/xDynamicsManager.h"
//#include "xdynamics_simulation/xSimulation.h"

class XDYNAMICS_API xDynamicsSimulator : public xSimulation
{
public:
	xDynamicsSimulator();
	xDynamicsSimulator(xDynamicsManager* _xdm);
	virtual ~xDynamicsSimulator();

	xMultiBodySimulation* setupMBDSimulation(xSimulation::MBDSolverType mst);
	bool xInitialize(bool exefromgui = false, double _dt = 0, unsigned int _st = 0, double _et = 0, xMultiBodySimulation* _xmbd = NULL, xDiscreteElementMethodSimulation* _xdem = NULL, xSmoothedParticleHydrodynamicsSimulation* _xsph = NULL);
	bool xRunSimulation();
	bool xRunSimulationThread(double ct, unsigned int pt);
	bool savePartData(double ct, unsigned int pt);
	void exportPartData();

private:

	xDynamicsManager* xdm;
	xMultiBodySimulation* xmbd;
	xDiscreteElementMethodSimulation* xdem;
	xSmoothedParticleHydrodynamicsSimulation* xsph;
};





#endif