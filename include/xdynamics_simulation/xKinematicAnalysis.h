#ifndef XKINEMATICANALYSIS_H
#define XKINEMATICANALYSIS_H

#include "xdynamics_decl.h"
#include "xdynamics_simulation/xMultiBodySimulation.h"

class XDYNAMICS_API xKinematicAnalysis : public xMultiBodySimulation
{
public:
	xKinematicAnalysis();
	virtual ~xKinematicAnalysis();

	virtual int Initialize(xMultiBodyModel* xmbd, bool is_set_result_memory = false);
	virtual int OneStepSimulation(double ct, unsigned int cstep);
};

#endif