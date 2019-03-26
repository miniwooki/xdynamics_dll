#ifndef XINTEGRATORRK4_H
#define XINTEGRATORRK4_H

#include "xdynamics_decl.h"
#include "xdynamics_simulation/xMultiBodySimulation.h"

class XDYNAMICS_API xIntegratorRK4 : public xMultiBodySimulation
{
public:
	xIntegratorRK4();
	virtual ~xIntegratorRK4();

	virtual int Initialize(xMultiBodyModel* xmbd);
	virtual int OneStepSimulation(double ct, unsigned int cstep);
};
 
#endif