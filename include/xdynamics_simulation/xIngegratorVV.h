#ifndef XINTEGRATORVV_H
#define XINTEGRATORVV_H

#include "xdynamics_decl.h"
#include "xdynamics_simulation/xDiscreteElementMethodSimulation.h"

class XDYNAMICS_API xIntegratorVV : public xDiscreteElementMethodSimulation
{
public:
	xIntegratorVV();
	virtual ~xIntegratorVV();

	virtual int Initialize(xDiscreteElementMethodModel* _xdem, xContactManager* _cm);
	virtual int OneStepSimulation(double ct, unsigned int cstep);

private:
	void updatePosition(double* dpos, double* dvel, double* dacc, 
		double* ep, double* ev, double* ea, unsigned int np);
	void updateVelocity(
		double *dvel, double* dacc, double* ep, 
		double *domega, double* dalpha, 
		double *dforce, double* dmoment, 
		double *dmass, double* dinertia, unsigned int np);
};

#endif