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

private:
	void setDICoordinate();
	bool SolveRK4_EOM();
	void Step1();
	void Step2();
	void Step3();
	void Step4();

	double _dt;
	int *jaco_index;
	int* u;
	int* v;
	int* ccd;
	xVectorD Prevforce;
	xVectorD PrevPos;
	xVectorD PrevVel;-9
	xVectorD k1;
	xVectorD k2;
	xVectorD k3;
	xVectorD k4;
	xMatrixD jaco_u;
	xMatrixD jaco_v;
	xVectorD Constraints;
	xMatrixD Jacobian;
	xVectorD qd_v;
	xVectorD pi_v_vd;
};
 
#endif