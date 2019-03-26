#ifndef XINTEGRATOR_HHT_H
#define XINTEGRATOR_HHT_H

#include "xdynamics_decl.h"
#include "xdynamics_simulation/xMultiBodySimulation.h"
#include "xdynamics_algebra/xParaboalPredictor.h"

class XDYNAMICS_API xIntegratorHHT : public xMultiBodySimulation
{
public:
	xIntegratorHHT();
	virtual ~xIntegratorHHT();

	void setImplicitTolerance(double _eps);
	void setParabolaPredictorEnable(bool _b);
	void setAlphaValue(double _alpha);

	double AlphaValue();
	double Tolerance();

	virtual int Initialize(xMultiBodyModel* xmbd);
	virtual int OneStepSimulation(double ct, unsigned int cstep);

private:
	void PredictionStep(double ct, unsigned int cstep);
	int CorrectionStep(double ct, unsigned int cstep);
	void MassJacobian(double mul);
	void ForceJacobian(double gt, double btt);
	void ConstructJacobian(double btt);

	bool using_parabola_predictor;
	double dt2accp;
	double dt2accv;
	double dt2acc;
	double divalpha;
	double divbeta;
	double alpha;
	double beta;
	double gamma;
	double eps;

	xVectorD pre;
	xVectorD ipp;
	xVectorD ipv;
	xVectorD ee;

	xParabolaPredictor* parabola;
};

#endif