#ifndef XMULTIBODYSIMULATION_H
#define XMULTIBODYSIMULATION_H

#include "xdynamics_decl.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_manager/xMultiBodyModel.h"
#include "xDynamicsError.h"

class xMultiBodyModel;

class XDYNAMICS_API xMultiBodySimulation : public xSimulation
{
public:
	//enum integratorType{};
	xMultiBodySimulation();
//	xMultiBodySimulation(xMultiBodyModel* _xmbd);
	virtual ~xMultiBodySimulation();

	//integratorType IntegrationType();
	//void setIntegrationType(integratorType _type);
	bool Initialized();
	//bool initialize(xMultiBodyModel* xmbd);
	//bool OneStepAnalysis(double ct, unsigned int cstep);
	//void setImplicitTolerance(double _eps);
	void SaveStepResult(unsigned int part, double ct);
	void ExportResults();
	virtual int Initialize(xMultiBodyModel* xmbd);
	virtual int OneStepSimulation(double ct, unsigned int cstep) = 0;

protected:
// 	bool initialize_implicit_hht(xMultiBodyModel* xmbd);
// 	bool initialize_explicit_rk4(xMultiBodyModel* xmbd);
	void ConstructMassMatrix(double mul = 1.0);
	void ConstructContraintJacobian();
	void ConstructForceVector(xVectorD& v);
	void ConstructConstraintEquation(xVectorD& v, int sr = 0.0, double mul = 1.0);
	void ConstructConstraintDerivative(xVectorD& v, int sr = 0.0, double mul = 1.0);
	void ConstructConstraintGamma(xVectorD& v, int sr = 0.0, double mul = 1.0);
	vector4d CalculateInertiaForce(const euler_parameters& ev, const matrix33d& J, const euler_parameters& ep);
	// HHT Integrator member functions
	
	bool isInitilize;		// initialize 

	unsigned int mdim;		// 
	unsigned int tdim;
	unsigned int sdim;
	int dof;

	double *lagMul;

	xVectorD q;				// generalized coordinates
	xVectorD qd;				// first derivative generalized coordinates
	//xVectorD qdd;			// second derivative generalized coordinates
	xVectorD rhs;
	xVectorD q_1;
	xMatrixD lhs;
	xSparseD cjaco;

	xPointMass* xpm;

	xMultiBodyModel* xmbd;
};

#endif