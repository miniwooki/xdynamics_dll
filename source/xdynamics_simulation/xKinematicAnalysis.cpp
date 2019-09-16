#include "xdynamics_simulation/xKinematicAnalysis.h"


xKinematicAnalysis::xKinematicAnalysis()
	: xMultiBodySimulation()
{

}

xKinematicAnalysis::~xKinematicAnalysis()
{

}

int xKinematicAnalysis::Initialize(xMultiBodyModel* xmbd)
{
	if (xMultiBodySimulation::Initialize(xmbd))
	{
		return xDynamicsError::xdynamicsErrorMultiBodyModelInitialization;
	}
	OneStepSimulation(0.0, 0);
	lagMul = new double[sdim];
	memset(lagMul, 0, sizeof(double) * sdim);
	return xDynamicsError::xdynamicsSuccess;
}

int xKinematicAnalysis::OneStepSimulation(double ct, unsigned int cstep)
{
	int niter = 0;
	double e_norm = 1;
	xmap<xstring, xPointMass*>::iterator it;
	while (1)
	{
		niter++;
		if (niter > 100)
			return xDynamicsError::xdynamicsErrorMultiBodySimulationHHTIterationOver;

 		lhs.zeros();
		rhs.zeros();

		ConstructContraintJacobian();
		ConstructConstraintEquation(rhs, 0, -1.0);
		for (unsigned int i = 0; i < cjaco.NNZ(); i++)
			lhs(cjaco.ridx[i], cjaco.cidx[i]) = cjaco.value[i];
		int info = LinearSolve(tdim, 1, lhs, tdim, rhs, tdim);
		e_norm = rhs.norm();
		for (unsigned int i = 0; i < mdim; i++)
			q(i + xModel::OneDOF()) += rhs(i);
		it = xmbd->Masses().begin();
		while (it.has_next())
		{
			it.value()->setNewPositionData(q);
		}
		if (e_norm <= 1e-5)	break;
	}
	// velocity analysis
	lhs.zeros();
	rhs.zeros();
	ConstructContraintJacobian();
	ConstructConstraintDerivative(rhs, 0, 1.0);
	
	for (unsigned int i = 0; i < cjaco.NNZ(); i++)
	{
		lhs(cjaco.ridx[i], cjaco.cidx[i]) = cjaco.value[i];
	}
	int info = LinearSolve(tdim, 1, lhs, tdim, rhs, tdim);
	for (unsigned int i = 0; i < mdim; i++)
		qd(i + xModel::OneDOF()) = rhs(i);
	it = xmbd->Masses().begin();
	for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())// (it.has_next())
		it.value()->setNewVelocityData(qd);
	//acceleration analysis
	lhs.zeros();
	rhs.zeros();
	ConstructContraintJacobian();
	ConstructConstraintGamma(rhs, 0, -1.0);
	for (unsigned int i = 0; i < cjaco.NNZ(); i++)
		lhs(cjaco.ridx[i], cjaco.cidx[i]) = cjaco.value[i];
	info = LinearSolve(tdim, 1, lhs, tdim, rhs, tdim);
	return 0;
}

