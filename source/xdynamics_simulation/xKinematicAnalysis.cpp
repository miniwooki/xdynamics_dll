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
	while (1)
	{
		niter++;
		if (niter > 100)
		{
			//errors::setError(errors::MBD_EXCEED_NR_ITERATION);
			//	n_NR_iteration = niter;
			//return -2;
			return xDynamicsError::xdynamicsErrorMultiBodySimulationHHTIterationOver;
		}

 		lhs.zeros();
		rhs.zeros();
// 		spar.zeroCount();
// 		r = 0;
		ConstructContraintJacobian();
		ConstructConstraintEquation(rhs, 0, -1.0);
		for (unsigned int i = 0; i < cjaco.NNZ(); i++)
		{
			lhs(cjaco.ridx[i], cjaco.cidx[i]) = cjaco.value[i];
		}
//		std::ofstream qf;
// 		qf.open("C:/xdynamics/lhs.txt", std::ios::out);
// 		for (unsigned int i = 0; i < tdim; i++)
// 		{
// 			for (unsigned int j = 0; j < tdim; j++)
// 			{
// 				qf << lhs(i, j) << " ";
// 			}
// 			qf << std::endl;
// 		}
// 		qf.close();
		int info = LinearSolve(tdim, 1, lhs, tdim, rhs, tdim);
		//dgesv_(&ptDof, &lapack_one, lhs.getDataPointer(), &ptDof, permutation, rhs.get_ptr(), &ptDof, &lapack_info);
		e_norm = rhs.norm();
		for (unsigned int i = 0; i < mdim; i++)
			q(i + xModel::OneDOF()) += rhs(i);
		foreach(xPointMass* xpm, xmbd->Masses())
		{
			xpm->setNewPositionData(q);
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
	foreach(xPointMass* xpm, xmbd->Masses())
	{
		xpm->setNewVelocityData(qd);
	}
	//acceleration analysis
	lhs.zeros();
	rhs.zeros();
	//	r = 0;
	//	spar.zeroCount();
	ConstructContraintJacobian();
	ConstructConstraintGamma(rhs, 0, -1.0);
	for (unsigned int i = 0; i < cjaco.NNZ(); i++)
	{
		lhs(cjaco.ridx[i], cjaco.cidx[i]) = cjaco.value[i];
	}
	info = LinearSolve(tdim, 1, lhs, tdim, rhs, tdim);
// 	for (unsigned int i = 0; i < mdim; i++)
// 		qd(i) = rhs(i);
	//	foreach(QString str, md->ConstraintList())
	//	{
	//		constraint* cs = md->Constraints()[str];
	//		cs->constraintJacobian(q, qd, spar, r);
	//		cs->derivative(q, qd, rhs, r);
	//		// 		if (cs->IsFixedWhenKinematicAnalysis())
	//		// 			r += 1;
	//		r += cs->NRow();
	//	}
	//	for (unsigned int i = 0; i < spar.nnz(); i++)
	//	{
	//		lhs(spar.ridx[i], spar.cidx[i]) = spar.value[i];
	//	}
	//	info = ls.solve(lhs.getDataPointer(), rhs.get_ptr());
	//	//dgesv_(&ptDof, &lapack_one, lhs.getDataPointer(), &ptDof, permutation, rhs.get_ptr(), &ptDof, &lapack_info);
	//	for (unsigned int i = 0; i < s_k; i++)
	//		qdd(i + 3) = rhs(i);
	//	//delete[] permutation;
	return 0;
}

