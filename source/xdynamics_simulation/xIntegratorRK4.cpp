#include "xdynamics_simulation/xIntegratorRK4.h"
#include<iostream>
using namespace std;
xIntegratorRK4::xIntegratorRK4()
	: xMultiBodySimulation()
{

}

xIntegratorRK4::~xIntegratorRK4()
{
	if (jaco_index) delete[] jaco_index; jaco_index = NULL;
	if (u) delete[] u; u = NULL;
	if (v) delete[] v; v = NULL;
	if (ccd) delete[] ccd; ccd = NULL;
}

int xIntegratorRK4::Initialize(xMultiBodyModel* xmbd)
{
	if (xMultiBodySimulation::Initialize(xmbd))
	{
		return xDynamicsError::xdynamicsErrorMultiBodyModelInitialization;
	}
	jaco_index = new int[(2 * mdim - tdim)];
	k1.alloc(2 * mdim);
	k4.alloc(2 * mdim);
	k2.alloc(2 * mdim);
	k3.alloc(2 * mdim);

	PrevPos.alloc(mdim);
	PrevVel.alloc(mdim);
	jaco_v.alloc(tdim - mdim, tdim - mdim);
	jaco_u.alloc(tdim - mdim, 2 * mdim - tdim);
	Constraints.alloc(tdim - mdim);
	Jacobian.alloc(tdim - mdim, mdim);
	qd_v.alloc(2 * mdim - tdim);
	pi_v_vd.alloc(tdim - mdim);
	u = new int[dof]; memset(u, 0, sizeof(int) * dof);
	v = new int[mdim - dof]; memset(v, 0, sizeof(int)* sdim);
	ccd = new int[mdim]; memset(ccd, 0, sizeof(int)* mdim);
	// 	ccd.alloc(mdim);
	// 	u.alloc(dof);
	// 	v.alloc(mdim - dof);
	qd_v.zeros();
	pi_v_vd.zeros();
	jaco_v.zeros();
	jaco_u.zeros();
	Jacobian.zeros();
	Constraints.zeros();
	ConstructMassMatrix(1.0);
	ConstructContraintJacobian();
	ConstructForceVector(rhs);
	ConstructConstraintGamma(rhs, mdim, -1.0);

	for (unsigned int i(0); i < cjaco.NNZ(); i++)
	{
		lhs(cjaco.ridx[i] + mdim, cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i] + mdim) = cjaco.value[i];
	}

	coordinatePartitioning(cjaco, jaco_index);
	setDICoordinate();
	int ret = LinearSolve(tdim, 1, lhs, tdim, rhs, tdim);
	if (ret)
		return xDynamicsError::xdynamicsErrorLinearEquationCalculation;
	lagMul = rhs.Data() + mdim;

	isInitilize = true;
	return xDynamicsError::xdynamicsSuccess;
	return 0;


}

int xIntegratorRK4::OneStepSimulation(double ct, unsigned int cstep)
{
	int niter = 0;
	double e_norm = 1;
	coordinatePartitioning(cjaco, jaco_index);
	//position analysis
	while (1)
	{
		niter++;
		if (niter > 100)
			return xDynamicsError::xdynamicsErrorMultiBodySimulationHHTIterationOver;
		jaco_v.zeros();
		pi_v_vd.zeros();
		ConstructContraintJacobian();
		ConstructConstraintEquation(Constraints, 0, -1.0);
		for (unsigned int i = 0; i < cjaco.NNZ(); i++)
		{
			unsigned int cid = ccd[cjaco.cidx[i]];
			if (cid == -1) continue;
			jaco_v(cjaco.ridx[i], ccd[cjaco.cidx[i]]) = cjaco.value[i];
		}
		e_norm = Constraints.norm();
		int info = LinearSolve(sdim, 1, jaco_v, sdim, Constraints, sdim);
		for (unsigned int i = 0; i < sdim; i++)
			q(v[i] + xModel::OneDOF()) += Constraints(i);
		//foreach(xPointMass* xpm, xmbd->Masses())
		for(xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
			it.value()->setNewPositionData(q);
		if (e_norm <= 1e-5)	break;
	}

	// velocity analysis
	jaco_v.zeros();
	jaco_u.zeros();
	pi_v_vd.zeros();
	ConstructContraintJacobian();
	for (unsigned int i = 0; i < cjaco.NNZ(); i++)
	{
		unsigned int cid = ccd[cjaco.cidx[i]];
		if (cid != -1) jaco_v(cjaco.ridx[i], cid) = cjaco.value[i];
		else pi_v_vd(cjaco.ridx[i]) -= cjaco.value[i] * qd(cjaco.cidx[i] + xModel::OneDOF());
	}
	for(xmap<xstring, xDrivingConstraint*>::iterator it = xmbd->Drivings().begin(); it != xmbd->Drivings().end(); it.next())
		it.value()->DerivateEquation(pi_v_vd, q, qd, -1, ct, -1.0);
	int	ret = LinearSolve(sdim, 1, jaco_v, sdim, pi_v_vd, sdim);
	for (unsigned int i = 0; i < sdim; i++)
		qd(v[i] + xModel::OneDOF()) = pi_v_vd(i);
	for (unsigned int i(0); i < mdim; i++)
	{
		PrevPos(i) = q(xModel::OneDOF() + i);
		PrevVel(i) = qd(xModel::OneDOF() + i);
	}

	for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
		it.value()->setNewVelocityData(qd);


	// RK4 Process
	Step1();		// Step 1
	Step2();		// Step 2
	Step3();		// Step 3
	Step4();		// Step 4
	return xDynamicsError::xdynamicsSuccess;
}

void xIntegratorRK4::setDICoordinate()
{
	int cv = 0;
	int cu = 0;
	for (int i = 0; i < mdim; i++)
	{
		if (jaco_index[cu] == i)
		{
			ccd[i] = -1;
			u[cu++] = i;
		}
		else
		{
			ccd[i] = cv;
			v[cv++] = i;
		}
	}
}

bool xIntegratorRK4::SolveRK4_EOM()
{
	ConstructMassMatrix(1.0);
	ConstructContraintJacobian();
	ConstructForceVector(rhs);
	ConstructConstraintGamma(rhs, mdim, -1.0);
	for (unsigned int i(0); i < cjaco.NNZ(); i++)
		lhs(cjaco.ridx[i] + mdim, cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i] + mdim) = cjaco.value[i];
	
	return LinearSolve(tdim, 1, lhs, tdim, rhs, tdim);
}

void xIntegratorRK4::Step1()
{
	if (!SolveRK4_EOM())
	{
		for (unsigned int i(0); i < mdim; i++)
		{
			k1(i) = qd(xModel::OneDOF() + i);
			k1(mdim + i) = rhs(i);
			q(xModel::OneDOF() + i) = PrevPos(i) + k1(i)*0.5*dt;
			qd(xModel::OneDOF() + i) = PrevVel(i) + k1(mdim + i)*0.5*dt;
		}
		for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
			it.value()->setNewData(q, qd);
	}
}

void xIntegratorRK4::Step2()
{
	if (!SolveRK4_EOM())
	{
		for (unsigned int i(0); i < mdim; i++)
		{
			k2(i) = qd(xModel::OneDOF() + i);
			k2(mdim + i) = rhs(i);
			q(xModel::OneDOF() + i) = PrevPos(i) + k2(i)*0.5*dt;
			qd(xModel::OneDOF() + i) = PrevVel(i) + k2(mdim + i)*0.5*dt;

		}
		for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
			it.value()->setNewData(q, qd);
	}
}

void xIntegratorRK4::Step3()
{
	if (!SolveRK4_EOM())
	{
		for (unsigned int i(0); i < mdim; i++)
		{
			k3(i) = qd(xModel::OneDOF() + i);
			k3(mdim + i) = rhs(i);
			q(xModel::OneDOF() + i) = PrevPos(i) + k3(i)*dt;
			qd(xModel::OneDOF() + i) = PrevVel(i) + k3(mdim + i)*dt;
		}
		for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
			it.value()->setNewData(q, qd);
	}
}

void xIntegratorRK4::Step4()
{
	if (!SolveRK4_EOM())
	{
		for (unsigned int i(0); i < mdim; i++)
		{
			k4(i) = qd(xModel::OneDOF() + i);
			k4(mdim + i) = rhs(i);
		}
		for (unsigned int i(0); i < mdim; i++)
		{
			q(xModel::OneDOF() + i) = PrevPos(i) + dt*(k1(i) + 2 * k2(i) + 2 * k3(i) + k4(i)) / 6;
			qd(xModel::OneDOF() + i) = PrevVel(i) + dt*(k1(mdim + i) + 2 * k2(mdim + i) + 2 * k3(mdim + i) + k4(mdim + i)) / 6;
		}
		for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
			it.value()->setNewData(q, qd);
	}
}