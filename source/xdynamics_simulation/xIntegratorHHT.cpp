#include "xdynamics_simulation/xIntegratorHHT.h"
#include "lapacke.h"

xIntegratorHHT::xIntegratorHHT()
	: xMultiBodySimulation()
	, dt2accp(0)
	, dt2accv(0)
	, dt2acc(0)
	, divalpha(0)
	, divbeta(0)
	, alpha(-0.3)
	, beta(0)
	, gamma(0)
	, eps(1e-5)
	, using_parabola_predictor(true)
	, parabola(NULL)
{

}

xIntegratorHHT::~xIntegratorHHT()
{
	if (parabola) delete parabola; parabola = NULL;
}

void xIntegratorHHT::setImplicitTolerance(double _eps)
{
	eps = _eps;
}

void xIntegratorHHT::setParabolaPredictorEnable(bool _b)
{
	using_parabola_predictor = _b;
}

void xIntegratorHHT::setAlphaValue(double _alpha)
{
	alpha = _alpha;
}

double xIntegratorHHT::AlphaValue()
{
	return alpha;
}

double xIntegratorHHT::Tolerance()
{
	return eps;
}

int xIntegratorHHT::Initialize(xMultiBodyModel* _xmbd, bool is_set_result_memory)
{
	if (xMultiBodySimulation::Initialize(_xmbd, is_set_result_memory))
	{
		return xDynamicsError::xdynamicsErrorMultiBodyModelInitialization;
	}

	alpha = -0.3;
	beta = (1 - alpha) * (1 - alpha) / 4;
	gamma = 0.5 - alpha;
	eps = 1E-5;
	pre.alloc(mdim);// = new double[mdim];
	ipp.alloc(mdim);// = new double[mdim];
	ipv.alloc(mdim);// = new double[mdim];
	
	ee.alloc(tdim);// = new double[tdim];

	dt2accp = xSimulation::dt*xSimulation::dt*(1.0 - 2.0 * beta)*0.5;
	dt2accv = xSimulation::dt*(1.0 - gamma);
	dt2acc = xSimulation::dt*xSimulation::dt*beta;
	divalpha = 1.0 / (1.0 + alpha);
	divbeta = -1.0 / (beta*xSimulation::dt*xSimulation::dt);
	ConstructMassMatrix(1.0);
	ConstructContraintJacobian();
	ConstructForceVector(rhs);
	for (unsigned int i(0); i < cjaco.NNZ(); i++){
		lhs(cjaco.ridx[i] + mdim, cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i] + mdim) = cjaco.value[i];
	}

	int ret = LinearSolve(tdim, 1, lhs, tdim, rhs, tdim);
	if (ret)
		return xDynamicsError::xdynamicsErrorLinearEquationCalculation;
	lagMul = rhs.Data() + mdim;// &(rhs[mdim]);
	if (using_parabola_predictor)
	{
		parabola = new xParabolaPredictor;
		parabola->init(rhs.Data(), rhs.Size());
		parabola->getTimeStep() = xSimulation::dt;
	}	
	isInitilize = true;
	return xDynamicsError::xdynamicsSuccess;
}

int xIntegratorHHT::OneStepSimulation(double ct, unsigned int cstep)
{
	PredictionStep(ct, cstep);
	int ret = CorrectionStep(ct, cstep);
	return ret;
}

void xIntegratorHHT::PredictionStep(double ct, unsigned int cstep)
{
	SetZeroBodyForce();
	ConstructForceVector(pre);
	ConstructContraintJacobian();
	for (unsigned int i = 0; i < cjaco.NNZ(); i++)
		pre(cjaco.cidx[i]) -= cjaco.value[i] * lagMul[cjaco.ridx[i]];
	pre *= alpha / (1.0 + alpha);
	unsigned int dof = xModel::OneDOF();
	for (unsigned int i = 0; i < mdim; i++)
	{
		ipp(i) = q(i + dof) + xSimulation::dt * qd(i + dof) + dt2accp * rhs(i);
		ipv(i) = qd(i + dof) + dt2accv * rhs(i);
	}
 	if (parabola)
 		parabola->apply(cstep);
	for (unsigned int i = 0; i < mdim; i++)
	{
		q_1(i + dof) = q(i + dof);
		q(i + dof) = ipp(i) + dt2acc * rhs(i);
		qd(i + dof) = ipv(i) + rhs(i) * xSimulation::dt * gamma;
	}
	for(xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
		it.value()->setNewData(q, qd);
}

int xIntegratorHHT::CorrectionStep(double ct, unsigned int cstep)
{
	int niter = 0;
	double e_norm = 1;
	unsigned int odof = xModel::OneDOF();
	while (1){
		niter++;
		if (niter > 100)
		{
			//errors::setError(errors::MBD_EXCEED_NR_ITERATION);
			//	n_NR_iteration = niter;
			//return -2;
			return xDynamicsError::xdynamicsErrorMultiBodySimulationHHTIterationOver;
		}
		ConstructForceVector(ee);
		ConstructContraintJacobian();
		for (unsigned int i = 0; i < cjaco.NNZ(); i++){
			ee(cjaco.cidx[i]) -= cjaco.value[i] * lagMul[cjaco.ridx[i]];
		}
		ConstructMassMatrix(divalpha);
		for (unsigned int k = 0; k < mdim; k += xModel::OneDOF())
		{
			ee(k + 0) += -lhs(k + 0, k + 0) * rhs(k + 0);
			ee(k + 1) += -lhs(k + 1, k + 1) * rhs(k + 1);
			ee(k + 2) += -lhs(k + 2, k + 2) * rhs(k + 2);
			for (int j = 3; j < xModel::OneDOF(); j++){
				ee(k + j) += -(lhs(k + j, k + 3) * rhs(k + 3) + lhs(k + j, k + 4)*rhs(k + 4) + lhs(k + j, k + 5)*rhs(k + 5) + lhs(k + j, k + 6)*rhs(k + 6));
			}
		}
		MassJacobian(divalpha * beta * dt * dt);
		ForceJacobian(gamma * dt, beta * dt * dt);
		ConstructJacobian(beta * dt * dt);
		for (unsigned int i(0); i < cjaco.NNZ(); i++){
			lhs(cjaco.ridx[i] + mdim, cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i] + mdim) = cjaco.value[i];
		}
		for (unsigned int i(0); i < mdim; i++) ee(i) -= pre(i);
		ConstructConstraintEquation(ee, mdim, divbeta);
		e_norm = ee.norm();
		int info = LinearSolve(tdim, 1, lhs, tdim, ee, tdim);
		if (info)
			return xDynamicsError::xdynamicsErrorLinearEquationCalculation;
		rhs += ee;
		for (unsigned int i = 0; i < mdim; i++)
		{
			q_1(i + odof) = q(i + odof);
			q(i + odof) = ipp(i) + dt2acc * rhs(i);
			qd(i + odof) = ipv(i) + xSimulation::dt * gamma * rhs(i);
		}

		for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
			it.value()->setNewData(q, qd);

		if (e_norm <= eps) break;
	}
	return xDynamicsError::xdynamicsSuccess;
}

void xIntegratorHHT::MassJacobian(double mul)
{
	euler_parameters e;
	euler_parameters ea;
	//xPointMass* xpm = xmbd->BeginPointMass();
	unsigned int id = 0;
	unsigned int src = 0;
	for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
	{
		xPointMass* xpm = it.value();
		id = xpm->xpmIndex() * xModel::OneDOF();
		src = (xpm->xpmIndex() - 1) * xModel::OneDOF();
		e = new_euler_parameters(q(id + 3), q(id + 4), q(id + 5), q(id + 6));
		ea = new_euler_parameters(rhs(src + 3), rhs(src + 4), rhs(src + 5), rhs(src + 6));
		matrix44d data = 4.0 * mul * (-GMatrix(e) * (xpm->Inertia() * GMatrix(ea)) + MMatrix(xpm->Inertia() * (GMatrix(e) * ea)));
		lhs.plus(src + 3, src + 3, data);
		//xpm = xmbd->NextPointMass();
	}
}

void xIntegratorHHT::ForceJacobian(double gt, double btt)
{
	euler_parameters e;
	euler_parameters ev;
	//xPointMass* xpm = xmbd->BeginPointMass();
	unsigned int id = 0, src = 0;
	for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
	{
		xPointMass* xpm = it.value();
		id = xpm->xpmIndex() * xModel::OneDOF();
		src = (xpm->xpmIndex() - 1) * xModel::OneDOF();
		e = new_euler_parameters(q(id + 3), q(id + 4), q(id + 5), q(id + 6));
		ev = new_euler_parameters(qd(id + 3), qd(id + 4), qd(id + 5), qd(id + 6));
		matrix44d GJG = GMatrix(ev) * (xpm->Inertia() * GMatrix(ev));
		matrix44d data = -8.0 * (gt * GJG + btt * (GMatrix(ev) * (xpm->Inertia() * GMatrix(e)) + MMatrix(xpm->Inertia() * (GMatrix(ev) * e))));
		lhs.plus(src + 3, src + 3, data);
	//	xpm = xmbd->NextPointMass();
	}
	for (xmap<xstring, xForce*>::iterator it = xmbd->Forces().begin(); it != xmbd->Forces().end(); it.next())// (xForce* xf, xmbd->Forces())
	{
		it.value()->xDerivate(lhs, q, qd, -btt);
		it.value()->xDerivateVelocity(lhs, q, qd, -gt);
	}
}

void xIntegratorHHT::ConstructJacobian(double btt)
{
	unsigned int sr = 0;
	//xKinematicConstraint* xkc = xmbd->BeginKinematicConstraint();
	for(xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
//	foreach(xKinematicConstraint* xkc, xmbd->Joints())//(xkc != xmbd->EndKinematicConstraint())
	{
		it.value()->DerivateJacobian(lhs, q, qd, lagMul + sr, sr, btt);
		sr += it.value()->NumConst();
		//xkc = xmbd->NextKinematicConstraint();
	}

	//xPointMass* xpm = xmbd->BeginPointMass();
	for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
	{
		//xPointMass* xpm = it.value();
		unsigned int id = (it.value()->xpmIndex() - 1) * xModel::OneDOF();
		double lm = 2.0 * btt * lagMul[sr++];
		lhs(id + 3, id + 3) += lm;
		lhs(id + 4, id + 4) += lm;
		lhs(id + 5, id + 5) += lm;
		lhs(id + 6, id + 6) += lm;
		//xpm = xmbd->NextPointMass();
	}
}
