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

int xIntegratorHHT::Initialize(xMultiBodyModel* _xmbd)
{
	if (xMultiBodySimulation::Initialize(_xmbd))
	{
		return xDynamicsError::xdynamicsErrorMultiBodyModelInitialization;
	}
	//multibodyDynamics::nnz = 0;
	alpha = -0.3;
	beta = (1 - alpha) * (1 - alpha) / 4;
	gamma = 0.5 - alpha;
	eps = 1E-5;
// 	int nm = xmbd->NumMass();
// 	int nr = 0;
// 	mdim = nm * xModel::OneDOF();
// 	sdim = nm;
// 	q.alloc(mdim + xModel::OneDOF());// = new double[mdim];
// 	q_1.alloc(mdim + xModel::OneDOF());
// 	qd.alloc(mdim + xModel::OneDOF());// = new double[mdim];
	pre.alloc(mdim);// = new double[mdim];
	ipp.alloc(mdim);// = new double[mdim];
	ipv.alloc(mdim);// = new double[mdim];
	
// 	unsigned int idx = 0;
// 	xPointMass* ground = xModel::Ground();
// 	q(idx + 0) = ground->Position().x;
// 	q(idx + 1) = ground->Position().y;
// 	q(idx + 2) = ground->Position().z;
// 	q(idx + 3) = ground->EulerParameters().e0;
// 	q(idx + 4) = ground->EulerParameters().e1;
// 	q(idx + 5) = ground->EulerParameters().e2;
// 	q(idx + 6) = ground->EulerParameters().e3;
// 	foreach(xPointMass* xpm, xmbd->Masses())
// 	{
// 		xpm->setXpmIndex(++idx);
// 		idx = xpm->xpmIndex() * xModel::OneDOF();
// 		q(idx + 0) = xpm->Position().x;			qd(idx + 0) = xpm->Velocity().x;
// 		q(idx + 1) = xpm->Position().y;			qd(idx + 1) = xpm->Velocity().y;
// 		q(idx + 2) = xpm->Position().z;			qd(idx + 2) = xpm->Velocity().z;
// 		q(idx + 3) = xpm->EulerParameters().e0;	qd(idx + 3) = xpm->DEulerParameters().e0;
// 		q(idx + 4) = xpm->EulerParameters().e1;	qd(idx + 4) = xpm->DEulerParameters().e1;
// 		q(idx + 5) = xpm->EulerParameters().e2;	qd(idx + 5) = xpm->DEulerParameters().e2;
// 		q(idx + 6) = xpm->EulerParameters().e3;	qd(idx + 6) = xpm->DEulerParameters().e3;
// 		xpm->setupInertiaMatrix();
// 		xpm->setupTransformationMatrix();
// 		xpm->AllocResultMomory(xSimulation::npart);
// 	}
// 	foreach(xKinematicConstraint* xkc, xmbd->Joints())
// 	{
// 		sdim += xkc->NumConst();
// 		std::string bn = xkc->BaseBodyName();
// 		std::string an = xkc->ActionBodyName();
// 		int base_idx = bn == "ground" ? 0 : xmbd->XMass(bn)->xpmIndex();
// 		int action_idx = an == "ground" ? 0 : xmbd->XMass(an)->xpmIndex();
// 		xkc->setBaseBodyIndex(base_idx);
// 		xkc->setActionBodyIndex(action_idx);
// 		xkc->AllocResultMemory(xSimulation::npart);
// 	}
// 	foreach(xForce* xf, xmbd->Forces())
// 	{
// 		std::string bn = xf->BaseBodyName();
// 		std::string an = xf->ActionBodyName();
// 		int base_idx = bn == "ground" ? 0 : xmbd->XMass(bn)->xpmIndex();
// 		int action_idx = an == "ground" ? 0 : xmbd->XMass(an)->xpmIndex();
// 		xf->setBaseBodyIndex(base_idx);
// 		xf->setActionBodyIndex(action_idx);
// 	}
// 	dof = mdim - sdim;
// 	if (dof < 0)
// 	{
// 		xLog::log("There are " + xstring(-dof) + "redundant constraint equations");
// 		return xDynamicsError::xdynamicsErrorMultiBodyModelRedundantCondition;
// 	}
// 	tdim = mdim + sdim;
// 	lhs.alloc(tdim, tdim);// new_matrix(tdim, tdim);
// 	cjaco.alloc(sdim * mdim);
	ee.alloc(tdim);// = new double[tdim];
//	rhs.alloc(tdim);// = new double[tdim];
// 	xPointMass* xpm = xmbd->BeginPointMass();
// 	while (xpm != xmbd->EndPointMass())
// 	{
// 		
// 		xpm = xmbd->NextPointMass();
// 	}
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
// 	std::ofstream qf;
// 	qf.open("C:/xdynamics/lhs.txt", std::ios::out);
// 	for (unsigned int i = 0; i < tdim; i++)
// 	{
// 		for (unsigned int j = 0; j < tdim; j++)
// 		{
// 			qf << lhs(i, j) << " ";
// 		}
// 		qf << std::endl;
// 	}
// 	qf.close();
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
	//return xDynamicsError::xdynamicsErrorLinearEquationCalculation;
}

void xIntegratorHHT::PredictionStep(double ct, unsigned int cstep)
{
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
	foreach(xPointMass* xpm, xmbd->Masses())
	{
		xpm->setNewData(q, qd);
	}
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
		foreach(xPointMass* xpm, xmbd->Masses())
		{
			xpm->setNewData(q, qd);
		}
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
	foreach(xPointMass* xpm, xmbd->Masses())// (xpm != xmbd->EndPointMass())
	{
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
	foreach(xPointMass* xpm, xmbd->Masses())//while (xpm != xmbd->EndPointMass())
	{
		id = xpm->xpmIndex() * xModel::OneDOF();
		src = (xpm->xpmIndex() - 1) * xModel::OneDOF();
		e = new_euler_parameters(q(id + 3), q(id + 4), q(id + 5), q(id + 6));
		ev = new_euler_parameters(qd(id + 3), qd(id + 4), qd(id + 5), qd(id + 6));
		matrix44d GJG = GMatrix(ev) * (xpm->Inertia() * GMatrix(ev));
		matrix44d data = -8.0 * (gt * GJG + btt * (GMatrix(ev) * (xpm->Inertia() * GMatrix(e)) + MMatrix(xpm->Inertia() * (GMatrix(ev) * e))));
		lhs.plus(src + 3, src + 3, data);
	//	xpm = xmbd->NextPointMass();
	}
	foreach(xForce* xf, xmbd->Forces())
	{
		xf->xDerivate(lhs, q, qd, -btt);
		xf->xDerivateVelocity(lhs, q, qd, -gt);
	}
}

void xIntegratorHHT::ConstructJacobian(double btt)
{
	unsigned int sr = 0;
	//xKinematicConstraint* xkc = xmbd->BeginKinematicConstraint();
	foreach(xKinematicConstraint* xkc, xmbd->Joints())//(xkc != xmbd->EndKinematicConstraint())
	{
		xkc->DerivateJacobian(lhs, q, qd, lagMul + sr, sr, btt);
		sr += xkc->NumConst();
		//xkc = xmbd->NextKinematicConstraint();
	}

	//xPointMass* xpm = xmbd->BeginPointMass();
	foreach(xPointMass* xpm, xmbd->Masses())// (xpm != xmbd->EndPointMass())
	{
		unsigned int id = (xpm->xpmIndex() - 1) * xModel::OneDOF();
		double lm = 2.0 * btt * lagMul[sr++];
		lhs(id + 3, id + 3) += lm;
		lhs(id + 4, id + 4) += lm;
		lhs(id + 5, id + 5) += lm;
		lhs(id + 6, id + 6) += lm;
		//xpm = xmbd->NextPointMass();
	}
}
