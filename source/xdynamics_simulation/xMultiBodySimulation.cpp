#include "xdynamics_simulation/xMultiBodySimulation.h"
#include "xdynamics_manager/xDynamicsManager.h"
#include <sstream>

std::fstream *ex_of = NULL;

xMultiBodySimulation::xMultiBodySimulation()
	: xSimulation()
	, mdim(0)
	, tdim(0)
	, sdim(0)
	, dof(0)
	, isInitilize(false)
	, xpm(NULL)
	, lagMul(NULL)
	, dem_tsda(NULL)
{

}

xMultiBodySimulation::~xMultiBodySimulation()
{
	if (xpm) delete[] xpm; xpm = NULL;
}

xMultiBodyModel * xMultiBodySimulation::Model()
{
	return xmbd;
}

bool xMultiBodySimulation::Initialized()
{
	return isInitilize;
}

void xMultiBodySimulation::SetDEMSpringDamper(xSpringDamperForce* dem_t)
{
	dem_tsda = dem_t;
	for (unsigned int i = 0; i < dem_tsda->NumSpringDamperBodyConnection(); i++)
	{
		xSpringDamperBodyConnectionInfo info = dem_tsda->xSpringDamperBodyConnectionInformation()[i];
		xPointMass* pm = xmbd->XMass(std::string(info.cbody));
		dem_tsda->ConvertGlobalToLocalOfBodyConnectionPosition(i, pm);
	}
}

void xMultiBodySimulation::SetZeroAllBodyForce()
{
	xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
	while (it.has_next())
	{
		it.value()->setZeroAllForce();
		it.next();
	}
}

void xMultiBodySimulation::SetZeroAxialBodyForce()
{
	xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
	while (it.has_next())
	{
		it.value()->setAxialForce(0, 0, 0);
		it.value()->setAxialMoment(0, 0, 0);
		it.value()->setEulerParameterMoment(0, 0, 0,0);
		it.next();
	}
}

bool xMultiBodySimulation::SaveStepResult(unsigned int part)
{
	xResultManager* xrm = xDynamicsManager::This()->XResult();
	xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
	while (it.has_next())
	{
		it.value()->setNewAccelerationData(rhs);
		xrm->save_mass_result(part, it.value());
		it.next();
	}
		
	unsigned int sr = 0;
	cjaco.zeros();
	for(xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
	{
		xKinematicConstraint::kinematicConstraint_result kcr = it.value()->GetStepResult(part, q, qd, lagMul, sr);
		xrm->save_joint_result(part, it.value()->Name(), kcr);
		sr += it.value()->NumConst();
	}
	for (xmap<xstring, xDrivingConstraint*>::iterator it = xmbd->Drivings().begin(); it != xmbd->Drivings().end(); it.next())
	{
		xKinematicConstraint::kinematicConstraint_result kcr = it.value()->GetStepResult(part, xSimulation::ctime, q, qd, lagMul, sr);
		xrm->save_joint_result(part, it.value()->Name(), kcr);
		if(it.value()->get_driving_type() == xDrivingConstraint::ROTATION_DRIVING)
			xrm->save_driving_rotation_result(part, it.value()->Name(), it.value()->RevolutionCount(), it.value()->DerivativeRevolutionCount(), it.value()->RotationAngle());
		sr++;
	}
	return xrm->save_generalized_coordinate_result(q.Data(), qd.Data(), q_1.Data(), rhs.Data());
}

void xMultiBodySimulation::ExportResults(std::fstream& of)
{
	xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
	while (it.has_next())
	{
		xPointMass* xpm = it.value();
		xpm->ExportResults(of);
		xpm->ExportInitialData();
		it.next();
	}
	unsigned int sr = 0;
	for (xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
	{
		it.value()->ExportResults(of);
		sr += it.value()->NumConst();
	}
	for (xmap<xstring, xDrivingConstraint*>::iterator it = xmbd->Drivings().begin(); it != xmbd->Drivings().end(); it.next())
	{
		it.value()->ExportResults(of);
		sr++;
	}
}


unsigned int xMultiBodySimulation::setupByLastSimulationFile(std::string lmr)
{
	unsigned int pt = 0;
	double ct = 0.0;
	std::fstream of;
	of.open(lmr, ios_base::in | ios_base::binary);
	
	
	/*while (!of.eof())
	{
		of.read((char*)&pt, sizeof(unsigned int));
		of.read((char*)&ct, sizeof(double));
		of.read((char*)q.Data(), sizeof(double) * mdim + xModel::OneDOF());
		of.read((char*)qd.Data(), sizeof(double) * mdim + xModel::OneDOF());
		of.read((char*)q_1.Data(), sizeof(double) * mdim + xModel::OneDOF());
		of.read((char*)rhs.Data(), sizeof(double) * tdim);
		lagMul = rhs.Data() + mdim;
		xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
		while (it.has_next())
		{
			xPointMass* xpm = it.value();
			of.read((char*)xpm->getForcePointer(), sizeof(double) * 18);
			xpm->SaveStepResult(pt, ct, q, qd, rhs);
			it.next();
		}
		unsigned int sr = 0;
		cjaco.zeros();
		for (xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
		{
			it.value()->SaveStepResult(pt, ct, q, qd, lagMul, sr);
			sr += it.value()->NumConst();
		}
		for (xmap<xstring, xDrivingConstraint*>::iterator it = xmbd->Drivings().begin(); it != xmbd->Drivings().end(); it.next())
		{
			it.value()->SaveStepResult(pt, ct, q, qd, lagMul, sr);
			sr++;
		}
	}
	
	for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
	{
		it.value()->setNewPositionData(q);
		it.value()->setNewVelocityData(qd);
	}*/		
	return pt;
}

unsigned int xMultiBodySimulation::num_generalized_coordinate()
{
	return mdim;
}

unsigned int xMultiBodySimulation::num_constraint_equations()
{
	return sdim;
}

unsigned int xMultiBodySimulation::set_mbd_data(double * _q, double * _dq, double * _q_1, double * _rhs)
{
	q.set(_q);
	qd.set(_dq);
	q_1.set(_q_1);
	rhs.set(_rhs);
	return 0;
}

int xMultiBodySimulation::Initialize(xMultiBodyModel* _xmbd, bool is_set_result_memory)
{
	//of = new std::fstream;
	xmbd = _xmbd;
	int nm = xmbd->NumMass();
	int nr = 0;
	mdim = nm * xModel::OneDOF();
	sdim = nm;
	if (is_set_result_memory)
		xDynamicsManager::This()->XResult()->set_num_generailzed_coordinates(mdim);
	q.alloc(mdim + xModel::OneDOF());// = new double[mdim];
	q_1.alloc(mdim + xModel::OneDOF());
	qd.alloc(mdim + xModel::OneDOF());// = new double[mdim];
	unsigned int idx = 0;
	xPointMass* ground = xModel::Ground();
	q(idx + 0) = ground->Position().x;
	q(idx + 1) = ground->Position().y;
	q(idx + 2) = ground->Position().z;
	q(idx + 3) = ground->EulerParameters().e0;
	q(idx + 4) = ground->EulerParameters().e1;
	q(idx + 5) = ground->EulerParameters().e2;
	q(idx + 6) = ground->EulerParameters().e3;
	unsigned int sid = 0;
	xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
	while (it.has_next())
	{
		xPointMass* xpm = it.value();
		xpm->setXpmIndex(++sid);
		xpm->ImportInitialData();
		idx = xpm->xpmIndex() * xModel::OneDOF();
		q(idx + 0) = xpm->Position().x;			qd(idx + 0) = xpm->Velocity().x;
		q(idx + 1) = xpm->Position().y;			qd(idx + 1) = xpm->Velocity().y;
		q(idx + 2) = xpm->Position().z;			qd(idx + 2) = xpm->Velocity().z;
		q(idx + 3) = xpm->EulerParameters().e0;	qd(idx + 3) = xpm->DEulerParameters().e0;
		q(idx + 4) = xpm->EulerParameters().e1;	qd(idx + 4) = xpm->DEulerParameters().e1;
		q(idx + 5) = xpm->EulerParameters().e2;	qd(idx + 5) = xpm->DEulerParameters().e2;
		q(idx + 6) = xpm->EulerParameters().e3;	qd(idx + 6) = xpm->DEulerParameters().e3;
		xpm->setupInertiaMatrix();
		xpm->setupTransformationMatrix();
	//	xpm->AllocResultMomory(xSimulation::npart);
		if(is_set_result_memory)
			xDynamicsManager::This()->XResult()->alloc_mass_result_memory(xpm->Name());
		it.next();
	}
	for (xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
	{
		xKinematicConstraint* xkc = it.value();
		sdim += xkc->NumConst();
		std::string bn = xkc->BaseBodyName();
		std::string an = xkc->ActionBodyName();
		int base_idx = bn == "ground" ? 0 : xmbd->XMass(bn)->xpmIndex();
		int action_idx = an == "ground" ? 0 : xmbd->XMass(an)->xpmIndex();
		xkc->setBaseBodyIndex(base_idx);
		xkc->setActionBodyIndex(action_idx);
		xkc->AllocResultMemory(xSimulation::npart);
		if (is_set_result_memory)
			xDynamicsManager::This()->XResult()->alloc_joint_result_memory(xkc->Name());
		//xkc->set
	}
	for (xmap<xstring, xDrivingConstraint*>::iterator it = xmbd->Drivings().begin(); it != xmbd->Drivings().end(); it.next())
	{
		it.value()->define(q);
		if (is_set_result_memory)
		{
			xDynamicsManager::This()->XResult()->alloc_joint_result_memory(it.value()->Name());
			if(it.value()->get_driving_type() == xDrivingConstraint::ROTATION_DRIVING)
				xDynamicsManager::This()->XResult()->alloc_driving_rotation_result_memory(it.value()->Name());
		}
			
		sdim++;
	}
	for(xmap<xstring, xForce*>::iterator it = xmbd->Forces().begin(); it != xmbd->Forces().end(); it.next())
	{
		xForce* xf = it.value();
		std::string bn = xf->BaseBodyName();
		std::string an = xf->ActionBodyName();
		int base_idx = bn == "ground" ? 0 : xmbd->XMass(bn)->xpmIndex();
		int action_idx = an == "ground" ? 0 : xmbd->XMass(an)->xpmIndex();
		xf->setBaseBodyIndex(base_idx);
		xf->setActionBodyIndex(action_idx);
		//xf->setBaseLocalCoordinate();
	}
	if (is_set_result_memory)
		xDynamicsManager::This()->XResult()->set_num_constraints_equations(sdim);
	dof = mdim - sdim;
	std::stringstream wss;
	wss << dof;
	if (dof < 0)
	{
		xLog::log(std::string("There are ") + wss.str() + std::string("redundant constraint equations"));
		return xDynamicsError::xdynamicsErrorMultiBodyModelRedundantCondition;
	}
	else
	{
		tdim = mdim + sdim;
		lhs.alloc(tdim, tdim);// new_matrix(tdim, tdim);
		cjaco.alloc(sdim, mdim);
		rhs.alloc(tdim);// = new double[tdim];
	}
	return xDynamicsError::xdynamicsSuccess;
}

void xMultiBodySimulation::ConstructMassMatrix(double mul)
{
	lhs.zeros();
	unsigned int src = 0;
	xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
	while (it.has_next())
	{
		xPointMass* xpm = it.value();
		src = (xpm->xpmIndex() - 1) * xModel::OneDOF();
		int idx = xpm->xpmIndex() * xModel::OneDOF();
		euler_parameters e = xpm->EulerParameters();// new_euler_parameters(q(idx + 3), q(idx + 4), q(idx + 5), q(idx + 6));
		lhs(src, src) = lhs(src + 1, src + 1) = lhs(src + 2, src + 2) = mul * xpm->Mass();
		matrix34d G = GMatrix(e);
		matrix44d LTJL = 4.0 * mul * (G * (xpm->Inertia() * G));
		lhs(src + 3, src + 3) = LTJL.a00; lhs(src + 3, src + 4) = LTJL.a01; lhs(src + 3, src + 5) = LTJL.a02; lhs(src + 3, src + 6) = LTJL.a03;
		lhs(src + 4, src + 3) = LTJL.a10; lhs(src + 4, src + 4) = LTJL.a11; lhs(src + 4, src + 5) = LTJL.a12; lhs(src + 4, src + 6) = LTJL.a13;
		lhs(src + 5, src + 3) = LTJL.a20; lhs(src + 5, src + 4) = LTJL.a21; lhs(src + 5, src + 5) = LTJL.a22; lhs(src + 5, src + 6) = LTJL.a23;
		lhs(src + 6, src + 3) = LTJL.a30; lhs(src + 6, src + 4) = LTJL.a31; lhs(src + 6, src + 5) = LTJL.a32; lhs(src + 6, src + 6) = LTJL.a33;
		it.next();
	}
}

void xMultiBodySimulation::ConstructContraintJacobian()
{
	unsigned int sr = 0;
	unsigned int sc = 0;
	cjaco.zeros();
	for (xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
	{
		it.value()->ConstraintJacobian(cjaco, q, qd, sr);
		sr += it.value()->NumConst();
	}
	for (xmap<xstring, xDrivingConstraint*>::iterator it = xmbd->Drivings().begin(); it != xmbd->Drivings().end(); it.next())
	{
		it.value()->ConstraintJacobian(cjaco, q, qd, sr, xSimulation::ctime);
		sr++;
	}
	unsigned int id = 0;
	xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
	while (it.has_next())
	{
		xPointMass* xpm = it.value();
		id = xpm->xpmIndex() * xModel::OneDOF();
		sc = (xpm->xpmIndex() - 1) * xModel::OneDOF();
		euler_parameters e = 2.0 * xpm->EulerParameters();// new_vector4d(q(id + 3), q(id + 4), q(id + 5), q(id + 6));
		cjaco.insert(sr++, sc + 3, new_vector4d(e.e0, e.e1, e.e2, e.e3));
		it.next();
	}
}

void xMultiBodySimulation::ConstructForceVector(xVectorD& v)
{
	euler_parameters e, ev;
	vector3d f;
	vector4d m;
	unsigned int j = 0;
	for (xmap<xstring, xForce*>::iterator it = xmbd->Forces().begin(); it != xmbd->Forces().end(); it.next())
		it.value()->xCalculateForce(q, qd);
	xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
	while (it.has_next())
	{
		xPointMass* xpm = it.value();
		e = xpm->EulerParameters();// new_euler_parameters(q(i + 3), q(i + 4), q(i + 5), q(i + 6));
		ev = xpm->DEulerParameters();// new_euler_parameters(qd(i + 3), qd(i + 4), qd(i + 5), qd(i + 6));
		f = xpm->Mass() * xModel::gravity + xpm->ContactForce() + xpm->AxialForce() + xpm->HydroForce();
	//	std::cout << "rk4" << " - " <<  xpm->ContactForce().x << " " << xpm->ContactForce().y << " " << xpm->ContactForce().z << std::endl;
		m = 2.0 * GMatrix(e) * (xpm->ContactMoment() + xpm->AxialMoment() + xpm->HydroMoment());
		m += CalculateInertiaForce(ev, xpm->Inertia(), e) + xpm->EulerParameterMoment();
		v(j + 0) = f.x; v(j + 1) = f.y; v(j + 2) = f.z;
		v(j + 3) = m.x; v(j + 4) = m.y; v(j + 5) = m.z; v(j + 6) = m.w;
		//i += xModel::OneDOF();
		j += xModel::OneDOF();
		it.next();
	}
}

void xMultiBodySimulation::ConstructConstraintEquation(xVectorD& v, int sr, double mul /*= 1.0*/)
{
	//unsigned int sr = mdim;
	for (xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
	{
		it.value()->ConstraintEquation(v, q, qd, sr, mul);
		sr += it.value()->NumConst();
	}
	for (xmap<xstring, xDrivingConstraint*>::iterator it = xmbd->Drivings().begin(); it != xmbd->Drivings().end(); it.next())
	{
		it.value()->ConstraintEquation(v, q, qd, sr, xSimulation::ctime, mul);
		sr++;
	}
	xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
	while (it.has_next())
	{
		xPointMass* xpm = it.value();
		unsigned int id = xpm->xpmIndex() * xModel::OneDOF();
		euler_parameters e = xpm->EulerParameters();// new_euler_parameters(q(id + 3), q(id + 4), q(id + 5), q(id + 6));
		v(sr++) = mul * (dot(e, e) - 1.0);
		it.next();
	}
}

void xMultiBodySimulation::ConstructConstraintDerivative(xVectorD& v, int sr, double mul /*= 1.0*/)
{
	for (xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
		sr += it.value()->NumConst();
	for (xmap<xstring, xDrivingConstraint*>::iterator it = xmbd->Drivings().begin(); it != xmbd->Drivings().end(); it.next())
	{
		it.value()->ConstraintDerivative(v, q, qd, sr, xSimulation::ctime, mul);
		sr++;
	}
	xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
	while (it.has_next())
	{
		xPointMass* xpm = it.value();
		unsigned int id = xpm->xpmIndex() * xModel::OneDOF();
		euler_parameters e = xpm->EulerParameters();// new_euler_parameters(q(id + 3), q(id + 4), q(id + 5), q(id + 6));
		v(sr++) = mul * (dot(e, e) - 1.0);
		it.next();
	}
}

void xMultiBodySimulation::ConstructConstraintGamma(xVectorD& v, int sr /*= 0.0*/, double mul /*= 1.0*/)
{
	//v.zeros();
	for (xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
	{
		it.value()->GammaFunction(v, q, qd, sr, mul);
		sr += it.value()->NumConst();
	}
	for(xmap<xstring, xDrivingConstraint*>::iterator it = xmbd->Drivings().begin(); it != xmbd->Drivings().end(); it.next())
	{
		it.value()->ConstraintGamma(v, q, qd, sr, xSimulation::ctime, mul);
		sr++;
	}
	xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin();
	while (it.has_next())
	{
		xPointMass* xpm = it.value();
		unsigned int id = xpm->xpmIndex() * xModel::OneDOF();
		euler_parameters e = xpm->DEulerParameters();// new_euler_parameters(qd(id + 3), qd(id + 4), qd(id + 5), qd(id + 6));
		v(sr++) = mul * 2.0 * dot(e, e);
		it.next();
	}
}

vector4d xMultiBodySimulation::CalculateInertiaForce(const euler_parameters& ev, const matrix33d& J, const euler_parameters& ep)
{
	matrix34d Gd = GMatrix(ev);
	return 8.0 * Gd * (J * (Gd * ep));
}
