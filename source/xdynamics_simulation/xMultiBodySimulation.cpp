#include "xdynamics_simulation/xMultiBodySimulation.h"
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
//	, of(NULL)
{

}

xMultiBodySimulation::~xMultiBodySimulation()
{
	if (xpm) delete[] xpm; xpm = NULL;
//	if (of) { of->close(); delete of; }
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
		xPointMass* pm = xmbd->XMass(info.cbody.toStdString());
		dem_tsda->ConvertGlobalToLocalOfBodyConnectionPosition(i, pm);
	}
}

void xMultiBodySimulation::SetZeroBodyForce()
{
	foreach(xPointMass* xpm, xmbd->Masses())
	{
		xpm->setZeroAllForce();
	}
}

void xMultiBodySimulation::SaveStepResult(unsigned int part, double ct)
{
	ex_of = new std::fstream;
	std::string fname = xModel::makeFilePath(xModel::getModelName() + ".lmr");
	//std::ifstream src(fname, ios_base::in | ios_base::binary);
	//if (!src.is_open())
	//{
	//	ex_of->close();
	//	ex_of->open(fname, ios_base::out | ios_base::binary);
	//}
	ex_of->open(fname, ios_base::out | ios_base::app | ios_base::binary);
	ex_of->write((char*)&part, sizeof(unsigned int));
	ex_of->write((char*)&ct, sizeof(double));
	ex_of->write((char*)q.Data(), sizeof(double) * mdim + xModel::OneDOF());
	ex_of->write((char*)qd.Data(), sizeof(double) * mdim + xModel::OneDOF());
	ex_of->write((char*)q_1.Data(), sizeof(double) * mdim + xModel::OneDOF());
	ex_of->write((char*)rhs.Data(), sizeof(double) * tdim);
	foreach(xPointMass* xpm, xmbd->Masses())
		ex_of->write((char*)xpm->getForcePointer(), sizeof(double) * 18);
	ex_of->close();
	delete ex_of;
	ex_of = NULL;
	foreach(xPointMass* xpm, xmbd->Masses())
	{
		xpm->SaveStepResult(part, ct, q, qd, rhs);
	}
	unsigned int sr = 0;
	cjaco.zeros();
	foreach(xKinematicConstraint* xkc, xmbd->Joints())
	{
		xkc->SaveStepResult(part, ct, q, qd, lagMul, sr);
		sr += xkc->NumConst();
	}
	foreach(xDrivingConstraint* xdc, xmbd->Drivings())
	{
		xdc->SaveStepResult(part, ct, q, qd, lagMul, sr);
		sr++;
	}
}

void xMultiBodySimulation::ExportResults(std::fstream& of)
{
	foreach(xPointMass* xpm, xmbd->Masses())
	{
		xpm->ExportResults(of);
		xpm->ExportInitialData();
	}
	unsigned int sr = 0;
	foreach(xKinematicConstraint* xkc, xmbd->Joints())
	{
		xkc->ExportResults(of);
		sr += xkc->NumConst();
	}
	foreach(xDrivingConstraint* xdc, xmbd->Drivings())
	{
		xdc->ExportResults(of);
		sr++;
	}
}


unsigned int xMultiBodySimulation::setupByLastSimulationFile(std::string lmr)
{
	unsigned int pt = 0;
	double ct = 0.0;
	std::fstream of;
	of.open(lmr, ios_base::in | ios_base::binary);
	while (!of.eof())
	{
		of.read((char*)&pt, sizeof(unsigned int));
		of.read((char*)&ct, sizeof(double));
		of.read((char*)q.Data(), sizeof(double) * mdim + xModel::OneDOF());
		of.read((char*)qd.Data(), sizeof(double) * mdim + xModel::OneDOF());
		of.read((char*)q_1.Data(), sizeof(double) * mdim + xModel::OneDOF());
		of.read((char*)rhs.Data(), sizeof(double) * tdim);
		lagMul = rhs.Data() + mdim;
		foreach(xPointMass* xpm, xmbd->Masses())
		{
			of.read((char*)xpm->getForcePointer(), sizeof(double) * 18);
			xpm->SaveStepResult(pt, ct, q, qd, rhs);
		}
		unsigned int sr = 0;
		cjaco.zeros();
		foreach(xKinematicConstraint* xkc, xmbd->Joints())
		{
			xkc->SaveStepResult(pt, ct, q, qd, lagMul, sr);
			sr += xkc->NumConst();
		}
		foreach(xDrivingConstraint* xdc, xmbd->Drivings())
		{
			xdc->SaveStepResult(pt, ct, q, qd, lagMul, sr);
			sr++;
		}
	}
	
	foreach(xPointMass* xpm, xmbd->Masses())
	{
		xpm->setNewPositionData(q);
		xpm->setNewVelocityData(qd);
	}		
	return pt;
}

int xMultiBodySimulation::Initialize(xMultiBodyModel* _xmbd)
{
	//of = new std::fstream;
	xmbd = _xmbd;
	int nm = xmbd->NumMass();
	int nr = 0;
	mdim = nm * xModel::OneDOF();
	sdim = nm;
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
	foreach(xPointMass* xpm, xmbd->Masses())
	{
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
		xpm->AllocResultMomory(xSimulation::npart);
	}
	foreach(xKinematicConstraint* xkc, xmbd->Joints())
	{
		sdim += xkc->NumConst();
		std::string bn = xkc->BaseBodyName();
		std::string an = xkc->ActionBodyName();
		int base_idx = bn == "ground" ? 0 : xmbd->XMass(bn)->xpmIndex();
		int action_idx = an == "ground" ? 0 : xmbd->XMass(an)->xpmIndex();
		xkc->setBaseBodyIndex(base_idx);
		xkc->setActionBodyIndex(action_idx);
		xkc->AllocResultMemory(xSimulation::npart);
		//xkc->set
	}
	foreach(xDrivingConstraint* xdc, xmbd->Drivings())
	{
		xdc->define(q);
		sdim++;
	}
	foreach(xForce* xf, xmbd->Forces())
	{
		std::string bn = xf->BaseBodyName();
		std::string an = xf->ActionBodyName();
		int base_idx = bn == "ground" ? 0 : xmbd->XMass(bn)->xpmIndex();
		int action_idx = an == "ground" ? 0 : xmbd->XMass(an)->xpmIndex();
		xf->setBaseBodyIndex(base_idx);
		xf->setActionBodyIndex(action_idx);
		//xf->setBaseLocalCoordinate();
	}
	dof = mdim - sdim;
	std::stringstream wss;
	wss << dof;
	if (dof < 0)
	{
		xLog::log(std::string("There are ") + wss.str() + std::string("redundant constraint equations"));
		return xDynamicsError::xdynamicsErrorMultiBodyModelRedundantCondition;
	}
// 	else if (dof == 0)
// 	{
// // 		tdim = sdim;
// // 		lhs.alloc(sdim, sdim);// new_matrix(tdim, tdim);
// // 		cjaco.alloc(sdim, sdim);
// // 		rhs.alloc(sdim);// = new double[tdim];
// 	}
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
	foreach(xPointMass* xpm, xmbd->Masses())
	{
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
	}
}

void xMultiBodySimulation::ConstructContraintJacobian()
{
	unsigned int sr = 0;
	unsigned int sc = 0;
	cjaco.zeros();
	foreach(xKinematicConstraint* xkc, xmbd->Joints())
	{
		xkc->ConstraintJacobian(cjaco, q, qd, sr);
		sr += xkc->NumConst();
	}
	foreach(xDrivingConstraint* xdc, xmbd->Drivings())
	{
		xdc->ConstraintJacobian(cjaco, q, qd, sr, xSimulation::ctime);
		sr++;
	}
	unsigned int id = 0;
	foreach(xPointMass* xpm, xmbd->Masses())
	{
		id = xpm->xpmIndex() * xModel::OneDOF();
		sc = (xpm->xpmIndex() - 1) * xModel::OneDOF();
		euler_parameters e = 2.0 * xpm->EulerParameters();// new_vector4d(q(id + 3), q(id + 4), q(id + 5), q(id + 6));
		cjaco.insert(sr++, sc + 3, new_vector4d(e.e0, e.e1, e.e2, e.e3));
	}
	
// 	xKinematicConstraint* xkc = xmbd->BeginKinematicConstraint();
// 	while (xkc != xmbd->EndKinematicConstraint())
// 	{
// 		xkc->ConstraintJacobian(cjaco, q, qd, sr);
// 		sr += xkc->NumConst();
// 		xkc = xmbd->NextKinematicConstraint();
// 	}
// 	xPointMass* xpm = xmbd->BeginPointMass();
// 	unsigned int id = 0;
// 	while (xpm != xmbd->EndPointMass())
// 	{
// 		id = xpm->xpmIndex() * xModel::OneDOF();
// 		sc = (xpm->xpmIndex() - 1) * xModel::OneDOF();
// 		vector4d e = 2.0 * new_vector4d(q(id + 3), q(id + 4), q(id + 5), q(id + 6));
// 		cjaco.insert(sr++, sc + 3, e);
// 		xpm = xmbd->NextPointMass();
// 	}
}

void xMultiBodySimulation::ConstructForceVector(xVectorD& v)
{
	euler_parameters e, ev;
	vector3d f;
	vector4d m;
	//xPointMass* xpm = xmbd->BeginPointMass();
	unsigned int j = 0;
	/*if (dem_tsda)
	{
		for (unsigned int i = 0; i < dem_tsda->NumSpringDamperBodyConnection(); i++)
		{
			xSpringDamperBodyConnectionInfo info = dem_tsda->xSpringDamperBodyConnectionInformation()[i];
			xPointMass* pm = xmbd->XMass(info.cbody.toStdString());
			dem_tsda->xCalculateForceFromDEM(i, pm, dem_pos, dem_vel);
		}

	}*/
		
	foreach(xForce* xf, xmbd->Forces())
	{
		xf->xCalculateForce(q, qd);
	}
	foreach(xPointMass* xpm, xmbd->Masses())
	{
	/*	if (xpm->Name() == "wheel")
		{
			xpm->setAxialForce(0, 0, 5.0);
			xpm->setAxialMoment(0.001, 0.001, 0.001);
		}*/
			
	//	i = xpm->xpmIndex() * 7;
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
	}

	
// 	while (xpm != xmbd->EndPointMass())
// 	{
// 		i = xpm->xpmIndex() * 7;
// 		e = new_euler_parameters(q(i + 3), q(i + 4), q(i + 5), q(i + 6));
// 		ev = new_euler_parameters(qd(i + 3), qd(i + 4), qd(i + 5), qd(i + 6));
// 		f = xpm->Mass() * xModel::gravity + xpm->ContactForce() + xpm->AxialForce() + xpm->HydroForce();
// 		m = 2.0 * (xpm->ContactMoment() + xpm->AxialMoment() + xpm->HydroMoment()) * GMatrix(e);
// 		m += CalculateInertiaForce(ev, xpm->Inertia(), e);
// 		v(j + 0) = f.x; v(j + 1) = f.y; v(j + 2) = f.z;
// 		v(j + 3) = m.x; v(j + 4) = m.y; v(j + 5) = m.z; v(j + 6) = m.w;
// 		i += xModel::OneDOF();
// 		j += xModel::OneDOF();
// 		xpm = xmbd->NextPointMass();
// 	}
}

void xMultiBodySimulation::ConstructConstraintEquation(xVectorD& v, int sr, double mul /*= 1.0*/)
{
	//unsigned int sr = mdim;
	foreach(xKinematicConstraint* xkc, xmbd->Joints())
	{
		xkc->ConstraintEquation(v, q, qd, sr, mul);
		sr += xkc->NumConst();
	}
	
// 	xKinematicConstraint* xkc = xmbd->BeginKinematicConstraint();
// 	while (xkc != xmbd->EndKinematicConstraint())
// 	{
// 		xkc->ConstraintEquation(v, q, qd, sr, mul);
// 		sr += xkc->NumConst();
// 		xkc = xmbd->NextKinematicConstraint();
// 	}
	foreach(xDrivingConstraint* xdc, xmbd->Drivings())
	{
		xdc->ConstraintEquation(v, q, qd, sr, xSimulation::ctime, mul);
		sr++;
	}
	foreach(xPointMass* xpm, xmbd->Masses())
	{
		unsigned int id = xpm->xpmIndex() * xModel::OneDOF();
		euler_parameters e = xpm->EulerParameters();// new_euler_parameters(q(id + 3), q(id + 4), q(id + 5), q(id + 6));
		v(sr++) = mul * (dot(e, e) - 1.0);
	}
	
// 	xPointMass* xpm = xmbd->BeginPointMass();
// 	while (xpm != xmbd->EndPointMass())
// 	{
// 		unsigned int id = xpm->xpmIndex() * xModel::OneDOF();
// 		euler_parameters e = new_euler_parameters(q(id + 3), q(id + 4), q(id + 5), q(id + 6));
// 		v(sr++) = mul * (dot(e, e) - 1.0);
// 		xpm = xmbd->NextPointMass();
// 	}
}

void xMultiBodySimulation::ConstructConstraintDerivative(xVectorD& v, int sr, double mul /*= 1.0*/)
{
	//v.zeros();
	foreach(xKinematicConstraint* xkc, xmbd->Joints())
	{
		sr += xkc->NumConst();
	}
	foreach(xDrivingConstraint* xdc, xmbd->Drivings())
	{
		xdc->ConstraintDerivative(v, q, qd, sr, xSimulation::ctime, mul);
		sr++;
	}
	foreach(xPointMass* xpm, xmbd->Masses())
	{
		unsigned int id = xpm->xpmIndex() * xModel::OneDOF();
		euler_parameters e = xpm->EulerParameters();// new_euler_parameters(q(id + 3), q(id + 4), q(id + 5), q(id + 6));
		v(sr++) = mul * (dot(e, e) - 1.0);
	}
}

void xMultiBodySimulation::ConstructConstraintGamma(xVectorD& v, int sr /*= 0.0*/, double mul /*= 1.0*/)
{
	//v.zeros();
	foreach(xKinematicConstraint* xkc, xmbd->Joints())
	{
		xkc->GammaFunction(v, q, qd, sr, mul);
		sr += xkc->NumConst();
	}
	foreach(xDrivingConstraint* xdc, xmbd->Drivings())
	{
		xdc->ConstraintGamma(v, q, qd, sr, xSimulation::ctime, mul);
		sr++;
	}
	foreach(xPointMass* xpm, xmbd->Masses())
	{
		unsigned int id = xpm->xpmIndex() * xModel::OneDOF();
		euler_parameters e = xpm->DEulerParameters();// new_euler_parameters(qd(id + 3), qd(id + 4), qd(id + 5), qd(id + 6));
		v(sr++) = mul * 2.0 * dot(e, e);
	}
}

vector4d xMultiBodySimulation::CalculateInertiaForce(const euler_parameters& ev, const matrix33d& J, const euler_parameters& ep)
{
	matrix34d Gd = GMatrix(ev);
	return 8.0 * Gd * (J * (Gd * ep));
// 	double GvP0 = -ev.e1*ep.e0 + ev.e0*ep.e1 + ev.e3*ep.e2 - ev.e2*ep.e3;
// 	double GvP1 = -ev.e2*ep.e0 - ev.e3*ep.e1 + ev.e0*ep.e2 + ev.e1*ep.e3;
// 	double GvP2 = -ev.e3*ep.e0 + ev.e2*ep.e1 - ev.e1*ep.e2 + ev.e0*ep.e3;
// 	return vector4d
// 	{
// 		8 * (-ev.e1*J.a00*GvP0 - ev.e2*J.a11*GvP1 - ev.e3*J.a22*GvP2),
// 		8 * (ev.e0*J.a00*GvP0 - ev.e3*J.a11*GvP1 + ev.e2*J.a22*GvP2),
// 		8 * (ev.e3*J.a00*GvP0 + ev.e0*J.a11*GvP1 - ev.e1*J.a22*GvP2),
// 		8 * (-ev.e2*J.a00*GvP0 + ev.e1*J.a11*GvP1 + ev.e0*J.a22*GvP2) 
// 	};
}
