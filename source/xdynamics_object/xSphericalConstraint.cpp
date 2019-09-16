//#include "stdafx.h"
#include "xdynamics_object/xSphericalConstraint.h"

xSphericalConstraint::xSphericalConstraint()
	: xKinematicConstraint()
{

}

xSphericalConstraint::xSphericalConstraint(std::string _name, std::string _i, std::string _j)
	: xKinematicConstraint(_name, xKinematicConstraint::SPHERICAL, _i, _j)
{

}

xSphericalConstraint::~xSphericalConstraint()
{

}

void xSphericalConstraint::ConstraintEquation(xVectorD& ce, xVectorD& q, xVectorD& dq, unsigned int sr, double mul)
{
	unsigned int si = i * xModel::OneDOF();
	unsigned int sj = j * xModel::OneDOF();
	vector3d v3 = mul * spherical_constraintEquation(
		i_ptr->Position(), j_ptr->Position(), 
		i_ptr->toGlobal(spi)/*Ai * spi*/, j_ptr->toGlobal(spj)/*Aj * spj*/);
	ce(sr + 0) = v3.x;
	ce(sr + 1) = v3.y;
	ce(sr + 2) = v3.z;
}

void xSphericalConstraint::ConstraintJacobian(xSparseD& lhs, xVectorD& q, xVectorD& qd, unsigned int sr)
{
	unsigned int sc = 0;
	unsigned int idx = 0;
	euler_parameters e;
//	matrix33d A;
	if (i)
	{
		//idx = i * xModel::OneDOF();
		sc = (i - 1) * xModel::OneDOF();
		lhs(sr + 0, sc + 0) = lhs(sr + 1, sc + 1) = lhs(sr + 2, sc + 2) = -1;
		e = i_ptr->EulerParameters();// new_euler_parameters(q(idx + 3), q(idx + 4), q(idx + 5), q(idx + 6));
		//A = GlobalTransformationMatrix(e);
		lhs.insert(sr + 0, sc + 3, spherical_constraintJacobian_e(e, -spi));
	}
	if (j)
	{
		//idx = j * xModel::OneDOF();
		sc = (j - 1) * xModel::OneDOF();
		lhs(sr + 0, sc + 0) = lhs(sr + 1, sc + 1) = lhs(sr + 2, sc + 2) = 1;
		e = j_ptr->EulerParameters();// q(idx + 3), q(idx + 4), q(idx + 5), q(idx + 6));
	//	A = GlobalTransformationMatrix(e);
		lhs.insert(sr + 0, sc + 3, spherical_constraintJacobian_e(e, spj));
	}
}

void xSphericalConstraint::DerivateJacobian(xMatrixD& lhs, xVectorD& q, xVectorD& qd, double* lm, unsigned int sr, double mul)
{
	//unsigned int si = i * xModel::OneDOF();
	//unsigned int sj = j * xModel::OneDOF();
	//euler_parameters ei = new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	//euler_parameters ej = new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));

	vector3d L = new_vector3d(lm[0], lm[1], lm[2]);
	spherical_differentialJacobian(lhs, mul * L);
}

xKinematicConstraint::kinematicConstraint_result xSphericalConstraint::GetStepResult(
	unsigned int part, xVectorD& q, xVectorD& qd, double* L, unsigned int sr)
{
	unsigned int sc = 0;
// 	unsigned int si = i * xModel::OneDOF();
// 	unsigned int sj = j * xModel::OneDOF();
	vector3d ri = i_ptr->Position();// new_vector3d(q(si + 0), q(si + 1), q(si + 2));
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	//matrix33d Ai = GlobalTransformationMatrix(ei);
	//matrix33d Aj = GlobalTransformationMatrix(ej);
	kinematicConstraint_result kcr = { 0, };
	double* lm = L + sr;
	vector3d L3 = new_vector3d(lm[0], lm[1], lm[2]);
	vector4d rf;
	matrix34d sce;
	if (i)
	{
		kcr.iaforce = L3;
		sce = spherical_constraintJacobian_e(ei, spi);
		rf = sce * L3;
		kcr.irforce = 0.5 * LMatrix(ei) * rf;
	}
	if (j)
	{
		kcr.jaforce = -L3;
		sce = spherical_constraintJacobian_e(ej, -spj);
		rf = sce * L3;
		kcr.jrforce = 0.5 * LMatrix(ej) * rf;
	}
	kcr.location = ri + i_ptr->toGlobal(spi);// Ai * spi;
//	kcrs.push_back(kcr);
	
	nr_part++;
	return kcr;
}

void xSphericalConstraint::GammaFunction(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double mul)
{
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	euler_parameters dei = i_ptr->DEulerParameters();// new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
	euler_parameters dej = j_ptr->DEulerParameters();// new_euler_parameters(qd(sj + 3), qd(sj + 4), qd(sj + 5), qd(sj + 6));
	vector3d v3 = mul * spherical_gamma(dei, dej, spi, spj);
	rhs(sr + 0) = v3.x;
	rhs(sr + 1) = v3.y;
	rhs(sr + 2) = v3.z;
}

