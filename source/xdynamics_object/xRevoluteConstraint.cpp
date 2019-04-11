//#include "stdafx.h"
#include "xdynamics_object/xRevoluteConstraint.h"
#include "xdynamics_object/xPointMass.h"

xRevoluteConstraint::xRevoluteConstraint()
	: xKinematicConstraint()
{

}

xRevoluteConstraint::xRevoluteConstraint(std::string _name, std::string _i, std::string _j)
	: xKinematicConstraint(_name, xKinematicConstraint::REVOLUTE, _i, _j)
{

}

xRevoluteConstraint::~xRevoluteConstraint()
{

}

void xRevoluteConstraint::ConstraintEquation(xVectorD& ce, xVectorD& q, xVectorD& dq, unsigned int sr, double mul)
{
// 	unsigned int idx = (i - 1) * xModel::OneDOF();
// 	unsigned int jdx = (j - 1) * xModel::OneDOF();
// 	unsigned int si = i * xModel::OneDOF();
// 	unsigned int sj = j * xModel::OneDOF();
	vector3d ri = i_ptr->Position();// new_vector3d(q(si + 0), q(si + 1), q(si + 2));
	vector3d rj = j_ptr->Position();// new_vector3d(q(sj + 0), q(sj + 1), q(sj + 2));
	//euler_parameters ei = new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
//	euler_parameters ej = new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
//	matrix33d Ai = GlobalTransformationMatrix(ei);
	//matrix33d Aj = GlobalTransformationMatrix(ej);
	matrix33d Ai = i_ptr->TransformationMatrix();// GlobalTransformationMatrix(ei);
	matrix33d Aj = j_ptr->TransformationMatrix();// GlobalTransformationMatrix(ej);
	vector3d v3 = mul * spherical_constraintEquation(ri, rj, Ai * spi, Aj * spj);
	//v3 = (rj + Aj * spj) - (ri + Ai * spi);
	ce(sr + 0) = v3.x;
	ce(sr + 1) = v3.y;
	ce(sr + 2) = v3.z;
	v3 = j_ptr->toGlobal(hj);// Aj * hj;
	ce(sr + 3) = mul * dot_1_constraintEquation(v3, Ai * fi);//dot(v3, Ai * fi);
	ce(sr + 4) = mul * dot_1_constraintEquation(v3, Ai * gi);
}

void xRevoluteConstraint::ConstraintJacobian(xSparseD& lhs, xVectorD& q, xVectorD& qd, unsigned int sr)
{
	unsigned int sc = 0;
// 	unsigned int si = i * xModel::OneDOF();
// 	unsigned int sj = j * xModel::OneDOF();
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	matrix33d Ai = i_ptr->TransformationMatrix();// GlobalTransformationMatrix(ei);
	matrix33d Aj = j_ptr->TransformationMatrix();// GlobalTransformationMatrix(ej);
	if (i)
	{
		//idx = i * xModel::OneDOF();
		sc = (i - 1) * xModel::OneDOF();
		lhs(sr + 0, sc + 0) = lhs(sr + 1, sc + 1) = lhs(sr + 2, sc + 2) = -1;
		//e = new_euler_parameters(q(idx + 3), q(idx + 4), q(idx + 5), q(idx + 6));
		//A = GlobalTransformationMatrix(e);
		lhs.insert(sr + 0, sc + 3, spherical_constraintJacobian_e(ei, -spi));
		lhs.insert(sr + 3, sc + 3, dot_1_constraintJacobian(Aj * hj, fi, ei));
		lhs.insert(sr + 4, sc + 3, dot_1_constraintJacobian(Aj * hj, gi, ei));
	}
	if (j)
	{
		//idx = j * xModel::OneDOF();
		sc = (j - 1) * xModel::OneDOF();
		lhs(sr + 0, sc + 0) = lhs(sr + 1, sc + 1) = lhs(sr + 2, sc + 2) = 1;
		//e = new_euler_parameters(q(idx + 3), q(idx + 4), q(idx + 5), q(idx + 6));
		//A = GlobalTransformationMatrix(e);
		lhs.insert(sr + 0, sc + 3, spherical_constraintJacobian_e(ej, spj));
		lhs.insert(sr + 3, sc + 3, dot_1_constraintJacobian(Ai * fi, hj, ej));// *BMatrix(e, hj));
		lhs.insert(sr + 4, sc + 3, dot_1_constraintJacobian(Ai * gi, hj, ej));// *BMatrix(e, hj));
	}
}

void xRevoluteConstraint::DerivateJacobian(xMatrixD& lhs, xVectorD& q, xVectorD& qd, double* lm, unsigned int sr, double mul)
{
// 	unsigned int si = i * xModel::OneDOF();
// 	unsigned int sj = j * xModel::OneDOF();
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	
	vector3d L = new_vector3d(lm[0], lm[1], lm[2]);
	spherical_differentialJacobian(lhs, mul * L);
	dot_1_differentialJacobian(lhs, fi, hj, ei, ej, mul * lm[3]);
	dot_1_differentialJacobian(lhs, gi, hj, ei, ej, mul * lm[4]);
}

void xRevoluteConstraint::SaveStepResult(
	unsigned int part, double ct, xVectorD& q, xVectorD& qd, double* L, unsigned int sr)
{
	unsigned int sc = 0;
// 	unsigned int si = i * xModel::OneDOF();
// 	unsigned int sj = j * xModel::OneDOF();
	vector3d ri = i_ptr->Position();// new_vector3d(q(si + 0), q(si + 1), q(si + 2));
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	matrix33d Ai = i_ptr->TransformationMatrix();// GlobalTransformationMatrix(ei);
	matrix33d Aj = j_ptr->TransformationMatrix();// GlobalTransformationMatrix(ej);
	kinematicConstraint_result kcr = { 0, };
	double* lm = L + sr;
	vector3d L3 = new_vector3d(lm[0], lm[1], lm[2]);
	vector4d rf;
	matrix34d sce;
	vector4d r0, r1;
	if (i)
	{
		kcr.iaforce = L3;
		sce = spherical_constraintJacobian_e(ei, spi);
		r0 = -dot_1_constraintJacobian(Aj * hj, fi, ei);
		r1 = -dot_1_constraintJacobian(Aj * hj, gi, ei);
		rf = sce * L3 + lm[3] * r0 + lm[4] * r1;
		kcr.irforce = 0.5 * LMatrix(ei) * rf;
	}
	if (j)
	{
		kcr.jaforce = -L3;
		sce = spherical_constraintJacobian_e(ej, -spj);
		r0 = -dot_1_constraintJacobian(Ai * fi, hj, ej);
		r1 = -dot_1_constraintJacobian(Ai * gi, hj, ej);
		rf = sce * L3 + lm[3] * r0 + lm[4] * r1;
		kcr.jrforce = 0.5 * LMatrix(ej) * rf;
	}
	kcr.time = ct;
	kcr.location = ri + Ai * spi;
	kcrs[part] = kcr;
	nr_part++;
}

void xRevoluteConstraint::GammaFunction(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double mul)
{
// 	unsigned int si = i * xModel::OneDOF();
// 	unsigned int sj = j * xModel::OneDOF();
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	euler_parameters dei = i_ptr->DEulerParameters();// new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
	euler_parameters dej = j_ptr->DEulerParameters();// new_euler_parameters(qd(sj + 3), qd(sj + 4), qd(sj + 5), qd(sj + 6));
// 	matrix33d Ai = GlobalTransformationMatrix(ei);
// 	matrix33d Aj = GlobalTransformationMatrix(ej);
	vector3d v3 = mul * spherical_gamma(dei, dej, spi, spj);
	rhs(sr + 0) = v3.x;
	rhs(sr + 1) = v3.y;
	rhs(sr + 2) = v3.z;
	rhs(sr + 3) = mul * dot_1_gamma(ei, ej, hj, fi, dei, dej);
	rhs(sr + 4) = mul * dot_1_gamma(ei, ej, hj, gi, dei, dej);
}